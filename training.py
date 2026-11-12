"""Models, flow-matching helpers and the training loop shared by the examples."""

import math

import jax
import jax.numpy as jnp
import optax


def layer(key, *shape):
    return jax.random.normal(key, shape) / math.sqrt(math.prod(shape[:-1])), jnp.zeros(
        shape[-1]
    )


def conv(h, p, stride=1):
    dims = ("NHWC", "HWIO", "NHWC")
    return (
        jax.lax.conv_general_dilated(
            h, p[0], (stride, stride), "SAME", dimension_numbers=dims
        )
        + p[1]
    )


def group_norm(h, groups=8):
    g = h.reshape(*h.shape[:3], groups, -1)
    g = (g - g.mean((1, 2, 4), keepdims=True)) / jnp.sqrt(
        g.var((1, 2, 4), keepdims=True) + 1e-5
    )
    return g.reshape(h.shape)


def mlp_init(key, sizes=(2, 64, 64, 1)):
    return [
        layer(k, m, n)
        for k, m, n in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1], sizes[1:])
    ]


def mlp(p, x, t):
    h = jnp.stack([x, t], -1)
    for W, b in p[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = p[-1]
    return (h @ W + b)[..., 0]


def block_init(key, c):
    k = jax.random.split(key, 3)
    return {
        "c1": layer(k[0], 3, 3, c, c),
        "c2": layer(k[1], 3, 3, c, c),
        "t": layer(k[2], 256, c),
    }


def block(p, h, e):
    W, b = p["t"]
    r = conv(jax.nn.silu(group_norm(h)), p["c1"]) + (e @ W + b)[:, None, None]
    return h + conv(jax.nn.silu(group_norm(r)), p["c2"])


def unet_init(key, labels=10):
    k = iter(jax.random.split(key, 14))
    c = lambda cin, cout: layer(next(k), 3, 3, cin, cout)
    return {
        "time": layer(next(k), 64, 256),
        "in": c(1, 32),
        "b1": block_init(next(k), 32),
        "down1": c(32, 64),
        "b2": block_init(next(k), 64),
        "down2": c(64, 128),
        "m1": block_init(next(k), 128),
        "m2": block_init(next(k), 128),
        "up2": c(128, 64),
        "b3": block_init(next(k), 64),
        "up1": c(64, 32),
        "b4": block_init(next(k), 32),
        "out": c(32, 1),
        "label": layer(next(k), 128, labels),
    }


def unet(p, x, t):
    freqs = jnp.exp(jnp.linspace(0, jnp.log(1000), 32))
    W, b = p["time"]
    e = jax.nn.silu(
        jnp.concatenate([jnp.sin(t[:, None] * freqs), jnp.cos(t[:, None] * freqs)], 1)
        @ W
        + b
    )
    h1 = block(p["b1"], conv(x[..., None], p["in"]), e)
    h2 = block(p["b2"], conv(h1, p["down1"], 2), e)
    h = block(p["m2"], block(p["m1"], conv(h2, p["down2"], 2), e), e)
    features = jax.nn.silu(group_norm(h)).mean((1, 2))
    h = block(p["b3"], conv(h, p["up2"]).repeat(2, 1).repeat(2, 2) + h2, e)
    h = block(p["b4"], conv(h, p["up1"]).repeat(2, 1).repeat(2, 2) + h1, e)
    W, b = p["label"]
    return (
        conv(jax.nn.silu(group_norm(h)), p["out"])[..., 0],
        features,
        features @ W + b,
    )


def cnn_init(key, labels=10):
    k = jax.random.split(key, 4)
    return [
        layer(k[0], 3, 3, 1, 32),
        layer(k[1], 3, 3, 32, 64),
        layer(k[2], 3, 3, 64, 128),
        layer(k[3], 128, labels),
    ]


def cnn(p, x):
    h = x[..., None]
    for c in p[:-1]:
        h = jax.nn.silu(conv(h, c, 2))
    W, b = p[-1]
    return h.mean((1, 2)) @ W + b


def classifier_init(key, din, d=64):
    k = iter(jax.random.split(key, 12))
    attention = lambda: [layer(next(k), d, d) for _ in range(3)] + [
        layer(next(k), d, 2 * d),
        layer(next(k), 2 * d, d),
    ]
    return {
        "in": layer(next(k), din, d),
        "blocks": [attention(), attention()],
        "out": layer(next(k), d, 1),
    }


def classifier(p, e, mask):
    norm = lambda h: (h - h.mean(-1, keepdims=True)) / jnp.sqrt(
        h.var(-1, keepdims=True) + 1e-5
    )
    W, b = p["in"]
    e = e @ W + b
    others = mask[:, None, :] & ~jnp.eye(mask.shape[1], dtype=bool)
    for (Wq, bq), (Wk, bk), (Wv, bv), (W1, b1), (W2, b2) in p["blocks"]:
        h = norm(e)
        q, k, v = h @ Wq + bq, h @ Wk + bk, h @ Wv + bv
        e = (
            e
            + jax.nn.softmax(
                jnp.where(others, q @ k.swapaxes(1, 2) / math.sqrt(e.shape[-1]), -1e9),
                -1,
            )
            @ v
        )
        e = e + jax.nn.silu(norm(e) @ W1 + b1) @ W2 + b2
    W, b = p["out"]
    return ((e * mask[..., None]).sum(1) @ W + b)[:, 0]


def interpolate(key, x1):
    k0, kt = jax.random.split(key)
    x0, t = jax.random.normal(k0, x1.shape), jax.random.uniform(kt, (len(x1),))
    tt = t.reshape(-1, *[1] * (x1.ndim - 1))
    return x0, (1 - tt) * x0 + tt * x1, t


def integrate(velocity, x, steps=100):
    for i in range(steps):
        x = x + velocity(x, (i + 0.5) / steps) / steps
    return x


def train(loss, params, data, steps, key, lr=1e-3, val=None, every=500, patience=10):
    optimizer = optax.adam(optax.linear_schedule(0.0, lr, 1000))

    @jax.jit
    def step(params, state, avg, key, i, data):
        updates, state = optimizer.update(jax.grad(loss)(params, key, *data), state)
        params = optax.apply_updates(params, updates)
        d = jnp.minimum(0.999, (1 + i) / (10 + i))
        return (
            params,
            state,
            jax.tree.map(lambda a, p: d * a + (1 - d) * p, avg, params),
        )

    state, avg = optimizer.init(params), params
    best, best_loss, since = params, float("inf"), 0
    for i, k in enumerate(jax.random.split(key, steps)):
        params, state, avg = step(params, state, avg, k, i, data)
        if val is not None and (i + 1) % every == 0:
            v = float(val(avg))
            best, best_loss, since = (
                (avg, v, 0) if v < best_loss else (best, best_loss, since + 1)
            )
            if since == patience:
                break
    return avg if val is None else best
