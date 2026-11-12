"""MNIST example: generate sets of 5 distinct digits with a joint flow + set classifier, on a grid of data sizes."""

import gzip
import math
import os
import time
import urllib.request

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from training import (
    cnn,
    cnn_init,
    integrate,
    interpolate,
    classifier,
    classifier_init,
    train,
    unet,
    unet_init,
)

SIZES = range(2, 11)
N_EVAL = 1000
LAMBDA = 0.1
LEVELS = (jnp.arange(20) + 0.5) / 20
DRAWS = 4
SETS = 256
COPIES = 4


def load_data():
    if not os.path.exists("data.npz"):
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        get = lambda f: gzip.decompress(urllib.request.urlopen(url + f).read())
        x = np.frombuffer(get("train-images-idx3-ubyte.gz")[16:], np.uint8).reshape(
            -1, 28, 28
        )
        y = np.frombuffer(get("train-labels-idx1-ubyte.gz")[8:], np.uint8)
        np.savez("data.npz", x=x, y=y)
    d = np.load("data.npz")
    return jnp.array(d["x"] / 127.5 - 1, jnp.float32), jnp.array(d["y"], jnp.int32)


def all_distinct(labels, mask):
    counts = ((labels[..., None] == jnp.arange(10)) & mask[..., None]).sum(1)
    return (counts.max(1) <= 1).astype(jnp.float32)


def log_validrate(n):
    pi = math.perm(10, n) / 10**n
    return math.log(pi / (1 - pi))


def labelled_sets(key, labels, count):
    per = count // (2 * len(SIZES)) + 1
    idx, mask, valid = [], [], []
    for n in SIZES:
        for want in (1.0, 0.0):
            found = []
            while sum(map(len, found)) < per:
                key, sub = jax.random.split(key)
                c = jax.random.randint(sub, (1_000_000, n), 0, len(labels))
                found.append(c[all_distinct(labels[c], jnp.ones((1, n), bool)) == want])
            idx.append(jnp.pad(jnp.concatenate(found)[:per], ((0, 0), (0, 10 - n))))
            mask.append(jnp.broadcast_to(jnp.arange(10) < n, (per, 10)))
            valid.append(jnp.full(per, want))
    interleave = lambda a: jnp.stack(a, 1).reshape(-1, *a[0].shape[1:])
    return interleave(idx), interleave(mask), interleave(valid)


def joint_objective(p, x0, xt, t, x1, y):
    v, _, logits = unet(p, xt, t)
    cross_entropy = jnp.mean(
        jax.nn.logsumexp(logits, 1) - logits[jnp.arange(len(y)), y]
    )
    return jnp.mean((v - (x1 - x0)) ** 2) + LAMBDA * cross_entropy


def joint_loss(p, key, X, Y):
    kb, kn = jax.random.split(key)
    i = jax.random.randint(kb, (128,), 0, len(X))
    return joint_objective(p, *interpolate(kn, X[i]), X[i], Y[i])


def judge_loss(p, key, X, Y):
    i = jax.random.randint(key, (128,), 0, len(X))
    logits = cnn(p, X[i])
    return jnp.mean(jax.nn.logsumexp(logits, 1) - logits[jnp.arange(128), Y[i]])


features = jax.jit(lambda p, x, t: unet(p, x, t)[1])


def feature_precompute(p, X, key):
    keys = jax.random.split(key, len(LEVELS) * DRAWS)
    cache = []
    for j, t in enumerate(LEVELS):
        row = []
        for d in range(DRAWS):
            xt = (1 - t) * jax.random.normal(keys[j * DRAWS + d], X.shape) + t * X
            f = jnp.concatenate(
                [
                    features(p, xt[i : i + 2000], jnp.full(len(xt[i : i + 2000]), t))
                    for i in range(0, len(X), 2000)
                ]
            )
            row.append(jnp.concatenate([f, jnp.full((len(X), 1), t)], 1))
        cache.append(jnp.stack(row))
    return jnp.stack(cache)


@jax.jit
def set_val_loss(p, f, mask, valid):
    logit = classifier(p, f, mask)
    return jnp.mean(jnp.logaddexp(0.0, logit) - valid * logit)


def set_loss(p, key, F, idx, mask, valid):
    kb, kl, kd = jax.random.split(key, 3)
    i = jax.random.randint(kb, (SETS,), 0, len(idx)).repeat(COPIES)
    level = jax.random.randint(kl, (len(i), 1), 0, len(LEVELS))
    draw = jax.random.randint(kd, (len(i), 1), 0, DRAWS)
    return set_val_loss(p, F[level, draw, idx[i]], mask[i], valid[i])


@jax.jit
def velocity(unet_p, head_p, b, scale, x, t):
    S, n = x.shape[:2]
    flat = lambda x: x.reshape(S * n, 28, 28)

    def log_p(x):
        f = unet(unet_p, flat(x), jnp.full(S * n, t))[1]
        tokens = jnp.concatenate([f, jnp.full((S * n, 1), t)], 1).reshape(S, n, -1)
        return -jnp.logaddexp(
            0.0, -(classifier(head_p, tokens, jnp.ones((S, n), bool)) + b)
        ).sum()

    v = unet(unet_p, flat(x), jnp.full(S * n, t))[0].reshape(x.shape)
    return v + scale * (1 - t) / t * jax.grad(log_p)(x)


if __name__ == "__main__":
    SIZE = 5
    start = time.time()
    X, Y = load_data()
    U_SIZES, S_SIZES = [100, 1000, 10_000, 50_000], [10, 100, 1000, 10_000, 100_000]
    pool, pool_y = X[50_000:], Y[50_000:]
    keys = iter(jax.random.split(jax.random.key(0), 100))

    judge = train(
        judge_loss, cnn_init(next(keys)), (X[:50_000], Y[:50_000]), 3000, next(keys)
    )
    judged = jax.jit(lambda x: cnn(judge, x).argmax(1))
    idx, mask, valid = labelled_sets(next(keys), pool_y, S_SIZES[-1])
    x0v, xtv, tv = interpolate(next(keys), pool[:2000])
    unet_val = jax.jit(
        lambda p: joint_objective(p, x0v, xtv, tv, pool[:2000], pool_y[:2000])
    )
    x_start = jax.random.normal(next(keys), (N_EVAL, SIZE, 28, 28))

    rates, unguided = np.zeros((len(U_SIZES), len(S_SIZES))), np.zeros(len(U_SIZES))
    for r, U in enumerate(U_SIZES):
        unet_p = train(
            joint_loss,
            unet_init(next(keys)),
            (X[:U], Y[:U]),
            20_000,
            next(keys),
            val=unet_val,
        )
        F = feature_precompute(unet_p, pool, next(keys))

        def valid_rate(head_p, b, scale):
            x = integrate(
                lambda x, t: velocity(unet_p, head_p, b, scale, x, t), x_start
            )
            return float(
                all_distinct(
                    judged(x.reshape(-1, 28, 28)).reshape(N_EVAL, SIZE),
                    jnp.ones((1, SIZE), bool),
                ).mean()
            )

        for c, S in enumerate(S_SIZES):
            n_val = min(max(S // 5, 2), 2000)
            level = jax.random.randint(next(keys), (n_val, 1), 0, len(LEVELS))
            draw = jax.random.randint(next(keys), (n_val, 1), 0, DRAWS)
            f_val = F[level, draw, idx[:n_val]]
            head_p = train(
                set_loss,
                classifier_init(next(keys), 129),
                (F, idx[n_val:S], mask[n_val:S], valid[n_val:S]),
                100_000,
                next(keys),
                val=lambda p: set_val_loss(p, f_val, mask[:n_val], valid[:n_val]),
            )
            rates[r, c] = valid_rate(head_p, log_validrate(SIZE), 1.0)
        unguided[r] = valid_rate(head_p, 0.0, 0.0)
        print(
            f"U={U:6d}  unguided {unguided[r]:.3f}  guided "
            + " ".join(f"{v:.3f}" for v in rates[r])
            + f"   ({time.time() - start:.0f}s)"
        )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    im = ax.imshow(rates, vmin=0, vmax=1, origin="lower")
    for r in range(len(U_SIZES)):
        for c in range(len(S_SIZES)):
            ax.text(
                c,
                r,
                f"{rates[r, c]:.2f}",
                ha="center",
                va="center",
            )
    ax.set_xticks(range(len(S_SIZES)), [f"{s:,}" for s in S_SIZES])
    ax.set_yticks(
        range(len(U_SIZES)), [f"{u:,}  ({g:.2f})" for u, g in zip(U_SIZES, unguided)]
    )
    fig.colorbar(im, label="valid rate")
    fig.tight_layout()
    fig.savefig("images.png", dpi=120)
