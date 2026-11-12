"""1D example: flow from N(0, 1) to a mixture of two Gaussians, then guided pairs that come from different components."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from training import integrate, interpolate, mlp, mlp_init, train

MU, SIGMA = 4.0, 1.0


def sample_target(key, n):
    k1, k2 = jax.random.split(key)
    return jnp.where(
        jax.random.bernoulli(k1, 0.5, (n,)), MU, -MU
    ) + SIGMA * jax.random.normal(k2, (n,))


def flow_loss(p, key):
    k1, k2 = jax.random.split(key)
    x1 = sample_target(k1, 1024)
    x0, xt, t = interpolate(k2, x1)
    return jnp.mean((mlp(p, xt, t) - (x1 - x0)) ** 2)


def classifier(x, t):
    var = (1 - t) ** 2 + (t * SIGMA) ** 2
    u = 2 * t * MU * x / var
    log_right, log_left = -jnp.logaddexp(0.0, -u), -jnp.logaddexp(0.0, u)
    return jnp.logaddexp(
        log_right[:, 0] + log_left[:, 1], log_left[:, 0] + log_right[:, 1]
    )


if __name__ == "__main__":
    k_init, k_train, k_pairs = jax.random.split(jax.random.key(0), 3)
    params = train(flow_loss, mlp_init(k_init), (), 5000, k_train)

    @jax.jit
    def velocity(x, t, scale):
        grad_log_p = jax.grad(lambda x: classifier(x, t).sum())(x)
        return mlp(params, x, jnp.full(x.shape, t)) + scale * (1 - t) / t * grad_log_p

    z = jax.random.normal(k_pairs, (5000, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, scale in zip(axes, [0.0, 1.0]):
        x = integrate(lambda x, t: velocity(x, t, scale), z)
        split = jnp.mean(x[:, 0] * x[:, 1] < 0)
        ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.3)

    fig.savefig("gaussian.png")
