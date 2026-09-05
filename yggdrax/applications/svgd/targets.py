"""Toy target distributions with analytic score functions for SVGD tests.

Each target exposes a ``score`` (gradient of log-density) and a ``sample``
(draw exact reference samples), so SVGD output can be compared against ground
truth. Scores are analytic and differentiable.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Target(NamedTuple):
    """A target distribution: a score function and an exact sampler."""

    score: Callable[[Float[Array, "n d"]], Float[Array, "n d"]]
    sample: Callable[[Array, int], Float[Array, "n d"]]
    dim: int


def gaussian(mean: Float[Array, " d"], cov_diag: Float[Array, " d"]) -> Target:
    """Diagonal Gaussian target.

    Args:
        mean: Mean vector, shape ``(d,)``.
        cov_diag: Diagonal of the covariance, shape ``(d,)``.

    Returns:
        A :class:`Target`.
    """
    mean = jnp.asarray(mean)
    cov_diag = jnp.asarray(cov_diag)
    dim = int(mean.shape[0])

    def score(x):
        return -(x - mean) / cov_diag

    def sample(key, n):
        std = jnp.sqrt(cov_diag)
        return mean + std * jax.random.normal(key, (n, dim))

    return Target(score=score, sample=sample, dim=dim)


def gaussian_mixture(
    means: Float[Array, "m d"],
    cov_diag: Float[Array, " d"],
    weights: Float[Array, " m"] | None = None,
) -> Target:
    """Isotropic-per-component Gaussian mixture target.

    Args:
        means: Component means, shape ``(m, d)``.
        cov_diag: Shared diagonal covariance, shape ``(d,)``.
        weights: Mixture weights, shape ``(m,)``; uniform if None.

    Returns:
        A :class:`Target`.
    """
    means = jnp.asarray(means)
    cov_diag = jnp.asarray(cov_diag)
    m, dim = means.shape
    if weights is None:
        weights = jnp.ones(m) / m
    log_w = jnp.log(weights)

    def log_prob_components(x):
        # x: (n, d) -> (n, m) per-component log densities (up to shared const).
        diff = x[:, None, :] - means[None, :, :]
        quad = -0.5 * jnp.sum(diff * diff / cov_diag, axis=-1)
        return quad + log_w

    def score(x):
        return jax.vmap(
            jax.grad(lambda xi: jax.nn.logsumexp(log_prob_components(xi[None, :])[0]))
        )(x)

    def sample(key, n):
        k_comp, k_norm = jax.random.split(key)
        comp = jax.random.choice(k_comp, m, (n,), p=weights)
        std = jnp.sqrt(cov_diag)
        return means[comp] + std * jax.random.normal(k_norm, (n, dim))

    return Target(score=score, sample=sample, dim=dim)


def banana(curvature: float = 0.5, scale: float = 2.0) -> Target:
    """2-D banana (Rosenbrock-style) target with analytic score.

    Density: :math:`x_0 \\sim N(0, scale^2)`,
    :math:`x_1 \\sim N(curvature (x_0^2 - scale^2), 1)`.

    Args:
        curvature: Bending strength of the banana.
        scale: Standard deviation of the first coordinate.

    Returns:
        A 2-D :class:`Target`.
    """

    def score(x):
        x0, x1 = x[:, 0], x[:, 1]
        mu1 = curvature * (x0**2 - scale**2)
        s0 = -x0 / scale**2 + (x1 - mu1) * curvature * 2.0 * x0
        s1 = -(x1 - mu1)
        return jnp.stack([s0, s1], axis=-1)

    def sample(key, n):
        k0, k1 = jax.random.split(key)
        x0 = scale * jax.random.normal(k0, (n,))
        x1 = curvature * (x0**2 - scale**2) + jax.random.normal(k1, (n,))
        return jnp.stack([x0, x1], axis=-1)

    return Target(score=score, sample=sample, dim=2)
