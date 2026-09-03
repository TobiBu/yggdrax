"""RBF kernel and Stein-update terms for SVGD.

Stein Variational Gradient Descent (Liu & Wang 2016) transports a set of
particles :math:`\\{x_i\\}` toward a target density :math:`p` by repeatedly
applying the empirical Stein update

.. math::

    \\phi(x_i) = \\frac{1}{N} \\sum_j \\big[\\, k(x_j, x_i)\\, \\nabla_{x_j}\\log p(x_j)
                 + \\nabla_{x_j} k(x_j, x_i) \\,\\big],

with the RBF kernel :math:`k(x, y) = \\exp(-\\lVert x-y\\rVert^2 / 2h^2)`. The
first (attractive) term pulls particles toward high-density regions; the second
(repulsive) term, :math:`\\nabla_{x_j} k(x_j, x_i) = k(x_j, x_i)(x_i-x_j)/h^2`,
spreads them out. Everything here is smooth in the particle positions and in
the bandwidth ``h``, which is what makes bandwidth learning (backpropagating a
validation loss through the sampler) possible.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def rbf_kernel(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
    h: float | Float[Array, ""],
) -> Float[Array, "..."]:
    """RBF kernel ``exp(-||x - y||^2 / (2 h^2))`` over the last axis.

    Args:
        x: Points, shape ``(..., d)``.
        y: Points broadcastable with ``x``, shape ``(..., d)``.
        h: Kernel bandwidth (scalar).

    Returns:
        Kernel values with the broadcast leading shape.
    """
    d2 = jnp.sum((x - y) ** 2, axis=-1)
    return jnp.exp(-d2 / (2.0 * h**2))


def median_heuristic(particles: Float[Array, "n d"]) -> Float[Array, ""]:
    """Median-heuristic bandwidth ``h = sqrt(0.5 * med / log(n+1))``.

    The standard SVGD default: ``med`` is the median squared pairwise distance.

    Args:
        particles: Particle positions, shape ``(n, d)``.

    Returns:
        Scalar bandwidth.
    """
    n = particles.shape[0]
    diff = particles[:, None, :] - particles[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    med = jnp.median(d2)
    return jnp.sqrt(0.5 * med / jnp.log(n + 1.0))


def stein_pair_terms(
    x_target: Float[Array, "... d"],
    x_source: Float[Array, "... d"],
    score_source: Float[Array, "... d"],
    h: float | Float[Array, ""],
) -> Float[Array, "... d"]:
    """Per-pair Stein contribution of ``x_source`` to the update at ``x_target``.

    Returns ``k(x_source, x_target) * score_source
    + grad_{x_source} k(x_source, x_target)`` (before the ``1/N`` average).

    Args:
        x_target: Target points ``x_i``, shape ``(..., d)``.
        x_source: Source points ``x_j``, shape ``(..., d)``.
        score_source: Target-density score at the source points, ``(..., d)``.
        h: Kernel bandwidth.

    Returns:
        Per-pair update vectors, shape ``(..., d)``.
    """
    k = rbf_kernel(x_source, x_target, h)[..., None]
    repulsive = k * (x_target - x_source) / (h**2)
    return k * score_source + repulsive
