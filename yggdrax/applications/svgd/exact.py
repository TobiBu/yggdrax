"""Reference O(N^2) SVGD for validating the tree-accelerated sampler.

Computes the exact empirical Stein update by summing over all particle pairs.
Used only at small N as ground truth; the tree sampler
(:mod:`yggdrax.applications.svgd.sampler`) must match it in distribution.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from yggdrax.applications.svgd.kernel import stein_pair_terms


def exact_phi(
    particles: Float[Array, "n d"],
    scores: Float[Array, "n d"],
    h: float | Float[Array, ""],
) -> Float[Array, "n d"]:
    """Exact Stein update direction phi(x_i) for every particle, O(N^2).

    Args:
        particles: Particle positions, shape ``(n, d)``.
        scores: Target score at each particle, shape ``(n, d)``.
        h: Kernel bandwidth.

    Returns:
        Update directions, shape ``(n, d)``.
    """
    n = particles.shape[0]
    # target axis 0 (i), source axis 1 (j): terms[i, j] contributes to phi[i].
    x_t = particles[:, None, :]
    x_s = particles[None, :, :]
    s_s = scores[None, :, :]
    terms = stein_pair_terms(x_t, x_s, s_s, h)  # (n, n, d)
    return jnp.sum(terms, axis=1) / n


def svgd_step(
    particles: Float[Array, "n d"],
    score_fn: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
    h: float | Float[Array, ""],
    step_size: float,
) -> Float[Array, "n d"]:
    """One exact SVGD step.

    Args:
        particles: Current particles, shape ``(n, d)``.
        score_fn: Target score function, ``(n, d) -> (n, d)``.
        h: Kernel bandwidth.
        step_size: Update step size.

    Returns:
        Updated particles, shape ``(n, d)``.
    """
    scores = score_fn(particles)
    return particles + step_size * exact_phi(particles, scores, h)


def run_svgd(
    particles: Float[Array, "n d"],
    score_fn: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
    h: float | Float[Array, ""],
    step_size: float,
    num_steps: int,
) -> Float[Array, "n d"]:
    """Run exact SVGD for ``num_steps`` steps.

    Args:
        particles: Initial particles, shape ``(n, d)``.
        score_fn: Target score function, ``(n, d) -> (n, d)``.
        h: Kernel bandwidth (fixed across steps).
        step_size: Update step size.
        num_steps: Number of SVGD steps.

    Returns:
        Final particles, shape ``(n, d)``.
    """

    def body(p, _):
        return svgd_step(p, score_fn, h, step_size), None

    final, _ = jax.lax.scan(body, particles, None, length=num_steps)
    return final
