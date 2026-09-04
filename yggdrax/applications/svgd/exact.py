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


def exact_phi(
    particles: Float[Array, "n d"],
    scores: Float[Array, "n d"],
    h: float | Float[Array, ""],
    *,
    block_size: int | None = None,
) -> Float[Array, "n d"]:
    """Exact Stein update direction phi(x_i) for every particle, O(N^2).

    The pair sum is contracted, not materialised. Writing it out,

    .. math::

        \\phi_i = \\frac{1}{N}\\Big[(K S)_i
                 + \\big(x_i (K \\mathbf{1})_i - (K X)_i\\big) / h^2\\Big],

    with :math:`K_{ij} = \\exp(-\\lVert x_i - x_j\\rVert^2 / 2h^2)`, so the whole
    update is one kernel matrix and **two matmuls**. The obvious form builds an
    ``(n, n, d)`` tensor of per-pair terms, which is *d* times the memory, runs
    at elementwise rather than GEMM throughput, and cannot be evaluated at all
    beyond N ~ 2e4 -- which mattered, because this is the baseline the tree
    update is judged against, and a weak baseline flatters the tree.

    Args:
        particles: Particle positions, shape ``(n, d)``.
        scores: Target score at each particle, shape ``(n, d)``.
        h: Kernel bandwidth.
        block_size: Targets per block. ``None`` does every target at once,
            which costs an ``(n, n)`` kernel matrix; pass a block size to cap
            that at ``(block_size, n)`` and reach large N.

    Returns:
        Update directions, shape ``(n, d)``.
    """
    n, d = particles.shape
    if block_size is None:
        block_size = n
    block = max(1, min(int(block_size), n))

    def _block(x_t: Array) -> Array:
        """Contribution of every source to one block of targets."""
        # (B, n) kernel, then two contractions -- no (B, n, d) tensor is ever
        # built. sum_j k_ij s_j is a matmul, and
        # sum_j k_ij (x_i - x_j) = x_i * sum_j k_ij - sum_j k_ij x_j is another.
        d2 = (
            jnp.sum(x_t * x_t, axis=-1)[:, None]
            - 2.0 * (x_t @ particles.T)
            + jnp.sum(particles * particles, axis=-1)[None, :]
        )
        k = jnp.exp(-jnp.maximum(d2, 0.0) / (2.0 * h**2))  # (B, n)
        attract = k @ scores
        repulse = x_t * jnp.sum(k, axis=1)[:, None] - k @ particles
        return attract + repulse / (h**2)

    if block >= n:
        return _block(particles) / n

    pad = (-n) % block
    padded = (
        particles
        if pad == 0
        else jnp.concatenate([particles, jnp.zeros((pad, d), particles.dtype)])
    )
    out = jax.lax.map(_block, padded.reshape(-1, block, d))
    return out.reshape(-1, d)[:n] / n


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
