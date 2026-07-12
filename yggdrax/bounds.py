"""Bounding-box helpers for tree construction."""

from __future__ import annotations

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped


@jaxtyped(typechecker=beartype)
def infer_bounds(positions: Array) -> tuple[Array, Array]:
    """Infer a padded axis-aligned bounding box from particle positions.

    The box spans the particle extent plus 5% padding per axis (at least
    ``1e-6``), so points on the boundary do not clip to the edge during Morton
    encoding.

    Parameters
    ----------
    positions
        Particle positions of shape ``(n, 3)``.

    Returns
    -------
    tuple of Array
        ``(min_corner, max_corner)``, each of shape ``(3,)``.
    """

    minimum = jnp.min(positions, axis=0)
    maximum = jnp.max(positions, axis=0)
    span = maximum - minimum
    padding = jnp.maximum(span * 0.05, jnp.full_like(span, 1e-6))
    return minimum - padding, maximum + padding


__all__ = ["infer_bounds"]
