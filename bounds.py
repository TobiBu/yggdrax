"""Bounding-box helpers for tree construction."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def infer_bounds(positions: Array) -> tuple[Array, Array]:
    """Infer generous tree bounds from particle positions."""

    minimum = jnp.min(positions, axis=0)
    maximum = jnp.max(positions, axis=0)
    span = maximum - minimum
    padding = jnp.maximum(span * 0.05, jnp.full_like(span, 1e-6))
    return minimum - padding, maximum + padding


__all__ = ["infer_bounds"]
