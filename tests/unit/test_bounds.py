"""Unit coverage for yggdrax bounds inference."""

import jax.numpy as jnp

from yggdrax import infer_bounds


def test_infer_bounds_adds_positive_padding_for_non_degenerate_cloud():
    positions = jnp.array(
        [
            [-1.0, 2.0, 0.5],
            [3.0, -2.0, 1.5],
            [0.5, 1.0, -4.0],
        ],
        dtype=jnp.float32,
    )
    lower, upper = infer_bounds(positions)
    min_pos = jnp.min(positions, axis=0)
    max_pos = jnp.max(positions, axis=0)
    assert jnp.all(lower < min_pos)
    assert jnp.all(upper > max_pos)


def test_infer_bounds_adds_minimum_padding_for_degenerate_cloud():
    positions = jnp.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    lower, upper = infer_bounds(positions)
    width = upper - lower
    # float32 rounding near 1e-6 can quantize to ~1.907e-6 total width.
    assert jnp.all(width >= jnp.array([1.9e-6, 1.9e-6, 1.9e-6], dtype=jnp.float32))
