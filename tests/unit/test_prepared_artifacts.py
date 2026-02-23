"""Yggdrax package-local regression tests."""

import jax
import jax.numpy as jnp

from yggdrax import TraversalResult, build_prepared_tree_artifacts


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(11)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def test_prepared_tree_artifacts_smoke():
    positions, masses = _sample_problem(n=48)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    artifacts = build_prepared_tree_artifacts(
        positions,
        masses,
        bounds,
        leaf_size=16,
        theta=0.6,
        mac_type="dehnen",
    )
    assert artifacts.positions_sorted.shape == positions.shape
    assert artifacts.masses_sorted.shape == masses.shape
    assert artifacts.tree.node_ranges.shape[0] == artifacts.geometry.center.shape[0]
    assert artifacts.interactions.sources.ndim == 1
    assert artifacts.neighbors.neighbors.ndim == 1
    assert isinstance(artifacts.traversal_result, TraversalResult)
    assert artifacts.traversal_result.far_pair_count.ndim == 0
