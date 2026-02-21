"""Tests for the two-pass dual-tree builder (count pass + fill pass).

This test compares the outputs when explicit capacities are supplied
against the automatic two-pass sizing path to ensure equivalence.
"""

import jax
import jax.numpy as jnp

from yggdrasil.tree import build_tree
from yggdrasil.geometry import compute_tree_geometry
from yggdrasil.interactions import build_interactions_and_neighbors


def _build_random_tree(n=1024, leaf_size=8, seed=0):
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (n, 3), minval=-1.0, maxval=1.0)
    masses = jnp.ones((n,))
    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions, masses, bounds, return_reordered=True, leaf_size=leaf_size
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    return tree, geometry


def test_two_pass_equivalence_small():
    tree, geometry = _build_random_tree(n=1024, leaf_size=8, seed=42)

    # Run with explicit (large) capacities to force the single fill pass.
    interactions_ref, neighbors_ref = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        max_interactions_per_node=128,
        max_neighbors_per_leaf=128,
        max_pair_queue=4096,
    )

    # Run without capacities to trigger the count pass + sized fill pass.
    interactions_auto, neighbors_auto = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        max_interactions_per_node=None,
        max_neighbors_per_leaf=128,
        max_pair_queue=None,
    )

    # Compare essential fields for equality / consistency.
    assert int(interactions_ref.offsets.shape[0]) == int(
        interactions_auto.offsets.shape[0]
    )
    assert int(interactions_ref.sources.shape[0]) == int(
        interactions_auto.sources.shape[0]
    )
    assert int(neighbors_ref.offsets.shape[0]) == int(neighbors_auto.offsets.shape[0])
    assert int(neighbors_ref.neighbors.shape[0]) == int(
        neighbors_auto.neighbors.shape[0]
    )

    # Compare content where lengths match.
    assert jnp.array_equal(interactions_ref.sources, interactions_auto.sources)
    assert jnp.array_equal(interactions_ref.targets, interactions_auto.targets)
    assert jnp.array_equal(neighbors_ref.neighbors, neighbors_auto.neighbors)
