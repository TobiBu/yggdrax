"""Backend-agnostic interaction contract tests (Radix + KD)."""

import jax
import jax.numpy as jnp
import pytest

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)


def _sample_problem(n: int = 96):
    key = jax.random.PRNGKey(2026)
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


def _assert_interaction_contract(tree, interactions, neighbors):
    total_nodes = int(tree.num_nodes)
    leaf_indices = jnp.asarray(neighbors.leaf_indices)

    offsets = jnp.asarray(interactions.offsets)
    counts = jnp.asarray(interactions.counts)
    sources = jnp.asarray(interactions.sources)
    targets = jnp.asarray(interactions.targets)

    assert offsets.shape == (total_nodes + 1,)
    assert counts.shape == (total_nodes,)
    assert sources.ndim == 1
    assert targets.ndim == 1
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == int(sources.shape[0]) == int(targets.shape[0])
    assert jnp.all(offsets[1:] >= offsets[:-1])
    assert jnp.all(counts >= 0)

    # Per-node counts should match offset deltas.
    deltas = offsets[1:] - offsets[:-1]
    assert jnp.array_equal(deltas, counts)

    # Indices are in range when non-empty.
    if sources.shape[0] > 0:
        assert int(jnp.min(sources)) >= 0
        assert int(jnp.max(sources)) < total_nodes
        assert int(jnp.min(targets)) >= 0
        assert int(jnp.max(targets)) < total_nodes

    n_offsets = jnp.asarray(neighbors.offsets)
    n_counts = jnp.asarray(neighbors.counts)
    n_data = jnp.asarray(neighbors.neighbors)

    assert leaf_indices.ndim == 1
    assert n_offsets.shape == (leaf_indices.shape[0] + 1,)
    assert n_counts.shape == (leaf_indices.shape[0],)
    assert int(n_offsets[0]) == 0
    assert int(n_offsets[-1]) == int(n_data.shape[0])
    assert jnp.all(n_offsets[1:] >= n_offsets[:-1])
    assert jnp.all(n_counts >= 0)
    assert jnp.array_equal(n_offsets[1:] - n_offsets[:-1], n_counts)

    if n_data.shape[0] > 0:
        assert int(jnp.min(n_data)) >= 0
        assert int(jnp.max(n_data)) < total_nodes


@pytest.mark.parametrize(
    "tree_type,leaf_size",
    [
        ("radix", 8),
        ("kdtree", 16),
    ],
)
def test_build_interactions_and_neighbors_contract_holds_for_backends(
    tree_type: str,
    leaf_size: int,
):
    positions, masses = _sample_problem(n=128)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        build_mode="adaptive",
        leaf_size=leaf_size,
        return_reordered=True,
    )

    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=32768,
        process_block=256,
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=8192,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )

    _assert_interaction_contract(tree, interactions, neighbors)
