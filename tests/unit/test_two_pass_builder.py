"""Tests for the two-pass dual-tree builder (count pass + fill pass).

This test compares the outputs when explicit capacities are supplied
against the automatic two-pass sizing path to ensure equivalence.
"""

import jax
import jax.numpy as jnp
import pytest

import yggdrax._interactions_impl as interactions_impl_private
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import build_interactions_and_neighbors
from yggdrax.tree import build_tree

_TEST_THETA = 0.6


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
    tree, geometry = _build_random_tree(n=128, leaf_size=8, seed=42)

    # Run with explicit (large) capacities to force the single fill pass.
    interactions_ref, neighbors_ref = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=_TEST_THETA,
        max_interactions_per_node=128,
        max_neighbors_per_leaf=128,
        max_pair_queue=4096,
    )

    # Run without capacities to trigger the count pass + sized fill pass.
    interactions_auto, neighbors_auto = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=_TEST_THETA,
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

    # Compare semantic content. The bounded fill-cap retry path may reorder
    # sources within the same target bucket while preserving the pair multiset.
    ref_pairs = jnp.stack([interactions_ref.targets, interactions_ref.sources], axis=1)
    auto_pairs = jnp.stack(
        [interactions_auto.targets, interactions_auto.sources], axis=1
    )
    ref_pairs_sorted = ref_pairs[jnp.lexsort((ref_pairs[:, 1], ref_pairs[:, 0]))]
    auto_pairs_sorted = auto_pairs[jnp.lexsort((auto_pairs[:, 1], auto_pairs[:, 0]))]
    assert jnp.array_equal(ref_pairs_sorted, auto_pairs_sorted)
    ref_neighbor_pairs = []
    auto_neighbor_pairs = []
    for leaf_idx in range(int(neighbors_ref.leaf_indices.shape[0])):
        ref_start = int(neighbors_ref.offsets[leaf_idx])
        ref_end = int(neighbors_ref.offsets[leaf_idx + 1])
        auto_start = int(neighbors_auto.offsets[leaf_idx])
        auto_end = int(neighbors_auto.offsets[leaf_idx + 1])
        ref_leaf = int(neighbors_ref.leaf_indices[leaf_idx])
        auto_leaf = int(neighbors_auto.leaf_indices[leaf_idx])
        assert ref_leaf == auto_leaf
        ref_neighbor_pairs.extend(
            (ref_leaf, int(v))
            for v in jnp.asarray(neighbors_ref.neighbors[ref_start:ref_end])
        )
        auto_neighbor_pairs.extend(
            (auto_leaf, int(v))
            for v in jnp.asarray(neighbors_auto.neighbors[auto_start:auto_end])
        )
    assert sorted(ref_neighbor_pairs) == sorted(auto_neighbor_pairs)


def test_count_pass_large_suggestions_still_use_compact_fill(monkeypatch):
    tree, geometry = _build_random_tree(n=64, leaf_size=8, seed=7)

    compact_fill_calls = []
    original_compact_fill = interactions_impl_private._dual_tree_walk_compact_fill_impl

    def fake_count_impl(*args, **kwargs):
        total_nodes = int(tree.parent.shape[0])
        num_internal = int(tree.left_child.shape[0])
        num_leaves = total_nodes - num_internal
        far_counts = jnp.full((total_nodes,), 512, dtype=jnp.int32)
        near_counts = jnp.full((num_leaves,), 64, dtype=jnp.int32)
        max_wf = jnp.asarray(1024, dtype=jnp.int32)
        return far_counts, near_counts, max_wf

    def fail_dense_fill(*args, **kwargs):
        raise AssertionError("dense fill path should not run under count-pass sizing")

    def wrapped_compact_fill(*args, **kwargs):
        compact_fill_calls.append(
            (int(kwargs["total_far_pairs"]), int(kwargs["total_near_pairs"]))
        )
        return original_compact_fill(*args, **kwargs)

    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_count_impl",
        fake_count_impl,
    )
    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_impl",
        fail_dense_fill,
    )
    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_compact_fill_impl",
        wrapped_compact_fill,
    )

    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=_TEST_THETA,
        max_interactions_per_node=None,
        max_neighbors_per_leaf=128,
        max_pair_queue=None,
    )

    assert compact_fill_calls
    assert compact_fill_calls[0][0] == int(interactions.sources.shape[0])
    assert compact_fill_calls[0][1] == int(neighbors.neighbors.shape[0])


def test_count_pass_uses_compact_fill_path(monkeypatch):
    tree, geometry = _build_random_tree(n=128, leaf_size=8, seed=11)

    compact_fill_calls = []
    original_compact_fill = interactions_impl_private._dual_tree_walk_compact_fill_impl

    def fail_dense_fill(*args, **kwargs):
        raise AssertionError("dense fill path should not run under count-pass sizing")

    def wrapped_compact_fill(*args, **kwargs):
        compact_fill_calls.append(int(kwargs["total_far_pairs"]))
        return original_compact_fill(*args, **kwargs)

    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_impl",
        fail_dense_fill,
    )
    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_compact_fill_impl",
        wrapped_compact_fill,
    )

    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=_TEST_THETA,
        max_interactions_per_node=None,
        max_neighbors_per_leaf=128,
        max_pair_queue=None,
    )

    assert compact_fill_calls
    assert compact_fill_calls[0] == int(interactions.sources.shape[0])
    assert int(neighbors.neighbors.shape[0]) >= 0


def test_compact_far_pair_return_can_skip_node_interaction_materialization():
    tree, geometry = _build_random_tree(n=128, leaf_size=8, seed=19)

    interactions, neighbors, compact_far_pairs = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=_TEST_THETA,
        max_interactions_per_node=None,
        max_neighbors_per_leaf=128,
        max_pair_queue=None,
        return_compact_far_pairs=True,
        return_interactions=False,
    )

    assert interactions is None
    assert int(compact_far_pairs.sources.shape[0]) == int(
        compact_far_pairs.targets.shape[0]
    )
    assert int(neighbors.neighbors.shape[0]) >= 0


def test_count_pass_rejects_int32_pair_total_overflow(monkeypatch):
    tree, geometry = _build_random_tree(n=64, leaf_size=8, seed=23)

    def fake_count_impl(*args, **kwargs):
        total_nodes = int(tree.parent.shape[0])
        num_internal = int(tree.left_child.shape[0])
        num_leaves = total_nodes - num_internal
        far_counts = jnp.zeros((total_nodes,), dtype=jnp.int32)
        far_counts = far_counts.at[0].set(jnp.int32((1 << 30) + 17))
        far_counts = far_counts.at[1].set(jnp.int32((1 << 30) + 19))
        near_counts = jnp.zeros((num_leaves,), dtype=jnp.int32)
        return far_counts, near_counts, jnp.asarray(32, dtype=jnp.int32)

    monkeypatch.setattr(
        interactions_impl_private,
        "_dual_tree_walk_count_impl",
        fake_count_impl,
    )

    with pytest.raises(RuntimeError, match="exceed signed int32 capacity"):
        build_interactions_and_neighbors(
            tree,
            geometry,
            theta=_TEST_THETA,
            max_interactions_per_node=None,
            max_neighbors_per_leaf=128,
            max_pair_queue=None,
        )
