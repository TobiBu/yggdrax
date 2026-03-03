"""Tests for generic traversal pair-policy plumbing."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from yggdrax import (
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
    build_tree,
    compute_tree_geometry,
)
from yggdrax.dtypes import INDEX_DTYPE, as_index

_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(23)
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


def _distance_bucket_policy(
    policy_state,
    *,
    valid_pairs,
    mac_ok,
    different_nodes,
    target_leaf,
    source_leaf,
    same_node,
    target_nodes,
    source_nodes,
    center_target,
    center_source,
    dist_sq,
    extent_target,
    extent_source,
):
    del (
        mac_ok,
        same_node,
        target_nodes,
        source_nodes,
        center_target,
        center_source,
        extent_target,
        extent_source,
    )
    accept = valid_pairs & different_nodes & (dist_sq >= policy_state["far_sq"])
    near = valid_pairs & different_nodes & (~accept) & target_leaf & source_leaf
    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=INDEX_DTYPE)
    actions = jnp.where(accept, as_index(_ACTION_ACCEPT), actions)
    actions = jnp.where(near, as_index(_ACTION_NEAR), actions)
    tags = jnp.where(
        dist_sq >= policy_state["tag_split_sq"],
        as_index(2),
        as_index(1),
    )
    tags = jnp.where(accept, tags, as_index(-1))
    return actions, tags


def _target_parity_policy(
    policy_state,
    *,
    valid_pairs,
    mac_ok,
    different_nodes,
    target_leaf,
    source_leaf,
    same_node,
    target_nodes,
    source_nodes,
    center_target,
    center_source,
    dist_sq,
    extent_target,
    extent_source,
):
    del (
        policy_state,
        mac_ok,
        target_leaf,
        source_leaf,
        same_node,
        source_nodes,
        center_target,
        center_source,
        dist_sq,
        extent_target,
        extent_source,
    )
    accept = valid_pairs & different_nodes
    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=INDEX_DTYPE)
    actions = jnp.where(accept, as_index(_ACTION_ACCEPT), actions)
    tags = jnp.where(accept, target_nodes % as_index(2), as_index(-1))
    return actions, tags


def test_pair_policy_returns_tagged_far_pairs():
    positions, masses = _sample_problem()
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=8,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=128,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )

    interactions, neighbors, result = build_interactions_and_neighbors(
        tree,
        geometry,
        traversal_config=traversal_cfg,
        pair_policy=_distance_bucket_policy,
        policy_state={
            "far_sq": jnp.asarray(0.18, dtype=jnp.float32),
            "tag_split_sq": jnp.asarray(0.75, dtype=jnp.float32),
        },
        return_result=True,
    )

    far_total = int(result.far_pair_count)
    assert interactions.sources.shape[0] == far_total
    assert result.interaction_tags.shape == result.interaction_sources.shape
    if far_total > 0:
        active_tags = result.interaction_tags[:far_total]
        assert jnp.all(active_tags >= 1)
        assert jnp.all(active_tags <= 2)
    assert neighbors.neighbors.ndim == 1


def test_pair_policy_keeps_directional_tags_aligned():
    positions, masses = _sample_problem()
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=8,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=128,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )

    _interactions, _neighbors, result = build_interactions_and_neighbors(
        tree,
        geometry,
        traversal_config=traversal_cfg,
        pair_policy=_target_parity_policy,
        policy_state=None,
        return_result=True,
    )

    far_total = int(result.far_pair_count)
    if far_total > 0:
        active_targets = result.interaction_targets[:far_total]
        active_tags = result.interaction_tags[:far_total]
        assert jnp.all(active_tags == (active_targets % as_index(2)))


def test_pair_policy_supports_outer_jit_with_auto_capacities():
    positions, masses = _sample_problem()
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=8,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)

    @jax.jit
    def run(tree_arg, geom_arg, far_sq, tag_sq):
        _interactions, _neighbors, result = build_interactions_and_neighbors(
            tree_arg,
            geom_arg,
            pair_policy=_distance_bucket_policy,
            policy_state={
                "far_sq": far_sq,
                "tag_split_sq": tag_sq,
            },
            return_result=True,
        )
        return result.far_pair_count, jnp.sum(result.interaction_tags >= 0)

    far_count, tagged_count = run(
        tree,
        geometry,
        jnp.asarray(0.18, dtype=jnp.float32),
        jnp.asarray(0.75, dtype=jnp.float32),
    )
    assert int(far_count) >= 0
    assert int(tagged_count) == int(far_count)


def test_pair_policy_supports_outer_jit():
    positions, masses = _sample_problem()
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=8,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=128,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )

    @jax.jit
    def run(tree_arg, geom_arg, far_sq, tag_sq):
        _interactions, _neighbors, result = build_interactions_and_neighbors(
            tree_arg,
            geom_arg,
            traversal_config=traversal_cfg,
            pair_policy=_distance_bucket_policy,
            policy_state={
                "far_sq": far_sq,
                "tag_split_sq": tag_sq,
            },
            return_result=True,
        )
        return result.far_pair_count, jnp.sum(result.interaction_tags >= 0)

    far_count, tagged_count = run(
        tree,
        geometry,
        jnp.asarray(0.18, dtype=jnp.float32),
        jnp.asarray(0.75, dtype=jnp.float32),
    )
    assert int(far_count) >= 0
    assert int(tagged_count) == int(far_count)
