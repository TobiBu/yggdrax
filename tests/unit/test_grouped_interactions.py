"""Tests for grouped far-field interaction buffers."""

import jax.numpy as jnp
import numpy as np

from yggdrax.geometry import compute_tree_geometry
from yggdrax.grouped_interactions import (
    _safe_cell_size,
    build_grouped_interactions,
    build_grouped_interactions_from_pairs,
)
from yggdrax.interactions import build_interactions_and_neighbors
from yggdrax.tree import build_fixed_depth_tree


def _sample_tree_state():
    positions = jnp.array(
        [
            [-0.8, -0.6, -0.2],
            [-0.5, 0.4, -0.1],
            [-0.1, -0.3, 0.5],
            [0.0, 0.2, -0.4],
            [0.3, -0.7, 0.1],
            [0.5, 0.6, 0.4],
            [0.7, -0.2, -0.6],
            [0.9, 0.8, 0.7],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.ones((positions.shape[0],), dtype=jnp.float64)
    bounds = (
        jnp.min(positions, axis=0) - 0.1,
        jnp.max(positions, axis=0) + 0.1,
    )

    tree, pos_sorted, _, _ = build_fixed_depth_tree(
        positions,
        masses,
        bounds,
        target_leaf_particles=2,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    interactions, _ = build_interactions_and_neighbors(tree, geometry, theta=0.6)
    return tree, geometry, interactions


def test_grouped_builder_matches_pair_wrapper():
    tree, geometry, interactions = _sample_tree_state()

    grouped_wrapper = build_grouped_interactions(tree, geometry, interactions)
    grouped_pairs = build_grouped_interactions_from_pairs(
        tree,
        geometry,
        interactions.sources,
        interactions.targets,
        level_offsets=interactions.level_offsets,
    )

    assert jnp.array_equal(grouped_wrapper.class_keys, grouped_pairs.class_keys)
    assert jnp.array_equal(grouped_wrapper.class_offsets, grouped_pairs.class_offsets)
    assert jnp.array_equal(grouped_wrapper.class_sources, grouped_pairs.class_sources)
    assert jnp.array_equal(grouped_wrapper.class_targets, grouped_pairs.class_targets)
    assert jnp.array_equal(grouped_wrapper.class_ids, grouped_pairs.class_ids)


def test_grouped_classes_partition_all_pairs_and_keys_match():
    tree, geometry, interactions = _sample_tree_state()
    grouped = build_grouped_interactions(tree, geometry, interactions)

    total_pairs = int(interactions.sources.shape[0])
    assert int(grouped.class_offsets[0]) == 0
    assert int(grouped.class_offsets[-1]) == total_pairs

    centers = np.asarray(geometry.center)
    levels = np.asarray(tree.node_level)
    bounds_min = np.asarray(tree.bounds_min)
    bounds_max = np.asarray(tree.bounds_max)
    root_extent = bounds_max - bounds_min

    for class_id in range(int(grouped.class_keys.shape[0])):
        start = int(grouped.class_offsets[class_id])
        end = int(grouped.class_offsets[class_id + 1])
        key = np.asarray(grouped.class_keys[class_id])
        for idx in range(start, end):
            source = int(grouped.class_sources[idx])
            target = int(grouped.class_targets[idx])
            delta = centers[target] - centers[source]
            cell = _safe_cell_size(
                root_extent, np.asarray([levels[target]], dtype=np.int32)
            )[0]
            disp = np.rint(delta / cell).astype(np.int32)
            actual = np.array(
                [
                    levels[target],
                    levels[source],
                    disp[0],
                    disp[1],
                    disp[2],
                ],
                dtype=np.int32,
            )
            assert np.array_equal(actual, key)


def test_grouped_empty_pairs_returns_empty_buffers():
    tree, geometry, interactions = _sample_tree_state()
    del interactions

    grouped = build_grouped_interactions_from_pairs(
        tree,
        geometry,
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
    )

    assert grouped.class_keys.shape == (0, 5)
    assert grouped.class_displacements.shape == (0, 3)
    assert grouped.class_sources.shape == (0,)
    assert grouped.class_targets.shape == (0,)
    assert grouped.class_ids.shape == (0,)
    assert jnp.array_equal(grouped.class_offsets, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(grouped.level_offsets, jnp.array([0], dtype=jnp.int32))


def test_grouped_non_empty_pairs_cover_grouping_path():
    tree, geometry, _ = _sample_tree_state()
    num_nodes = int(tree.parent.shape[0])
    assert num_nodes >= 4

    sources = jnp.array([0, 1, 2, 1], dtype=jnp.int32) % num_nodes
    targets = jnp.array([2, 3, 1, 3], dtype=jnp.int32) % num_nodes

    grouped = build_grouped_interactions_from_pairs(
        tree,
        geometry,
        sources,
        targets,
        level_offsets=jnp.array([0, 2, 4], dtype=jnp.int32),
    )

    assert grouped.class_keys.shape[1] == 5
    assert int(grouped.class_offsets[-1]) == int(sources.shape[0])
    assert grouped.class_sources.shape == sources.shape
    assert grouped.class_targets.shape == targets.shape
    assert grouped.class_displacements.shape[0] == grouped.class_keys.shape[0]
    assert jnp.all(grouped.class_ids >= 0)
