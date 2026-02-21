"""Tests for dense interaction builder."""

import jax.numpy as jnp

from yggdrasil.dense_interactions import build_dense_interactions, densify_interactions
from yggdrasil.geometry import compute_tree_geometry
from yggdrasil.interactions import build_interactions_and_neighbors
from yggdrasil.tree import build_fixed_depth_tree


def _build_fixed_depth_state(theta: float = 0.6):
    positions = jnp.array(
        [
            [-0.7, -0.3, 0.2],
            [-0.5, 0.4, -0.1],
            [-0.2, -0.5, -0.4],
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.4],
            [0.5, -0.2, 0.6],
            [0.7, 0.6, -0.3],
            [0.9, -0.7, 0.8],
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
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
    )
    dense = densify_interactions(tree, geometry, interactions)
    return tree, geometry, interactions, dense


def _node_slot_lookup(level_nodes):
    nodes = jnp.asarray(level_nodes)
    mapping = {}
    num_levels, max_nodes = nodes.shape
    for level in range(num_levels):
        for slot in range(max_nodes):
            node = int(nodes[level, slot])
            if node >= 0:
                mapping[node] = (level, slot)
    return mapping


def test_dense_sources_match_sparse_lists():
    tree, _geom, interactions, dense = _build_fixed_depth_state()
    node_lookup = _node_slot_lookup(dense.geometry.node_indices)

    offsets = jnp.asarray(interactions.offsets)
    counts = jnp.asarray(interactions.counts)
    sources = jnp.asarray(interactions.sources)

    for node, (level, slot) in node_lookup.items():
        start = int(offsets[node])
        count = int(counts[node])
        sparse = set(map(int, sources[start : start + count]))
        mask = jnp.asarray(dense.m2l_mask[level, slot])
        dense_sources = jnp.asarray(dense.m2l_sources[level, slot])[mask]
        assert sparse == set(map(int, dense_sources))


def test_dense_displacements_align_with_centers():
    tree, geometry, interactions, dense = _build_fixed_depth_state()
    del interactions
    node_lookup = _node_slot_lookup(dense.geometry.node_indices)
    centers = jnp.asarray(geometry.center)

    for node, (level, slot) in node_lookup.items():
        mask = jnp.asarray(dense.m2l_mask[level, slot])
        sources = jnp.asarray(dense.m2l_sources[level, slot])
        displacements = jnp.asarray(dense.m2l_displacements[level, slot])
        target_center = centers[node]
        for idx, valid in enumerate(mask):
            if not bool(valid):
                continue
            source_idx = int(sources[idx])
            expected = target_center - centers[source_idx]
            assert jnp.allclose(displacements[idx], expected)


def test_build_dense_interactions_matches_manual_result():
    tree, geometry, interactions, dense_manual = _build_fixed_depth_state()
    dense_built = build_dense_interactions(tree, geometry)

    assert jnp.array_equal(dense_built.m2l_sources, dense_manual.m2l_sources)
    assert jnp.array_equal(dense_built.m2l_mask, dense_manual.m2l_mask)
    assert jnp.array_equal(dense_built.m2l_counts, dense_manual.m2l_counts)
    assert jnp.allclose(
        dense_built.m2l_displacements,
        dense_manual.m2l_displacements,
    )
    assert jnp.array_equal(
        dense_built.geometry.node_indices,
        dense_manual.geometry.node_indices,
    )

    # Ensure the builder returns fresh sparse structures as well.
    assert jnp.array_equal(
        jnp.asarray(dense_built.sparse_interactions.sources),
        jnp.asarray(interactions.sources),
    )
