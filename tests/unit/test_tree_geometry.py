"""Tests for tree geometry helpers."""

import jax.numpy as jnp

from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree import build_tree

DEFAULT_TEST_LEAF_SIZE = 1


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    masses = jnp.ones((positions.shape[0],))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    return tree, pos_sorted


def test_geometry_shapes_match_tree():
    tree, pos_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)

    total_nodes = tree.parent.shape[0]
    assert geom.center.shape == (total_nodes, 3)
    assert geom.half_extent.shape == (total_nodes, 3)
    assert geom.radius.shape == (total_nodes,)
    assert geom.max_extent.shape == (total_nodes,)


def test_leaf_geometry_matches_positions():
    tree, pos_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)

    num_internal = tree.num_internal_nodes
    leaf_centers = geom.center[num_internal:]
    leaf_extents = geom.half_extent[num_internal:]

    # compute_tree_geometry clamps half-extents to a small epsilon to avoid
    # zero-size boxes (which would lead to divisions by zero in MAC logic).
    eps = 1e-6

    assert jnp.allclose(leaf_centers, pos_sorted)
    assert jnp.allclose(leaf_extents, eps * jnp.ones_like(leaf_centers))
    assert jnp.allclose(geom.max_extent[num_internal:], eps)
    assert jnp.allclose(
        geom.radius[num_internal:],
        jnp.sqrt(3.0) * eps,
    )


def test_root_geometry_spans_all_points():
    tree, pos_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_mask = parents == -1
    assert int(root_mask.sum()) == 1
    from yggdrax.dtypes import INDEX_DTYPE

    root_index = int(jnp.argmax(root_mask.astype(INDEX_DTYPE)))

    pts_min = jnp.min(pos_sorted, axis=0)
    pts_max = jnp.max(pos_sorted, axis=0)
    expected_center = 0.5 * (pts_min + pts_max)
    expected_extent = 0.5 * (pts_max - pts_min)

    assert jnp.allclose(geom.center[root_index], expected_center)
    assert jnp.allclose(geom.half_extent[root_index], expected_extent)
    assert jnp.isclose(geom.max_extent[root_index], jnp.max(expected_extent))
