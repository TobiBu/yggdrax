"""Tests for the experimental KD-tree style neighbor-query API."""

import jax
import jax.numpy as jnp
import numpy as np

from yggdrax import build_kdtree, count_neighbors, query_neighbors


def _sample_points(n: int = 32, dim: int = 3) -> jnp.ndarray:
    key = jax.random.PRNGKey(123)
    return jax.random.uniform(key, (n, dim), minval=-1.0, maxval=1.0, dtype=jnp.float32)


def test_query_neighbors_returns_expected_shapes():
    points = _sample_points(n=20, dim=3)
    queries = _sample_points(n=7, dim=3)
    tree = build_kdtree(points)

    indices, distances = query_neighbors(tree, queries, k=4, backend="tiled")

    assert indices.shape == (7, 4)
    assert distances.shape == (7, 4)
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    assert jnp.all(distances >= 0.0)


def test_query_neighbors_supports_jit():
    points = _sample_points(n=24, dim=2)
    queries = _sample_points(n=6, dim=2)
    tree = build_kdtree(points)

    jitted = jax.jit(lambda q: query_neighbors(tree, q, k=3, backend="tiled"))
    indices, distances = jitted(queries)

    assert indices.shape == (6, 3)
    assert distances.shape == (6, 3)


def test_build_kdtree_supports_jit():
    points = _sample_points(n=24, dim=3)
    jitted = jax.jit(lambda p: build_kdtree(p, leaf_size=8))
    tree = jitted(points)
    assert tree.indices.shape == (24,)
    assert int(tree.node_start[0]) == 0
    assert int(tree.node_end[0]) == 24


def test_query_neighbors_exclude_self_omits_diagonal():
    points = _sample_points(n=10, dim=3)
    tree = build_kdtree(points)
    indices, _ = query_neighbors(tree, points, k=1, exclude_self=True, backend="tiled")

    assert not np.any(np.asarray(indices[:, 0]) == np.arange(10))


def test_count_neighbors_radius_and_autodiff():
    points = _sample_points(n=16, dim=3)
    tree = build_kdtree(points)
    counts = count_neighbors(
        tree,
        points,
        radius=0.5,
        include_self=False,
        backend="tree",
    )

    assert counts.shape == (16,)
    assert jnp.issubdtype(counts.dtype, jnp.integer)

    def smooth_loss(p: jnp.ndarray) -> jnp.ndarray:
        local_tree = build_kdtree(p)
        _, distances = query_neighbors(
            local_tree,
            p,
            k=2,
            exclude_self=True,
            backend="tiled",
        )
        return jnp.mean(distances[:, 0])

    grad = jax.grad(smooth_loss)(points)
    assert grad.shape == points.shape
    assert jnp.all(jnp.isfinite(grad))


def test_kdtree_topology_fields_are_well_formed():
    points = _sample_points(n=33, dim=3)
    tree = build_kdtree(points, leaf_size=4)

    num_nodes = int(tree.node_start.shape[0])
    assert tree.indices.shape == (33,)
    assert tree.node_end.shape == (num_nodes,)
    assert tree.left_child.shape == (num_nodes,)
    assert tree.right_child.shape == (num_nodes,)
    assert tree.split_dim.shape == (num_nodes,)
    assert tree.split_value.shape == (num_nodes,)
    assert tree.bbox_min.shape == (num_nodes, 3)
    assert tree.bbox_max.shape == (num_nodes, 3)
    assert tree.leaf_nodes.ndim == 1
    assert tree.leaf_start.shape == tree.leaf_nodes.shape
    assert tree.leaf_end.shape == tree.leaf_nodes.shape
    assert tree.leaf_point_ids.ndim == 2
    assert tree.leaf_valid_mask.shape == tree.leaf_point_ids.shape
    assert int(tree.node_start[0]) == 0
    assert int(tree.node_end[0]) == 33


def test_dense_and_tiled_queries_match():
    points = _sample_points(n=64, dim=3)
    queries = _sample_points(n=13, dim=3)
    tree = build_kdtree(points, leaf_size=8)

    idx_dense, d_dense = query_neighbors(tree, queries, k=5, backend="dense")
    idx_tiled, d_tiled = query_neighbors(
        tree,
        queries,
        k=5,
        backend="tiled",
        point_block_size=16,
    )

    assert jnp.array_equal(idx_dense, idx_tiled)
    assert jnp.allclose(d_dense, d_tiled, atol=1e-6, rtol=1e-6)


def test_tree_backend_matches_dense_queries():
    points = _sample_points(n=96, dim=3)
    queries = _sample_points(n=21, dim=3)
    tree = build_kdtree(points, leaf_size=8)

    idx_dense, d_dense = query_neighbors(tree, queries, k=6, backend="dense")
    idx_tree, d_tree = query_neighbors(tree, queries, k=6, backend="tree")

    assert jnp.array_equal(idx_dense, idx_tree)
    assert jnp.allclose(d_dense, d_tree, atol=1e-6, rtol=1e-6)


def test_tree_backend_queries_support_jit():
    points = _sample_points(n=96, dim=3)
    queries = _sample_points(n=21, dim=3)
    tree = build_kdtree(points, leaf_size=8)
    jitted = jax.jit(lambda q: query_neighbors(tree, q, k=6, backend="tree"))
    idx, dist = jitted(queries)
    assert idx.shape == (21, 6)
    assert dist.shape == (21, 6)


def test_query_neighbors_return_squared_matches_distance_square():
    points = _sample_points(n=72, dim=3)
    queries = _sample_points(n=15, dim=3)
    tree = build_kdtree(points, leaf_size=8)

    idx_d, d = query_neighbors(tree, queries, k=5, backend="tree")
    idx_s, d2 = query_neighbors(
        tree,
        queries,
        k=5,
        backend="tree",
        return_squared=True,
    )

    assert jnp.array_equal(idx_d, idx_s)
    assert jnp.allclose(d2, d * d, atol=1e-6, rtol=1e-6)


def test_tree_backend_matches_tiled_radius_counts():
    points = _sample_points(n=96, dim=3)
    queries = _sample_points(n=23, dim=3)
    tree = build_kdtree(points, leaf_size=8)

    counts_tiled = count_neighbors(tree, queries, radius=0.4, backend="tiled")
    counts_tree = count_neighbors(tree, queries, radius=0.4, backend="tree")

    assert jnp.array_equal(counts_tiled, counts_tree)


def test_count_neighbors_supports_vector_radii():
    points = _sample_points(n=96, dim=3)
    queries = _sample_points(n=23, dim=3)
    tree = build_kdtree(points, leaf_size=8)
    radii = jnp.asarray([0.25, 0.4, 0.6], dtype=jnp.float32)

    counts_tiled = count_neighbors(tree, queries, radius=radii, backend="tiled")
    counts_tree = count_neighbors(tree, queries, radius=radii, backend="tree")

    assert counts_tiled.shape == (23, 3)
    assert counts_tree.shape == (23, 3)
    assert jnp.array_equal(counts_tiled, counts_tree)


def test_tree_backend_counts_support_jit():
    points = _sample_points(n=96, dim=3)
    queries = _sample_points(n=23, dim=3)
    tree = build_kdtree(points, leaf_size=8)
    jitted = jax.jit(lambda q: count_neighbors(tree, q, radius=0.4, backend="tree"))
    counts = jitted(queries)
    assert counts.shape == (23,)


def test_tree_backend_matches_dense_across_leaf_sizes():
    points = _sample_points(n=128, dim=3)
    queries = _sample_points(n=19, dim=3)

    for leaf_size in (1, 4, 16, 32):
        tree = build_kdtree(points, leaf_size=leaf_size)
        idx_dense, d_dense = query_neighbors(tree, queries, k=6, backend="dense")
        idx_tree, d_tree = query_neighbors(tree, queries, k=6, backend="tree")
        assert jnp.array_equal(idx_dense, idx_tree)
        assert jnp.allclose(d_dense, d_tree, atol=1e-6, rtol=1e-6)


def test_auto_backend_matches_dense_for_small_problem():
    points = _sample_points(n=48, dim=3)
    queries = _sample_points(n=11, dim=3)
    tree = build_kdtree(points, leaf_size=8)

    idx_dense, d_dense = query_neighbors(tree, queries, k=4, backend="dense")
    idx_auto, d_auto = query_neighbors(tree, queries, k=4, backend="auto")

    assert jnp.array_equal(idx_dense, idx_auto)
    assert jnp.allclose(d_dense, d_auto, atol=1e-6, rtol=1e-6)
