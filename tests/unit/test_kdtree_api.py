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

    indices, distances = query_neighbors(tree, queries, k=4)

    assert indices.shape == (7, 4)
    assert distances.shape == (7, 4)
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    assert jnp.all(distances >= 0.0)


def test_query_neighbors_supports_jit():
    points = _sample_points(n=24, dim=2)
    queries = _sample_points(n=6, dim=2)
    tree = build_kdtree(points)

    jitted = jax.jit(lambda q: query_neighbors(tree, q, k=3))
    indices, distances = jitted(queries)

    assert indices.shape == (6, 3)
    assert distances.shape == (6, 3)


def test_query_neighbors_exclude_self_omits_diagonal():
    points = _sample_points(n=10, dim=3)
    tree = build_kdtree(points)
    indices, _ = query_neighbors(tree, points, k=1, exclude_self=True)

    assert not np.any(np.asarray(indices[:, 0]) == np.arange(10))


def test_count_neighbors_radius_and_autodiff():
    points = _sample_points(n=16, dim=3)
    tree = build_kdtree(points)
    counts = count_neighbors(tree, points, radius=0.5, include_self=False)

    assert counts.shape == (16,)
    assert jnp.issubdtype(counts.dtype, jnp.integer)

    def smooth_loss(p: jnp.ndarray) -> jnp.ndarray:
        local_tree = build_kdtree(p)
        _, distances = query_neighbors(local_tree, p, k=2, exclude_self=True)
        return jnp.mean(distances[:, 0])

    grad = jax.grad(smooth_loss)(points)
    assert grad.shape == points.shape
    assert jnp.all(jnp.isfinite(grad))
