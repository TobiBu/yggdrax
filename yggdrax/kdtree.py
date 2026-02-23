"""Experimental KD-tree style nearest-neighbor API built on JAX primitives."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped


@dataclass(frozen=True)
class KDTree:
    """Container for KD-tree style neighbor queries.

    Notes:
        This initial implementation keeps exact reference points and executes
        brute-force pairwise queries. The interface is intentionally small so
        that a true partitioned KD backend can replace internals later.
    """

    points: Array

    @property
    def num_points(self) -> int:
        """Return the number of reference points in the tree."""

        return int(self.points.shape[0])

    @property
    def dimension(self) -> int:
        """Return spatial dimensionality of reference points."""

        return int(self.points.shape[1])


def _pairwise_squared_distances(queries: Array, points: Array) -> Array:
    """Return squared pairwise distances with shape ``(n_queries, n_points)``."""

    deltas = queries[:, None, :] - points[None, :, :]
    return jnp.sum(deltas * deltas, axis=-1)


@jaxtyped(typechecker=beartype)
def build_kdtree(points: Array) -> KDTree:
    """Build an experimental KD-tree style container from reference points."""

    points_arr = jnp.asarray(points)
    if points_arr.ndim != 2:
        raise ValueError(
            "points must have shape (n_points, dim); "
            f"received ndim={points_arr.ndim}"
        )
    if points_arr.shape[0] < 1:
        raise ValueError("points must contain at least one row")
    if points_arr.shape[1] < 1:
        raise ValueError("points must have dim >= 1")
    return KDTree(points=points_arr)


@jaxtyped(typechecker=beartype)
def query_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    k: int = 1,
    exclude_self: bool = False,
) -> tuple[Array, Array]:
    """Return nearest-neighbor indices and distances for each query point.

    Args:
        tree: Reference point container.
        queries: Query points with shape ``(n_queries, dim)``.
        k: Number of nearest neighbors to return.
        exclude_self: If ``True`` and ``queries.shape[0] == tree.num_points``,
            masks diagonal elements so self-matches are not returned.

    Returns:
        Tuple ``(indices, distances)`` with shapes ``(n_queries, k)``.
    """

    queries_arr = jnp.asarray(queries)
    if queries_arr.ndim != 2:
        raise ValueError(
            "queries must have shape (n_queries, dim); "
            f"received ndim={queries_arr.ndim}"
        )
    if queries_arr.shape[1] != tree.dimension:
        raise ValueError(
            "queries and tree points must share last-dimension size; "
            f"received {queries_arr.shape[1]} and {tree.dimension}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, received {k}")
    if k > tree.num_points:
        raise ValueError(f"k must be <= num_points={tree.num_points}, received {k}")

    distances_sq = _pairwise_squared_distances(queries_arr, tree.points)

    if exclude_self and (queries_arr.shape[0] == tree.num_points):
        diag_len = queries_arr.shape[0]
        diag_idx = jnp.arange(diag_len)
        distances_sq = distances_sq.at[diag_idx, diag_idx].set(jnp.inf)

    top_scores, indices = jax.lax.top_k(-distances_sq, k)
    distances = jnp.sqrt(jnp.maximum(-top_scores, 0.0))
    return indices, distances


@jaxtyped(typechecker=beartype)
def count_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    radius: float,
    include_self: bool = True,
) -> Array:
    """Count reference points within ``radius`` for each query point."""

    queries_arr = jnp.asarray(queries)
    if queries_arr.ndim != 2:
        raise ValueError(
            "queries must have shape (n_queries, dim); "
            f"received ndim={queries_arr.ndim}"
        )
    if queries_arr.shape[1] != tree.dimension:
        raise ValueError(
            "queries and tree points must share last-dimension size; "
            f"received {queries_arr.shape[1]} and {tree.dimension}"
        )
    if radius < 0.0:
        raise ValueError(f"radius must be >= 0, received {radius}")

    radius_sq = jnp.asarray(radius, dtype=queries_arr.dtype) ** 2
    distances_sq = _pairwise_squared_distances(queries_arr, tree.points)
    within = distances_sq <= radius_sq

    if not include_self and (queries_arr.shape[0] == tree.num_points):
        diag_len = queries_arr.shape[0]
        diag_idx = jnp.arange(diag_len)
        within = within.at[diag_idx, diag_idx].set(False)

    return jnp.sum(within, axis=1)


@jaxtyped(typechecker=beartype)
def build_and_query(
    points: Array, queries: Array, *, k: int = 1
) -> tuple[Array, Array]:
    """Convenience function to build a tree and run nearest-neighbor queries."""

    tree = build_kdtree(points)
    return query_neighbors(tree, queries, k=k)


__all__ = [
    "KDTree",
    "build_and_query",
    "build_kdtree",
    "count_neighbors",
    "query_neighbors",
]
