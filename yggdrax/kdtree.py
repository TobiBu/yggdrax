"""Experimental KD-tree API with JAX-friendly exact query kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped


@dataclass(frozen=True)
class KDTree:
    """KD-tree container plus precomputed topology metadata."""

    points: Array
    indices: Array
    node_start: Array
    node_end: Array
    left_child: Array
    right_child: Array
    split_dim: Array
    split_value: Array
    bbox_min: Array
    bbox_max: Array
    leaf_nodes: Array
    leaf_start: Array
    leaf_end: Array
    leaf_size: int

    @property
    def num_points(self) -> int:
        """Return the number of reference points in the tree."""

        return int(self.points.shape[0])

    @property
    def dimension(self) -> int:
        """Return spatial dimensionality of reference points."""

        return int(self.points.shape[1])


def _register_kdtree_pytree() -> None:
    if getattr(KDTree, "_yggdrax_pytree_registered", False):
        return

    def flatten(tree: KDTree):
        children = (
            tree.points,
            tree.indices,
            tree.node_start,
            tree.node_end,
            tree.left_child,
            tree.right_child,
            tree.split_dim,
            tree.split_value,
            tree.bbox_min,
            tree.bbox_max,
            tree.leaf_nodes,
            tree.leaf_start,
            tree.leaf_end,
        )
        aux = (tree.leaf_size,)
        return children, aux

    def unflatten(aux, children):
        (leaf_size,) = aux
        (
            points,
            indices,
            node_start,
            node_end,
            left_child,
            right_child,
            split_dim,
            split_value,
            bbox_min,
            bbox_max,
            leaf_nodes,
            leaf_start,
            leaf_end,
        ) = children
        return KDTree(
            points=points,
            indices=indices,
            node_start=node_start,
            node_end=node_end,
            left_child=left_child,
            right_child=right_child,
            split_dim=split_dim,
            split_value=split_value,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            leaf_nodes=leaf_nodes,
            leaf_start=leaf_start,
            leaf_end=leaf_end,
            leaf_size=leaf_size,
        )

    jax.tree_util.register_pytree_node(KDTree, flatten, unflatten)
    setattr(KDTree, "_yggdrax_pytree_registered", True)


_register_kdtree_pytree()


def _pairwise_squared_distances(queries: Array, points: Array) -> Array:
    """Return squared pairwise distances with shape ``(n_queries, n_points)``."""

    deltas = queries[:, None, :] - points[None, :, :]
    return jnp.sum(deltas * deltas, axis=-1)


def _validate_points(points: Array) -> Array:
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
    return points_arr


def _validate_queries(tree: KDTree, queries: Array) -> Array:
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
    return queries_arr


def _build_kdtree_topology(points: Array, leaf_size: int) -> tuple[Array, ...]:
    """Build a balanced KD-tree topology using cyclic-axis median splits."""

    n, dim = int(points.shape[0]), int(points.shape[1])
    order = jnp.arange(n, dtype=jnp.int32)
    node_start: list[int] = []
    node_end: list[int] = []
    left_child: list[int] = []
    right_child: list[int] = []
    split_dim_vals: list[Array] = []
    node_depth: list[int] = []
    bbox_min: list[Array] = []
    bbox_max: list[Array] = []
    leaf_nodes: list[int] = []

    def append_node(start: int, end: int, depth: int) -> int:
        node_start.append(start)
        node_end.append(end)
        node_depth.append(depth)
        left_child.append(-1)
        right_child.append(-1)
        split_dim_vals.append(jnp.asarray(-1, dtype=jnp.int32))
        bbox_min.append(jnp.zeros((dim,), dtype=points.dtype))
        bbox_max.append(jnp.zeros((dim,), dtype=points.dtype))
        return len(node_start) - 1

    root = append_node(0, n, 0)
    stack = [root]

    while stack:
        node = stack.pop()
        start = node_start[node]
        end = node_end[node]
        depth = node_depth[node]
        idx = order[start:end]
        pts = points[idx]
        mins = jnp.min(pts, axis=0)
        maxs = jnp.max(pts, axis=0)
        bbox_min[node] = mins
        bbox_max[node] = maxs
        count = end - start

        if count <= leaf_size:
            leaf_nodes.append(node)
            continue

        spreads = maxs - mins
        axis = jnp.argmax(spreads).astype(jnp.int32)
        split_dim_vals[node] = axis

        values = jnp.take(pts, axis, axis=1)
        local_sort = jnp.argsort(values, stable=True)
        order = order.at[start:end].set(idx[local_sort])
        mid = start + count // 2

        left = append_node(start, mid, depth + 1)
        right = append_node(mid, end, depth + 1)
        left_child[node] = left
        right_child[node] = right

        # Push right then left to keep left branch denser in memory order.
        stack.append(right)
        stack.append(left)

    node_start_arr = jnp.asarray(node_start, dtype=jnp.int32)
    node_end_arr = jnp.asarray(node_end, dtype=jnp.int32)
    split_dim_arr = jnp.stack(split_dim_vals).astype(jnp.int32)
    mid_arr = (node_start_arr + node_end_arr) // 2
    safe_mid = jnp.clip(mid_arr, 0, n - 1)
    safe_dim = jnp.clip(split_dim_arr, 0, dim - 1)
    safe_point_idx = order[safe_mid]
    gathered = points[safe_point_idx, safe_dim]
    split_value_arr = jnp.where(split_dim_arr >= 0, gathered, jnp.nan)

    return (
        order,
        node_start_arr,
        node_end_arr,
        jnp.asarray(left_child, dtype=jnp.int32),
        jnp.asarray(right_child, dtype=jnp.int32),
        split_dim_arr,
        split_value_arr,
        jnp.stack(bbox_min, axis=0),
        jnp.stack(bbox_max, axis=0),
        jnp.asarray(leaf_nodes, dtype=jnp.int32),
        jnp.asarray([node_start[x] for x in leaf_nodes], dtype=jnp.int32),
        jnp.asarray([node_end[x] for x in leaf_nodes], dtype=jnp.int32),
    )


@jaxtyped(typechecker=beartype)
def build_kdtree(points: Array, *, leaf_size: int = 32) -> KDTree:
    """Build an experimental KD-tree style container from reference points."""

    points_arr = _validate_points(points)
    if leaf_size < 1:
        raise ValueError(f"leaf_size must be >= 1, received {leaf_size}")

    topology = _build_kdtree_topology(points_arr, int(leaf_size))
    return KDTree(
        points=points_arr,
        indices=topology[0],
        node_start=topology[1],
        node_end=topology[2],
        left_child=topology[3],
        right_child=topology[4],
        split_dim=topology[5],
        split_value=topology[6],
        bbox_min=topology[7],
        bbox_max=topology[8],
        leaf_nodes=topology[9],
        leaf_start=topology[10],
        leaf_end=topology[11],
        leaf_size=int(leaf_size),
    )


def _query_neighbors_dense(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
) -> tuple[Array, Array]:
    distances_sq = _pairwise_squared_distances(queries_arr, tree.points)
    if exclude_self and (queries_arr.shape[0] == tree.num_points):
        diag_len = queries_arr.shape[0]
        diag_idx = jnp.arange(diag_len)
        distances_sq = distances_sq.at[diag_idx, diag_idx].set(jnp.inf)

    top_scores, indices = jax.lax.top_k(-distances_sq, k)
    distances = jnp.sqrt(jnp.maximum(-top_scores, 0.0))
    return indices, distances


def _query_neighbors_tiled(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
    point_block_size: int,
) -> tuple[Array, Array]:
    num_points = tree.num_points
    if point_block_size < 1:
        raise ValueError(f"point_block_size must be >= 1, received {point_block_size}")

    n_queries = queries_arr.shape[0]
    block_size = int(point_block_size)
    num_blocks = (num_points + block_size - 1) // block_size
    padded_size = num_blocks * block_size
    pad = padded_size - num_points

    if pad > 0:
        points_pad = jnp.zeros((pad, tree.dimension), dtype=tree.points.dtype)
        points_padded = jnp.concatenate([tree.points, points_pad], axis=0)
        ids_pad = jnp.full((pad,), -1, dtype=jnp.int32)
        point_ids = jnp.concatenate([jnp.arange(num_points, dtype=jnp.int32), ids_pad])
    else:
        points_padded = tree.points
        point_ids = jnp.arange(num_points, dtype=jnp.int32)

    best_d2 = jnp.full((n_queries, k), jnp.inf, dtype=queries_arr.dtype)
    best_idx = jnp.full((n_queries, k), -1, dtype=jnp.int32)
    query_ids = jnp.arange(n_queries, dtype=jnp.int32)[:, None]

    def body(block_idx, state):
        d2_best, idx_best = state
        start = block_idx * block_size
        block_points = jax.lax.dynamic_slice(
            points_padded, (start, 0), (block_size, tree.dimension)
        )
        block_ids = jax.lax.dynamic_slice(point_ids, (start,), (block_size,))
        d2 = _pairwise_squared_distances(queries_arr, block_points)
        valid = block_ids >= 0
        d2 = jnp.where(valid[None, :], d2, jnp.inf)

        if exclude_self and (n_queries == num_points):
            self_mask = query_ids == block_ids[None, :]
            d2 = jnp.where(self_mask, jnp.inf, d2)

        candidate_d2 = jnp.concatenate([d2_best, d2], axis=1)
        candidate_idx = jnp.concatenate(
            [idx_best, jnp.broadcast_to(block_ids[None, :], (n_queries, block_size))],
            axis=1,
        )
        top_scores, top_cols = jax.lax.top_k(-candidate_d2, k)
        new_d2 = -top_scores
        new_idx = jnp.take_along_axis(candidate_idx, top_cols, axis=1)
        return new_d2, new_idx

    best_d2, best_idx = jax.lax.fori_loop(0, num_blocks, body, (best_d2, best_idx))
    return best_idx, jnp.sqrt(jnp.maximum(best_d2, 0.0))


def _point_to_bbox_distance_sq(query: Array, bbox_min: Array, bbox_max: Array) -> Array:
    clipped = jnp.clip(query[None, :], bbox_min, bbox_max)
    delta = query[None, :] - clipped
    return jnp.sum(delta * delta, axis=1)


def _query_neighbors_tree(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
) -> tuple[Array, Array]:
    points_sorted = tree.points[tree.indices]
    ids_sorted = jnp.asarray(tree.indices, dtype=jnp.int32)
    leaf_bbox_min = tree.bbox_min[tree.leaf_nodes]
    leaf_bbox_max = tree.bbox_max[tree.leaf_nodes]
    leaf_start = jnp.asarray(tree.leaf_start, dtype=jnp.int32)
    leaf_end = jnp.asarray(tree.leaf_end, dtype=jnp.int32)
    leaf_sizes = leaf_end - leaf_start
    max_leaf_points = max(int(jnp.max(leaf_sizes)), 1)

    pad = max_leaf_points - 1
    if pad > 0:
        points_pad = jnp.zeros((pad, tree.dimension), dtype=tree.points.dtype)
        ids_pad = jnp.full((pad,), -1, dtype=jnp.int32)
        points_sorted_padded = jnp.concatenate([points_sorted, points_pad], axis=0)
        ids_sorted_padded = jnp.concatenate([ids_sorted, ids_pad], axis=0)
    else:
        points_sorted_padded = points_sorted
        ids_sorted_padded = ids_sorted

    num_leaves = int(leaf_start.shape[0])
    n_queries = int(queries_arr.shape[0])
    num_points = tree.num_points
    use_self_mask = bool(exclude_self and (n_queries == num_points))
    query_ids = jnp.arange(n_queries, dtype=jnp.int32)

    def single_query(query: Array, query_id: Array) -> tuple[Array, Array]:
        lower_bounds = _point_to_bbox_distance_sq(query, leaf_bbox_min, leaf_bbox_max)
        order = jnp.argsort(lower_bounds)

        best_d2 = jnp.full((k,), jnp.inf, dtype=queries_arr.dtype)
        best_idx = jnp.full((k,), -1, dtype=jnp.int32)

        def cond_fun(state):
            i, _d2, _idx, done = state
            return (i < num_leaves) & (~done)

        def body_fun(state):
            i, d2_best, idx_best, _done = state
            leaf_ord = order[i]
            lb = lower_bounds[leaf_ord]
            have_k = idx_best[k - 1] >= 0
            worst = d2_best[k - 1]
            should_stop = have_k & (lb > worst)

            start = leaf_start[leaf_ord]
            end = leaf_end[leaf_ord]
            count = end - start
            zero = jnp.asarray(0, dtype=start.dtype)
            block_points = jax.lax.dynamic_slice(
                points_sorted_padded,
                (start, zero),
                (max_leaf_points, tree.dimension),
            )
            block_ids = jax.lax.dynamic_slice(
                ids_sorted_padded, (start,), (max_leaf_points,)
            )
            valid = jnp.arange(max_leaf_points, dtype=jnp.int32) < count
            d2 = jnp.sum((block_points - query[None, :]) ** 2, axis=1)
            d2 = jnp.where(valid, d2, jnp.inf)

            if use_self_mask:
                d2 = jnp.where(block_ids == query_id, jnp.inf, d2)

            candidate_d2 = jnp.concatenate([d2_best, d2], axis=0)
            candidate_idx = jnp.concatenate([idx_best, block_ids], axis=0)
            top_scores, top_cols = jax.lax.top_k(-candidate_d2, k)
            next_d2 = -top_scores
            next_idx = jnp.take_along_axis(candidate_idx, top_cols, axis=0)

            d2_best = jnp.where(should_stop, d2_best, next_d2)
            idx_best = jnp.where(should_stop, idx_best, next_idx)
            return i + 1, d2_best, idx_best, should_stop

        _, best_d2, best_idx, _ = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (0, best_d2, best_idx, False),
        )
        return best_idx, jnp.sqrt(jnp.maximum(best_d2, 0.0))

    return jax.vmap(single_query, in_axes=(0, 0))(queries_arr, query_ids)


@jaxtyped(typechecker=beartype)
def query_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    k: int = 1,
    exclude_self: bool = False,
    backend: Literal["dense", "tiled", "tree"] = "tiled",
    point_block_size: int = 2048,
) -> tuple[Array, Array]:
    """Return nearest-neighbor indices and distances for each query point.

    Args:
        tree: Reference point container.
        queries: Query points with shape ``(n_queries, dim)``.
        k: Number of nearest neighbors to return.
        exclude_self: If ``True`` and ``queries.shape[0] == tree.num_points``,
            masks diagonal elements so self-matches are not returned.
        backend: Query backend. ``dense`` builds a full pairwise matrix;
            ``tiled`` computes exact neighbors blockwise with lower memory;
            ``tree`` uses KD leaf bounding boxes to prune exact search.
        point_block_size: Block size used by the tiled backend.

    Returns:
        Tuple ``(indices, distances)`` with shapes ``(n_queries, k)``.
    """

    queries_arr = _validate_queries(tree, queries)
    if k < 1:
        raise ValueError(f"k must be >= 1, received {k}")
    if k > tree.num_points:
        raise ValueError(f"k must be <= num_points={tree.num_points}, received {k}")
    if backend not in {"dense", "tiled", "tree"}:
        raise ValueError("backend must be one of: 'dense', 'tiled', 'tree'")

    if backend == "dense":
        return _query_neighbors_dense(tree, queries_arr, k=k, exclude_self=exclude_self)
    if backend == "tree":
        return _query_neighbors_tree(tree, queries_arr, k=k, exclude_self=exclude_self)
    return _query_neighbors_tiled(
        tree,
        queries_arr,
        k=k,
        exclude_self=exclude_self,
        point_block_size=point_block_size,
    )


def _count_neighbors_tiled(
    tree: KDTree,
    queries_arr: Array,
    *,
    radius: float,
    include_self: bool,
    point_block_size: int,
) -> Array:
    num_points = tree.num_points
    if point_block_size < 1:
        raise ValueError(f"point_block_size must be >= 1, received {point_block_size}")

    n_queries = queries_arr.shape[0]
    radius_sq = jnp.asarray(radius, dtype=queries_arr.dtype) ** 2
    block_size = int(point_block_size)
    num_blocks = (num_points + block_size - 1) // block_size
    padded_size = num_blocks * block_size
    pad = padded_size - num_points

    if pad > 0:
        points_pad = jnp.zeros((pad, tree.dimension), dtype=tree.points.dtype)
        points_padded = jnp.concatenate([tree.points, points_pad], axis=0)
        ids_pad = jnp.full((pad,), -1, dtype=jnp.int32)
        point_ids = jnp.concatenate([jnp.arange(num_points, dtype=jnp.int32), ids_pad])
    else:
        points_padded = tree.points
        point_ids = jnp.arange(num_points, dtype=jnp.int32)

    query_ids = jnp.arange(n_queries, dtype=jnp.int32)[:, None]

    def body(block_idx, counts):
        start = block_idx * block_size
        block_points = jax.lax.dynamic_slice(
            points_padded, (start, 0), (block_size, tree.dimension)
        )
        block_ids = jax.lax.dynamic_slice(point_ids, (start,), (block_size,))
        d2 = _pairwise_squared_distances(queries_arr, block_points)
        valid = block_ids >= 0
        within = (d2 <= radius_sq) & valid[None, :]

        if (not include_self) and (n_queries == num_points):
            self_mask = query_ids == block_ids[None, :]
            within = within & (~self_mask)

        return counts + jnp.sum(within, axis=1, dtype=counts.dtype)

    init = jnp.zeros((n_queries,), dtype=jnp.int32)
    return jax.lax.fori_loop(0, num_blocks, body, init)


def _count_neighbors_tree(
    tree: KDTree,
    queries_arr: Array,
    *,
    radius: float,
    include_self: bool,
) -> Array:
    radius_sq = jnp.asarray(radius, dtype=queries_arr.dtype) ** 2
    points_sorted = tree.points[tree.indices]
    ids_sorted = jnp.asarray(tree.indices, dtype=jnp.int32)
    leaf_bbox_min = tree.bbox_min[tree.leaf_nodes]
    leaf_bbox_max = tree.bbox_max[tree.leaf_nodes]
    leaf_start = jnp.asarray(tree.leaf_start, dtype=jnp.int32)
    leaf_end = jnp.asarray(tree.leaf_end, dtype=jnp.int32)
    leaf_sizes = leaf_end - leaf_start
    max_leaf_points = max(int(jnp.max(leaf_sizes)), 1)

    pad = max_leaf_points - 1
    if pad > 0:
        points_pad = jnp.zeros((pad, tree.dimension), dtype=tree.points.dtype)
        ids_pad = jnp.full((pad,), -1, dtype=jnp.int32)
        points_sorted_padded = jnp.concatenate([points_sorted, points_pad], axis=0)
        ids_sorted_padded = jnp.concatenate([ids_sorted, ids_pad], axis=0)
    else:
        points_sorted_padded = points_sorted
        ids_sorted_padded = ids_sorted

    num_leaves = int(leaf_start.shape[0])
    n_queries = int(queries_arr.shape[0])
    num_points = tree.num_points
    use_self_mask = bool((not include_self) and (n_queries == num_points))
    query_ids = jnp.arange(n_queries, dtype=jnp.int32)

    def single_query(query: Array, query_id: Array) -> Array:
        lower_bounds = _point_to_bbox_distance_sq(query, leaf_bbox_min, leaf_bbox_max)
        order = jnp.argsort(lower_bounds)

        def cond_fun(state):
            i, _count = state
            return (i < num_leaves) & (lower_bounds[order[i]] <= radius_sq)

        def body_fun(state):
            i, count_acc = state
            leaf_ord = order[i]
            start = leaf_start[leaf_ord]
            end = leaf_end[leaf_ord]
            count = end - start
            zero = jnp.asarray(0, dtype=start.dtype)
            block_points = jax.lax.dynamic_slice(
                points_sorted_padded,
                (start, zero),
                (max_leaf_points, tree.dimension),
            )
            block_ids = jax.lax.dynamic_slice(
                ids_sorted_padded, (start,), (max_leaf_points,)
            )
            valid = jnp.arange(max_leaf_points, dtype=jnp.int32) < count
            d2 = jnp.sum((block_points - query[None, :]) ** 2, axis=1)
            within = (d2 <= radius_sq) & valid
            if use_self_mask:
                within = within & (block_ids != query_id)
            return i + 1, count_acc + jnp.sum(within, dtype=count_acc.dtype)

        _, total = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (0, jnp.asarray(0, dtype=jnp.int32)),
        )
        return total

    return jax.vmap(single_query, in_axes=(0, 0))(queries_arr, query_ids)


@jaxtyped(typechecker=beartype)
def count_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    radius: float,
    include_self: bool = True,
    backend: Literal["tiled", "tree"] = "tiled",
    point_block_size: int = 2048,
) -> Array:
    """Count reference points within ``radius`` for each query point."""

    queries_arr = _validate_queries(tree, queries)
    if radius < 0.0:
        raise ValueError(f"radius must be >= 0, received {radius}")
    if backend not in {"tiled", "tree"}:
        raise ValueError("backend must be one of: 'tiled', 'tree'")
    if backend == "tree":
        return _count_neighbors_tree(
            tree,
            queries_arr,
            radius=radius,
            include_self=include_self,
        )
    return _count_neighbors_tiled(
        tree,
        queries_arr,
        radius=radius,
        include_self=include_self,
        point_block_size=point_block_size,
    )


@jaxtyped(typechecker=beartype)
def build_and_query(
    points: Array,
    queries: Array,
    *,
    k: int = 1,
    leaf_size: int = 32,
    backend: Literal["dense", "tiled", "tree"] = "tiled",
) -> tuple[Array, Array]:
    """Convenience function to build a tree and run nearest-neighbor queries."""

    tree = build_kdtree(points, leaf_size=leaf_size)
    return query_neighbors(tree, queries, k=k, backend=backend)


__all__ = [
    "KDTree",
    "build_and_query",
    "build_kdtree",
    "count_neighbors",
    "query_neighbors",
]
