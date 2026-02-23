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
    particle_indices: Array
    node_start: Array
    node_end: Array
    node_ranges: Array
    parent: Array
    left_child: Array
    right_child: Array
    num_internal_nodes: int
    num_particles: Array
    use_morton_geometry: Array
    split_dim: Array
    split_value: Array
    bbox_min: Array
    bbox_max: Array
    leaf_nodes: Array
    leaf_start: Array
    leaf_end: Array
    leaf_point_ids: Array
    leaf_valid_mask: Array
    node_to_leaf: Array
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
            tree.particle_indices,
            tree.node_start,
            tree.node_end,
            tree.node_ranges,
            tree.parent,
            tree.left_child,
            tree.right_child,
            tree.num_particles,
            tree.use_morton_geometry,
            tree.split_dim,
            tree.split_value,
            tree.bbox_min,
            tree.bbox_max,
            tree.leaf_nodes,
            tree.leaf_start,
            tree.leaf_end,
            tree.leaf_point_ids,
            tree.leaf_valid_mask,
            tree.node_to_leaf,
        )
        aux = (tree.leaf_size, tree.num_internal_nodes)
        return children, aux

    def unflatten(aux, children):
        leaf_size, num_internal_nodes = aux
        (
            points,
            indices,
            particle_indices,
            node_start,
            node_end,
            node_ranges,
            parent,
            left_child,
            right_child,
            num_particles,
            use_morton_geometry,
            split_dim,
            split_value,
            bbox_min,
            bbox_max,
            leaf_nodes,
            leaf_start,
            leaf_end,
            leaf_point_ids,
            leaf_valid_mask,
            node_to_leaf,
        ) = children
        return KDTree(
            points=points,
            indices=indices,
            particle_indices=particle_indices,
            node_start=node_start,
            node_end=node_end,
            node_ranges=node_ranges,
            parent=parent,
            left_child=left_child,
            right_child=right_child,
            num_internal_nodes=num_internal_nodes,
            num_particles=num_particles,
            use_morton_geometry=use_morton_geometry,
            split_dim=split_dim,
            split_value=split_value,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            leaf_nodes=leaf_nodes,
            leaf_start=leaf_start,
            leaf_end=leaf_end,
            leaf_point_ids=leaf_point_ids,
            leaf_valid_mask=leaf_valid_mask,
            node_to_leaf=node_to_leaf,
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


def _heap_inorder_and_ranges(n: int) -> tuple[Array, Array]:
    """Return inorder node order and inclusive subtree ranges for heap nodes."""

    left = [2 * i + 1 if (2 * i + 1) < n else -1 for i in range(n)]
    right = [2 * i + 2 if (2 * i + 2) < n else -1 for i in range(n)]

    inorder_nodes: list[int] = []
    stack: list[int] = []
    cur = 0 if n > 0 else -1
    while stack or (cur >= 0):
        while cur >= 0:
            stack.append(cur)
            cur = left[cur]
        cur = stack.pop()
        inorder_nodes.append(cur)
        cur = right[cur]

    inorder_pos = [0] * n
    for pos, node in enumerate(inorder_nodes):
        inorder_pos[node] = pos

    subtree_sizes = [1] * n
    for i in range(n - 1, -1, -1):
        l = left[i]
        r = right[i]
        size = 1
        if l >= 0:
            size += subtree_sizes[l]
        if r >= 0:
            size += subtree_sizes[r]
        subtree_sizes[i] = size

    ranges = []
    for i in range(n):
        l = left[i]
        l_size = subtree_sizes[l] if l >= 0 else 0
        start = inorder_pos[i] - l_size
        end = start + subtree_sizes[i] - 1
        ranges.append((start, end))

    return (
        jnp.asarray(inorder_nodes, dtype=jnp.int32),
        jnp.asarray(ranges, dtype=jnp.int32),
    )


def _build_kdtree_topology(points: Array, leaf_size: int) -> tuple[Array, ...]:
    """Build KD-tree topology in heap order using JAX primitives.

    The build follows Wald-style level scans (as in jaxkd), then derives
    child links, subtree extents, and leaf buffers expected by this API.
    """

    leaf_size_int = int(leaf_size)
    n = int(points.shape[0])
    dim = int(points.shape[1])

    array_index = jnp.arange(n, dtype=jnp.int32)
    n_levels = n.bit_length()

    def step(carry, level):
        nodes, indices, split_dims = carry
        pts_sorted = points[indices]
        dim_max = jax.ops.segment_max(pts_sorted, nodes, num_segments=n)
        dim_min = jax.ops.segment_min(pts_sorted, nodes, num_segments=n)
        new_split_dims = jnp.asarray(jnp.argmax(dim_max - dim_min, axis=-1), dtype=jnp.int32)
        split_dims = jnp.where(array_index < (1 << level) - 1, split_dims, new_split_dims)
        points_along_dim = jnp.take_along_axis(
            pts_sorted,
            split_dims[nodes][:, None],
            axis=-1,
        ).squeeze(-1)

        nodes, _, indices = jax.lax.sort(
            (nodes, points_along_dim, indices),
            dimension=0,
            num_keys=2,
        )

        height = n_levels - level - 1
        n_left_siblings = nodes - ((1 << level) - 1)
        branch_start = (
            ((1 << level) - 1)
            + n_left_siblings * ((1 << height) - 1)
            + jnp.minimum(n_left_siblings * (1 << height), n - ((1 << (n_levels - 1)) - 1))
        )

        left_child = 2 * nodes + 1
        child_height = jnp.maximum(0, height - 1)
        first_left_leaf = ~((~left_child) << child_height)
        left_branch_size = ((1 << child_height) - 1) + jnp.minimum(
            1 << child_height,
            jnp.maximum(0, n - first_left_leaf),
        )

        pivot_position = branch_start + left_branch_size
        right_child = 2 * nodes + 2
        nodes = jax.lax.select(
            (array_index == pivot_position) | (array_index < (1 << level) - 1),
            nodes,
            jax.lax.select(array_index < pivot_position, left_child, right_child),
        )
        return (nodes, indices, split_dims), None

    nodes0 = jnp.zeros(n, dtype=jnp.int32)
    indices0 = jnp.arange(n, dtype=jnp.int32)
    split_dims0 = -jnp.ones(n, dtype=jnp.int32)
    (_, order, split_dim_arr), _ = jax.lax.scan(
        step,
        (nodes0, indices0, split_dims0),
        jnp.arange(n_levels, dtype=jnp.int32),
    )
    split_dim_arr = split_dim_arr.at[n // 2 :].set(jnp.asarray(-1, dtype=jnp.int32))

    node_ids = jnp.arange(n, dtype=jnp.int32)
    left_raw = 2 * node_ids + 1
    right_raw = left_raw + 1
    left_child_full = jnp.where(left_raw < n, left_raw, -1).astype(jnp.int32)
    right_child_full = jnp.where(right_raw < n, right_raw, -1).astype(jnp.int32)

    num_internal = n // 2
    left_child = left_child_full[:num_internal]
    right_child = right_child_full[:num_internal]
    parent = jnp.where(
        node_ids == 0,
        jnp.asarray(-1, dtype=jnp.int32),
        (node_ids - 1) // 2,
    ).astype(jnp.int32)

    safe_dim = jnp.clip(split_dim_arr, 0, dim - 1)
    safe_idx = order
    split_value_arr = jnp.where(
        split_dim_arr >= 0,
        points[safe_idx, safe_dim],
        jnp.nan,
    )

    subtree_size = jnp.zeros((n,), dtype=jnp.int32)

    def size_body(t, sizes):
        i = n - 1 - t
        l = left_child_full[i]
        r = right_child_full[i]
        l_size = jnp.where(l >= 0, sizes[l], jnp.asarray(0, dtype=sizes.dtype))
        r_size = jnp.where(r >= 0, sizes[r], jnp.asarray(0, dtype=sizes.dtype))
        return sizes.at[i].set(1 + l_size + r_size)

    subtree_size = jax.lax.fori_loop(0, n, size_body, subtree_size)
    split_dim_arr = jnp.where(
        subtree_size <= leaf_size_int,
        jnp.asarray(-1, dtype=jnp.int32),
        split_dim_arr,
    )
    node_start = jnp.maximum(0, node_ids + 1 - subtree_size).astype(jnp.int32)
    node_end = (node_start + subtree_size).astype(jnp.int32)

    point_for_node = points[order]
    bbox_min = jnp.zeros((n, dim), dtype=points.dtype)
    bbox_max = jnp.zeros((n, dim), dtype=points.dtype)

    def bbox_body(t, state):
        mins, maxs = state
        i = n - 1 - t
        p = point_for_node[i]
        l = left_child_full[i]
        r = right_child_full[i]
        l_min = jnp.where(l >= 0, mins[l], p)
        l_max = jnp.where(l >= 0, maxs[l], p)
        r_min = jnp.where(r >= 0, mins[r], p)
        r_max = jnp.where(r >= 0, maxs[r], p)
        mn = jnp.minimum(p, jnp.minimum(l_min, r_min))
        mx = jnp.maximum(p, jnp.maximum(l_max, r_max))
        mins = mins.at[i].set(mn)
        maxs = maxs.at[i].set(mx)
        return mins, maxs

    bbox_min, bbox_max = jax.lax.fori_loop(0, n, bbox_body, (bbox_min, bbox_max))

    leaf_mask = split_dim_arr < 0
    leaf_nodes = jnp.nonzero(leaf_mask, size=n, fill_value=-1)[0]
    num_leaves = jnp.sum(leaf_mask, dtype=jnp.int32)
    valid_leaf_rows = jnp.arange(n, dtype=jnp.int32) < num_leaves
    leaf_nodes = jnp.where(valid_leaf_rows, leaf_nodes, -1)
    safe_leaf_nodes = jnp.clip(leaf_nodes, 0, n - 1)

    leaf_start = jnp.where(valid_leaf_rows, node_start[safe_leaf_nodes], 0).astype(jnp.int32)
    leaf_end = jnp.where(valid_leaf_rows, node_end[safe_leaf_nodes], 0).astype(jnp.int32)

    max_leaf_points = leaf_size_int
    leaf_point_ids = -jnp.ones((n, max_leaf_points), dtype=jnp.int32)
    leaf_valid_mask = jnp.zeros((n, max_leaf_points), dtype=jnp.bool_)

    def leaf_body(i, state):
        i32 = jnp.asarray(i, dtype=jnp.int32)
        ids_arr, valid_arr = state
        is_valid_row = i32 < num_leaves
        root = safe_leaf_nodes[i32]

        def fill_row(_):
            ids, valid = _collect_subtree_points(
                order,
                root=root,
                num_nodes=n,
                max_points=max_leaf_points,
            )
            return ids, valid

        row_ids, row_valid = jax.lax.cond(
            is_valid_row,
            fill_row,
            lambda _: (
                -jnp.ones((max_leaf_points,), dtype=jnp.int32),
                jnp.zeros((max_leaf_points,), dtype=jnp.bool_),
            ),
            operand=None,
        )
        ids_arr = ids_arr.at[i32].set(row_ids)
        valid_arr = valid_arr.at[i32].set(row_valid)
        return ids_arr, valid_arr

    leaf_point_ids, leaf_valid_mask = jax.lax.fori_loop(
        0,
        n,
        leaf_body,
        (leaf_point_ids, leaf_valid_mask),
    )

    node_to_leaf = -jnp.ones((n,), dtype=jnp.int32)

    def node_to_leaf_body(i, arr):
        i32 = jnp.asarray(i, dtype=jnp.int32)
        return jax.lax.cond(
            i32 < num_leaves,
            lambda _: arr.at[safe_leaf_nodes[i32]].set(i32),
            lambda _: arr,
            operand=None,
        )

    node_to_leaf = jax.lax.fori_loop(0, n, node_to_leaf_body, node_to_leaf)

    inorder_nodes, node_ranges = _heap_inorder_and_ranges(n)
    particle_indices = order[inorder_nodes]
    num_particles = jnp.asarray(n, dtype=jnp.int32)
    use_morton_geometry = jnp.asarray(False)

    return (
        order.astype(jnp.int32),
        particle_indices.astype(jnp.int32),
        node_start,
        node_end,
        node_ranges,
        parent,
        left_child,
        right_child,
        num_internal,
        num_particles,
        use_morton_geometry,
        split_dim_arr.astype(jnp.int32),
        split_value_arr,
        bbox_min,
        bbox_max,
        leaf_nodes.astype(jnp.int32),
        leaf_start,
        leaf_end,
        leaf_point_ids.astype(jnp.int32),
        leaf_valid_mask.astype(jnp.bool_),
        node_to_leaf,
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
        particle_indices=topology[1],
        node_start=topology[2],
        node_end=topology[3],
        node_ranges=topology[4],
        parent=topology[5],
        left_child=topology[6],
        right_child=topology[7],
        num_internal_nodes=int(topology[8]),
        num_particles=topology[9],
        use_morton_geometry=topology[10],
        split_dim=topology[11],
        split_value=topology[12],
        bbox_min=topology[13],
        bbox_max=topology[14],
        leaf_nodes=topology[15],
        leaf_start=topology[16],
        leaf_end=topology[17],
        leaf_point_ids=topology[18],
        leaf_valid_mask=topology[19],
        node_to_leaf=topology[20],
        leaf_size=int(leaf_size),
    )


def _query_neighbors_dense(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
    return_squared: bool,
) -> tuple[Array, Array]:
    distances_sq = _pairwise_squared_distances(queries_arr, tree.points)
    if exclude_self and (queries_arr.shape[0] == tree.num_points):
        diag_len = queries_arr.shape[0]
        diag_idx = jnp.arange(diag_len)
        distances_sq = distances_sq.at[diag_idx, diag_idx].set(jnp.inf)

    top_scores, indices = jax.lax.top_k(-distances_sq, k)
    best_d2 = jnp.maximum(-top_scores, 0.0)
    if return_squared:
        return indices, best_d2
    return indices, jnp.sqrt(best_d2)


def _query_neighbors_tiled(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
    point_block_size: int,
    return_squared: bool,
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
    best_d2 = jnp.maximum(best_d2, 0.0)
    if return_squared:
        return best_idx, best_d2
    return best_idx, jnp.sqrt(best_d2)


def _point_to_bbox_distance_sq(query: Array, bbox_min: Array, bbox_max: Array) -> Array:
    clipped = jnp.clip(query[None, :], bbox_min, bbox_max)
    delta = query[None, :] - clipped
    return jnp.sum(delta * delta, axis=1)


def _point_to_bbox_distance_sq_single(
    query: Array,
    bbox_min: Array,
    bbox_max: Array,
) -> Array:
    clipped = jnp.clip(query, bbox_min, bbox_max)
    delta = query - clipped
    return jnp.sum(delta * delta)


def _topk_merge(
    best_d2: Array, best_idx: Array, cand_d2: Array, cand_idx: Array
) -> tuple[Array, Array]:
    k = best_d2.shape[0]
    merged_d2 = jnp.concatenate([best_d2, cand_d2], axis=0)
    merged_idx = jnp.concatenate([best_idx, cand_idx], axis=0)
    top_scores, top_cols = jax.lax.top_k(-merged_d2, k)
    new_d2 = -top_scores
    new_idx = jnp.take_along_axis(merged_idx, top_cols, axis=0)
    return new_d2, new_idx


def _collect_subtree_points(
    indices: Array,
    *,
    root: Array,
    num_nodes: int,
    max_points: int,
) -> tuple[Array, Array]:
    """Collect point ids from a subtree rooted at ``root`` (bounded by ``max_points``)."""

    root_i32 = jnp.asarray(root, dtype=jnp.int32)
    stack = jnp.full((max_points,), -1, dtype=jnp.int32).at[0].set(root_i32)
    stack_size = jnp.asarray(1, dtype=jnp.int32)
    out_ids = jnp.full((max_points,), -1, dtype=jnp.int32)
    out_size = jnp.asarray(0, dtype=jnp.int32)

    def cond_fun(state):
        _stk, stk_size, _ids, out_n = state
        return (stk_size > 0) & (out_n < max_points)

    def body_fun(state):
        stk, stk_size, ids, out_n = state
        stk_size = stk_size - jnp.asarray(1, dtype=jnp.int32)
        node = stk[stk_size]
        point_id = indices[node]
        ids = ids.at[out_n].set(point_id)
        out_n = out_n + jnp.asarray(1, dtype=jnp.int32)

        left = 2 * node + 1
        right = left + 1
        has_left = left < num_nodes
        has_right = right < num_nodes

        stk, stk_size = jax.lax.cond(
            has_right & (stk_size < max_points),
            lambda _: (stk.at[stk_size].set(right), stk_size + jnp.asarray(1, dtype=jnp.int32)),
            lambda _: (stk, stk_size),
            operand=None,
        )
        stk, stk_size = jax.lax.cond(
            has_left & (stk_size < max_points),
            lambda _: (stk.at[stk_size].set(left), stk_size + jnp.asarray(1, dtype=jnp.int32)),
            lambda _: (stk, stk_size),
            operand=None,
        )
        return stk, stk_size, ids, out_n

    _stack, _stack_size, out_ids, out_size = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (stack, stack_size, out_ids, out_size),
    )
    valid = jnp.arange(max_points, dtype=jnp.int32) < out_size
    return out_ids, valid


def _query_neighbors_tree(
    tree: KDTree,
    queries_arr: Array,
    *,
    k: int,
    exclude_self: bool,
    return_squared: bool,
) -> tuple[Array, Array]:
    points = tree.points
    indices = jnp.asarray(tree.indices, dtype=jnp.int32)
    split_dims = jnp.asarray(tree.split_dim, dtype=jnp.int32)
    leaf_point_ids = jnp.asarray(tree.leaf_point_ids, dtype=jnp.int32)
    leaf_valid = jnp.asarray(tree.leaf_valid_mask, dtype=jnp.bool_)
    node_to_leaf = jnp.asarray(tree.node_to_leaf, dtype=jnp.int32)
    num_nodes = int(tree.node_start.shape[0])
    n_queries = int(queries_arr.shape[0])
    num_points = tree.num_points
    use_self_mask = bool(exclude_self and (n_queries == num_points))
    query_ids = jnp.arange(n_queries, dtype=jnp.int32)

    def single_query(query: Array, query_id: Array) -> tuple[Array, Array]:
        best_d2 = jnp.full((k,), jnp.inf, dtype=queries_arr.dtype)
        best_idx = jnp.full((k,), -1, dtype=jnp.int32)
        square_radius = jnp.asarray(jnp.inf, dtype=queries_arr.dtype)
        root = jnp.asarray(0, dtype=jnp.int32)
        root_parent = (root - 1) // 2

        def cond_fun(state):
            current, _prev, _d2, _idx, _radius = state
            return current != root_parent

        def body_fun(state):
            current, previous, d2_state, idx_state, radius_state = state
            parent = (current - 1) // 2
            is_cluster_leaf = split_dims[current] < 0

            def update_node(_):
                point_id = indices[current]
                point = points[point_id]
                point_d2 = jnp.sum((query - point) ** 2)
                if use_self_mask:
                    point_d2 = jnp.where(point_id == query_id, jnp.inf, point_d2)
                worst_slot = jnp.argmax(d2_state)
                better = point_d2 < d2_state[worst_slot]
                next_d2 = jnp.where(better, d2_state.at[worst_slot].set(point_d2), d2_state)
                next_idx = jnp.where(
                    better,
                    idx_state.at[worst_slot].set(point_id),
                    idx_state,
                )
                next_radius = jnp.max(next_d2)
                return next_d2, next_idx, next_radius

            def update_cluster(_):
                leaf_ord = node_to_leaf[current]
                cand_idx = leaf_point_ids[leaf_ord]
                valid = leaf_valid[leaf_ord]
                safe_idx = jnp.clip(cand_idx, 0, num_points - 1)
                cand_points = points[safe_idx]
                cand_d2 = jnp.sum((cand_points - query[None, :]) ** 2, axis=1)
                cand_d2 = jnp.where(valid, cand_d2, jnp.inf)
                if use_self_mask:
                    cand_d2 = jnp.where(cand_idx == query_id, jnp.inf, cand_d2)
                next_d2, next_idx = _topk_merge(d2_state, idx_state, cand_d2, cand_idx)
                return next_d2, next_idx, jnp.max(next_d2)

            d2_state, idx_state, radius_state = jax.lax.cond(
                previous == parent,
                lambda _: jax.lax.cond(
                    is_cluster_leaf,
                    update_cluster,
                    update_node,
                    operand=None,
                ),
                lambda _: (d2_state, idx_state, radius_state),
                operand=None,
            )

            def next_for_cluster(_):
                return parent

            def next_for_internal(_):
                split_dim_raw = split_dims[current]
                split_dim = jnp.where(
                    split_dim_raw >= 0,
                    split_dim_raw,
                    jnp.asarray(0, dtype=jnp.int32),
                )
                split_distance = query[split_dim] - points[indices[current], split_dim]
                near_side = jnp.asarray(split_distance > 0, dtype=jnp.int32)
                near_child = 2 * current + 1 + near_side
                far_child = 2 * current + 2 - near_side
                far_in_range = (split_distance * split_distance) <= radius_state
                return jax.lax.select(
                    (previous == near_child) | ((previous == parent) & (near_child >= num_nodes)),
                    jax.lax.select((far_child < num_nodes) & far_in_range, far_child, parent),
                    jax.lax.select(previous == parent, near_child, parent),
                )

            next_node = jax.lax.cond(
                is_cluster_leaf & (previous == parent),
                next_for_cluster,
                next_for_internal,
                operand=None,
            )
            return next_node, current, d2_state, idx_state, radius_state

        _current, _previous, best_d2, best_idx, _radius = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (root, root_parent, best_d2, best_idx, square_radius),
        )
        best_d2 = jnp.maximum(best_d2, 0.0)
        if return_squared:
            best_d2, best_idx = jax.lax.sort((best_d2, best_idx), dimension=0, num_keys=2)
            return best_idx, best_d2
        safe_idx = jnp.clip(best_idx, 0, num_points - 1)
        distances = jnp.linalg.norm(points[safe_idx] - query[None, :], axis=-1)
        distances = jnp.where(best_idx >= 0, distances, jnp.inf)
        distances, best_idx = jax.lax.sort((distances, best_idx), dimension=0, num_keys=2)
        return best_idx, distances

    return jax.vmap(single_query, in_axes=(0, 0))(queries_arr, query_ids)


def _resolve_query_backend(tree: KDTree, queries_arr: Array, backend: str, *, k: int) -> str:
    if backend != "auto":
        return backend
    n = int(tree.num_points)
    q = int(queries_arr.shape[0])
    work = n * q
    platform = jax.devices()[0].platform

    if platform == "gpu":
        if (work <= 8_000_000) and (k <= 16):
            return "tiled"
        return "tree"

    # CPU: dense is excellent for small work; tiled is best medium; tree wins large.
    if (work <= 1_500_000) and (k <= 32):
        return "dense"
    if (work <= 20_000_000) and (k <= 16):
        return "tiled"
    return "tree"


@jaxtyped(typechecker=beartype)
def query_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    k: int = 1,
    exclude_self: bool = False,
    backend: Literal["dense", "tiled", "tree", "auto"] = "tiled",
    point_block_size: int = 2048,
    return_squared: bool = False,
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
            ``tree`` uses KD leaf bounding boxes to prune exact search;
            ``auto`` selects a backend by problem size.
        point_block_size: Block size used by the tiled backend.
        return_squared: If ``True``, return squared Euclidean distances.

    Returns:
        Tuple ``(indices, distances)`` with shapes ``(n_queries, k)``.
    """

    queries_arr = _validate_queries(tree, queries)
    if k < 1:
        raise ValueError(f"k must be >= 1, received {k}")
    if k > tree.num_points:
        raise ValueError(f"k must be <= num_points={tree.num_points}, received {k}")
    if backend not in {"dense", "tiled", "tree", "auto"}:
        raise ValueError("backend must be one of: 'dense', 'tiled', 'tree', 'auto'")

    resolved_backend = _resolve_query_backend(tree, queries_arr, backend, k=k)

    if resolved_backend == "dense":
        return _query_neighbors_dense(
            tree,
            queries_arr,
            k=k,
            exclude_self=exclude_self,
            return_squared=return_squared,
        )
    if resolved_backend == "tree":
        return _query_neighbors_tree(
            tree,
            queries_arr,
            k=k,
            exclude_self=exclude_self,
            return_squared=return_squared,
        )
    return _query_neighbors_tiled(
        tree,
        queries_arr,
        k=k,
        exclude_self=exclude_self,
        point_block_size=point_block_size,
        return_squared=return_squared,
    )


def _prepare_radius_inputs(queries_arr: Array, radius: float | Array) -> tuple[Array, int]:
    radius_arr = jnp.asarray(radius, dtype=queries_arr.dtype)
    if not isinstance(radius_arr, jax.core.Tracer):
        if bool(jnp.any(radius_arr < 0.0)):
            raise ValueError(f"radius must be >= 0, received {radius}")
    if radius_arr.ndim == 0:
        radius_qr = jnp.broadcast_to(radius_arr[None, None], (queries_arr.shape[0], 1))
        return radius_qr * radius_qr, 0
    if radius_arr.ndim == 1:
        radius_qr = jnp.broadcast_to(radius_arr[None, :], (queries_arr.shape[0], radius_arr.shape[0]))
        return radius_qr * radius_qr, 1
    if radius_arr.ndim == 2:
        if radius_arr.shape[0] != queries_arr.shape[0]:
            raise ValueError(
                "radius with ndim=2 must have shape (n_queries, n_radii); "
                f"received {radius_arr.shape} for n_queries={queries_arr.shape[0]}"
            )
        return radius_arr * radius_arr, 2
    raise ValueError(f"radius must be a scalar, 1D, or 2D array; received ndim={radius_arr.ndim}")


def _count_neighbors_tiled(
    tree: KDTree,
    queries_arr: Array,
    *,
    radius_sq_qr: Array,
    include_self: bool,
    point_block_size: int,
) -> Array:
    num_points = tree.num_points
    if point_block_size < 1:
        raise ValueError(f"point_block_size must be >= 1, received {point_block_size}")

    n_queries = queries_arr.shape[0]
    n_r = int(radius_sq_qr.shape[1])
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
        within = (d2[:, :, None] <= radius_sq_qr[:, None, :]) & valid[None, :, None]

        if (not include_self) and (n_queries == num_points):
            self_mask = query_ids == block_ids[None, :]
            within = within & (~self_mask[:, :, None])

        return counts + jnp.sum(within, axis=1, dtype=counts.dtype)

    init = jnp.zeros((n_queries, n_r), dtype=jnp.int32)
    return jax.lax.fori_loop(0, num_blocks, body, init)


def _count_neighbors_tree(
    tree: KDTree,
    queries_arr: Array,
    *,
    radius_sq_qr: Array,
    include_self: bool,
) -> Array:
    points = tree.points
    indices = jnp.asarray(tree.indices, dtype=jnp.int32)
    num_nodes = int(tree.node_start.shape[0])
    split_dims = jnp.asarray(tree.split_dim, dtype=jnp.int32)
    leaf_point_ids = jnp.asarray(tree.leaf_point_ids, dtype=jnp.int32)
    leaf_valid = jnp.asarray(tree.leaf_valid_mask, dtype=jnp.bool_)
    node_to_leaf = jnp.asarray(tree.node_to_leaf, dtype=jnp.int32)
    n_queries = int(queries_arr.shape[0])
    n_r = int(radius_sq_qr.shape[1])
    num_points = tree.num_points
    use_self_mask = bool((not include_self) and (n_queries == num_points))
    query_ids = jnp.arange(n_queries, dtype=jnp.int32)

    def single_query(query: Array, query_id: Array) -> Array:
        radius_sq = radius_sq_qr[query_id]
        radius_sq_max = jnp.max(radius_sq)
        root = jnp.asarray(0, dtype=jnp.int32)
        root_parent = (root - 1) // 2

        def cond_fun(state):
            current, _prev, _count = state
            return current != root_parent

        def body_fun(state):
            current, previous, count = state
            parent = (current - 1) // 2
            is_cluster_leaf = split_dims[current] < 0

            def update_node(_):
                point_id = indices[current]
                point = points[point_id]
                point_d2 = jnp.sum((query - point) ** 2)
                within = point_d2 <= radius_sq
                if use_self_mask:
                    within = within & (point_id != query_id)
                return count + within.astype(jnp.int32)

            def update_cluster(_):
                leaf_ord = node_to_leaf[current]
                cand_idx = leaf_point_ids[leaf_ord]
                valid = leaf_valid[leaf_ord]
                safe_idx = jnp.clip(cand_idx, 0, num_points - 1)
                cand_points = points[safe_idx]
                cand_d2 = jnp.sum((cand_points - query[None, :]) ** 2, axis=1)
                within = (cand_d2[:, None] <= radius_sq[None, :]) & valid[:, None]
                if use_self_mask:
                    within = within & (cand_idx[:, None] != query_id)
                return count + jnp.sum(within, axis=0, dtype=jnp.int32)

            count = jax.lax.cond(
                previous == parent,
                lambda _: jax.lax.cond(
                    is_cluster_leaf,
                    update_cluster,
                    update_node,
                    operand=None,
                ),
                lambda _: count,
                operand=None,
            )

            def next_for_cluster(_):
                return parent

            def next_for_internal(_):
                split_dim_raw = split_dims[current]
                split_dim = jnp.where(
                    split_dim_raw >= 0,
                    split_dim_raw,
                    jnp.asarray(0, dtype=jnp.int32),
                )
                split_distance = query[split_dim] - points[indices[current], split_dim]
                near_side = jnp.asarray(split_distance > 0, dtype=jnp.int32)
                near_child = 2 * current + 1 + near_side
                far_child = 2 * current + 2 - near_side
                far_in_range = (split_distance * split_distance) <= radius_sq_max
                return jax.lax.select(
                    (previous == near_child)
                    | ((previous == parent) & (near_child >= num_nodes)),
                    jax.lax.select((far_child < num_nodes) & far_in_range, far_child, parent),
                    jax.lax.select(previous == parent, near_child, parent),
                )

            next_node = jax.lax.cond(
                is_cluster_leaf & (previous == parent),
                next_for_cluster,
                next_for_internal,
                operand=None,
            )
            return next_node, current, count

        _current, _previous, total = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (
                root,
                root_parent,
                jnp.zeros((n_r,), dtype=jnp.int32),
            ),
        )
        return total

    return jax.vmap(single_query, in_axes=(0, 0))(queries_arr, query_ids)


def _resolve_count_backend(
    tree: KDTree,
    queries_arr: Array,
    backend: str,
    *,
    num_radii: int,
) -> str:
    if backend != "auto":
        return backend
    work = int(tree.num_points) * int(queries_arr.shape[0]) * int(num_radii)
    platform = jax.devices()[0].platform
    if platform == "gpu":
        if work <= 20_000_000:
            return "tiled"
        return "tree"
    if work <= 3_000_000:
        return "tiled"
    return "tree"


@jaxtyped(typechecker=beartype)
def count_neighbors(
    tree: KDTree,
    queries: Array,
    *,
    radius: float | Array,
    include_self: bool = True,
    backend: Literal["tiled", "tree", "auto"] = "tiled",
    point_block_size: int = 2048,
) -> Array:
    """Count reference points within radius/radii for each query point."""

    queries_arr = _validate_queries(tree, queries)
    radius_sq_qr, radius_ndim = _prepare_radius_inputs(queries_arr, radius)
    if backend not in {"tiled", "tree", "auto"}:
        raise ValueError("backend must be one of: 'tiled', 'tree', 'auto'")
    resolved_backend = _resolve_count_backend(
        tree,
        queries_arr,
        backend,
        num_radii=int(radius_sq_qr.shape[1]),
    )
    if resolved_backend == "tree":
        counts = _count_neighbors_tree(
            tree,
            queries_arr,
            radius_sq_qr=radius_sq_qr,
            include_self=include_self,
        )
    else:
        counts = _count_neighbors_tiled(
            tree,
            queries_arr,
            radius_sq_qr=radius_sq_qr,
            include_self=include_self,
            point_block_size=point_block_size,
        )
    if radius_ndim == 0:
        return counts[:, 0]
    return counts


@jaxtyped(typechecker=beartype)
def build_and_query(
    points: Array,
    queries: Array,
    *,
    k: int = 1,
    leaf_size: int = 32,
    backend: Literal["dense", "tiled", "tree"] = "tiled",
    return_squared: bool = False,
) -> tuple[Array, Array]:
    """Convenience function to build a tree and run nearest-neighbor queries."""

    tree = build_kdtree(points, leaf_size=leaf_size)
    return query_neighbors(tree, queries, k=k, backend=backend, return_squared=return_squared)


__all__ = [
    "KDTree",
    "build_and_query",
    "build_kdtree",
    "count_neighbors",
    "query_neighbors",
]
