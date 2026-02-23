"""Geometry helpers for radix trees.

This module derives per-node bounding geometry from Morton-sorted
particles so the Fast Multipole Method can reason about node extents
when deciding whether a multipole expansion is admissible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import core as jax_core
from jax import lax
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index
from .morton import _compact3_u64
from .tree import (
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
    get_num_levels,
    require_morton_topology,
)


class TreeGeometry(NamedTuple):
    """Bounding information for every node in a radix tree."""

    center: Array
    half_extent: Array
    radius: Array
    max_extent: Array


class LevelMajorTreeGeometry(NamedTuple):
    """Tree geometry reshaped into dense level-major buffers."""

    centers: Array
    half_extents: Array
    radii: Array
    max_extents: Array
    level_counts: Array
    node_indices: Array


def _validate_inputs(tree: object, positions_sorted: Array) -> None:
    total_nodes = tree.parent.shape[0]
    if tree.node_ranges.shape[0] != total_nodes:
        raise ValueError("tree.node_ranges must align with tree.parent shape")
    num_particles = jnp.asarray(tree.num_particles)
    if not isinstance(num_particles, jax_core.Tracer):
        if positions_sorted.shape[0] != int(num_particles):
            raise ValueError("positions_sorted must match tree.num_particles")
    if positions_sorted.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")


@jaxtyped(typechecker=beartype)
def _compute_leaf_bounds(
    positions_sorted: Array,
    leaf_starts: Array,
    leaf_counts: Array,
) -> tuple[Array, Array]:
    index_dtype = leaf_starts.dtype
    dtype = positions_sorted.dtype
    dim = positions_sorted.shape[1]
    one = as_index(1)
    zero = as_index(0)
    inf_vec = jnp.full((dim,), jnp.inf, dtype=dtype)
    neg_inf_vec = jnp.full((dim,), -jnp.inf, dtype=dtype)

    def single(start, count):
        row_start = lax.convert_element_type(start, index_dtype)
        row_count = lax.convert_element_type(count, index_dtype)

        def cond_fun(state):
            i, _mn, _mx = state
            return i < row_count

        def body_fun(state):
            i, mn, mx = state
            idx = row_start + i
            point = lax.dynamic_index_in_dim(
                positions_sorted, idx, axis=0, keepdims=False
            )
            return i + one, jnp.minimum(mn, point), jnp.maximum(mx, point)

        _, mins, maxs = lax.while_loop(
            cond_fun,
            body_fun,
            (zero, inf_vec, neg_inf_vec),
        )
        return mins, maxs

    mins, maxs = jax.vmap(single)(leaf_starts, leaf_counts)
    return mins, maxs


_MAX_MORTON_LEVEL = 21


def _compute_leaf_bounds_from_morton(tree: object) -> tuple[Array, Array]:
    require_morton_topology(tree)

    bounds_min = jnp.asarray(tree.bounds_min)
    bounds_max = jnp.asarray(tree.bounds_max)
    domain = bounds_max - bounds_min

    leaf_codes = jnp.asarray(tree.leaf_codes, dtype=jnp.uint64)
    leaf_depths = jnp.asarray(tree.leaf_depths, dtype=INDEX_DTYPE)

    depth_clamped = jnp.clip(
        leaf_depths,
        as_index(0),
        as_index(_MAX_MORTON_LEVEL),
    )
    shift = jnp.asarray(_MAX_MORTON_LEVEL, dtype=jnp.uint64) - depth_clamped.astype(
        jnp.uint64
    )

    x_coords = _compact3_u64(leaf_codes)
    y_coords = _compact3_u64(leaf_codes >> jnp.uint64(1))
    z_coords = _compact3_u64(leaf_codes >> jnp.uint64(2))

    x_idx = x_coords >> shift
    y_idx = y_coords >> shift
    z_idx = z_coords >> shift

    indices = jnp.stack([x_idx, y_idx, z_idx], axis=1).astype(bounds_min.dtype)

    counts = jnp.left_shift(
        jnp.ones_like(depth_clamped, dtype=jnp.uint64),
        depth_clamped.astype(jnp.uint64),
    )
    counts = jnp.maximum(counts, jnp.uint64(1))
    counts = counts.astype(bounds_min.dtype)

    cell_sizes = domain[None, :] / counts[:, None]
    mins = bounds_min[None, :] + cell_sizes * indices
    maxs = mins + cell_sizes
    return mins, maxs


@jaxtyped(typechecker=beartype)
def compute_tree_geometry(
    tree: object,
    positions_sorted: Array,
) -> TreeGeometry:
    """Compute per-node bounding boxes and helper radii.

    Parameters
    ----------
    tree : object
        Topology object exposing tree-structure arrays.
    positions_sorted : Array
        Particle positions reordered into Morton order. This should be the
        ``positions`` output from ``build_tree(..., return_reordered=True)``.

    Returns
    -------
    TreeGeometry
        Named tuple containing node centers, half-extents, and bounding radii.
    """

    _validate_inputs(tree, positions_sorted)

    ranges = tree.node_ranges.astype(INDEX_DTYPE)
    num_internal = tree.left_child.shape[0]
    num_nodes = ranges.shape[0]

    use_morton_raw = getattr(tree, "use_morton_geometry", False)
    use_morton_bounds = jnp.asarray(use_morton_raw, dtype=jnp.bool_)
    has_morton_fields = all(
        hasattr(tree, name)
        for name in ("bounds_min", "bounds_max", "leaf_codes", "leaf_depths")
    )
    if (not has_morton_fields) and (not isinstance(use_morton_raw, jax_core.Tracer)):
        if bool(jnp.asarray(use_morton_raw)):
            require_morton_topology(tree)

    def _leaf_bounds_from_ranges(_):
        leaf_ranges = ranges[num_internal:]
        leaf_starts = leaf_ranges[:, 0]
        leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)
        return _compute_leaf_bounds(
            positions_sorted,
            leaf_starts,
            leaf_counts,
        )

    if has_morton_fields:
        leaf_min, leaf_max = lax.cond(
            use_morton_bounds,
            lambda _: _compute_leaf_bounds_from_morton(tree),
            _leaf_bounds_from_ranges,
            operand=None,
        )
    else:
        leaf_min, leaf_max = _leaf_bounds_from_ranges(None)

    mins = jnp.zeros((num_nodes, 3), dtype=positions_sorted.dtype)
    maxs = jnp.zeros((num_nodes, 3), dtype=positions_sorted.dtype)
    mins = mins.at[num_internal:].set(leaf_min)
    maxs = maxs.at[num_internal:].set(leaf_max)

    left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
    right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)

    indices_full = jnp.arange(num_nodes, dtype=INDEX_DTYPE)
    processed = indices_full >= as_index(num_internal)

    def cond_fun(state):
        _mins_state, _maxs_state, processed_state = state
        return jnp.any(~processed_state[:num_internal])

    def body_fun(state):
        mins_state, maxs_state, processed_state = state
        child_min = jnp.minimum(
            mins_state[left_child],
            mins_state[right_child],
        )
        child_max = jnp.maximum(
            maxs_state[left_child],
            maxs_state[right_child],
        )
        ready = processed_state[left_child] & processed_state[right_child]
        update_mask = (~processed_state[:num_internal]) & ready
        mins_internal = jnp.where(
            update_mask[:, None],
            child_min,
            mins_state[:num_internal],
        )
        maxs_internal = jnp.where(
            update_mask[:, None],
            child_max,
            maxs_state[:num_internal],
        )
        mins_state = mins_state.at[:num_internal].set(mins_internal)
        maxs_state = maxs_state.at[:num_internal].set(maxs_internal)
        processed_internal = processed_state[:num_internal] | update_mask
        processed_state = processed_state.at[:num_internal].set(processed_internal)
        return mins_state, maxs_state, processed_state

    if num_internal > 0:
        mins, maxs, _ = lax.while_loop(
            cond_fun,
            body_fun,
            (mins, maxs, processed),
        )

    centers = 0.5 * (mins + maxs)
    half_extents = 0.5 * (maxs - mins)
    min_extent = jnp.asarray(1e-6, dtype=half_extents.dtype)
    half_extents = jnp.maximum(half_extents, min_extent)
    radii = jnp.linalg.norm(half_extents, axis=1)
    max_extents = jnp.max(half_extents, axis=1)

    return TreeGeometry(centers, half_extents, radii, max_extents)


@jaxtyped(typechecker=beartype)
def geometry_to_level_major(
    tree: object,
    geometry: TreeGeometry,
) -> LevelMajorTreeGeometry:
    """Materialise padded, level-major views of node geometry."""

    node_levels = get_node_levels(tree)
    num_levels = get_num_levels(tree, node_levels=node_levels)
    if num_levels <= 0:
        raise ValueError("tree must contain at least one populated level")

    level_offsets = get_level_offsets(tree, node_levels=node_levels)
    level_counts_full = level_offsets[1:] - level_offsets[:-1]
    level_counts = level_counts_full[:num_levels]
    max_nodes = int(jnp.max(jnp.maximum(level_counts, 1)))

    def _make_buffer(template, fill_value=0):
        shape = (num_levels, max_nodes) + template.shape[1:]
        return jnp.full(shape, fill_value, dtype=template.dtype)

    centers_buf = _make_buffer(geometry.center)
    extents_buf = _make_buffer(geometry.half_extent)
    radii_buf = _make_buffer(geometry.radius)
    max_extents_buf = _make_buffer(geometry.max_extent)
    node_idx_buf = jnp.full((num_levels, max_nodes), -1, dtype=INDEX_DTYPE)

    nodes_by_level = get_nodes_by_level(tree, node_levels=node_levels)

    total_nodes = geometry.center.shape[0]
    node_indices = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)

    def body(level_idx, state):
        centers_arr, extents_arr, radii_arr, max_extents_arr, idx_arr = state
        count = level_counts[level_idx]
        start = level_offsets[level_idx]
        slots = jnp.arange(max_nodes, dtype=INDEX_DTYPE)
        valid = slots < count
        node_positions = start + slots
        safe_pos = jnp.where(valid, node_positions, as_index(0))
        level_nodes = jnp.where(
            valid,
            node_indices[safe_pos],
            as_index(-1),
        )

        safe_nodes = jnp.clip(level_nodes, min=0, max=total_nodes - 1)
        centers_values = geometry.center[safe_nodes]
        extents_values = geometry.half_extent[safe_nodes]
        radii_values = geometry.radius[safe_nodes]
        max_extents_values = geometry.max_extent[safe_nodes]

        centers_values = jnp.where(valid[:, None], centers_values, 0.0)
        extents_values = jnp.where(valid[:, None], extents_values, 0.0)
        radii_values = jnp.where(valid, radii_values, 0.0)
        max_extents_values = jnp.where(valid, max_extents_values, 0.0)

        centers_arr = centers_arr.at[level_idx].set(centers_values)
        extents_arr = extents_arr.at[level_idx].set(extents_values)
        radii_arr = radii_arr.at[level_idx].set(radii_values)
        max_extents_arr = max_extents_arr.at[level_idx].set(max_extents_values)
        idx_arr = idx_arr.at[level_idx].set(level_nodes)
        return centers_arr, extents_arr, radii_arr, max_extents_arr, idx_arr

    (
        centers_buf,
        extents_buf,
        radii_buf,
        max_extents_buf,
        node_idx_buf,
    ) = lax.fori_loop(
        0,
        num_levels,
        body,
        (centers_buf, extents_buf, radii_buf, max_extents_buf, node_idx_buf),
    )

    return LevelMajorTreeGeometry(
        centers=centers_buf,
        half_extents=extents_buf,
        radii=radii_buf,
        max_extents=max_extents_buf,
        level_counts=level_counts,
        node_indices=node_idx_buf,
    )


__all__ = [
    "LevelMajorTreeGeometry",
    "TreeGeometry",
    "compute_tree_geometry",
    "geometry_to_level_major",
]
