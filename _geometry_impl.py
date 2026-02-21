"""Geometry helpers for radix trees.

This module derives per-node bounding geometry from Morton-sorted
particles so the Fast Multipole Method can reason about node extents
when deciding whether a multipole expansion is admissible.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index
from .morton import _compact3_u64
from .tree import RadixTree


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


def _validate_inputs(tree: RadixTree, positions_sorted: Array) -> None:
    total_nodes = tree.parent.shape[0]
    if tree.node_ranges.shape[0] != total_nodes:
        raise ValueError("tree.node_ranges must align with tree.parent shape")
    if positions_sorted.shape[0] != tree.num_particles:
        raise ValueError("positions_sorted must match tree.num_particles")
    if positions_sorted.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")


@partial(jax.jit, static_argnames=("max_leaf_size",))
@jaxtyped(typechecker=beartype)
def _compute_leaf_bounds(
    padded_positions: Array,
    leaf_starts: Array,
    leaf_counts: Array,
    *,
    max_leaf_size: int,
) -> tuple[Array, Array]:
    max_leaf = int(max_leaf_size)

    index_dtype = leaf_starts.dtype

    def single(start, count):
        row_start = lax.convert_element_type(start, index_dtype)
        zero = jnp.asarray(0, dtype=index_dtype)
        segment = lax.dynamic_slice(
            padded_positions,
            (row_start, zero),
            (max_leaf, padded_positions.shape[1]),
        )
        idx = jnp.arange(max_leaf, dtype=index_dtype)
        mask = (idx < count)[:, None]
        min_mask = jnp.where(mask, segment, jnp.inf)
        max_mask = jnp.where(mask, segment, -jnp.inf)
        return jnp.min(min_mask, axis=0), jnp.max(max_mask, axis=0)

    mins, maxs = jax.vmap(single)(leaf_starts, leaf_counts)
    return mins, maxs


_MAX_MORTON_LEVEL = 21


def _compute_leaf_bounds_from_morton(tree: RadixTree) -> tuple[Array, Array]:
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
    tree: RadixTree,
    positions_sorted: Array,
) -> TreeGeometry:
    """Compute per-node bounding boxes and helper radii.

    Parameters
    ----------
    tree : RadixTree
        Radix tree constructed by :func:`yggdrasil.tree.build_tree`.
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
    num_internal = int(tree.num_internal_nodes)
    num_nodes = ranges.shape[0]

    use_morton_bounds = bool(
        jnp.asarray(tree.use_morton_geometry, dtype=jnp.bool_).item()
    )

    if use_morton_bounds:
        leaf_min, leaf_max = _compute_leaf_bounds_from_morton(tree)
    else:
        leaf_ranges = ranges[num_internal:]
        leaf_starts = leaf_ranges[:, 0]
        leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)

        max_leaf_size = int(jnp.max(leaf_counts))

        pad_rows = max(max_leaf_size - 1, 0)
        if pad_rows > 0:
            padding = jnp.zeros(
                (pad_rows, positions_sorted.shape[1]),
                dtype=positions_sorted.dtype,
            )
            padded_positions = jnp.concatenate(
                [positions_sorted, padding],
                axis=0,
            )
        else:
            padded_positions = positions_sorted

        leaf_min, leaf_max = _compute_leaf_bounds(
            padded_positions,
            leaf_starts,
            leaf_counts,
            max_leaf_size=max_leaf_size,
        )

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
    tree: RadixTree,
    geometry: TreeGeometry,
) -> LevelMajorTreeGeometry:
    """Materialise padded, level-major views of node geometry."""

    num_levels = int(tree.num_levels)
    if num_levels <= 0:
        raise ValueError("tree must contain at least one populated level")

    level_offsets = jnp.asarray(tree.level_offsets, dtype=INDEX_DTYPE)
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

    nodes_by_level = jnp.asarray(tree.nodes_by_level, dtype=INDEX_DTYPE)

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
