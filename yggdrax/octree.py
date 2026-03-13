"""Explicit octree metadata derived from Morton/radix topology."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from .dtypes import INDEX_DTYPE

_MORTON_BITS = 63
_MAX_MORTON_LEVEL = 21


class ExplicitOctreeMetadata(NamedTuple):
    """Explicit octree buffers derived from a radix-compatible topology."""

    oct_parent: jnp.ndarray
    oct_children: jnp.ndarray
    oct_child_counts: jnp.ndarray
    oct_child_mask: jnp.ndarray
    oct_valid_mask: jnp.ndarray
    oct_node_codes: jnp.ndarray
    oct_node_depths: jnp.ndarray
    oct_node_ranges: jnp.ndarray
    oct_nodes_by_level: jnp.ndarray
    oct_level_offsets: jnp.ndarray
    oct_num_levels: jnp.ndarray
    oct_leaf_mask: jnp.ndarray
    oct_leaf_nodes: jnp.ndarray
    radix_node_to_oct: jnp.ndarray
    radix_leaf_to_oct: jnp.ndarray


class OctreeTopology(NamedTuple):
    """Radix-compatible topology augmented with explicit octree metadata."""

    parent: jnp.ndarray
    left_child: jnp.ndarray
    right_child: jnp.ndarray
    left_is_leaf: jnp.ndarray
    right_is_leaf: jnp.ndarray
    particle_indices: jnp.ndarray
    morton_codes: jnp.ndarray
    node_ranges: jnp.ndarray
    num_particles: int
    num_internal_nodes: int
    node_level: jnp.ndarray
    level_offsets: jnp.ndarray
    nodes_by_level: jnp.ndarray
    num_levels: jnp.ndarray
    bounds_min: jnp.ndarray
    bounds_max: jnp.ndarray
    leaf_codes: jnp.ndarray
    leaf_depths: jnp.ndarray
    use_morton_geometry: jnp.ndarray
    oct_parent: jnp.ndarray
    oct_children: jnp.ndarray
    oct_child_counts: jnp.ndarray
    oct_child_mask: jnp.ndarray
    oct_valid_mask: jnp.ndarray
    oct_node_codes: jnp.ndarray
    oct_node_depths: jnp.ndarray
    oct_node_ranges: jnp.ndarray
    oct_nodes_by_level: jnp.ndarray
    oct_level_offsets: jnp.ndarray
    oct_num_levels: jnp.ndarray
    oct_leaf_mask: jnp.ndarray
    oct_leaf_nodes: jnp.ndarray
    radix_node_to_oct: jnp.ndarray
    radix_leaf_to_oct: jnp.ndarray


def _prefix_code(codes: jnp.ndarray, depths: jnp.ndarray) -> jnp.ndarray:
    codes_u64 = jnp.asarray(codes, dtype=jnp.uint64)
    depths_i32 = jnp.asarray(depths, dtype=INDEX_DTYPE)
    shifts = jnp.maximum(0, _MORTON_BITS - 3 * depths_i32).astype(jnp.uint64)
    prefixed = jnp.left_shift(jnp.right_shift(codes_u64, shifts), shifts)
    return jnp.where(depths_i32 > 0, prefixed, jnp.zeros_like(prefixed))


def _octant_at_depth(codes: jnp.ndarray, depths: jnp.ndarray) -> jnp.ndarray:
    codes_u64 = jnp.asarray(codes, dtype=jnp.uint64)
    depths_i32 = jnp.asarray(depths, dtype=INDEX_DTYPE)
    shifts = jnp.maximum(0, _MORTON_BITS - 3 * depths_i32).astype(jnp.uint64)
    octants = jnp.right_shift(codes_u64, shifts) & jnp.uint64(0x7)
    return jnp.where(depths_i32 > 0, octants, jnp.zeros_like(octants))


def _common_depth_from_codes(
    first_codes: jnp.ndarray, last_codes: jnp.ndarray
) -> jnp.ndarray:
    xor = jnp.bitwise_xor(
        jnp.asarray(first_codes, dtype=jnp.uint64),
        jnp.asarray(last_codes, dtype=jnp.uint64),
    )
    clz = lax.clz(xor).astype(INDEX_DTYPE)
    common_bits = jnp.where(
        xor == jnp.uint64(0),
        jnp.asarray(_MORTON_BITS, dtype=INDEX_DTYPE),
        jnp.maximum(clz - jnp.asarray(1, dtype=INDEX_DTYPE), 0),
    )
    return common_bits // jnp.asarray(3, dtype=INDEX_DTYPE)


def _resolved_leaf_cells(topology: object) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_internal = int(getattr(topology, "num_internal_nodes"))
    node_ranges = jnp.asarray(getattr(topology, "node_ranges"), dtype=INDEX_DTYPE)
    morton_codes = jnp.asarray(getattr(topology, "morton_codes"), dtype=jnp.uint64)
    leaf_codes_raw = jnp.asarray(getattr(topology, "leaf_codes"), dtype=jnp.uint64)
    leaf_depths_raw = jnp.asarray(getattr(topology, "leaf_depths"), dtype=INDEX_DTYPE)

    leaf_ranges = node_ranges[num_internal:]
    leaf_first = morton_codes[leaf_ranges[:, 0]]
    leaf_last = morton_codes[leaf_ranges[:, 1]]
    derived_depths = _common_depth_from_codes(leaf_first, leaf_last)
    resolved_depths = jnp.where(leaf_depths_raw >= 0, leaf_depths_raw, derived_depths)
    resolved_codes = jnp.where(
        leaf_depths_raw >= 0,
        _prefix_code(leaf_codes_raw, resolved_depths),
        _prefix_code(leaf_first, resolved_depths),
    )
    return resolved_codes, resolved_depths


def build_explicit_octree_metadata(topology: object) -> ExplicitOctreeMetadata:
    """Derive explicit octree cells and child tables from a radix topology."""

    node_ranges = jnp.asarray(getattr(topology, "node_ranges"), dtype=INDEX_DTYPE)
    morton_codes = jnp.asarray(getattr(topology, "morton_codes"), dtype=jnp.uint64)
    num_nodes = node_ranges.shape[0]
    num_internal = int(getattr(topology, "num_internal_nodes"))

    leaf_codes, leaf_depths = _resolved_leaf_cells(topology)
    node_first = morton_codes[node_ranges[:, 0]]
    node_last = morton_codes[node_ranges[:, 1]]
    node_depths = _common_depth_from_codes(node_first, node_last)
    node_codes = _prefix_code(node_first, node_depths)

    node_depths = node_depths.at[num_internal:].set(leaf_depths)
    node_codes = node_codes.at[num_internal:].set(leaf_codes)

    order = jnp.lexsort((node_codes.astype(jnp.int64), node_depths))
    depths_sorted = node_depths[order]
    codes_sorted = node_codes[order]
    ranges_sorted = node_ranges[order]

    is_new = jnp.ones((num_nodes,), dtype=jnp.bool_)
    is_new = is_new.at[1:].set(
        (depths_sorted[1:] != depths_sorted[:-1])
        | (codes_sorted[1:] != codes_sorted[:-1])
    )
    unique_index = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - jnp.asarray(
        1, dtype=INDEX_DTYPE
    )
    num_unique = jnp.sum(is_new.astype(INDEX_DTYPE))
    oct_valid_mask = jnp.arange(num_nodes, dtype=INDEX_DTYPE) < num_unique

    oct_node_depths = jnp.full((num_nodes,), -1, dtype=INDEX_DTYPE)
    oct_node_depths = oct_node_depths.at[unique_index].set(depths_sorted)
    oct_node_codes = jnp.zeros((num_nodes,), dtype=jnp.uint64)
    oct_node_codes = oct_node_codes.at[unique_index].set(codes_sorted)

    max_index = jnp.asarray(num_nodes, dtype=INDEX_DTYPE)
    oct_starts = jnp.full((num_nodes,), max_index, dtype=INDEX_DTYPE)
    oct_starts = oct_starts.at[unique_index].min(ranges_sorted[:, 0])
    oct_ends = jnp.full((num_nodes,), -1, dtype=INDEX_DTYPE)
    oct_ends = oct_ends.at[unique_index].max(ranges_sorted[:, 1])
    oct_node_ranges = jnp.stack([oct_starts, oct_ends], axis=1)
    oct_node_ranges = jnp.where(
        oct_valid_mask[:, None],
        oct_node_ranges,
        jnp.asarray([0, -1], dtype=INDEX_DTYPE),
    )

    inverse_order = jnp.empty_like(order)
    inverse_order = inverse_order.at[order].set(
        jnp.arange(num_nodes, dtype=INDEX_DTYPE)
    )
    radix_node_to_oct = unique_index[inverse_order]
    radix_leaf_to_oct = radix_node_to_oct[num_internal:]

    oct_num_levels = jnp.asarray(
        jnp.max(jnp.where(oct_valid_mask, oct_node_depths, 0)) + 1,
        dtype=INDEX_DTYPE,
    )
    level_counts = jnp.bincount(
        jnp.clip(oct_node_depths, 0, _MAX_MORTON_LEVEL),
        weights=oct_valid_mask.astype(INDEX_DTYPE),
        length=_MAX_MORTON_LEVEL + 1,
    ).astype(INDEX_DTYPE)
    oct_level_offsets = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(level_counts, dtype=INDEX_DTYPE),
        ],
        axis=0,
    )
    sort_depths = jnp.where(
        oct_valid_mask,
        oct_node_depths,
        jnp.asarray(_MAX_MORTON_LEVEL + 1, dtype=oct_node_depths.dtype),
    )
    oct_nodes_by_level = jnp.argsort(sort_depths, kind="stable").astype(INDEX_DTYPE)

    parent_depths = oct_node_depths - jnp.asarray(1, dtype=INDEX_DTYPE)
    parent_codes = _prefix_code(oct_node_codes, parent_depths)
    valid = oct_valid_mask
    matches = (
        valid[:, None]
        & valid[None, :]
        & (oct_node_depths[None, :] == parent_depths[:, None])
        & (oct_node_codes[None, :] == parent_codes[:, None])
    )
    has_parent = jnp.any(matches, axis=1)
    oct_parent = jnp.where(
        valid & has_parent,
        jnp.argmax(matches, axis=1).astype(INDEX_DTYPE),
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    octants = _octant_at_depth(oct_node_codes, oct_node_depths).astype(INDEX_DTYPE)
    valid_child = valid & (oct_parent >= 0)
    child_rows = jnp.where(valid_child, oct_parent, 0)
    child_cols = jnp.where(valid_child, octants, 0)
    child_vals = jnp.where(valid_child, jnp.arange(num_nodes, dtype=INDEX_DTYPE) + 1, 0)
    children_plus = jnp.zeros((num_nodes, 8), dtype=INDEX_DTYPE)
    children_plus = children_plus.at[child_rows, child_cols].max(child_vals)
    oct_children = children_plus - jnp.asarray(1, dtype=INDEX_DTYPE)
    oct_child_mask = valid[:, None] & (oct_children >= 0)
    oct_child_counts = jnp.sum(oct_child_mask.astype(INDEX_DTYPE), axis=1)
    oct_leaf_mask = valid & (oct_child_counts == 0)
    oct_leaf_nodes = jnp.nonzero(
        oct_leaf_mask,
        size=num_nodes,
        fill_value=jnp.asarray(-1, dtype=INDEX_DTYPE),
    )[0].astype(INDEX_DTYPE)

    return ExplicitOctreeMetadata(
        oct_parent=oct_parent,
        oct_children=oct_children,
        oct_child_counts=oct_child_counts,
        oct_child_mask=oct_child_mask,
        oct_valid_mask=oct_valid_mask,
        oct_node_codes=oct_node_codes,
        oct_node_depths=oct_node_depths,
        oct_node_ranges=oct_node_ranges,
        oct_nodes_by_level=oct_nodes_by_level,
        oct_level_offsets=oct_level_offsets,
        oct_num_levels=oct_num_levels,
        oct_leaf_mask=oct_leaf_mask,
        oct_leaf_nodes=oct_leaf_nodes,
        radix_node_to_oct=radix_node_to_oct,
        radix_leaf_to_oct=radix_leaf_to_oct,
    )


def augment_radix_topology_with_octree(topology: object) -> OctreeTopology:
    """Return an octree-augmented topology preserving radix compatibility."""

    metadata = build_explicit_octree_metadata(topology)
    base_fields = tuple(getattr(topology, name) for name in topology._fields)
    return OctreeTopology(*base_fields, *metadata)


__all__ = [
    "ExplicitOctreeMetadata",
    "OctreeTopology",
    "augment_radix_topology_with_octree",
    "build_explicit_octree_metadata",
]
