"""Explicit octree metadata derived from Morton/radix topology."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dtypes import INDEX_DTYPE

_MORTON_BITS = 63
_MAX_CODE = (1 << 64) - 1


class ExplicitOctreeMetadata(NamedTuple):
    """Explicit octree buffers derived from a radix-compatible topology."""

    oct_parent: jnp.ndarray
    oct_children: jnp.ndarray
    oct_child_counts: jnp.ndarray
    oct_child_mask: jnp.ndarray
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


def _prefix_code(code: int, depth: int) -> int:
    if depth <= 0:
        return 0
    shift = _MORTON_BITS - 3 * depth
    if shift <= 0:
        return code & _MAX_CODE
    return code & (_MAX_CODE ^ ((1 << shift) - 1))


def _octant_at_depth(code: int, depth: int) -> int:
    if depth <= 0:
        return 0
    shift = _MORTON_BITS - 3 * depth
    return (code >> shift) & 0x7


def _common_depth_from_codes(code_a: int, code_b: int) -> int:
    if code_a == code_b:
        return 21
    xor = code_a ^ code_b
    return max(0, (_MORTON_BITS - xor.bit_length()) // 3)


def _radix_node_cell(
    topology: object,
    node_index: int,
    *,
    leaf_codes: np.ndarray,
    leaf_depths: np.ndarray,
    node_ranges: np.ndarray,
    morton_codes: np.ndarray,
) -> tuple[int, int]:
    num_internal = int(getattr(topology, "num_internal_nodes"))
    if node_index >= num_internal:
        leaf_idx = node_index - num_internal
        depth = int(leaf_depths[leaf_idx])
        code = int(leaf_codes[leaf_idx])
        return _prefix_code(code, depth), depth

    start = int(node_ranges[node_index, 0])
    end = int(node_ranges[node_index, 1])
    first = int(morton_codes[start])
    last = int(morton_codes[end])
    depth = _common_depth_from_codes(first, last)
    return _prefix_code(first, depth), depth


def _resolved_leaf_cells(topology: object) -> tuple[np.ndarray, np.ndarray]:
    num_internal = int(getattr(topology, "num_internal_nodes"))
    node_ranges, morton_codes, leaf_codes_raw, leaf_depths_raw = jax.device_get(
        (
            getattr(topology, "node_ranges"),
            getattr(topology, "morton_codes"),
            getattr(topology, "leaf_codes"),
            getattr(topology, "leaf_depths"),
        )
    )
    node_ranges = np.asarray(node_ranges, dtype=np.int64)
    morton_codes = np.asarray(morton_codes, dtype=np.uint64)
    leaf_codes_raw = np.asarray(leaf_codes_raw, dtype=np.uint64)
    leaf_depths_raw = np.asarray(leaf_depths_raw, dtype=np.int64)

    num_leaves = node_ranges.shape[0] - num_internal
    leaf_codes = np.zeros((num_leaves,), dtype=np.uint64)
    leaf_depths = np.zeros((num_leaves,), dtype=np.int64)

    for leaf_idx in range(num_leaves):
        depth = int(leaf_depths_raw[leaf_idx])
        if depth >= 0:
            code = int(leaf_codes_raw[leaf_idx])
            leaf_codes[leaf_idx] = np.uint64(_prefix_code(code, depth))
            leaf_depths[leaf_idx] = depth
            continue

        node_idx = num_internal + leaf_idx
        start = int(node_ranges[node_idx, 0])
        end = int(node_ranges[node_idx, 1])
        first = int(morton_codes[start])
        last = int(morton_codes[end])
        depth = _common_depth_from_codes(first, last)
        leaf_codes[leaf_idx] = np.uint64(_prefix_code(first, depth))
        leaf_depths[leaf_idx] = depth

    return leaf_codes, leaf_depths


def build_explicit_octree_metadata(topology: object) -> ExplicitOctreeMetadata:
    """Derive explicit octree cells and child tables from a radix topology."""

    node_ranges, morton_codes = jax.device_get(
        (getattr(topology, "node_ranges"), getattr(topology, "morton_codes"))
    )
    node_ranges = np.asarray(node_ranges, dtype=np.int64)
    morton_codes = np.asarray(morton_codes, dtype=np.uint64)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(getattr(topology, "num_internal_nodes"))
    leaf_codes, leaf_depths = _resolved_leaf_cells(topology)

    node_records: dict[tuple[int, int], list[int]] = {}
    for leaf_idx, (code_raw, depth_raw) in enumerate(zip(leaf_codes, leaf_depths)):
        node_idx = num_internal + leaf_idx
        start = int(node_ranges[node_idx, 0])
        end = int(node_ranges[node_idx, 1])
        code = int(code_raw)
        depth = int(depth_raw)
        for level in range(depth + 1):
            key = (level, _prefix_code(code, level))
            record = node_records.get(key)
            if record is None:
                node_records[key] = [start, end]
            else:
                record[0] = min(record[0], start)
                record[1] = max(record[1], end)

    keys = sorted(node_records.keys(), key=lambda item: (item[0], item[1]))
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    num_oct_nodes = len(keys)

    oct_parent = np.full((num_oct_nodes,), -1, dtype=np.int32)
    oct_children = np.full((num_oct_nodes, 8), -1, dtype=np.int32)
    oct_node_codes = np.zeros((num_oct_nodes,), dtype=np.uint64)
    oct_node_depths = np.zeros((num_oct_nodes,), dtype=np.int32)
    oct_node_ranges = np.zeros((num_oct_nodes, 2), dtype=np.int32)

    for idx, (depth, code) in enumerate(keys):
        oct_node_codes[idx] = np.uint64(code)
        oct_node_depths[idx] = depth
        start, end = node_records[(depth, code)]
        oct_node_ranges[idx] = np.asarray([start, end], dtype=np.int32)
        if depth <= 0:
            continue
        parent_key = (depth - 1, _prefix_code(code, depth - 1))
        parent_idx = key_to_index[parent_key]
        octant = _octant_at_depth(code, depth)
        oct_parent[idx] = parent_idx
        oct_children[parent_idx, octant] = idx

    oct_child_mask = oct_children >= 0
    oct_child_counts = np.sum(oct_child_mask, axis=1, dtype=np.int32)
    oct_leaf_mask = oct_child_counts == 0
    oct_leaf_nodes = np.flatnonzero(oct_leaf_mask).astype(np.int32)

    level_counts = np.bincount(
        oct_node_depths,
        minlength=int(oct_node_depths.max()) + 1 if num_oct_nodes > 0 else 0,
    ).astype(np.int32)
    oct_level_offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int32), np.cumsum(level_counts, dtype=np.int32)]
    )
    oct_nodes_by_level = np.arange(num_oct_nodes, dtype=np.int32)
    oct_num_levels = np.int32(level_counts.shape[0])

    radix_node_to_oct = np.full((num_nodes,), -1, dtype=np.int32)
    for node_idx in range(num_nodes):
        code, depth = _radix_node_cell(
            topology,
            node_idx,
            leaf_codes=leaf_codes,
            leaf_depths=leaf_depths,
            node_ranges=node_ranges,
            morton_codes=morton_codes,
        )
        radix_node_to_oct[node_idx] = key_to_index[(depth, code)]

    radix_leaf_to_oct = radix_node_to_oct[num_internal:].copy()

    return ExplicitOctreeMetadata(
        oct_parent=jnp.asarray(oct_parent, dtype=INDEX_DTYPE),
        oct_children=jnp.asarray(oct_children, dtype=INDEX_DTYPE),
        oct_child_counts=jnp.asarray(oct_child_counts, dtype=INDEX_DTYPE),
        oct_child_mask=jnp.asarray(oct_child_mask, dtype=jnp.bool_),
        oct_node_codes=jnp.asarray(oct_node_codes, dtype=jnp.uint64),
        oct_node_depths=jnp.asarray(oct_node_depths, dtype=INDEX_DTYPE),
        oct_node_ranges=jnp.asarray(oct_node_ranges, dtype=INDEX_DTYPE),
        oct_nodes_by_level=jnp.asarray(oct_nodes_by_level, dtype=INDEX_DTYPE),
        oct_level_offsets=jnp.asarray(oct_level_offsets, dtype=INDEX_DTYPE),
        oct_num_levels=jnp.asarray(oct_num_levels, dtype=INDEX_DTYPE),
        oct_leaf_mask=jnp.asarray(oct_leaf_mask, dtype=jnp.bool_),
        oct_leaf_nodes=jnp.asarray(oct_leaf_nodes, dtype=INDEX_DTYPE),
        radix_node_to_oct=jnp.asarray(radix_node_to_oct, dtype=INDEX_DTYPE),
        radix_leaf_to_oct=jnp.asarray(radix_leaf_to_oct, dtype=INDEX_DTYPE),
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
