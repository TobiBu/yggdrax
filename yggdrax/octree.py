"""Explicit octree metadata derived from Morton/radix topology."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from .dtypes import INDEX_DTYPE
from .morton import _compact3_u64

_MORTON_BITS = 63
_MAX_MORTON_LEVEL = 21


class ExplicitOctreeMetadata(NamedTuple):
    """Explicit octree buffers derived from octree leaf-cell partitions."""

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


class ExplicitOctreeBoxGeometry(NamedTuple):
    """Axis-aligned box geometry for explicit octree cells."""

    centers: jnp.ndarray
    half_extents: jnp.ndarray
    radii: jnp.ndarray
    max_extents: jnp.ndarray


class ExplicitOctreeTraversalView(NamedTuple):
    """Source-owned octree traversal/geometry package for native consumers."""

    valid_mask: jnp.ndarray
    parent: jnp.ndarray
    children: jnp.ndarray
    child_counts: jnp.ndarray
    node_codes: jnp.ndarray
    node_depths: jnp.ndarray
    node_ranges: jnp.ndarray
    nodes_by_level: jnp.ndarray
    level_offsets: jnp.ndarray
    num_levels: jnp.ndarray
    leaf_mask: jnp.ndarray
    leaf_nodes: jnp.ndarray
    radix_node_to_oct: jnp.ndarray
    radix_leaf_to_oct: jnp.ndarray
    oct_to_radix_node: jnp.ndarray
    oct_to_radix_leaf: jnp.ndarray
    num_valid_nodes: jnp.ndarray
    num_leaf_nodes: jnp.ndarray
    box_centers: jnp.ndarray
    box_half_extents: jnp.ndarray
    box_radii: jnp.ndarray
    box_max_extents: jnp.ndarray


class OctreeTopology(NamedTuple):
    """Octree-native metadata plus shared compatibility topology fields."""

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
    leaf_size: int | None
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


def _oct_address(codes: jnp.ndarray, depths: jnp.ndarray) -> jnp.ndarray:
    """Unique sortable uint64 key for (depth, code) pairs.

    Maps each (depth, code) pair to a uint64 value that preserves
    lexicographic ordering on (depth, code).  Specifically::

        address = (8**depth - 1) // 7 + (code >> (63 - 3 * depth))

    The first term counts all nodes in a full octree above depth ``depth``,
    ensuring keys at different depths never overlap.  This is used to
    enable O(n log n) parent lookup via :func:`jnp.searchsorted`.
    """
    depths_u64 = jnp.asarray(
        jnp.maximum(0, depths), dtype=jnp.uint64
    )  # clamp negative sentinel values (e.g. -1 padding) before uint64 cast
    codes_u64 = jnp.asarray(codes, dtype=jnp.uint64)
    bit_shift = jnp.uint64(_MORTON_BITS) - jnp.uint64(3) * depths_u64
    level_prefix = jnp.right_shift(codes_u64, bit_shift)
    pow8d = jnp.left_shift(jnp.uint64(1), jnp.uint64(3) * depths_u64)
    level_offset = (pow8d - jnp.uint64(1)) // jnp.uint64(7)
    return level_offset + level_prefix


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
    node_ranges = jnp.asarray(getattr(topology, "node_ranges"), dtype=INDEX_DTYPE)
    # Derive num_internal from the static shape (total_nodes = 2*num_leaves - 1).
    num_internal = (node_ranges.shape[0] - 1) // 2
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


def _resolved_leaf_partitions(
    topology: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Morton-sorted leaf partitions as the octree source of truth."""

    node_ranges = jnp.asarray(getattr(topology, "node_ranges"), dtype=INDEX_DTYPE)
    # Derive num_internal from the static shape (total_nodes = 2*num_leaves - 1).
    num_internal = (node_ranges.shape[0] - 1) // 2
    leaf_ranges = node_ranges[num_internal:]
    leaf_starts = jnp.asarray(leaf_ranges[:, 0], dtype=INDEX_DTYPE)
    leaf_ends_exclusive = jnp.asarray(leaf_ranges[:, 1] + 1, dtype=INDEX_DTYPE)
    leaf_codes, leaf_depths = _resolved_leaf_cells(topology)
    return leaf_starts, leaf_ends_exclusive, leaf_codes, leaf_depths


def build_explicit_octree_metadata_from_leaf_partitions(
    *,
    num_particles: int,
    leaf_starts: jnp.ndarray,
    leaf_ends_exclusive: jnp.ndarray,
    leaf_codes: jnp.ndarray,
    leaf_depths: jnp.ndarray,
) -> ExplicitOctreeMetadata:
    """Build explicit octree cells directly from Morton leaf-cell partitions."""

    leaf_starts = jnp.asarray(leaf_starts, dtype=INDEX_DTYPE)
    leaf_ends_exclusive = jnp.asarray(leaf_ends_exclusive, dtype=INDEX_DTYPE)
    leaf_codes = jnp.asarray(leaf_codes, dtype=jnp.uint64)
    leaf_depths = jnp.asarray(leaf_depths, dtype=INDEX_DTYPE)
    num_leaves = int(leaf_starts.shape[0])
    if num_leaves == 0:
        raise ValueError(
            "build_explicit_octree_metadata_from_leaf_partitions requires at least "
            "one leaf; got 0 (empty leaf_starts)."
        )
    candidate_depths = jnp.arange(_MAX_MORTON_LEVEL + 1, dtype=INDEX_DTYPE)
    max_candidates = max(1, num_leaves * (_MAX_MORTON_LEVEL + 1))

    depth_grid = jnp.broadcast_to(
        candidate_depths[None, :], (num_leaves, _MAX_MORTON_LEVEL + 1)
    )
    code_grid = jnp.broadcast_to(
        leaf_codes[:, None], (num_leaves, _MAX_MORTON_LEVEL + 1)
    )
    valid_grid = depth_grid <= leaf_depths[:, None]
    prefix_grid = _prefix_code(code_grid, depth_grid)
    address_grid = _oct_address(prefix_grid, depth_grid)
    uint64_max = jnp.asarray(jnp.iinfo(jnp.uint64).max, dtype=jnp.uint64)
    address_grid = jnp.where(valid_grid, address_grid, uint64_max)

    starts_grid = jnp.broadcast_to(
        leaf_starts[:, None], (num_leaves, _MAX_MORTON_LEVEL + 1)
    )
    ends_grid = jnp.broadcast_to(
        (leaf_ends_exclusive - 1)[:, None], (num_leaves, _MAX_MORTON_LEVEL + 1)
    )

    flat_addresses = address_grid.reshape((max_candidates,))
    flat_codes = prefix_grid.reshape((max_candidates,))
    flat_depths = depth_grid.reshape((max_candidates,))
    flat_starts = starts_grid.reshape((max_candidates,))
    flat_ends = ends_grid.reshape((max_candidates,))
    flat_valid = valid_grid.reshape((max_candidates,))

    order = jnp.argsort(flat_addresses, stable=True)
    addresses_sorted = flat_addresses[order]
    codes_sorted = flat_codes[order]
    depths_sorted = flat_depths[order]
    starts_sorted = flat_starts[order]
    ends_sorted = flat_ends[order]
    valid_sorted = flat_valid[order] & (addresses_sorted != uint64_max)

    is_new = valid_sorted
    is_new = is_new.at[1:].set(
        valid_sorted[1:]
        & ((addresses_sorted[1:] != addresses_sorted[:-1]) | (~valid_sorted[:-1]))
    )
    unique_index = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - jnp.asarray(
        1, dtype=INDEX_DTYPE
    )
    num_unique = jnp.sum(is_new.astype(INDEX_DTYPE))
    oct_valid_mask = jnp.arange(max_candidates, dtype=INDEX_DTYPE) < num_unique

    oct_node_depths = jnp.full((max_candidates,), -1, dtype=INDEX_DTYPE)
    oct_node_depths = oct_node_depths.at[unique_index].max(
        jnp.where(valid_sorted, depths_sorted, jnp.asarray(-1, dtype=INDEX_DTYPE))
    )
    oct_node_codes = jnp.zeros((max_candidates,), dtype=jnp.uint64)
    oct_node_codes = oct_node_codes.at[unique_index].max(
        jnp.where(valid_sorted, codes_sorted, jnp.asarray(0, dtype=jnp.uint64))
    )

    max_index = jnp.asarray(max(num_particles, 1), dtype=INDEX_DTYPE)
    oct_starts = jnp.full((max_candidates,), max_index, dtype=INDEX_DTYPE)
    oct_starts = oct_starts.at[unique_index].min(
        jnp.where(valid_sorted, starts_sorted, max_index)
    )
    oct_ends = jnp.full((max_candidates,), -1, dtype=INDEX_DTYPE)
    oct_ends = oct_ends.at[unique_index].max(
        jnp.where(valid_sorted, ends_sorted, jnp.asarray(-1, dtype=INDEX_DTYPE))
    )
    oct_node_ranges = jnp.stack([oct_starts, oct_ends], axis=1)
    oct_node_ranges = jnp.where(
        oct_valid_mask[:, None],
        oct_node_ranges,
        jnp.asarray([0, -1], dtype=INDEX_DTYPE),
    )

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
    oct_nodes_by_level = jnp.argsort(sort_depths, stable=True).astype(INDEX_DTYPE)

    oct_addresses = jnp.where(
        oct_valid_mask,
        _oct_address(oct_node_codes, oct_node_depths),
        uint64_max,
    )
    parent_depths = oct_node_depths - jnp.asarray(1, dtype=INDEX_DTYPE)
    parent_codes = _prefix_code(oct_node_codes, parent_depths)
    parent_addresses = _oct_address(parent_codes, parent_depths)
    parent_pos = jnp.searchsorted(oct_addresses, parent_addresses, side="left").astype(
        INDEX_DTYPE
    )
    parent_pos_safe = jnp.minimum(
        parent_pos, jnp.asarray(max_candidates - 1, dtype=INDEX_DTYPE)
    )
    address_match = oct_addresses[parent_pos_safe] == parent_addresses
    oct_parent = jnp.where(
        oct_valid_mask
        & (oct_node_depths > jnp.asarray(0, dtype=INDEX_DTYPE))
        & address_match,
        parent_pos,
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    octants = _octant_at_depth(oct_node_codes, oct_node_depths).astype(INDEX_DTYPE)
    valid_child = oct_valid_mask & (oct_parent >= 0)
    child_rows = jnp.where(valid_child, oct_parent, 0)
    child_cols = jnp.where(valid_child, octants, 0)
    child_vals = jnp.where(
        valid_child, jnp.arange(max_candidates, dtype=INDEX_DTYPE) + 1, 0
    )
    children_plus = jnp.zeros((max_candidates, 8), dtype=INDEX_DTYPE)
    children_plus = children_plus.at[child_rows, child_cols].max(child_vals)
    oct_children = children_plus - jnp.asarray(1, dtype=INDEX_DTYPE)
    oct_child_mask = oct_valid_mask[:, None] & (oct_children >= 0)
    oct_child_counts = jnp.sum(oct_child_mask.astype(INDEX_DTYPE), axis=1)
    oct_leaf_mask = oct_valid_mask & (oct_child_counts == 0)
    oct_leaf_nodes = jnp.nonzero(
        oct_leaf_mask,
        size=max_candidates,
        fill_value=jnp.asarray(-1, dtype=INDEX_DTYPE),
    )[0].astype(INDEX_DTYPE)

    leaf_addresses = _oct_address(leaf_codes, leaf_depths)
    radix_leaf_to_oct = jnp.searchsorted(
        oct_addresses, leaf_addresses, side="left"
    ).astype(INDEX_DTYPE)
    radix_node_to_oct = jnp.full((max(2 * num_leaves - 1, 1),), -1, dtype=INDEX_DTYPE)
    if num_leaves > 0:
        num_internal = max(num_leaves - 1, 0)
        radix_node_to_oct = radix_node_to_oct.at[num_internal:].set(radix_leaf_to_oct)

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


def compute_explicit_octree_box_geometry(
    *,
    valid_mask: jnp.ndarray,
    node_codes: jnp.ndarray,
    node_depths: jnp.ndarray,
    bounds_min: jnp.ndarray,
    bounds_max: jnp.ndarray,
) -> ExplicitOctreeBoxGeometry:
    """Compute explicit octree box geometry directly from code/depth pairs."""

    bounds_min = jnp.asarray(bounds_min)
    bounds_max = jnp.asarray(bounds_max)
    domain = bounds_max - bounds_min
    depth_clamped = jnp.clip(
        jnp.asarray(node_depths, dtype=INDEX_DTYPE),
        jnp.asarray(0, dtype=INDEX_DTYPE),
        jnp.asarray(_MAX_MORTON_LEVEL, dtype=INDEX_DTYPE),
    )
    shift = jnp.asarray(_MAX_MORTON_LEVEL, dtype=jnp.uint64) - depth_clamped.astype(
        jnp.uint64
    )
    node_codes = jnp.asarray(node_codes, dtype=jnp.uint64)
    x_coords = _compact3_u64(node_codes)
    y_coords = _compact3_u64(node_codes >> jnp.uint64(1))
    z_coords = _compact3_u64(node_codes >> jnp.uint64(2))

    x_idx = x_coords >> shift
    y_idx = y_coords >> shift
    z_idx = z_coords >> shift
    indices = jnp.stack([x_idx, y_idx, z_idx], axis=1).astype(bounds_min.dtype)

    counts = jnp.left_shift(
        jnp.ones_like(depth_clamped, dtype=jnp.uint64),
        depth_clamped.astype(jnp.uint64),
    )
    counts = jnp.maximum(counts, jnp.uint64(1)).astype(bounds_min.dtype)
    cell_sizes = domain[None, :] / counts[:, None]
    mins = bounds_min[None, :] + cell_sizes * indices
    maxs = mins + cell_sizes
    centers = 0.5 * (mins + maxs)
    half_extents = 0.5 * (maxs - mins)
    radii = jnp.linalg.norm(half_extents, axis=1)
    max_extents = jnp.max(half_extents, axis=1)

    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    centers = jnp.where(valid_mask[:, None], centers, 0.0)
    half_extents = jnp.where(valid_mask[:, None], half_extents, 0.0)
    radii = jnp.where(valid_mask, radii, 0.0)
    max_extents = jnp.where(valid_mask, max_extents, 0.0)
    return ExplicitOctreeBoxGeometry(
        centers=centers,
        half_extents=half_extents,
        radii=radii,
        max_extents=max_extents,
    )


def build_explicit_octree_metadata(topology: object) -> ExplicitOctreeMetadata:
    """Derive explicit octree cells from octree leaves and map compat nodes onto them."""

    node_ranges = jnp.asarray(getattr(topology, "node_ranges"), dtype=INDEX_DTYPE)
    morton_codes = jnp.asarray(getattr(topology, "morton_codes"), dtype=jnp.uint64)
    num_nodes = int(node_ranges.shape[0])
    num_internal = (num_nodes - 1) // 2
    leaf_starts, leaf_ends_exclusive, leaf_codes, leaf_depths = (
        _resolved_leaf_partitions(topology)
    )
    metadata = build_explicit_octree_metadata_from_leaf_partitions(
        num_particles=int(morton_codes.shape[0]),
        leaf_starts=leaf_starts,
        leaf_ends_exclusive=leaf_ends_exclusive,
        leaf_codes=leaf_codes,
        leaf_depths=leaf_depths,
    )

    node_first = morton_codes[node_ranges[:, 0]]
    node_last = morton_codes[node_ranges[:, 1]]
    node_depths = _common_depth_from_codes(node_first, node_last)
    node_codes = _prefix_code(node_first, node_depths)
    node_depths = node_depths.at[num_internal:].set(leaf_depths)
    node_codes = node_codes.at[num_internal:].set(leaf_codes)

    uint64_max = jnp.asarray(jnp.iinfo(jnp.uint64).max, dtype=jnp.uint64)
    oct_addresses = jnp.where(
        metadata.oct_valid_mask,
        _oct_address(metadata.oct_node_codes, metadata.oct_node_depths),
        uint64_max,
    )
    radix_addresses = _oct_address(node_codes, node_depths)
    radix_node_to_oct = jnp.searchsorted(
        oct_addresses,
        radix_addresses,
        side="left",
    ).astype(INDEX_DTYPE)
    radix_leaf_to_oct = radix_node_to_oct[num_internal:]

    return ExplicitOctreeMetadata(
        oct_parent=metadata.oct_parent,
        oct_children=metadata.oct_children,
        oct_child_counts=metadata.oct_child_counts,
        oct_child_mask=metadata.oct_child_mask,
        oct_valid_mask=metadata.oct_valid_mask,
        oct_node_codes=metadata.oct_node_codes,
        oct_node_depths=metadata.oct_node_depths,
        oct_node_ranges=metadata.oct_node_ranges,
        oct_nodes_by_level=metadata.oct_nodes_by_level,
        oct_level_offsets=metadata.oct_level_offsets,
        oct_num_levels=metadata.oct_num_levels,
        oct_leaf_mask=metadata.oct_leaf_mask,
        oct_leaf_nodes=metadata.oct_leaf_nodes,
        radix_node_to_oct=radix_node_to_oct,
        radix_leaf_to_oct=radix_leaf_to_oct,
    )


def augment_radix_topology_with_octree(topology: object) -> OctreeTopology:
    """Return an octree-augmented topology preserving radix compatibility."""

    metadata = build_explicit_octree_metadata(topology)
    base_fields = tuple(getattr(topology, name) for name in topology._fields)
    return OctreeTopology(*base_fields, *metadata)


def build_explicit_octree_traversal_view(
    topology: object,
) -> ExplicitOctreeTraversalView:
    """Package explicit octree structure, mappings, and box geometry."""

    required = (
        "oct_valid_mask",
        "oct_parent",
        "oct_children",
        "oct_child_counts",
        "oct_node_codes",
        "oct_node_depths",
        "oct_node_ranges",
        "oct_nodes_by_level",
        "oct_level_offsets",
        "oct_num_levels",
        "oct_leaf_mask",
        "oct_leaf_nodes",
        "radix_node_to_oct",
        "radix_leaf_to_oct",
        "bounds_min",
        "bounds_max",
    )
    missing = tuple(name for name in required if not hasattr(topology, name))
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(
            f"topology is missing explicit octree traversal fields: {missing_txt}"
        )

    valid_mask = jnp.asarray(getattr(topology, "oct_valid_mask"), dtype=jnp.bool_)
    parent = jnp.asarray(getattr(topology, "oct_parent"), dtype=INDEX_DTYPE)
    children = jnp.asarray(getattr(topology, "oct_children"), dtype=INDEX_DTYPE)
    child_counts = jnp.asarray(getattr(topology, "oct_child_counts"), dtype=INDEX_DTYPE)
    node_codes = jnp.asarray(getattr(topology, "oct_node_codes"), dtype=jnp.uint64)
    node_depths = jnp.asarray(getattr(topology, "oct_node_depths"), dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(getattr(topology, "oct_node_ranges"), dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(
        getattr(topology, "oct_nodes_by_level"), dtype=INDEX_DTYPE
    )
    level_offsets = jnp.asarray(
        getattr(topology, "oct_level_offsets"), dtype=INDEX_DTYPE
    )
    num_levels = jnp.asarray(getattr(topology, "oct_num_levels"), dtype=INDEX_DTYPE)
    leaf_mask = jnp.asarray(getattr(topology, "oct_leaf_mask"), dtype=jnp.bool_)
    leaf_nodes = jnp.asarray(getattr(topology, "oct_leaf_nodes"), dtype=INDEX_DTYPE)
    radix_node_to_oct = jnp.asarray(
        getattr(topology, "radix_node_to_oct"), dtype=INDEX_DTYPE
    )
    radix_leaf_to_oct = jnp.asarray(
        getattr(topology, "radix_leaf_to_oct"), dtype=INDEX_DTYPE
    )

    full_oct_nodes = valid_mask.shape[0]
    radix_nodes = radix_node_to_oct.shape[0]
    radix_node_ids = jnp.arange(radix_nodes, dtype=INDEX_DTYPE)
    node_fill = jnp.asarray(radix_nodes, dtype=INDEX_DTYPE)
    oct_to_radix_node = jnp.full((full_oct_nodes,), node_fill, dtype=INDEX_DTYPE)
    oct_to_radix_node = oct_to_radix_node.at[radix_node_to_oct].min(radix_node_ids)
    oct_to_radix_node = jnp.where(
        valid_mask & (oct_to_radix_node < node_fill),
        oct_to_radix_node,
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    num_internal = int(jnp.asarray(getattr(topology, "left_child")).shape[0])
    radix_leaf_nodes = jnp.arange(
        num_internal,
        num_internal + radix_leaf_to_oct.shape[0],
        dtype=INDEX_DTYPE,
    )
    leaf_fill = jnp.asarray(
        num_internal + radix_leaf_to_oct.shape[0], dtype=INDEX_DTYPE
    )
    oct_to_radix_leaf = jnp.full((full_oct_nodes,), leaf_fill, dtype=INDEX_DTYPE)
    oct_to_radix_leaf = oct_to_radix_leaf.at[radix_leaf_to_oct].min(radix_leaf_nodes)
    oct_to_radix_leaf = jnp.where(
        leaf_mask & (oct_to_radix_leaf < leaf_fill),
        oct_to_radix_leaf,
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    num_valid_nodes = jnp.sum(valid_mask.astype(INDEX_DTYPE))
    num_leaf_nodes = jnp.sum(leaf_mask.astype(INDEX_DTYPE))
    box_geometry = compute_explicit_octree_box_geometry(
        valid_mask=valid_mask,
        node_codes=node_codes,
        node_depths=node_depths,
        bounds_min=jnp.asarray(getattr(topology, "bounds_min")),
        bounds_max=jnp.asarray(getattr(topology, "bounds_max")),
    )
    return ExplicitOctreeTraversalView(
        valid_mask=valid_mask,
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_codes=node_codes,
        node_depths=node_depths,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=num_levels,
        leaf_mask=leaf_mask,
        leaf_nodes=leaf_nodes,
        radix_node_to_oct=radix_node_to_oct,
        radix_leaf_to_oct=radix_leaf_to_oct,
        oct_to_radix_node=oct_to_radix_node,
        oct_to_radix_leaf=oct_to_radix_leaf,
        num_valid_nodes=num_valid_nodes,
        num_leaf_nodes=num_leaf_nodes,
        box_centers=box_geometry.centers,
        box_half_extents=box_geometry.half_extents,
        box_radii=box_geometry.radii,
        box_max_extents=box_geometry.max_extents,
    )


__all__ = [
    "ExplicitOctreeBoxGeometry",
    "ExplicitOctreeMetadata",
    "ExplicitOctreeTraversalView",
    "OctreeTopology",
    "augment_radix_topology_with_octree",
    "build_explicit_octree_traversal_view",
    "compute_explicit_octree_box_geometry",
    "build_explicit_octree_metadata",
]
