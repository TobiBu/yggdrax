"""Occupancy-bucketed, padding-free near-field W-tiles for the adaptive octree.

The adaptive octree execution view (:func:`build_adaptive_octree_execution_view_device`)
exposes an occupancy-bounded near list as a fixed-width CSR (``u_offsets`` / ``u_neighbors``
over leaf ROW ids, self-excluded). The straightforward near-field kernel groups each leaf
into fixed-``B`` blocks and gives every block a source list of width ``(1 + u_capacity) *
ceil(max_leaf_size / B)`` -- with ``u_capacity`` (256) hugely over-provisioned vs the ~27-56
real neighbours, and the block factor sized by the *max* leaf occupancy. That ~48x padding is
the whole near-field wall on the concentrated-galaxy IC.

This module builds the SAME near tile-pairs but **compacted**: it re-packs each adaptive cell
into full uniform **W-tiles** (small width ``W`` = 32/64, cell-respecting so tile-pairs map
1:1 onto cell-pairs -- no cross-cell masking, no double-counting), and it densely packs each
tile's source-tile ids into a fixed capacity sized by the *actual* near work (probed), not the
worst-case product. Overflow (a leaf needing more tiles than ``max_tiles_per_leaf``, a tile
needing more sources than ``max_source_tiles``, or more tiles than ``num_tiles_cap``) is
reported as a single boolean the caller checks EAGER-only -- never per-step under a scan.

The output :class:`OctreeNearfieldTiles` drops straight into jaccpot's ``_octree_near_field``:
``tile_particle_idx``/``tile_mask`` become the ``(num_tiles, W, ...)`` gather table (tiles as
"leaves", ``W`` as ``leaf_width``) and ``tile_source_ids``/``tile_source_valid`` become the
per-tile source-id list for the topology-agnostic ``nearfield_leafpair`` Pallas kernel or the
pure-JAX chunked edge scan. Intra-tile self (``i != j``) is handled by the caller's existing
self-block path; this builder EXCLUDES the target tile from its own source list.

Everything is static-shape (shapes depend only on ``(num_tiles_cap, W, max_source_tiles)``),
so the build compiles once and is reused across timesteps.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from .dtypes import INDEX_DTYPE
from .octree_uvwx import UniformOctreeExecutionView

__all__ = [
    "OctreeNearfieldTiles",
    "build_octree_nearfield_tiles",
]


class OctreeNearfieldTiles(NamedTuple):
    """Compact, cell-respecting near-field W-tiles for the adaptive octree.

    All array shapes are static functions of ``(num_tiles, W, max_source_tiles)``; the two
    ``num_*`` scalars are data-dependent (traced under jit) and ``overflow`` is the eager-only
    capacity gate. ``sentinel`` marks unused padded rows/slots.
    """

    tile_particle_idx: Array  # (num_tiles, W) Morton particle indices per tile
    tile_mask: Array  # (num_tiles, W) bool: real particle vs pad
    tile_leaf_row: Array  # (num_tiles,) owning leaf ROW id (sentinel for unused tiles)
    tile_source_ids: (
        Array  # (num_tiles, max_source_tiles) source TILE ids (self-excluded)
    )
    tile_source_valid: Array  # (num_tiles, max_source_tiles) bool
    num_tiles_used: Array  # () traced: total real tiles (<= num_tiles cap)
    num_source_tiles_max: (
        Array  # () traced: max source tiles over target tiles (incl self slot)
    )
    overflow: Array  # () bool: any capacity exceeded (check eager-only)


def _compact_rows_to_width(
    values: Array,
    valid: Array,
    *,
    out_width: int,
    fill: int,
    index_dtype: DTypeLike,
) -> tuple[Array, Array, Array]:
    """Pack the valid entries of each row to the front of a fixed-width buffer.

    ``values`` / ``valid`` are ``(M, C)``. Returns ``(packed, packed_valid, counts)`` where
    ``packed`` / ``packed_valid`` are ``(M, out_width)`` (valid entries of row ``m`` in
    ``values[m]`` moved to columns ``0..counts[m]-1``, padded with ``fill`` / ``False``) and
    ``counts`` is ``(M,)`` the number of valid entries per row (may exceed ``out_width``; the
    overflow is dropped and reported via ``counts > out_width``).
    """

    m = int(values.shape[0])
    w = int(out_width)
    valid_i = valid.astype(index_dtype)
    # exclusive prefix sum along the row = destination column of each valid entry
    prefix = jnp.cumsum(valid_i, axis=1) - valid_i  # (M, C)
    counts = jnp.sum(valid_i, axis=1)  # (M,)
    place = valid & (prefix < w)  # entries that fit
    row = jnp.arange(m, dtype=index_dtype)[:, None]
    oor = jnp.asarray(m * w, dtype=index_dtype)  # out-of-range slot -> dropped
    dest = jnp.where(place, row * w + prefix, oor)
    packed = (
        jnp.full((m * w,), fill, dtype=index_dtype)
        .at[dest]
        .set(jnp.where(place, values, fill).astype(index_dtype), mode="drop")
        .reshape(m, w)
    )
    packed_valid = (
        jnp.zeros((m * w,), dtype=bool).at[dest].set(place, mode="drop").reshape(m, w)
    )
    return packed, packed_valid, counts


def build_octree_nearfield_tiles(
    view: UniformOctreeExecutionView,
    *,
    tile_width: int,
    num_tiles_cap: int,
    max_source_tiles: int,
    max_leaf_occupancy: int,
    index_dtype: DTypeLike = INDEX_DTYPE,
) -> OctreeNearfieldTiles:
    """Build compact, cell-respecting near-field W-tiles from an adaptive octree view.

    Args:
        view: adaptive octree execution view built with ``interactions=True`` (must carry
            ``leaf_indices``, ``node_ranges`` and the near CSR ``u_offsets`` / ``u_neighbors``).
        tile_width: ``W``, the fixed tile width (e.g. 32 or 64). Small ``W`` collapses the
            within-tile padding on under-full cells.
        num_tiles_cap: static upper bound on the number of tiles; a safe value is
            ``ceil(N / W) + leaf_capacity`` (each cell contributes at most one partial tail
            tile beyond ``floor(occ / W)``).
        max_source_tiles: static per-target-tile source-id capacity ``S``. Size from an eager
            probe of the near CSR (max over tiles of the summed source-tile count, self slot
            included) with a modest safety factor; under-sizing trips ``overflow`` (loud, not
            silent) rather than corrupting forces.
        max_leaf_occupancy: static bound on per-leaf particle count (the adaptive
            ``leaf_size``); ``ceil(max_leaf_occupancy / W)`` is the per-leaf tile cap used to
            expand source cells. A leaf exceeding it (a ``max_depth``-forced dense leaf) trips
            ``overflow``.
        index_dtype: integer dtype for id arrays.

    Returns:
        :class:`OctreeNearfieldTiles`.

    A tile ``t`` owned by leaf row ``r`` has as its source tiles: all tiles of ``r``'s near
    leaves (the CSR row ``u_neighbors[r]``, self-excluded) plus ``r``'s OWN other tiles
    (intra-cell near), with ``t`` itself removed (intra-tile ``i != j`` self handled by the
    caller). Because tiles never span cells, this reproduces exactly the leaf-pair near set the
    non-compact path evaluates, up to the compaction.
    """

    idt = index_dtype
    w = int(tile_width)
    num_tiles = int(num_tiles_cap)
    s_cap = int(max_source_tiles)
    mbpl = (int(max_leaf_occupancy) + w - 1) // w  # static max tiles per leaf

    node_ranges = jnp.asarray(view.node_ranges, dtype=idt)
    leaf_indices = jnp.asarray(view.leaf_indices, dtype=idt)
    num_nodes = int(node_ranges.shape[0])
    r_leaves = int(leaf_indices.shape[0])  # leaf_capacity (rows, sentinel-padded)
    u_cap = int(jnp.asarray(view.u_neighbors).shape[0] // max(r_leaves, 1))

    # ---- per-leaf geometry (Morton particle span) ----
    valid_leaf = leaf_indices < num_nodes  # real leaf vs sentinel pad
    leaf_safe = jnp.where(valid_leaf, leaf_indices, 0)
    lo = node_ranges[leaf_safe, 0]  # (R,) Morton start index
    occ = jnp.where(
        valid_leaf,
        node_ranges[leaf_safe, 1] - node_ranges[leaf_safe, 0] + 1,
        0,
    )  # (R,) particle count

    # ---- per-leaf tile counts + global (level-agnostic) tile numbering ----
    tiles_per_leaf = jnp.where(valid_leaf, (occ + (w - 1)) // w, 0)  # (R,)
    cum = jnp.cumsum(tiles_per_leaf)  # (R,) inclusive
    tile_base = cum - tiles_per_leaf  # (R,) first global tile id of leaf r
    total_tiles = jnp.sum(tiles_per_leaf)

    g = jnp.arange(num_tiles, dtype=idt)  # (num_tiles,)
    valid_tile = g < total_tiles
    tile_leaf_row = jnp.clip(
        jnp.searchsorted(cum, g, side="right"), 0, max(r_leaves - 1, 0)
    ).astype(idt)
    r_of_tile = tile_leaf_row
    local = g - tile_base[r_of_tile]  # (num_tiles,) local tile index within its leaf

    # ---- tile -> particle gather indices (Morton order), padded ----
    slot = jnp.arange(w, dtype=idt)  # (W,)
    within = local[:, None] * w + slot[None, :]  # (num_tiles, W) local particle offset
    tile_mask = valid_tile[:, None] & (within < occ[r_of_tile][:, None])
    tile_particle_idx = jnp.where(tile_mask, lo[r_of_tile][:, None] + within, 0)

    # ---- per-leaf compacted source-tile list ----
    # neighbour leaf rows = own leaf (self, first slot) + U-neighbours (CSR, self-excluded).
    src_rows = jnp.asarray(view.u_neighbors, dtype=idt).reshape(r_leaves, u_cap)
    own_row = jnp.arange(r_leaves, dtype=idt)[:, None]
    nbr = jnp.concatenate([own_row, src_rows], axis=1)  # (R, 1+u_cap)
    nbr_valid = jnp.concatenate(
        [
            valid_leaf[:, None],
            (src_rows < r_leaves) & (src_rows != own_row),  # guard against a self entry
        ],
        axis=1,
    )
    nbr_safe = jnp.where(nbr_valid, nbr, 0)

    # expand each neighbour leaf into its (up to mbpl) tiles
    k = jnp.arange(mbpl, dtype=idt)  # (mbpl,)
    cand = tile_base[nbr_safe][:, :, None] + k[None, None, :]  # (R, 1+u_cap, mbpl)
    cand_valid = nbr_valid[:, :, None] & (
        k[None, None, :] < tiles_per_leaf[nbr_safe][:, :, None]
    )
    c_width = (1 + u_cap) * mbpl
    cand = cand.reshape(r_leaves, c_width)
    cand_valid = cand_valid.reshape(r_leaves, c_width)

    leaf_src_ids, leaf_src_valid, leaf_src_counts = _compact_rows_to_width(
        cand, cand_valid, out_width=s_cap, fill=0, index_dtype=idt
    )

    # ---- broadcast per-leaf source list to tiles, then exclude the target tile itself ----
    tile_source_ids = leaf_src_ids[r_of_tile]  # (num_tiles, S)
    tile_source_valid = (
        leaf_src_valid[r_of_tile]
        & valid_tile[:, None]
        & (
            tile_source_ids != g[:, None]
        )  # drop the self tile (intra-tile handled elsewhere)
    )
    tile_source_ids = jnp.where(tile_source_ids < total_tiles, tile_source_ids, 0)
    tile_source_ids = jnp.where(tile_source_valid, tile_source_ids, 0)

    # ---- unused-tile sentinel + capacity gate (traced; caller checks eager-only) ----
    sentinel = jnp.asarray(num_tiles, dtype=idt)
    tile_leaf_row = jnp.where(valid_tile, r_of_tile, sentinel)

    tiles_per_leaf_max = (
        jnp.max(tiles_per_leaf) if r_leaves > 0 else jnp.asarray(0, idt)
    )
    num_source_tiles_max = (
        jnp.max(leaf_src_counts) if r_leaves > 0 else jnp.asarray(0, idt)
    )
    overflow = (
        (total_tiles > jnp.asarray(num_tiles, idt))
        | (tiles_per_leaf_max > jnp.asarray(mbpl, idt))
        | (num_source_tiles_max > jnp.asarray(s_cap, idt))
    )

    return OctreeNearfieldTiles(
        tile_particle_idx=tile_particle_idx.astype(idt),
        tile_mask=tile_mask,
        tile_leaf_row=tile_leaf_row.astype(idt),
        tile_source_ids=tile_source_ids.astype(idt),
        tile_source_valid=tile_source_valid,
        num_tiles_used=total_tiles.astype(idt),
        num_source_tiles_max=num_source_tiles_max.astype(idt),
        overflow=overflow,
    )
