"""Uniform-octree colleague / U / V interaction-list construction (O(N) FMM foundation).

Tree + interaction-list construction lives in yggdrax; the FMM operators that consume
these lists (P2M/M2M/M2L/L2L/L2P) live in jaccpot (``runtime/_octree_fmm``). The
per-leaf treecode is O(N log N) (it re-does each far interaction per leaf, no L2L
sharing); a level-restricted octree + classic Greengard-Rokhlin U/V interaction lists
restores O(N) via M2L into internal-node locals + an L2L cascade.

This module builds the interaction STRUCTURE (the novel, validated piece):

* a uniform-depth Morton octree (all leaves at depth ``L``; the reusable, ultra-fast
  Morton encode+sort foundation, here self-contained for the reference),
* per-node COLLEAGUE lists (same-level cells adjacent in Morton coords, ``|dcoord|<=1``),
* U lists (leaf colleagues -> near P2P) and V lists
  (``children(colleagues(parent)) \\ colleagues(self)`` -> M2L at the node's level).

Correctness contract (``test_octree_uvwx_coverage``): for every ordered leaf-cell pair,
the interaction is handled EXACTLY ONCE -- either U (adjacent at leaf level) or a V
interaction at the unique coarsest well-separated level. This partition is what
guarantees a correct O(N) FMM once operators are wired.

The FMM OPERATORS (P2M/M2M/M2L/L2L/L2P) are deliberately NOT wired here: jaccpot's raw
*reference* harmonic operators do not converge with order (recorded in the RESUME);
the operators must come from the unit-validated ``runtime/_octree_fmm`` kernels. So this
module intentionally stops at the (validated) interaction lists. Uniform depth is a
correctness scaffold; the clustered-galaxy regime needs the adaptive 2:1-balanced tree +
W/X lists (Phase 2).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .morton import _compact3_u64, _spread3_u64


class OctreeUVLists(NamedTuple):
    """Uniform-octree structure + U/V interaction lists (host-built reference)."""

    centers: np.ndarray  # (num_nodes, 3) cell geometric centers
    parent: np.ndarray  # (num_nodes,) parent node id (-1 at root)
    level: np.ndarray  # (num_nodes,) octree level (0..L)
    is_leaf: np.ndarray  # (num_nodes,) bool
    node_ranges: (
        np.ndarray
    )  # (num_nodes, 2) inclusive particle range (leaves; (-1,-1) internal)
    leaf_of: np.ndarray  # (num_particles,) leaf node id per (Morton-sorted) particle
    v_src: np.ndarray  # (num_v_pairs,) V-list source node ids
    v_tgt: np.ndarray  # (num_v_pairs,) V-list target node ids
    u_offsets: np.ndarray  # (num_leaves + 1,) CSR offsets into u_neighbors
    u_neighbors: np.ndarray  # near source-leaf node ids (self INCLUDED, i.e. U-list)
    leaf_indices: np.ndarray  # (num_leaves,) leaf node ids, CSR row order
    perm: np.ndarray  # (num_particles,) Morton sort permutation applied to inputs


def _morton_encode(coords: np.ndarray, nbits: int) -> np.ndarray:
    code = np.zeros(coords.shape[0], np.int64)
    for b in range(nbits):
        for d in range(3):
            code |= ((coords[:, d] >> b) & 1).astype(np.int64) << (3 * b + d)
    return code


def _decode_cell(cellcode: int, level: int) -> np.ndarray:
    c = int(cellcode)
    x = y = z = 0
    for b in range(level):
        x |= ((c >> (3 * b + 0)) & 1) << b
        y |= ((c >> (3 * b + 1)) & 1) << b
        z |= ((c >> (3 * b + 2)) & 1) << b
    return np.array([x, y, z])


def _encode_coord(coord: np.ndarray, level: int) -> int:
    code = 0
    for b in range(level):
        code |= int((coord[0] >> b) & 1) << (3 * b + 0)
        code |= int((coord[1] >> b) & 1) << (3 * b + 1)
        code |= int((coord[2] >> b) & 1) << (3 * b + 2)
    return code


def build_uniform_octree_uv(positions: np.ndarray, depth: int) -> OctreeUVLists:
    """Build a uniform-depth Morton octree + colleague-based U/V interaction lists.

    ``positions`` (N, 3); ``depth`` = octree levels (leaves at level ``depth``). Points
    are Morton-sorted (``perm`` returns the permutation); ``node_ranges`` / ``leaf_of``
    index the sorted order.
    """
    pts = np.asarray(positions, dtype=np.float64)
    L = int(depth)
    lo = pts.min(0)
    span = (pts.max(0) - lo).max() * (1 + 1e-6)
    g = np.minimum(((pts - lo) / span * (2**L)).astype(np.int64), 2**L - 1)
    leaf_code = _morton_encode(g, L)
    perm = np.argsort(leaf_code, kind="stable")
    leaf_code = leaf_code[perm]

    node_id: dict[tuple[int, int], int] = {}
    centers: list[np.ndarray] = []
    level_of: list[int] = []
    parent: list[int] = []
    nranges: list[tuple[int, int]] = []
    is_leaf: list[bool] = []
    node_cell: list[int] = []
    csz = [span / (2**lev) for lev in range(L + 1)]
    for lev in range(L + 1):
        cells = np.unique(leaf_code >> (3 * (L - lev)))
        for c in cells:
            nid = len(centers)
            node_id[(lev, int(c))] = nid
            coord = _decode_cell(c, lev)
            centers.append(lo + (coord + 0.5) * csz[lev])
            level_of.append(lev)
            is_leaf.append(lev == L)
            node_cell.append(int(c))
            parent.append(-1 if lev == 0 else node_id[(lev - 1, int(c) >> 3)])
            if lev == L:
                idx = np.where(leaf_code == c)[0]
                nranges.append((int(idx[0]), int(idx[-1])))
            else:
                nranges.append((-1, -1))
    num_nodes = len(centers)
    centers_a = np.array(centers)
    parent_a = np.array(parent)
    level_a = np.array(level_of)
    is_leaf_a = np.array(is_leaf)
    nranges_a = np.array(nranges)
    node_cell_a = np.array(node_cell)

    def colleagues(nid: int) -> set[int]:
        lev = level_a[nid]
        if lev == 0:
            return set()
        coord = _decode_cell(node_cell_a[nid], lev)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nc = coord + [dx, dy, dz]
                    if np.any(nc < 0) or np.any(nc >= 2**lev):
                        continue
                    code = _encode_coord(nc, lev)
                    if (lev, code) in node_id:
                        out.append(node_id[(lev, code)])
        return set(out)

    coll = {nid: colleagues(nid) for nid in range(num_nodes)}
    children: dict[int, list[int]] = {nid: [] for nid in range(num_nodes)}
    for nid in range(num_nodes):
        if parent_a[nid] >= 0:
            children[int(parent_a[nid])].append(nid)

    v_src: list[int] = []
    v_tgt: list[int] = []
    for C in range(num_nodes):
        p = int(parent_a[C])
        if p < 0:
            continue
        near_self = coll[C] | {C}
        for pc in coll[p] | {p}:
            for s in children[pc]:
                if s not in near_self:
                    v_src.append(s)
                    v_tgt.append(C)

    leaf_indices = np.array(
        [C for C in range(num_nodes) if is_leaf_a[C]], dtype=np.int64
    )
    u_off = np.zeros(len(leaf_indices) + 1, np.int64)
    u_nb: list[int] = []
    for r, C in enumerate(leaf_indices):
        near = [s for s in coll[int(C)] if is_leaf_a[s]] + [int(C)]
        u_nb.extend(near)
        u_off[r + 1] = u_off[r] + len(near)

    leaf_of = np.zeros(pts.shape[0], np.int64)
    for C in range(num_nodes):
        if is_leaf_a[C]:
            s, e = nranges_a[C]
            leaf_of[s : e + 1] = C

    return OctreeUVLists(
        centers=centers_a,
        parent=parent_a,
        level=level_a,
        is_leaf=is_leaf_a,
        node_ranges=nranges_a,
        leaf_of=leaf_of,
        v_src=np.array(v_src, np.int64),
        v_tgt=np.array(v_tgt, np.int64),
        u_offsets=u_off,
        u_neighbors=np.array(u_nb, np.int64),
        leaf_indices=leaf_indices,
        perm=perm,
    )


class UniformOctreeExecutionView(NamedTuple):
    """Kernel-ready uniform-octree execution view + U/V interaction lists.

    The node-topology fields map 1:1 onto the execution view consumed by the FMM
    operators (jaccpot ``runtime/_octree_fmm``): natural node layout, root at index 0,
    NO reserved sentinel node (the operators drop invalid scatter slots out of range),
    and ``num_valid_nodes`` is the TOTAL node count (the operators derive their per-level
    batch width from ``level_offsets`` themselves). Nodes are level-major, so ``parent <
    child index`` and the deepest level is the last, densest block.
    """

    # --- node topology / geometry (execution view) ---
    valid_mask: np.ndarray  # (num_nodes,) bool; all True for a uniform octree
    parent: np.ndarray  # (num_nodes,) parent id (-1 at root)
    children: np.ndarray  # (num_nodes, 8) child ids (-1 padded)
    child_counts: np.ndarray  # (num_nodes,) number of children
    node_depths: np.ndarray  # (num_nodes,) octree level (0..depth)
    node_ranges: (
        np.ndarray
    )  # (num_nodes, 2) inclusive particle span (internal = child span rollup)
    nodes_by_level: np.ndarray  # (num_nodes,) node ids sorted by level (stable)
    level_offsets: np.ndarray  # (depth + 2,) CSR offsets into nodes_by_level per level
    num_levels: int  # depth + 1
    leaf_mask: np.ndarray  # (num_nodes,) bool
    leaf_nodes: np.ndarray  # (num_nodes,) leaf ids, -1 padded to num_nodes
    num_valid_nodes: int  # total node count
    num_leaf_nodes: int  # number of leaves
    centers: np.ndarray  # (num_nodes, 3) cell geometric centers
    # --- interaction lists + particle permutation ---
    v_src: np.ndarray  # V-list M2L source node ids
    v_tgt: np.ndarray  # V-list M2L target node ids
    u_offsets: np.ndarray  # (num_leaves + 1,) CSR offsets into u_neighbors
    u_neighbors: np.ndarray  # near source-leaf node ids (self INCLUDED)
    leaf_indices: np.ndarray  # (num_leaves,) leaf node ids, CSR row order
    perm: np.ndarray  # (num_particles,) Morton sort permutation applied to inputs


def build_uniform_octree_execution_view(
    positions: np.ndarray, depth: int
) -> UniformOctreeExecutionView:
    """Build a uniform-octree execution view (topology + geometry + U/V lists).

    Wraps :func:`build_uniform_octree_uv` and derives the node-topology fields the FMM
    operators consume (child table, particle-span rollup for internal nodes, level-major
    ordering + per-level CSR offsets) in the natural node layout (root at index 0, no
    reserved sentinel node). ``positions`` (N, 3); ``depth`` = octree levels.
    """
    uv = build_uniform_octree_uv(positions, depth)
    L = int(depth)
    parent = np.asarray(uv.parent, np.int64)
    level = np.asarray(uv.level, np.int64)
    is_leaf = np.asarray(uv.is_leaf, bool)
    leaf_ranges = np.asarray(uv.node_ranges, np.int64)
    num_nodes = int(parent.shape[0])

    # child table from parent pointers (level-major order => parent < child)
    children = np.full((num_nodes, 8), -1, np.int64)
    slot = np.zeros(num_nodes, np.int64)
    for c in range(num_nodes):
        p = int(parent[c])
        if p >= 0:
            children[p, slot[p]] = c
            slot[p] += 1
    child_counts = (children >= 0).sum(1).astype(np.int64)

    # particle-span rollup: leaves from the U/V build; internal = min/max over children
    node_ranges = np.full((num_nodes, 2), -1, np.int64)
    node_ranges[is_leaf] = leaf_ranges[is_leaf]
    for c in sorted(range(num_nodes), key=lambda x: -int(level[x])):
        if not is_leaf[c]:
            ch = children[c][children[c] >= 0]
            node_ranges[c, 0] = node_ranges[ch, 0].min()
            node_ranges[c, 1] = node_ranges[ch, 1].max()

    nodes_by_level = np.argsort(level, kind="stable").astype(np.int64)
    counts = np.bincount(level, minlength=L + 1)
    level_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    leaf_indices = np.asarray(uv.leaf_indices, np.int64)
    leaf_nodes = np.full(num_nodes, -1, np.int64)
    leaf_nodes[: len(leaf_indices)] = leaf_indices

    return UniformOctreeExecutionView(
        valid_mask=np.ones(num_nodes, bool),
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_depths=level,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=L + 1,
        leaf_mask=is_leaf,
        leaf_nodes=leaf_nodes,
        num_valid_nodes=num_nodes,
        num_leaf_nodes=int(len(leaf_indices)),
        centers=np.asarray(uv.centers),
        v_src=np.asarray(uv.v_src, np.int64),
        v_tgt=np.asarray(uv.v_tgt, np.int64),
        u_offsets=np.asarray(uv.u_offsets, np.int64),
        u_neighbors=np.asarray(uv.u_neighbors, np.int64),
        leaf_indices=leaf_indices,
        perm=np.asarray(uv.perm, np.int64),
    )


_U64 = jnp.uint64


def _neighbor_offsets_26() -> np.ndarray:
    """The 26 (dx,dy,dz) colleague offsets (all of {-1,0,1}^3 except origin)."""
    offs = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]
    return np.asarray(offs, np.int64)


def _cell_coords(cell_code: jnp.ndarray) -> jnp.ndarray:
    """De-interleave Morton cell codes -> integer (x, y, z) cell coords (stacked -1)."""
    c = cell_code.astype(_U64)
    x = _compact3_u64(c)
    y = _compact3_u64(c >> _U64(1))
    z = _compact3_u64(c >> _U64(2))
    return jnp.stack([x, y, z], axis=-1).astype(jnp.int64)


def _encode_coords(coords: jnp.ndarray) -> jnp.ndarray:
    """Interleave integer (x, y, z) cell coords (..., 3) -> Morton cell codes (uint64)."""
    x = _spread3_u64(coords[..., 0].astype(_U64))
    y = _spread3_u64(coords[..., 1].astype(_U64))
    z = _spread3_u64(coords[..., 2].astype(_U64))
    return x | (y << _U64(1)) | (z << _U64(2))


def build_uniform_octree_execution_view_device(
    positions: jnp.ndarray,
    depth: int,
    *,
    bounds: Optional[tuple] = None,
    v_capacity: int = 216,
    u_capacity: int = 26,
    index_dtype=jnp.int64,
) -> UniformOctreeExecutionView:
    """On-device (JAX), STATIC-SHAPE uniform-octree execution view + U/V lists.

    Unlike :func:`build_uniform_octree_execution_view` (host numpy, occupied-only nodes),
    this builds a DENSE node space -- every cell at every level ``0..L`` gets a slot, so
    the node count is ``(8^(L+1)-1)/7`` and depends ONLY on ``depth`` (never on the data).
    Together with fixed-capacity U/V lists and ``searchsorted`` particle ranges, every
    array shape is a function of ``(N, depth)`` alone: the build compiles once and is
    reused across all timesteps (positions/masses change, shapes do not) -- no
    recompilation inside a time-integration loop. Empty cells carry empty particle ranges
    and contribute nothing.

    Node id of cell ``c`` at level ``l`` is ``level_offsets[l] + c``. Parent/children are
    arithmetic; colleagues and V lists use geometric (integer-coord) tests. Invalid far /
    near list entries use the out-of-range sentinel ``num_nodes`` (dropped downstream).

    ``v_capacity`` (max V-list sources per node; 216 = 27 parent-near cells * 8 children)
    and ``u_capacity`` (max U-list neighbours per leaf; 26 colleagues) are static caps.
    """
    L = int(depth)
    idt = index_dtype
    pts = jnp.asarray(positions)
    n_particles = int(pts.shape[0])

    if bounds is None:
        lo = jnp.min(pts, axis=0)
        span = jnp.max(jnp.max(pts, axis=0) - lo) * (1.0 + 1e-6)
    else:
        lo = jnp.asarray(bounds[0])
        span = jnp.max(jnp.asarray(bounds[1]) - lo) * (1.0 + 1e-6)

    # level-L integer cell coords -> Morton leaf code; sort particles by it
    g = jnp.clip(((pts - lo) / span * (2**L)).astype(jnp.int64), 0, 2**L - 1)
    leaf_code = _encode_coords(g)  # (N,) uint64 in [0, 8^L)
    perm = jnp.argsort(leaf_code)
    sorted_code = leaf_code[perm]

    # ---- dense node tables (STATIC: derived from L only) ----
    level_sizes = [8**l for l in range(L + 1)]
    off_np = np.concatenate([[0], np.cumsum(level_sizes)]).astype(np.int64)
    num_nodes = int(off_np[L + 1])
    node_level_np = np.concatenate([np.full(8**l, l, np.int64) for l in range(L + 1)])
    node_cell_np = np.concatenate(
        [np.arange(8**l, dtype=np.int64) for l in range(L + 1)]
    )
    off = jnp.asarray(off_np, idt)
    node_level = jnp.asarray(node_level_np, idt)
    node_cell = jnp.asarray(node_cell_np)  # uint-safe as int64

    # ---- particle ranges via searchsorted (a cell c at level l owns leaf codes in
    #      [c<<3(L-l), (c+1)<<3(L-l)) ) ----
    shift = (3 * (L - node_level)).astype(_U64)
    cell_u = node_cell.astype(_U64)
    lo_code = cell_u << shift
    hi_code = (cell_u + _U64(1)) << shift
    start = jnp.searchsorted(sorted_code, lo_code, side="left").astype(idt)
    end_excl = jnp.searchsorted(sorted_code, hi_code, side="left").astype(idt)
    count = end_excl - start
    node_ranges = jnp.stack([start, end_excl - 1], axis=-1).astype(idt)

    # ---- parent / children (arithmetic) ----
    parent = jnp.where(
        node_level == 0,
        jnp.asarray(-1, idt),
        off[jnp.clip(node_level - 1, 0, L)] + (node_cell >> 3).astype(idt),
    ).astype(idt)
    child_base = off[jnp.clip(node_level + 1, 0, L)] + (node_cell * 8).astype(idt)
    children = child_base[:, None] + jnp.arange(8, dtype=idt)[None, :]
    children = jnp.where((node_level < L)[:, None], children, jnp.asarray(-1, idt))
    child_counts = jnp.where(node_level < L, jnp.asarray(8, idt), jnp.asarray(0, idt))

    leaf_mask = node_level == L
    leaf_indices = jnp.arange(off_np[L], num_nodes, dtype=idt)  # all level-L cells
    leaf_nodes = (
        jnp.full((num_nodes,), -1, idt)
        .at[jnp.arange(leaf_indices.shape[0], dtype=idt)]
        .set(leaf_indices)
    )
    nodes_by_level = jnp.arange(num_nodes, dtype=idt)  # dense = already level-major
    level_offsets = off

    # ---- centres ----
    coord = _cell_coords(node_cell).astype(jnp.float64)  # (M,3) at each node's level
    csz = span / jnp.power(2.0, node_level.astype(jnp.float64))
    centers = lo + (coord + 0.5) * csz[:, None]

    # ---- colleagues: same-level cells within +/-1 coord (26 offsets) ----
    neigh = jnp.asarray(_neighbor_offsets_26(), idt)  # (26,3)
    ncoord = coord.astype(idt)[:, None, :] + neigh[None, :, :]  # (M,26,3)
    two_l = jnp.power(2, node_level).astype(idt)[:, None, None]
    in_bounds = jnp.all((ncoord >= 0) & (ncoord < two_l), axis=-1)  # (M,26)
    ncode = _encode_coords(ncoord)  # (M,26) uint64
    colleague = off[node_level][:, None] + ncode.astype(idt)  # (M,26) node ids
    colleague = jnp.where(in_bounds, colleague, jnp.asarray(-1, idt))

    sentinel = jnp.asarray(num_nodes, idt)  # out-of-range invalid marker

    # ---- V list: children of (parent's colleagues + parent) that are NOT adjacent to
    #      self at this level (|dcoord|_inf > 1). Geometric test, fixed capacity. ----
    safe_parent = jnp.clip(parent, 0, num_nodes - 1)
    # parent-near set: parent + parent's 26 colleagues  (M, 27) node ids (-1 invalid)
    pnear = jnp.concatenate([parent[:, None], colleague[safe_parent]], axis=1)  # (M,27)
    pnear = jnp.where(parent[:, None] >= 0, pnear, jnp.asarray(-1, idt))
    pnear_cell = node_cell[
        jnp.clip(pnear, 0, num_nodes - 1)
    ]  # (M,27) cells at level l-1
    # their 8 children (cells at level l): 8*pc + [0..7]
    cand_cell = pnear_cell[:, :, None] * 8 + jnp.arange(8, dtype=idt)[None, None, :]
    cand_valid0 = (pnear >= 0)[:, :, None] & jnp.broadcast_to(
        (node_level >= 1)[:, None, None], cand_cell.shape
    )
    cand_cell = jnp.reshape(cand_cell, (num_nodes, 27 * 8))  # (M,216)
    cand_valid0 = jnp.reshape(cand_valid0, (num_nodes, 27 * 8))
    cand_coord = _cell_coords(cand_cell.astype(_U64)).astype(idt)  # (M,216,3)
    dcoord = jnp.abs(cand_coord - coord.astype(idt)[:, None, :])
    well_sep = jnp.max(dcoord, axis=-1) > 1  # (M,216) not adjacent at this level
    cand_node = off[node_level][:, None] + cand_cell.astype(idt)  # (M,216) node ids
    v_ok = cand_valid0 & well_sep  # (M,216)
    if int(27 * 8) != int(v_capacity):
        v_capacity = 27 * 8
    v_tgt_full = jnp.broadcast_to(
        jnp.arange(num_nodes, dtype=idt)[:, None], (num_nodes, v_capacity)
    )
    v_src = jnp.where(v_ok, cand_node, sentinel).reshape(-1)
    v_tgt = jnp.where(v_ok, v_tgt_full, sentinel).reshape(-1)

    # ---- U list (leaves only): occupied colleague leaves, self excluded, as ROW ids ----
    num_leaves = int(leaf_indices.shape[0])
    leaf_coll = colleague[leaf_indices]  # (num_leaves, 26) node ids (-1 invalid)
    coll_row = leaf_coll - jnp.asarray(int(off_np[L]), idt)  # node id -> leaf row
    coll_count = count[jnp.clip(leaf_coll, 0, num_nodes - 1)]  # occupancy of colleague
    coll_ok = (leaf_coll >= 0) & (coll_count > 0)  # valid + occupied
    u_capacity = 26
    u_neighbors = jnp.where(coll_ok, coll_row, sentinel).reshape(-1)  # (num_leaves*26,)
    u_offsets = jnp.arange(0, (num_leaves + 1) * u_capacity, u_capacity, dtype=idt)

    return UniformOctreeExecutionView(
        valid_mask=jnp.ones((num_nodes,), bool),
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_depths=node_level,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=L + 1,
        leaf_mask=leaf_mask,
        leaf_nodes=leaf_nodes,
        num_valid_nodes=num_nodes,
        num_leaf_nodes=num_leaves,
        centers=centers,
        v_src=v_src,
        v_tgt=v_tgt,
        u_offsets=u_offsets,
        u_neighbors=u_neighbors,
        leaf_indices=leaf_indices,
        perm=perm.astype(idt),
    )


def build_sparse_uniform_octree_execution_view_device(
    positions: jnp.ndarray,
    depth: int,
    *,
    node_capacity: int,
    leaf_capacity: int,
    level_batch_width_cap: Optional[int] = None,
    bounds: Optional[tuple] = None,
    index_dtype=jnp.int64,
) -> UniformOctreeExecutionView:
    """On-device (JAX), STATIC-SHAPE *sparse* uniform-octree execution view + U/V lists.

    Same output contract as :func:`build_uniform_octree_execution_view_device` (a
    :class:`UniformOctreeExecutionView` that drops straight into jaccpot's
    ``octree_fmm_uvwx``), but only the OCCUPIED cells at levels ``0..L`` (``L = depth``)
    become nodes. The dense builder allocates all ``(8^(L+1)-1)/7`` cells; on a
    concentrated galaxy disk that OOMs long before the leaf occupancy is small enough. The
    sparse builder instead lists the occupied ``(level, cell)`` keys, level-major and
    within-level Morton-sorted, padded to a fixed ``node_capacity`` (nodes) / ``leaf_capacity``
    (leaves) so every array shape is a function of ``(N, depth, node_capacity, leaf_capacity)``
    alone -- never of the data values. The build therefore still compiles once and is reused
    across timesteps with no recompilation.

    Nodes are stored in sorted-key order, so the node id IS the array index: parent /
    children / colleagues / V-list candidates are resolved by ``searchsorted`` on the sorted
    node-key array (a cell that is not occupied returns the out-of-range sentinel), rather
    than by the dense builder's arithmetic node ids. Invalid far/near list entries and node
    padding use the out-of-range sentinel ``node_capacity`` (dropped downstream, matching the
    dense builder's ``num_nodes`` convention); absent children/colleagues use ``-1``.

    ``node_capacity`` / ``leaf_capacity`` are static caps: the number of valid nodes / leaves
    is data-dependent (returned as ``num_valid_nodes`` / ``num_leaf_nodes``), but the array
    shapes are static. Pick ``leaf_capacity >= 8^L`` is unnecessary -- size it to the occupied
    leaf count, and ``node_capacity`` to the total occupied node count, both with headroom.
    Overflow (more occupied cells than a cap) degrades detectably rather than silently: the
    deepest cells are dropped and ``num_valid_nodes == node_capacity`` (no sentinel padding);
    callers should assert ``num_valid_nodes < node_capacity`` and
    ``num_leaf_nodes < leaf_capacity``.

    ``level_batch_width_cap`` (default ``leaf_capacity``) is the fixed per-level batch window
    the caller must pass STATIC to jaccpot's M2M / L2L kernels (``level_batch_width``); it must
    be ``>=`` the max per-level node count (the leaf level dominates). The kernels
    ``dynamic_slice`` ``level_batch_width`` nodes starting at ``level_offsets[l]``, so for the
    batched levels (M2M/L2L touch parent levels ``0..L-1``, the deepest being ``L-1``) to stay
    aligned rather than clamp, the caps must satisfy
    ``level_offsets[L - 1] + level_batch_width_cap <= node_capacity``. ``level_offsets[L - 1]``
    is the count of nodes ABOVE the deepest-parent level (small -- the leaf level itself is
    never a batch parent), so it suffices to size ``node_capacity`` a little above
    ``leaf_capacity`` (on the 200k disk at ``L=7``: ``level_offsets[6] = 2629``, so
    ``node_capacity = 40000`` clears ``2629 + 32768 = 35397`` comfortably).
    """
    L = int(depth)
    idt = index_dtype
    node_capacity = int(node_capacity)
    leaf_capacity = int(leaf_capacity)
    if level_batch_width_cap is None:
        level_batch_width_cap = leaf_capacity
    level_batch_width_cap = int(level_batch_width_cap)
    pts = jnp.asarray(positions)

    if bounds is None:
        lo = jnp.min(pts, axis=0)
        span = jnp.max(jnp.max(pts, axis=0) - lo) * (1.0 + 1e-6)
    else:
        lo = jnp.asarray(bounds[0])
        span = jnp.max(jnp.asarray(bounds[1]) - lo) * (1.0 + 1e-6)

    # level-L integer cell coords -> Morton leaf code; sort particles by it
    g = jnp.clip(((pts - lo) / span * (2**L)).astype(jnp.int64), 0, 2**L - 1)
    leaf_code = _encode_coords(g)  # (N,) uint64 in [0, 8^L)
    perm = jnp.argsort(leaf_code)
    sorted_code = leaf_code[perm]

    three_L = 3 * L
    cell_mask = (1 << three_L) - 1
    fill_key = (L + 1) << three_L  # decodes to level L+1 (> any real key), cell 0
    sentinel = jnp.asarray(
        node_capacity, idt
    )  # out-of-range marker (dropped downstream)

    # ---- OCCUPIED nodes: level-major, within-level Morton-sorted unique (level, cell) ----
    # key = (level << 3L) | cell_code_at_level  -> lexicographic sort = level-major/Morton.
    levels_col = jnp.arange(L + 1, dtype=jnp.int64)  # (L+1,)
    shifts = (three_L - 3 * levels_col).astype(_U64)  # (L+1,)
    cells_all = (sorted_code[:, None] >> shifts[None, :]).astype(jnp.int64)  # (N, L+1)
    keys_all = (levels_col[None, :] << three_L) | cells_all  # (N, L+1)
    node_key = jnp.unique(
        keys_all.reshape(-1), size=node_capacity, fill_value=fill_key
    ).astype(jnp.int64)
    valid_mask = node_key != jnp.asarray(fill_key, jnp.int64)
    node_level = (node_key >> three_L).astype(idt)  # padding -> L+1
    node_cell = (node_key & cell_mask).astype(idt)
    num_valid = jnp.sum(valid_mask.astype(idt))
    node_key_j = node_key  # sorted; searchsorted target array for lookups

    # per-level contiguous ranges (level-major): level_offsets[l] = first slot with level l
    level_offsets = jnp.searchsorted(
        node_level, jnp.arange(L + 2, dtype=idt), side="left"
    ).astype(idt)
    off_L = level_offsets[L]  # first leaf-level (== L) node id

    def lookup_node(level_arr, cell_arr):
        """Node id of an occupied ``(level, cell)`` via searchsorted; else ``sentinel``."""
        lvl = jnp.asarray(level_arr, idt)
        cll = jnp.asarray(cell_arr, idt)
        in_level = (lvl >= 0) & (lvl <= L)
        lvl_safe = jnp.clip(lvl, 0, L)
        target = (lvl_safe << three_L) | (cll & jnp.asarray(cell_mask, idt))
        pos = jnp.searchsorted(node_key_j, target, side="left").astype(idt)
        pos_safe = jnp.clip(pos, 0, node_capacity - 1)
        found = in_level & (node_key_j[pos_safe] == target)
        return jnp.where(found, pos_safe, sentinel)

    # ---- particle ranges via searchsorted (cell c at level l owns leaf codes in
    #      [c<<3(L-l), (c+1)<<3(L-l)) ); padding slots forced empty ----
    lvl_r = jnp.minimum(node_level, L)  # avoid negative shift for padding (level L+1)
    shift = (3 * (L - lvl_r)).astype(_U64)
    cell_u = node_cell.astype(_U64)
    lo_code = cell_u << shift
    hi_code = (cell_u + _U64(1)) << shift
    start = jnp.searchsorted(sorted_code, lo_code, side="left").astype(idt)
    end_excl = jnp.searchsorted(sorted_code, hi_code, side="left").astype(idt)
    nr = jnp.stack([start, end_excl - 1], axis=-1)
    node_ranges = jnp.where(valid_mask[:, None], nr, jnp.asarray([0, -1], idt))

    # ---- parent / children via cell arithmetic + occupancy lookup ----
    parent = lookup_node(node_level - 1, node_cell >> 3)
    parent = jnp.where(node_level == 0, jnp.asarray(-1, idt), parent)
    parent = jnp.where(valid_mask, parent, jnp.asarray(-1, idt))

    child_cells = node_cell[:, None] * 8 + jnp.arange(8, dtype=idt)[None, :]  # (M,8)
    child_levels = jnp.broadcast_to((node_level + 1)[:, None], child_cells.shape)
    child_lookup = lookup_node(child_levels, child_cells)  # (M,8) node id / sentinel
    children = jnp.where(child_lookup == sentinel, jnp.asarray(-1, idt), child_lookup)
    children = jnp.where(valid_mask[:, None], children, jnp.asarray(-1, idt))
    child_counts = (children >= 0).sum(1).astype(idt)

    # ---- leaves (level L) ----
    leaf_mask = valid_mask & (node_level == L)
    num_leaves = jnp.sum(leaf_mask.astype(idt))
    leaf_row = jnp.arange(leaf_capacity, dtype=idt)
    leaf_indices = jnp.where(leaf_row < num_leaves, off_L + leaf_row, sentinel)
    row_node = jnp.arange(node_capacity, dtype=idt)
    leaf_nodes = jnp.where(
        row_node < num_leaves, off_L + row_node, jnp.asarray(-1, idt)
    )
    nodes_by_level = jnp.arange(node_capacity, dtype=idt)  # already level-major

    # ---- centres ----
    coord_int = _cell_coords(node_cell)  # (M,3) int cell coords at each node's level
    lvl_f = jnp.minimum(node_level, L).astype(jnp.float64)
    csz = span / jnp.power(2.0, lvl_f)
    centers = lo + (coord_int.astype(jnp.float64) + 0.5) * csz[:, None]

    # ---- colleagues: same-level cells within +/-1 coord (26 offsets), occupancy lookup ----
    neigh = jnp.asarray(_neighbor_offsets_26(), idt)  # (26,3)
    ncoord = coord_int[:, None, :] + neigh[None, :, :]  # (M,26,3)
    two_l = jnp.power(2, jnp.minimum(node_level, L)).astype(idt)[:, None, None]
    in_bounds = jnp.all((ncoord >= 0) & (ncoord < two_l), axis=-1)  # (M,26)
    ncell = _encode_coords(ncoord).astype(idt)  # (M,26)
    coll_levels = jnp.broadcast_to(node_level[:, None], ncell.shape)
    colleague = lookup_node(coll_levels, ncell)  # (M,26) node id / sentinel
    colleague = jnp.where(in_bounds, colleague, jnp.asarray(-1, idt))

    # ---- V list: children of (parent + parent's colleagues) NOT adjacent to self ----
    safe_parent = jnp.clip(parent, 0, node_capacity - 1)
    pnear = jnp.concatenate([parent[:, None], colleague[safe_parent]], axis=1)  # (M,27)
    pnear = jnp.where(parent[:, None] >= 0, pnear, jnp.asarray(-1, idt))
    pnear_valid = (pnear >= 0) & (pnear < node_capacity)  # excludes -1 and sentinel
    pnear_cell = node_cell[
        jnp.clip(pnear, 0, node_capacity - 1)
    ]  # (M,27) level l-1 cells
    cand_cell = pnear_cell[:, :, None] * 8 + jnp.arange(8, dtype=idt)[None, None, :]
    cand_valid0 = pnear_valid[:, :, None] & (node_level >= 1)[:, None, None]  # (M,27,1)
    cand_cell = cand_cell.reshape(node_capacity, 27 * 8)  # (M,216)
    cand_valid0 = jnp.broadcast_to(cand_valid0, (node_capacity, 27, 8)).reshape(
        node_capacity, 27 * 8
    )
    cand_coord = _cell_coords(cand_cell.astype(_U64)).astype(jnp.int32)  # (M,216,3)
    dcoord = jnp.abs(cand_coord - coord_int.astype(jnp.int32)[:, None, :])
    well_sep = jnp.max(dcoord, axis=-1) > 1  # (M,216) not adjacent at this level
    cand_levels = jnp.broadcast_to(node_level[:, None], cand_cell.shape)
    cand_node = lookup_node(
        cand_levels, cand_cell
    )  # (M,216) occupied node id / sentinel
    v_ok = cand_valid0 & well_sep & (cand_node != sentinel) & valid_mask[:, None]
    v_self = jnp.broadcast_to(
        jnp.arange(node_capacity, dtype=idt)[:, None], cand_cell.shape
    )
    v_src = jnp.where(v_ok, cand_node, sentinel).reshape(-1)
    v_tgt = jnp.where(v_ok, v_self, sentinel).reshape(-1)

    # ---- U list (leaves only): occupied colleague leaves, self excluded, as ROW ids ----
    leaf_nid = jnp.clip(off_L + leaf_row, 0, node_capacity - 1)  # (leaf_capacity,)
    leaf_coll = colleague[leaf_nid]  # (leaf_capacity, 26) node id / sentinel / -1
    coll_is_node = (leaf_coll >= 0) & (leaf_coll < node_capacity)  # found occupied leaf
    coll_ok = coll_is_node & (leaf_row < num_leaves)[:, None]
    coll_row = leaf_coll - off_L  # leaf node id -> leaf row
    u_neighbors = jnp.where(coll_ok, coll_row, sentinel).reshape(
        -1
    )  # (leaf_capacity*26,)
    u_offsets = jnp.arange(0, (leaf_capacity + 1) * 26, 26, dtype=idt)

    node_depths = jnp.where(valid_mask, node_level, jnp.asarray(L, idt))

    return UniformOctreeExecutionView(
        valid_mask=valid_mask,
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_depths=node_depths,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=L + 1,
        leaf_mask=leaf_mask,
        leaf_nodes=leaf_nodes,
        num_valid_nodes=num_valid,
        num_leaf_nodes=num_leaves,
        centers=centers,
        v_src=v_src,
        v_tgt=v_tgt,
        u_offsets=u_offsets,
        u_neighbors=u_neighbors,
        leaf_indices=leaf_indices,
        perm=perm.astype(idt),
    )


def build_adaptive_octree_execution_view_device(
    positions: jnp.ndarray,
    max_depth: int,
    leaf_size: int,
    *,
    node_capacity: int,
    leaf_capacity: int,
    level_batch_width_cap: Optional[int] = None,
    bounds: Optional[tuple] = None,
    index_dtype=jnp.int64,
) -> UniformOctreeExecutionView:
    """On-device (JAX), STATIC-SHAPE *adaptive-depth* octree execution view.

    Same output contract as :func:`build_sparse_uniform_octree_execution_view_device` (a
    :class:`UniformOctreeExecutionView` that drops straight into jaccpot's
    ``octree_fmm_uvwx``), but the node set is the ADAPTIVE tree rather than every occupied
    cell to a uniform depth. A uniform-depth octree pads every leaf to the MAX leaf
    occupancy while the MEAN is far lower (a concentrated galaxy disk at ``L=7`` has mean
    occupancy ~7.5 but max ~128), so its near field wastes ~max/mean per leaf. The
    adaptive tree instead subdivides ONLY where a cell is dense, so every leaf holds at
    most ``leaf_size`` particles (except leaves forced at ``max_depth``) and is ~full: on
    the 200k disk at ``leaf_size = 256`` that is ~800 leaves, ~40x fewer than the 32768
    uniform-``L7`` leaf slots, collapsing the padding.

    ADAPTIVE NODE SET (the only conceptual change from the sparse builder). For every
    occupied cell ``(level l, cell c)`` with occupancy ``occ`` (particle count):

    * INTERNAL iff ``occ > leaf_size`` AND ``l < max_depth`` (it will be subdivided);
    * ACTIVE (a node in the tree) iff it is occupied AND (``l == 0`` OR its parent cell
      ``(l - 1, c >> 3)`` is internal), i.e. all of its proper ancestors are internal;
    * a LEAF iff it is active and NOT internal (``occ <= leaf_size`` or ``l == max_depth``).

    Because active-ness of a cell is fixed by its parent's occupancy (shared by all
    particles in the cell), it is well defined per cell. Every particle lands in exactly
    one leaf -- the first cell along its root-to-``max_depth`` path that is not internal --
    so the leaves' particle ranges partition ``[0, N)`` exactly once, with leaves at
    VARIABLE levels.

    Construction (static shape throughout). Particles are Morton-sorted at ``max_depth``;
    the per-``(particle, level)`` cell keys ``(l << 3L) | cell`` are formed exactly as in
    the sparse builder, and each cell's ``occ`` / parent ``occ`` come from ``searchsorted``
    on the sorted leaf codes (parent ``occ`` at level ``l`` is simply ``occ`` at level
    ``l - 1``). Keys of non-active cells are masked to the fill value and the ACTIVE keys
    are compacted with ``jnp.unique(size=node_capacity)`` (level-major / within-level
    Morton-sorted, so the node id is the sorted-array index). ``parent`` / ``children`` /
    ``node_ranges`` / ``centers`` then follow the sparse builder's ``searchsorted``
    ``lookup_node`` (a leaf's children are absent from the active set, hence ``-1``).

    STATIC shapes: every array shape is a function of ``(N, max_depth, node_capacity,
    leaf_capacity)`` only, never of the data values, so the build compiles once and is
    reused across timesteps with no recompilation. ``num_valid_nodes`` / ``num_leaf_nodes``
    are data-dependent (returned as traced scalars); ``num_levels = max_depth + 1`` is a
    static python int. Padding / invalid list entries use the out-of-range sentinel
    ``node_capacity`` (dropped downstream, matching the sparse/dense builders); absent
    children use ``-1``. ``node_capacity`` / ``leaf_capacity`` are static caps sized to the
    active-node / leaf count with headroom (``node_capacity >= leaf_capacity``); overflow
    degrades detectably rather than silently -- the deepest active cells are dropped and
    ``num_valid_nodes == node_capacity`` (callers should assert ``num_valid_nodes <
    node_capacity`` and ``num_leaf_nodes < leaf_capacity``).

    STAGE 1 SCOPE: this builds the adaptive node topology + geometry ONLY. The interaction
    lists (``v_src`` / ``v_tgt`` / ``u_offsets`` / ``u_neighbors``) are NOT built here --
    cross-level colleague / U / V logic for a variable-depth tree is Stage 2. They are
    returned as shape-correct PLACEHOLDERS (``v_src`` / ``v_tgt`` full sentinel,
    ``u_offsets`` all zeros, ``u_neighbors`` empty) so the NamedTuple is well-formed; do
    NOT feed this view to the FMM's far/near passes until Stage 2 populates them.

    ``level_batch_width_cap`` (default ``leaf_capacity``) is accepted for API parity with
    the sparse builder (the caller passes it STATIC to jaccpot's M2M / L2L kernels); it is
    not consumed internally.
    """
    L = int(max_depth)
    idt = index_dtype
    leaf_size = int(leaf_size)
    node_capacity = int(node_capacity)
    leaf_capacity = int(leaf_capacity)
    if level_batch_width_cap is None:
        level_batch_width_cap = leaf_capacity
    level_batch_width_cap = int(level_batch_width_cap)
    pts = jnp.asarray(positions)
    n_particles = int(pts.shape[0])

    if bounds is None:
        lo = jnp.min(pts, axis=0)
        span = jnp.max(jnp.max(pts, axis=0) - lo) * (1.0 + 1e-6)
    else:
        lo = jnp.asarray(bounds[0])
        span = jnp.max(jnp.asarray(bounds[1]) - lo) * (1.0 + 1e-6)

    # level-L integer cell coords -> Morton leaf code; sort particles by it
    g = jnp.clip(((pts - lo) / span * (2**L)).astype(jnp.int64), 0, 2**L - 1)
    leaf_code = _encode_coords(g)  # (N,) uint64 in [0, 8^L)
    perm = jnp.argsort(leaf_code)
    sorted_code = leaf_code[perm]

    three_L = 3 * L
    cell_mask = (1 << three_L) - 1
    fill_key = (L + 1) << three_L  # decodes to level L+1 (> any real key), cell 0
    sentinel = jnp.asarray(
        node_capacity, idt
    )  # out-of-range marker (dropped downstream)

    # ---- per-(particle, level) cell keys + occupancy (same keys as the sparse build) ----
    # key = (level << 3L) | cell_code_at_level  -> lexicographic sort = level-major/Morton.
    levels_col = jnp.arange(L + 1, dtype=jnp.int64)  # (L+1,)
    shifts = (three_L - 3 * levels_col).astype(_U64)  # (L+1,) = 3*(L - l)
    cells_all = (sorted_code[:, None] >> shifts[None, :]).astype(jnp.int64)  # (N, L+1)
    keys_all = (levels_col[None, :] << three_L) | cells_all  # (N, L+1)

    # occupancy of each particle's level-l cell (Morton span [c<<3(L-l), (c+1)<<3(L-l)))
    cells_u = cells_all.astype(_U64)
    lo_code = cells_u << shifts[None, :]
    hi_code = (cells_u + _U64(1)) << shifts[None, :]
    occ = (
        jnp.searchsorted(sorted_code, hi_code, side="left")
        - jnp.searchsorted(sorted_code, lo_code, side="left")
    ).astype(
        jnp.int64
    )  # (N, L+1)

    # parent occupancy: occ of the level-(l-1) cell == occ shifted one level shallower.
    # Column 0 is a dummy (root is always active) so its value is irrelevant.
    big = jnp.asarray(n_particles + 1, jnp.int64)
    parent_occ = jnp.concatenate(
        [jnp.full((n_particles, 1), big, jnp.int64), occ[:, :-1]], axis=1
    )  # (N, L+1)

    # ACTIVE iff root OR parent is internal (parent occ > leaf_size). Mask the rest away.
    active = (levels_col[None, :] == 0) | (parent_occ > leaf_size)  # (N, L+1)
    active_keys = jnp.where(active, keys_all, jnp.asarray(fill_key, jnp.int64))

    # ---- compact the ACTIVE keys -> node set (level-major / within-level Morton-sorted) ----
    node_key = jnp.unique(
        active_keys.reshape(-1), size=node_capacity, fill_value=fill_key
    ).astype(jnp.int64)
    valid_mask = node_key != jnp.asarray(fill_key, jnp.int64)
    node_level = (node_key >> three_L).astype(idt)  # padding -> L+1
    node_cell = (node_key & cell_mask).astype(idt)
    num_valid = jnp.sum(valid_mask.astype(idt))
    node_key_j = node_key  # sorted; searchsorted target array for lookups

    # per-level contiguous ranges (level-major): level_offsets[l] = first slot with level l
    level_offsets = jnp.searchsorted(
        node_level, jnp.arange(L + 2, dtype=idt), side="left"
    ).astype(idt)

    def lookup_node(level_arr, cell_arr):
        """Node id of an active ``(level, cell)`` via searchsorted; else ``sentinel``."""
        lvl = jnp.asarray(level_arr, idt)
        cll = jnp.asarray(cell_arr, idt)
        in_level = (lvl >= 0) & (lvl <= L)
        lvl_safe = jnp.clip(lvl, 0, L)
        target = (lvl_safe << three_L) | (cll & jnp.asarray(cell_mask, idt))
        pos = jnp.searchsorted(node_key_j, target, side="left").astype(idt)
        pos_safe = jnp.clip(pos, 0, node_capacity - 1)
        found = in_level & (node_key_j[pos_safe] == target)
        return jnp.where(found, pos_safe, sentinel)

    # ---- particle ranges + node occupancy via searchsorted (padding forced empty) ----
    lvl_r = jnp.minimum(node_level, L)  # avoid negative shift for padding (level L+1)
    shift = (3 * (L - lvl_r)).astype(_U64)
    cell_uu = node_cell.astype(_U64)
    lo_c = cell_uu << shift
    hi_c = (cell_uu + _U64(1)) << shift
    start = jnp.searchsorted(sorted_code, lo_c, side="left").astype(idt)
    end_excl = jnp.searchsorted(sorted_code, hi_c, side="left").astype(idt)
    node_occ = (end_excl - start).astype(idt)
    nr = jnp.stack([start, end_excl - 1], axis=-1)
    node_ranges = jnp.where(valid_mask[:, None], nr, jnp.asarray([0, -1], idt))

    # ---- internal / leaf classification ----
    internal = valid_mask & (node_occ > leaf_size) & (node_level < L)
    leaf_mask = valid_mask & (~internal)  # active and not internal

    # ---- parent / children via cell arithmetic + active-set lookup ----
    parent = lookup_node(node_level - 1, node_cell >> 3)
    parent = jnp.where(node_level == 0, jnp.asarray(-1, idt), parent)
    parent = jnp.where(valid_mask, parent, jnp.asarray(-1, idt))

    child_cells = node_cell[:, None] * 8 + jnp.arange(8, dtype=idt)[None, :]  # (M,8)
    child_levels = jnp.broadcast_to((node_level + 1)[:, None], child_cells.shape)
    child_lookup = lookup_node(child_levels, child_cells)  # (M,8) node id / sentinel
    children = jnp.where(child_lookup == sentinel, jnp.asarray(-1, idt), child_lookup)
    children = jnp.where(valid_mask[:, None], children, jnp.asarray(-1, idt))
    child_counts = (children >= 0).sum(1).astype(idt)

    # ---- centres ----
    coord_int = _cell_coords(node_cell)  # (M,3) int cell coords at each node's level
    lvl_f = jnp.minimum(node_level, L).astype(jnp.float64)
    csz = span / jnp.power(2.0, lvl_f)
    centers = lo + (coord_int.astype(jnp.float64) + 0.5) * csz[:, None]

    # ---- leaves: scattered across levels, compacted to leaf_capacity ----
    num_leaves = jnp.sum(leaf_mask.astype(idt))
    node_ids = jnp.arange(node_capacity, dtype=idt)
    leaf_pos = jnp.where(leaf_mask, node_ids, sentinel)  # leaf node id else sentinel
    leaf_pos_sorted = jnp.sort(leaf_pos)  # real leaf ids first, sentinel padding last
    leaf_indices = leaf_pos_sorted[:leaf_capacity]  # sentinel-padded (node_capacity)
    leaf_nodes = jnp.where(
        leaf_pos_sorted < node_capacity, leaf_pos_sorted, jnp.asarray(-1, idt)
    )  # (node_capacity,), -1 padded
    nodes_by_level = jnp.arange(node_capacity, dtype=idt)  # already level-major
    node_depths = jnp.where(valid_mask, node_level, jnp.asarray(L, idt))

    # ---- Stage 2 placeholders: interaction lists NOT built yet (see docstring) ----
    v_src = jnp.full((node_capacity,), sentinel, idt)
    v_tgt = jnp.full((node_capacity,), sentinel, idt)
    u_offsets = jnp.zeros((leaf_capacity + 1,), idt)
    u_neighbors = jnp.zeros((0,), idt)

    return UniformOctreeExecutionView(
        valid_mask=valid_mask,
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_depths=node_depths,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=L + 1,
        leaf_mask=leaf_mask,
        leaf_nodes=leaf_nodes,
        num_valid_nodes=num_valid,
        num_leaf_nodes=num_leaves,
        centers=centers,
        v_src=v_src,
        v_tgt=v_tgt,
        u_offsets=u_offsets,
        u_neighbors=u_neighbors,
        leaf_indices=leaf_indices,
        perm=perm.astype(idt),
    )


__all__ = [
    "OctreeUVLists",
    "build_uniform_octree_uv",
    "UniformOctreeExecutionView",
    "build_uniform_octree_execution_view",
    "build_uniform_octree_execution_view_device",
    "build_sparse_uniform_octree_execution_view_device",
    "build_adaptive_octree_execution_view_device",
]
