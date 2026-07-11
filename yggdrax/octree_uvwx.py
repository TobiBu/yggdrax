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

from typing import NamedTuple

import numpy as np


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


__all__ = [
    "OctreeUVLists",
    "build_uniform_octree_uv",
    "UniformOctreeExecutionView",
    "build_uniform_octree_execution_view",
]
