"""Exact k-nearest neighbours on the GPU: cell-aligned buckets + Pallas top-k.

The kNN of :mod:`yggdrax.kdtree` (``query_neighbors(backend="tree")``) walks one
query at a time and visits one point per loop iteration: the CPU KD-tree design,
which a GPU runs as divergent serial threads (1.1 s at 1e6 on an A100, where
JZ-Tree takes 17 ms). This module is the GPU design.

**Tree.** Morton count buckets of at most ``leaf_size`` points that never cross
the parent cell of their adaptive cell leaf
(:func:`~yggdrax._cell_partition.adaptive_cell_leaf_partition`), over a balanced
static structure. Plain count buckets straddle Morton jumps (at 1e6 one spanned
the box); cell leaves alone are 5-8x under-filled. A leaf is a dense ``W``-lane
block (``W`` the next power of two).

**Method "traverse" (default): packet traversal.** The data get their own
tree -- the same buckets under a Morton-prefix (Karras) radix structure, whose
internal nodes are compact (2.5x fewer node tests than a count-balanced
structure) -- and separate queries their own cell-aligned blocks of ``W``, so
every source block is all data and every target block all queries (in one
shared tree a leaf was ~40 % of each). One Pallas program per query block
(:func:`~yggdrax.pallas.knn_topk.leaf_topk_preorder_pallas`): its lanes merge
a centre-out window of Morton-neighbour data leaves (with ``k == W`` the centre
leaf fills the empty top-k directly), then walk the tree together in
pre-order -- entering a node is ``p + 1``, skipping it ``p + size`` -- loading
node boxes a line at a time; a node is entered when any lane's k-th ball
reaches its box. Reached leaves merge into each lane's running top-k (query
lanes on the thread axis, so each lane's list is in its own registers).
Exact by construction; the only static sizes are bucket counts.

**Method "walk": mutual walk + certificate.** The Pallas mutual walk (Dehnen
MAC at ``theta``) gives every leaf a near list; the top-k runs over it
(:func:`~yggdrax.pallas.knn_topk.leaf_topk_pallas`). A far pair ``(P, Q)``
puts every point of ``Q`` at least ``|c_P - c_Q| - r_P - r_Q`` from every point
of ``P``; a query whose k-th distance is within the smallest such gap over its
leaf's ancestors is exact (``certified``). On clustered data, queries in voids
may stay uncertified at any affordable ``theta``; it is kept for comparison.

Data and queries may be one set (``queries=None``: self excluded) or two; two
sets share one tree, so only leaves holding queries are targets and only leaves
holding data are sources. Open boundaries only so far.
"""

from __future__ import annotations

import functools
import math
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._cell_partition import MORTON_LEVELS, adaptive_cell_leaf_partition
from ._tree_impl import (
    RadixTree,
    _build_balanced_bucket_structure,
    _build_tree_from_leaf_partitions,
    _morton_sort,
)
from .dtypes import INDEX_DTYPE, as_index
from .geometry import compute_tree_geometry
from .interactions import dual_tree_walk_mutual, mutual_walk_inputs
from .morton import morton_encode
from .pallas._compat import pallas_triton_supported
from .pallas.knn_topk import (
    leaf_topk_pallas,
    leaf_topk_preorder_pallas,
    leaf_topk_xla,
    next_pow2,
)

__all__ = ["FastKNN", "FastKNNPlan", "fast_knn", "plan_fast_knn"]

#: Relative and absolute pad on the bounding radii the certificate uses: the
#: geometry's radii are the max point distance up to float32 rounding (1e-6).
_RADIUS_REL_PAD = 1e-5
_RADIUS_ABS_PAD = 1e-30


class FastKNNPlan(NamedTuple):
    """Static sizes of one :func:`fast_knn` problem (hashable: a ``jit`` static).

    Attributes
    ----------
    num_data
        Point counts; ``num_queries == 0`` means the queries are the data.
    num_queries
        As ``num_data``.
    k
        Neighbours per query.
    leaf_size
        Bucket size of the tree.
    theta
        Walk MAC opening angle.
    max_pair_queue
        Walk capacities.
    far_cap
        As ``max_pair_queue``.
    near_cap
        As ``max_pair_queue``.
    max_sources
        Bound on any target leaf's source count (the XLA top-k needs it).
    bucket_capacity
        Static leaf (bucket) count of the tree; unused buckets are empty.
    method
        ``"traverse"`` (packet traversal, exact by construction) or ``"walk"``
        (mutual walk + CSR + certificate); see :func:`fast_knn`.
    query_capacity
        Traversal with separate queries: static count of query blocks (the
        data tree then holds the data alone, ``bucket_capacity`` buckets).
    """

    num_data: int
    num_queries: int
    k: int
    leaf_size: int
    theta: float
    max_pair_queue: int
    far_cap: int
    near_cap: int
    max_sources: int
    bucket_capacity: int
    method: str = "traverse"
    query_capacity: int = 0


class FastKNN(NamedTuple):
    """Result of :func:`fast_knn`.

    Attributes
    ----------
    indices
        ``(Q, k)`` int32 rows of ``data``, nearest first (``-1``: fewer than
        ``k`` data points reachable, only possible with fewer than ``k`` data).
    distances
        ``(Q, k)`` Euclidean distances (``inf`` where ``indices`` is ``-1``).
    certified
        ``(Q,)`` bool: the query's k-th distance is within its leaf's exclusion
        distance, so its neighbours are exact (see the module docstring).
    overflow
        Scalar bool: a walk or source capacity overflowed; nothing is trusted.
    stats
        Diagnostics: pair counts, the largest source count, the walk's peak.
    """

    indices: Any
    distances: Any
    certified: Any
    overflow: Any
    stats: dict


def _points(data: Any, queries: Any) -> tuple[Any, np.ndarray, np.ndarray, int]:
    nd = int(data.shape[0])
    if queries is None:
        flags = np.ones((nd,), bool)
        return data, flags, flags, nd
    pts = jnp.concatenate([data, jnp.asarray(queries, data.dtype)])
    nq = int(queries.shape[0])
    is_data = np.concatenate([np.ones(nd, bool), np.zeros(nq, bool)])
    return pts, is_data, ~is_data, nq


def _bounds(pts: Any) -> tuple[Any, Any]:
    lo = jnp.min(pts, axis=0)
    hi = jnp.max(pts, axis=0)
    pad = 1e-4 * jnp.maximum(jnp.max(hi - lo), 1e-30)
    return lo - pad, hi + pad


class _Built(NamedTuple):
    topo: Any
    ps: Any
    inv: Any
    geom: Any
    num_internal: int
    num_leaves: int
    width: int
    num_buckets: Any
    overflow: Any


@functools.lru_cache(maxsize=32)
def _bucket_structure(capacity: int) -> tuple:
    """Balanced static structure over ``capacity`` buckets (host, cached).

    The node ranges come back in BUCKET units (first, last bucket of each
    node); the device maps them to particle ranges.
    """
    starts = np.arange(capacity, dtype=np.int64)
    return _build_balanced_bucket_structure(starts, starts + 1)


def _aligned_buckets(scodes: Any, width: int, leaf_size: int) -> tuple[Any, Any]:
    """Bucket start flags and count for Morton-sorted codes (see module doc)."""
    n = int(scodes.shape[0])
    cells = adaptive_cell_leaf_partition(scodes, leaf_size=int(leaf_size), capacity=n)
    live = jnp.arange(n) < cells.num_leaves
    mark = (
        jnp.zeros((n + 1,), jnp.int32)
        .at[jnp.where(live, cells.leaf_starts, n)]
        .set(1, mode="drop")[:n]
    )
    lid = jnp.cumsum(mark) - 1
    depth = cells.leaf_depths[jnp.clip(lid, 0, n - 1)]
    glev = jnp.maximum(depth - 1, 0).astype(jnp.uint64)
    shift = jnp.uint64(3) * (jnp.uint64(MORTON_LEVELS) - glev)
    key = jnp.right_shift(scodes.astype(jnp.uint64), shift)
    i = jnp.arange(n)
    brk = jnp.concatenate(
        [
            jnp.ones((1,), bool),
            (key[1:] != key[:-1]) | (glev[1:] != glev[:-1]),
        ]
    )
    run0 = lax.cummax(jnp.where(brk, i, 0))
    flag = ((i - run0) % int(width)) == 0
    return flag, jnp.sum(flag.astype(jnp.int32))


def _build(pts: Any, leaf_size: int, capacity: int, geometry: bool = True) -> _Built:
    """Cell-aligned bucket tree over ``pts`` with ``capacity`` buckets."""
    n = int(pts.shape[0])
    width = next_pow2(int(leaf_size))
    codes = morton_encode(pts, _bounds(pts))
    perm, scodes = _morton_sort(codes)
    flag, nb = _aligned_buckets(scodes, width, leaf_size)
    starts = jnp.nonzero(flag, size=capacity, fill_value=n)[0].astype(INDEX_DTYPE)
    ends = jnp.concatenate([starts[1:], jnp.full((1,), n, INDEX_DTYPE)])
    (
        parent_np,
        left_np,
        right_np,
        lil_np,
        ril_np,
        nr_np,
        nlev_np,
        loff_np,
        nbl_np,
        num_levels,
    ) = _bucket_structure(int(capacity))
    node_ranges = jnp.stack(
        [starts[jnp.asarray(nr_np[:, 0])], ends[jnp.asarray(nr_np[:, 1])] - 1], axis=1
    ).astype(INDEX_DTYPE)
    lo, hi = _bounds(pts)
    topo = RadixTree(
        parent=jnp.asarray(parent_np, INDEX_DTYPE),
        left_child=jnp.asarray(left_np, INDEX_DTYPE),
        right_child=jnp.asarray(right_np, INDEX_DTYPE),
        left_is_leaf=jnp.asarray(lil_np, jnp.bool_),
        right_is_leaf=jnp.asarray(ril_np, jnp.bool_),
        particle_indices=perm.astype(INDEX_DTYPE),
        morton_codes=scodes,
        node_ranges=node_ranges,
        num_particles=n,
        num_internal_nodes=int(capacity) - 1,
        node_level=jnp.asarray(nlev_np, INDEX_DTYPE),
        level_offsets=jnp.asarray(loff_np, INDEX_DTYPE),
        nodes_by_level=jnp.asarray(nbl_np, INDEX_DTYPE),
        num_levels=as_index(num_levels),
        bounds_min=lo,
        bounds_max=hi,
        leaf_codes=scodes[jnp.clip(starts, 0, n - 1)],
        leaf_depths=jnp.full((int(capacity),), -1, INDEX_DTYPE),
        use_morton_geometry=jnp.asarray(False, dtype=jnp.bool_),
        leaf_size=width,
    )
    ps = pts[perm]
    inv = jnp.zeros((n,), INDEX_DTYPE).at[perm].set(jnp.arange(n, dtype=INDEX_DTYPE))
    geom = compute_tree_geometry(topo, ps, max_leaf_size=width) if geometry else None
    ni = int(capacity) - 1
    return _Built(topo, ps, inv, geom, ni, int(capacity), width, nb, nb > capacity)


def _build_radix(
    pts: Any, leaf_size: int, capacity: int, bounds: Optional[tuple] = None
) -> _Built:
    """Cell-aligned buckets under a Morton-prefix (Karras) radix structure.

    Same leaves as :func:`_build`, but the internal nodes split at Morton
    prefixes instead of balancing bucket counts, so every internal node is
    compact (its points share a Morton cell). The packet traversal tests 2.5x
    fewer nodes on it (1e6 uniform: 502 -> 193 at k = 8). Padding buckets get
    the largest code, so they sit in one subtree at the right end.
    """
    n = int(pts.shape[0])
    width = next_pow2(int(leaf_size))
    if bounds is None:
        bounds = _bounds(pts)
    codes = morton_encode(pts, bounds)
    perm, scodes = _morton_sort(codes)
    flag, nb = _aligned_buckets(scodes, width, leaf_size)
    starts = jnp.nonzero(flag, size=capacity, fill_value=n)[0].astype(INDEX_DTYPE)
    ends = jnp.concatenate([starts[1:], jnp.full((1,), n, INDEX_DTYPE)])
    live = starts < n
    sentinel = jnp.asarray(np.uint64(2**63 - 1), dtype=jnp.uint64)
    leaf_codes = jnp.where(live, scodes[jnp.minimum(starts, n - 1)], sentinel)
    topo = _build_tree_from_leaf_partitions(
        pts,
        jnp.ones((n,), pts.dtype),
        perm.astype(INDEX_DTYPE),
        scodes,
        starts,
        ends,
        bounds,
        leaf_size=width,
        return_reordered=False,
        workspace=None,
        return_workspace=False,
        leaf_codes_override=leaf_codes,
    )
    ps = pts[perm]
    inv = jnp.zeros((n,), INDEX_DTYPE).at[perm].set(jnp.arange(n, dtype=INDEX_DTYPE))
    ni = int(capacity) - 1
    return _Built(topo, ps, inv, None, ni, int(capacity), width, nb, nb > capacity)


class _QueryBlocks(NamedTuple):
    x: Any
    y: Any
    z: Any
    valid: Any
    qid: Any
    first_code: Any
    count: Any


def _query_blocks(
    qpts: Any, bounds: tuple, leaf_size: int, capacity: int
) -> _QueryBlocks:
    """Queries in cell-aligned Morton buckets of at most ``W``: the targets.

    Same partition rule as the data leaves (:func:`_aligned_buckets`), in the
    data's Morton frame, so a block is compact and its first code locates it
    among the data leaves (the seed window). ``capacity`` blocks, padded.
    """
    nq = int(qpts.shape[0])
    width = next_pow2(int(leaf_size))
    codes = morton_encode(qpts, bounds)
    perm, scodes = _morton_sort(codes)
    flag, nb = _aligned_buckets(scodes, width, leaf_size)
    starts = jnp.nonzero(flag, size=capacity, fill_value=nq)[0].astype(INDEX_DTYPE)
    ends = jnp.concatenate([starts[1:], jnp.full((1,), nq, INDEX_DTYPE)])
    lane = jnp.arange(width)
    valid = lane[None, :] < (ends - starts)[:, None]
    slot = jnp.where(valid, starts[:, None] + lane[None, :], 0)
    qs = qpts[perm]
    p = qs[slot]
    qid = jnp.where(valid, perm[slot], -1).astype(jnp.int32)
    sentinel = jnp.asarray(np.uint64(2**63 - 1), dtype=jnp.uint64)
    first = jnp.where(starts < nq, scodes[jnp.minimum(starts, nq - 1)], sentinel)
    return _QueryBlocks(p[..., 0], p[..., 1], p[..., 2], valid, qid, first, nb)


def _run_traverse_two(data, queries, plan, interpret, num_warps, seed):
    """Separate queries: a data-only tree and query-only target blocks.

    In one shared tree a leaf held ~13 data and ~13 query lanes of 32, so every
    tile ran 40 % full on both sides; here every source block is all data and
    every target block all queries.
    """
    dq = jax.lax.stop_gradient(data)
    qq = jax.lax.stop_gradient(jnp.asarray(queries, data.dtype))
    nd, nq = int(dq.shape[0]), int(qq.shape[0])
    bounds = _bounds(jnp.concatenate([dq, qq]))
    b = _build_radix(dq, plan.leaf_size, plan.bucket_capacity, bounds=bounds)
    t = _tables(b, np.ones((nd,), bool), np.zeros((nd,), bool))
    pre, leaves = _radix_layout(b, t)
    qb = _query_blocks(qq, bounds, plan.leaf_size, plan.query_capacity)
    seedc = jnp.clip(
        jnp.searchsorted(b.topo.leaf_codes, qb.first_code, side="right") - 1,
        0,
        b.num_leaves - 1,
    ).astype(jnp.int32)
    d2q, sq = leaf_topk_preorder_pallas(
        t.x,
        t.y,
        t.z,
        t.dmask,
        t.qmask,
        t.slot,
        leaves,
        pre,
        qb.qid,
        targets=(qb.x, qb.y, qb.z, qb.valid, jnp.full_like(qb.qid, -1), seedc),
        num_queries=nq,
        k=int(plan.k),
        exclude_self=False,
        seed=int(seed),
        interpret=bool(interpret),
        num_warps=num_warps,
    )
    rows = jnp.where(sq >= 0, b.topo.particle_indices[jnp.clip(sq, 0)], -1)
    overflow = b.overflow | (qb.count > plan.query_capacity)
    return d2q, rows.astype(jnp.int32), overflow, {"num_leaves": b.num_buckets}


def _radix_layout(b: _Built, t: "_Tables") -> tuple[tuple, tuple]:
    """Pre-order node arrays and leaf spheres of a radix-built tree (on device).

    Leaf spheres: AABB centre and the exact largest point distance. Internal
    nodes: AABB of the union (relaxed bottom-up to a fixpoint) with its
    half-diagonal as radius. A node's leaf range ``[nlo, nhi]`` comes from the
    same relaxation, its subtree size is ``2 (nhi - nlo + 1) - 1`` and its
    pre-order position is ``2 nlo + (left turns on its root path)``, the latter
    summed by pointer jumping. Returns ``(pre, leaves)`` as
    :func:`~yggdrax.pallas.knn_topk.leaf_topk_preorder_pallas` takes them.
    """
    topo = b.topo
    ni, nl = b.num_internal, b.num_leaves
    nodes = ni + nl
    left = topo.left_child.astype(jnp.int32)
    right = topo.right_child.astype(jnp.int32)
    parent = topo.parent.astype(jnp.int32)
    valid = t.slot >= 0
    inf = jnp.asarray(jnp.inf, t.x.dtype)
    coords = (t.x, t.y, t.z)
    llo = jnp.stack([jnp.min(jnp.where(valid, c, inf), axis=1) for c in coords], -1)
    lhi = jnp.stack([jnp.max(jnp.where(valid, c, -inf), axis=1) for c in coords], -1)
    has = jnp.any(valid, axis=1)
    lcen = jnp.where(has[:, None], 0.5 * (llo + lhi), 0.0)
    d2 = sum((c - lcen[:, i : i + 1]) ** 2 for i, c in enumerate(coords))
    lrad = jnp.where(has, jnp.sqrt(jnp.max(jnp.where(valid, d2, 0.0), axis=1)), -inf)

    leaf_ids = jnp.arange(nl, dtype=jnp.int32)
    lo0 = jnp.concatenate([jnp.full((ni, 3), inf, llo.dtype), llo])
    hi0 = jnp.concatenate([jnp.full((ni, 3), -inf, lhi.dtype), lhi])
    a0 = jnp.concatenate([jnp.full((ni,), nl, jnp.int32), leaf_ids])
    z0 = jnp.concatenate([jnp.full((ni,), -1, jnp.int32), leaf_ids])

    def relax(st):
        lo, hi, a, z, _ = st
        nlo_ = jnp.minimum(lo[left], lo[right])
        nhi_ = jnp.maximum(hi[left], hi[right])
        na = jnp.minimum(a[left], a[right])
        nz = jnp.maximum(z[left], z[right])
        changed = jnp.any(na != a[:ni]) | jnp.any(nz != z[:ni])
        return (
            lo.at[:ni].set(nlo_),
            hi.at[:ni].set(nhi_),
            a.at[:ni].set(na),
            z.at[:ni].set(nz),
            changed,
        )

    lo, hi, nlo, nhi, _ = lax.while_loop(
        lambda st: st[4], relax, (lo0, hi0, a0, z0, jnp.asarray(True))
    )
    alive = jnp.all(lo <= hi, axis=1)
    cen = jnp.where(alive[:, None], 0.5 * (lo + hi), 0.0)
    half = 0.5 * jnp.sqrt(jnp.sum((hi - lo) ** 2, axis=1))
    rad = jnp.where(alive, half, -inf).at[ni:].set(lrad)
    cen = cen.at[ni:].set(lcen)
    rad = jnp.where(rad > -inf, rad * (1.0 + _RADIUS_REL_PAD) + _RADIUS_ABS_PAD, rad)

    # left turns on the root path, by pointer jumping
    node_ids = jnp.arange(nodes, dtype=jnp.int32)
    is_root = parent < 0
    par = jnp.where(is_root, node_ids, parent)
    acc = jnp.where(is_root, 0, (left[jnp.clip(par, 0, ni - 1)] == node_ids)).astype(
        jnp.int32
    )
    anc = par
    for _ in range(int(nodes).bit_length() + 1):
        acc = acc + jnp.where(anc == node_ids, 0, acc[anc])
        anc = anc[anc]
    pos = 2 * nlo + acc
    size = 2 * (nhi - nlo + 1) - 1
    order = jnp.zeros((nodes,), jnp.int32).at[pos].set(node_ids)
    leaf_row = jnp.where(order >= ni, order - ni, -1)
    lo = lo.at[ni:].set(llo)
    hi = hi.at[ni:].set(lhi)
    pre = (
        lo[order, 0],
        lo[order, 1],
        lo[order, 2],
        hi[order, 0],
        hi[order, 1],
        hi[order, 2],
        leaf_row,
        size[order],
        nlo[order],
        nhi[order],
    )
    leaves = (llo[:, 0], llo[:, 1], llo[:, 2], lhi[:, 0], lhi[:, 1], lhi[:, 2])
    return pre, leaves


def _walk(
    b: _Built, theta: float, q: int, far_cap: int, near_cap: int, backend, interpret
):
    w = mutual_walk_inputs(b.topo, b.geom, mac_type="dehnen", mask_empty_nodes=True)
    return dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        float(theta),
        w.root,
        max_pair_queue=int(q),
        far_cap=int(far_cap),
        near_cap=int(near_cap),
        mac_type="dehnen",
        node_active=w.node_active,
        backend=backend,
        pallas_interpret=bool(interpret),
    )


class _Tables(NamedTuple):
    x: Any
    y: Any
    z: Any
    dmask: Any
    qmask: Any
    slot: Any
    flat_of_slot: Any
    cx: Any
    cy: Any
    cz: Any
    cr: Any


def _tables(b: _Built, is_data: np.ndarray, is_query: np.ndarray) -> _Tables:
    ni, nl, width = b.num_internal, b.num_leaves, b.width
    npts = int(b.ps.shape[0])
    rng = b.topo.node_ranges[ni:]
    start = rng[:, 0]
    cnt = jnp.clip(rng[:, 1] - rng[:, 0] + 1, 0, width)
    lane = jnp.arange(width)
    valid = (lane[None, :] < cnt[:, None]) & (start[:, None] < npts)
    slot = jnp.where(valid, start[:, None] + lane[None, :], 0).astype(jnp.int32)
    p = b.ps[slot]
    orig = b.topo.particle_indices[slot]
    dmask = valid & jnp.asarray(is_data)[orig]
    qmask = valid & jnp.asarray(is_query)[orig]
    flat = jnp.arange(nl * width, dtype=jnp.int32).reshape(nl, width)
    flat_of_slot = (
        jnp.zeros((npts,), jnp.int32)
        .at[jnp.where(valid, slot, npts)]
        .set(flat, mode="drop")
    )
    if (
        b.geom is None
    ):  # radix build (traversal): its node boxes come from _radix_layout
        c = jnp.zeros((nl, 3), b.ps.dtype)
        r = jnp.zeros((nl,), b.ps.dtype)
    else:
        c = b.geom.center[ni:]
        r = b.geom.radius[ni:] * (1.0 + _RADIUS_REL_PAD) + _RADIUS_ABS_PAD
    return _Tables(
        p[..., 0],
        p[..., 1],
        p[..., 2],
        dmask,
        qmask,
        jnp.where(valid, slot, -1).astype(jnp.int32),
        flat_of_slot,
        c[:, 0],
        c[:, 1],
        c[:, 2],
        r,
    )


def _csr(b: _Built, t: _Tables, walk: Any) -> tuple[Any, Any, Any]:
    ni, nl = b.num_internal, b.num_leaves
    cap = int(walk.near_a.shape[0])
    live = (jnp.arange(cap) < walk.near_count) & (walk.near_a >= 0) & (walk.near_b >= 0)
    a = jnp.clip(walk.near_a - ni, 0, nl - 1)
    bb = jnp.clip(walk.near_b - ni, 0, nl - 1)
    selfl = jnp.arange(nl, dtype=a.dtype)
    tgt = jnp.concatenate([a, bb, selfl])
    src = jnp.concatenate([bb, a, selfl])
    has_q = jnp.any(t.qmask, axis=1)
    has_d = jnp.any(t.dmask, axis=1)
    ok = jnp.concatenate([live, live, jnp.ones((nl,), bool)]) & has_q[tgt] & has_d[src]
    key = jnp.sqrt(
        (t.cx[tgt] - t.cx[src]) ** 2
        + (t.cy[tgt] - t.cy[src]) ** 2
        + (t.cz[tgt] - t.cz[src]) ** 2
    )
    # one int32 key: target leaf in the high bits, the centre distance
    # quantised into the rest (only the order within a target matters)
    tbits = int(nl).bit_length()
    dbits = 31 - tbits
    if dbits >= 6:
        dmax = jnp.maximum(jnp.max(jnp.where(ok, key, 0.0)), 1e-30)
        dq = jnp.clip(
            jnp.floor(key / dmax * ((1 << dbits) - 1)), 0, (1 << dbits) - 1
        ).astype(jnp.int32)
        packed = (jnp.where(ok, tgt, nl).astype(jnp.int32) << dbits) | dq
        _, sources = lax.sort((packed, src.astype(jnp.int32)), num_keys=1)
    else:  # too many leaves to pack: two-key sort
        order = jnp.lexsort((key, jnp.where(ok, tgt, nl)))
        sources = src[order].astype(jnp.int32)
    counts = jax.ops.segment_sum(ok.astype(jnp.int32), jnp.where(ok, tgt, nl), nl + 1)[
        :nl
    ]
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(counts).astype(jnp.int32)]
    )
    return offsets, sources, jnp.max(counts)


def _exclusion(b: _Built, walk: Any) -> Any:
    """Per-leaf exclusion distance ``G`` from the far list (see module docstring)."""
    nodes = int(b.topo.node_ranges.shape[0])
    cap = int(walk.far_a.shape[0])
    live = (jnp.arange(cap) < walk.far_count) & (walk.far_a >= 0) & (walk.far_b >= 0)
    fa = jnp.clip(walk.far_a, 0, nodes - 1)
    fb = jnp.clip(walk.far_b, 0, nodes - 1)
    c = b.geom.center
    r = b.geom.radius * (1.0 + _RADIUS_REL_PAD) + _RADIUS_ABS_PAD
    g = jnp.sqrt(jnp.sum((c[fa] - c[fb]) ** 2, axis=-1)) - r[fa] - r[fb]
    g = jnp.where(live, g, jnp.inf)
    drop = jnp.full_like(fa, nodes)
    val = (
        jnp.full((nodes,), jnp.inf, g.dtype)
        .at[jnp.where(live, fa, drop)]
        .min(g, mode="drop")
        .at[jnp.where(live, fb, drop)]
        .min(g, mode="drop")
    )
    parent = b.topo.parent
    anc = jnp.where(parent < 0, jnp.arange(nodes, dtype=parent.dtype), parent)
    for _ in range(max(1, math.ceil(math.log2(nodes))) + 1):
        val = jnp.minimum(val, val[anc])
        anc = anc[anc]
    return val[b.num_internal :]


def _resolve_backend(backend: str, interpret: bool) -> str:
    if backend not in ("auto", "pallas", "xla"):
        raise ValueError(f"backend must be 'auto', 'pallas' or 'xla', got {backend!r}")
    if backend == "auto":
        return "pallas" if (interpret or pallas_triton_supported()) else "xla"
    return backend


def fast_knn(
    data: Any,
    queries: Optional[Any] = None,
    *,
    plan: FastKNNPlan,
    exclude_self: Optional[bool] = None,
    backend: str = "auto",
    walk_backend: str = "auto",
    interpret: bool = False,
    differentiable: bool = False,
    num_warps: Optional[int] = None,
    seed: int = 8,
) -> FastKNN:
    """Exact kNN of ``queries`` (default: ``data``, self excluded) among ``data``.

    Traceable under ``jit`` with ``plan`` static (make it with
    :func:`plan_fast_knn`). Read ``overflow`` and ``certified``: the neighbours
    of a query are exact when ``overflow`` is false and its ``certified`` is
    true.

    Parameters
    ----------
    data
        ``(N, 3)`` data points.
    queries
        ``(Q, 3)`` query points, or ``None`` for the data themselves.
    plan
        Static sizes from :func:`plan_fast_knn` for these shapes.
    exclude_self
        Skip each query's own point; default ``True`` iff ``queries is None``.
    backend
        Top-k kernel: ``"pallas"``, ``"xla"`` or ``"auto"`` (Pallas on Ampere+).
    walk_backend
        Walk backend of :func:`~yggdrax.interactions.dual_tree_walk_mutual`.
    interpret
        Run the Pallas kernels in interpret mode (CPU tests).
    differentiable
        Recompute ``distances`` from the live positions at the found indices
        (indices carry no gradient), so ``jax.grad`` reaches data and queries.
        ``False`` returns the kernel's distances.
    num_warps
        Triton warps per top-k program (``None``: the kernel's default).
    seed
        Traversal only: Morton-neighbour leaves merged first on each side of a
        target leaf; the walk skips subtrees inside that window.

    Returns
    -------
    FastKNN
        Indices, distances, certificate, overflow flag and diagnostics.

    Raises
    ------
    ValueError
        If the shapes do not match ``plan``, or the points are not 3-D.
    """
    data = jnp.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"data must be (N, 3), got {data.shape}")
    nq_expect = 0 if queries is None else int(jnp.shape(queries)[0])
    if int(data.shape[0]) != plan.num_data or nq_expect != plan.num_queries:
        raise ValueError(
            f"plan is for {plan.num_data} data / {plan.num_queries} queries, got "
            f"{int(data.shape[0])} / {nq_expect}"
        )
    k = int(plan.k)
    if exclude_self is None:
        exclude_self = queries is None
    pts, is_data, is_query, nq = _points(data, queries)
    kb = _resolve_backend(backend, interpret)
    if plan.method == "traverse" and kb == "pallas" and queries is not None:
        d2q, rows, overflow, stats = _run_traverse_two(
            data, queries, plan, interpret, num_warps, seed
        )
        certified = jnp.ones((int(nq),), bool)
    elif plan.method == "traverse" and kb == "xla":
        # CPU reference: exact brute force (the traversal is a GPU kernel)
        qpts = data if queries is None else jnp.asarray(queries, data.dtype)
        d2q, rows = _brute_topk(data, qpts, k, bool(exclude_self))
        certified = jnp.ones((int(nq),), bool)
        overflow = jnp.asarray(False)
        stats: dict = {}
    else:
        if plan.method == "traverse":
            b = _build_radix(
                jax.lax.stop_gradient(pts), plan.leaf_size, plan.bucket_capacity
            )
        else:
            b = _build(jax.lax.stop_gradient(pts), plan.leaf_size, plan.bucket_capacity)
        t = _tables(b, is_data, is_query)
        if plan.method == "traverse":
            d2q, sq, certified, overflow, stats = _run_traverse(
                b,
                t,
                plan,
                queries is not None,
                int(nq),
                bool(exclude_self),
                interpret,
                num_warps,
                seed,
            )
        else:
            d2q, sq, certified, overflow, stats = _run_walk(
                b,
                t,
                plan,
                queries is not None,
                kb,
                bool(exclude_self),
                walk_backend,
                interpret,
                num_warps,
            )
        stats["num_leaves"] = b.num_buckets
        rows = jnp.where(sq >= 0, b.topo.particle_indices[jnp.clip(sq, 0)], -1).astype(
            jnp.int32
        )
    found = rows >= 0
    if differentiable:
        qpts = data if queries is None else jnp.asarray(queries, data.dtype)
        nb = data[jnp.clip(rows, 0)]
        dist = jnp.where(
            found,
            jnp.sqrt(jnp.sum((qpts[:, None, :] - nb) ** 2, axis=-1)),
            jnp.inf,
        )
    else:
        dist = jnp.where(found, jnp.sqrt(jnp.maximum(d2q, 0.0)), jnp.inf)
    return FastKNN(rows, dist, certified, overflow, stats)


def _run_traverse(b, t, plan, separate, nq, exclude_self, interpret, num_warps, seed):
    """Packet traversal: query-ordered ``(d2, slots, certified, overflow, stats)``."""
    nd = plan.num_data
    pre, leaves = _radix_layout(b, t)
    # output row of each query lane: its index among the queries
    orig = b.topo.particle_indices[jnp.clip(t.slot, 0)]
    qid = jnp.where(t.qmask, orig - (nd if separate else 0), -1).astype(jnp.int32)
    d2q, sq = leaf_topk_preorder_pallas(
        t.x,
        t.y,
        t.z,
        t.dmask,
        t.qmask,
        t.slot,
        leaves,
        pre,
        qid,
        num_queries=nq,
        k=int(plan.k),
        exclude_self=exclude_self,
        seed=int(seed),
        interpret=bool(interpret),
        num_warps=num_warps,
    )
    return d2q, sq, jnp.ones((nq,), bool), b.overflow, {}


def _run_walk(
    b, t, plan, separate, kb, exclude_self, walk_backend, interpret, num_warps
):
    """Walk + CSR + certificate: query-ordered ``(d2, slots, certified, ...)``."""
    k = int(plan.k)
    walk = _walk(
        b,
        plan.theta,
        plan.max_pair_queue,
        plan.far_cap,
        plan.near_cap,
        walk_backend,
        interpret,
    )
    offsets, sources, max_src = _csr(b, t, walk)
    args = (
        t.x,
        t.y,
        t.z,
        t.dmask,
        t.qmask,
        t.slot,
        t.cx,
        t.cy,
        t.cz,
        t.cr,
        offsets,
        sources,
    )
    if kb == "pallas":
        d2, idx = leaf_topk_pallas(
            *args,
            k=k,
            exclude_self=exclude_self,
            interpret=bool(interpret),
            num_warps=num_warps,
        )
        src_overflow = jnp.asarray(False)
    else:
        d2, idx = leaf_topk_xla(
            *args, k=k, exclude_self=exclude_self, max_sources=plan.max_sources
        )
        src_overflow = max_src > plan.max_sources
    nd = plan.num_data
    qslot = b.inv[nd:] if separate else b.inv
    flat = t.flat_of_slot[qslot]
    d2q = d2.reshape(-1, k)[flat]
    sq = idx.reshape(-1, k)[flat]
    excl = _exclusion(b, walk)
    kth = jnp.where(sq[:, -1] >= 0, jnp.sqrt(jnp.maximum(d2q[:, -1], 0.0)), jnp.inf)
    certified = kth <= excl[flat // b.width]
    overflow = (
        b.overflow
        | jnp.asarray(walk.far_overflow)
        | jnp.asarray(walk.near_overflow)
        | jnp.asarray(walk.queue_overflow)
        | src_overflow
    )
    stats = {
        "near_count": walk.near_count,
        "far_count": walk.far_count,
        "peak_wavefront": walk.peak_wavefront,
        "max_sources": max_src,
    }
    return d2q, sq, certified, overflow, stats


def _brute_topk(data: Any, qpts: Any, k: int, exclude_self: bool, chunk: int = 256):
    """Exact top-``k`` by dense distances, ``chunk`` queries at a time (CPU)."""
    nq = int(qpts.shape[0])
    nchunk = -(-nq // chunk)
    pad = nchunk * chunk - nq
    qp = jnp.concatenate([qpts, jnp.zeros((pad, 3), qpts.dtype)]).reshape(
        nchunk, chunk, 3
    )
    ids = jnp.arange(nchunk * chunk).reshape(nchunk, chunk)

    def one(args):
        qq, ii = args
        d2 = jnp.sum((qq[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        if exclude_self:
            d2 = jnp.where(
                ii[:, None] == jnp.arange(data.shape[0])[None, :], jnp.inf, d2
            )
        neg, j = lax.top_k(-d2, k)
        return -neg, j

    d2, j = lax.map(one, (qp, ids))
    d2 = d2.reshape(-1, k)[:nq]
    j = j.reshape(-1, k)[:nq]
    return d2, jnp.where(jnp.isfinite(d2), j, -1).astype(jnp.int32)


def plan_fast_knn(
    data: Any,
    queries: Optional[Any] = None,
    *,
    k: int,
    leaf_size: int = 32,
    theta: Optional[float] = None,
    walk_backend: str = "auto",
    interpret: bool = False,
    headroom: float = 1.25,
    max_theta_steps: int = 8,
    method: str = "traverse",
) -> FastKNNPlan:
    """Size the static capacities of :func:`fast_knn` from the points (eager).

    Runs the build and walk with growing capacities until nothing overflows,
    then sets each capacity to ``headroom`` times what was used (powers of two).
    Positions that move afterwards can need more: read ``FastKNN.overflow``.

    Parameters
    ----------
    data
        As :func:`fast_knn`.
    queries
        As ``data``.
    k
        Neighbours per query.
    leaf_size
        Bucket size (``<= 64`` keeps the kernel's register blocks small).
    theta
        Walk MAC opening angle; see the module docstring for the trade-off.
        ``None`` (default) starts from ``1 / (1 + 1.2 (k / W)^(1/3))`` and
        shrinks it by 10 % until a run on these points certifies every query
        (at most ``max_theta_steps`` tries); a given value is used as is.
    walk_backend
        As :func:`fast_knn`.
    interpret
        As ``walk_backend``.
    headroom
        Safety factor on the measured sizes.
    max_theta_steps
        Tries of the automatic ``theta``.
    method
        ``"traverse"`` (default) or ``"walk"``; see :func:`fast_knn`. The
        traversal needs only the bucket capacity, so its plan is one count.

    Returns
    -------
    FastKNNPlan
        The static sizes.

    Raises
    ------
    ValueError
        If ``k`` is not below the number of usable data points. (With
        ``method="walk"``, a ``RuntimeError`` also propagates when the walk
        capacities do not converge.)
    """
    data = jnp.asarray(data)
    nd = int(data.shape[0])
    nq = 0 if queries is None else int(jnp.shape(queries)[0])
    if k < 1 or k > nd - (1 if queries is None else 0):
        raise ValueError(f"k={k} needs at least {k + (queries is None)} data points")
    pts, is_data, is_query, _ = _points(data, queries)

    width = next_pow2(int(leaf_size))

    @jax.jit
    def count_buckets(p):
        codes = morton_encode(p, _bounds(p))
        _, scodes = _morton_sort(codes)
        return _aligned_buckets(scodes, width, leaf_size)[1]

    capacity = int(math.ceil(1.05 * int(count_buckets(pts)))) + 8
    if method == "traverse":
        query_capacity = 0
        if queries is not None:
            # two structures in one Morton frame: data buckets, query blocks
            bounds = _bounds(pts)

            @jax.jit
            def count_in_frame(p):
                _, sc = _morton_sort(morton_encode(p, bounds))
                return _aligned_buckets(sc, width, leaf_size)[1]

            capacity = int(math.ceil(1.05 * int(count_in_frame(data)))) + 8
            query_capacity = (
                int(math.ceil(1.05 * int(count_in_frame(jnp.asarray(queries))))) + 8
            )
        return FastKNNPlan(
            num_data=nd,
            num_queries=nq,
            k=int(k),
            leaf_size=int(leaf_size),
            theta=0.0,
            max_pair_queue=0,
            far_cap=0,
            near_cap=0,
            max_sources=0,
            bucket_capacity=capacity,
            method="traverse",
            query_capacity=query_capacity,
        )
    if method != "walk":
        raise ValueError(f"method must be 'traverse' or 'walk', got {method!r}")
    b = _build(pts, leaf_size, capacity)
    t = _tables(b, is_data, is_query)

    def size(th: float) -> FastKNNPlan:
        q, far_cap, near_cap = 1 << 14, 1 << 16, 1 << 16
        for _ in range(20):
            w = _walk(b, th, q, far_cap, near_cap, walk_backend, interpret)
            qo, fo, no = (
                bool(w.queue_overflow),
                bool(w.far_overflow),
                bool(w.near_overflow),
            )
            if not (qo or fo or no):
                break
            q, far_cap, near_cap = (
                q * (4 if qo else 1),
                far_cap * (4 if fo else 1),
                near_cap * (4 if no else 1),
            )
        else:
            raise RuntimeError("fast_knn walk capacities did not converge")
        _, _, max_src = _csr(b, t, w)
        return FastKNNPlan(
            num_data=nd,
            num_queries=nq,
            k=int(k),
            leaf_size=int(leaf_size),
            theta=float(th),
            max_pair_queue=next_pow2(int(headroom * int(w.peak_wavefront)) + 1),
            far_cap=next_pow2(int(headroom * int(w.far_count)) + 1),
            near_cap=next_pow2(int(headroom * int(w.near_count)) + 1),
            max_sources=int(math.ceil(headroom * int(max_src))) + 1,
            bucket_capacity=capacity,
            method="walk",
        )

    if theta is not None:
        return size(float(theta))
    th = 1.0 / (1.0 + 1.2 * (float(k) / width) ** (1.0 / 3.0))
    plan = size(th)
    for _ in range(int(max_theta_steps)):
        res = jax.jit(
            lambda a, qq: fast_knn(
                a, qq, plan=plan, walk_backend=walk_backend, interpret=interpret
            )
        )(data, queries)
        if bool(jnp.all(res.certified)) and not bool(res.overflow):
            return plan
        th *= 0.9
        plan = size(th)
    return plan
