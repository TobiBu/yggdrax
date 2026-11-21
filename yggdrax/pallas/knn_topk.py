"""Leaf-block exact top-k: the kernel of :func:`yggdrax.fast_knn.fast_knn`.

One Pallas program per *target* leaf. The leaf's ``W`` lanes are queries (lanes
that are not are masked); its source leaves arrive through a CSR offset table,
nearest first. For each source leaf the program

1. skips it unless some query lane can reach its bounding sphere within that
   lane's current k-th distance (a lower bound ``|q - c_s| - r_s``), so after
   the first few leaves most sources cost one sphere test;
2. forms the ``(W, W)`` squared-distance block in registers;
3. merges the block into each lane's running best ``k``: while any lane's
   nearest remaining candidate beats its k-th distance, every lane takes its
   nearest candidate and, if it beats the k-th, replaces its current worst.

The running best is kept unsorted (replace-the-worst); one final pass sorts it.
Self-pairs are excluded by sorted slot when ``exclude_self``.

:func:`leaf_topk_xla` is the pure-XLA twin with the same inputs and outputs: the
CPU reference the parity tests compare against, and the fallback off Ampere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import lax

from ._compat import KernelRef, pallas_backend_kwargs

if TYPE_CHECKING:
    # Under a type checker Pallas is present, so the kernel bodies type-check.
    # At runtime it is optional (the XLA twin and brute force still run).
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
else:
    try:  # pragma: no cover - import is environment-dependent
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plgpu
    except Exception:  # pragma: no cover - Pallas is optional
        pl = None
        plgpu = None

__all__ = [
    "leaf_topk_pallas",
    "leaf_topk_traverse_pallas",
    "leaf_topk_xla",
    "next_pow2",
]

_INF = float("inf")


def next_pow2(x: int) -> int:
    """Smallest power of two ``>= max(x, 1)``.

    Parameters
    ----------
    x
        A positive integer.

    Returns
    -------
    int
        The power of two.
    """
    return 1 << max(int(x) - 1, 0).bit_length()


# Layout: query lanes run along the LAST axis of every tile -- candidates are
# ``(Ws, W)`` and the running top-k ``(Kp, W)`` -- so Triton spreads the lanes
# over threads and each thread keeps its query's candidates and best-k in its
# own registers: the per-slot reductions are in-thread loops, not shuffles.


def _lane_state(
    x_ref, y_ref, z_ref, qmask_ref, slot_ref, t, *, width: int, k: int, kpad: int
):
    """Query lanes of target leaf ``t`` and the empty running top-k."""
    lanes = pl.dslice(0, width)
    qx = x_ref[t, lanes]
    qy = y_ref[t, lanes]
    qz = z_ref[t, lanes]
    qv = qmask_ref[t, lanes] != 0
    qslot = slot_ref[t, lanes]
    inf = jnp.float32(_INF)
    neg = jnp.float32(-1.0)
    iota_k = lax.broadcasted_iota(jnp.int32, (kpad, width), 0)
    # padding slots hold -1: never the worst, so never replaced
    best_d = jnp.where(iota_k >= kpad - k, inf, neg)
    best_i = jnp.full((kpad, width), -1, jnp.int32)
    # non-query lanes get k-th = -1, so they never accept and never ask for a leaf
    kth = jnp.where(qv, inf, neg)
    return (qx, qy, qz, qv, qslot), (best_d, best_i, kth)


def _need(q, cx, cy, cz, cr, kt):
    """Lanes whose k-th ball reaches the sphere ``(c, r)``, and whether any do."""
    qx, qy, qz = q[0], q[1], q[2]
    ex = qx - cx
    ey = qy - cy
    ez = qz - cz
    gap = jnp.sqrt(ex * ex + ey * ey + ez * ez) - cr
    lb2 = jnp.where(gap > 0, gap * gap, jnp.float32(0.0))
    need = lb2 < kt
    return need, jnp.max(jnp.where(need, 1, 0)) > 0


def _make_merge(
    x_ref, y_ref, z_ref, dmask_ref, slot_ref, q, *, width, kpad, exclude_self
):
    """``merge(s, need, carry)``: fold source leaf ``s`` into the running top-k."""
    qx, qy, qz, qv, qslot = q
    lanes = pl.dslice(0, width)
    inf = jnp.float32(_INF)
    neg = jnp.float32(-1.0)
    iota_s = lax.broadcasted_iota(jnp.int32, (width, width), 0)
    iota_k = lax.broadcasted_iota(jnp.int32, (kpad, width), 0)

    def merge(s: Any, need: Any, carry: tuple) -> tuple:
        bd, bi, kt = carry
        sx = x_ref[s, lanes]
        sy = y_ref[s, lanes]
        sz = z_ref[s, lanes]
        sv = dmask_ref[s, lanes] != 0
        ss = slot_ref[s, lanes]
        # a leaf's lanes are consecutive sorted slots: candidate j is s0 + j
        s0 = slot_ref[s, 0]
        dx = sx[:, None] - qx[None, :]
        dy = sy[:, None] - qy[None, :]
        dz = sz[:, None] - qz[None, :]
        d2 = dx * dx + dy * dy + dz * dz  # (Ws, W): candidate x query lane
        ok = sv[:, None] & need[None, :]
        if exclude_self:
            ok = ok & (ss[:, None] != qslot[None, :])
        d2 = jnp.where(ok, d2, inf)
        mn = jnp.min(d2, axis=0)

        def cond_fn(st: tuple) -> Any:
            _, mn_, _, _, kt_ = st
            return jnp.max(jnp.where(mn_ < kt_, 1, 0)) > 0

        def body_fn(st: tuple) -> tuple:
            d2_, mn_, bd_, bi_, kt_ = st
            # first-index argmin/argmax by hand: Triton's argreduce wants an
            # i32 index type, which jnp.argmin does not give under x64. The
            # k-th distance IS the slot maximum, so the worst slot needs no max.
            j = jnp.min(jnp.where(d2_ == mn_[None, :], iota_s, width), axis=0)
            take = mn_ < kt_
            worst = jnp.min(jnp.where(bd_ == kt_[None, :], iota_k, kpad), axis=0)
            put = (iota_k == worst[None, :]) & take[None, :]
            bd_ = jnp.where(put, mn_[None, :], bd_)
            bi_ = jnp.where(put, (s0 + j)[None, :], bi_)
            kt_ = jnp.where(qv, jnp.max(bd_, axis=0), neg)
            d2_ = jnp.where(iota_s == j[None, :], inf, d2_)
            return d2_, jnp.min(d2_, axis=0), bd_, bi_, kt_

        _, _, bd, bi, kt = lax.while_loop(cond_fn, body_fn, (d2, mn, bd, bi, kt))
        return bd, bi, kt

    return merge


def _sorted(best_d, best_i, *, width, kpad):
    """The running top-k sorted ascending per lane (padding -1 rows first)."""
    inf = jnp.float32(_INF)
    iota_k = lax.broadcasted_iota(jnp.int32, (kpad, width), 0)

    def _sort(r: Any, st: tuple) -> tuple:
        bd, sd, si = st
        v = jnp.min(bd, axis=0)
        j = jnp.min(jnp.where(bd == v[None, :], iota_k, kpad), axis=0)
        hit = iota_k == j[None, :]
        idx = jnp.sum(jnp.where(hit, best_i, 0), axis=0, dtype=jnp.int32)
        # an all-inf remainder may re-pick an extracted slot: report it as -1
        idx = jnp.where(v < inf, idx, -1)
        row = iota_k == r
        sd = jnp.where(row, v[None, :], sd)
        si = jnp.where(row, idx[None, :], si)
        return jnp.where(hit, inf, bd), sd, si

    _, sd, si = lax.fori_loop(
        jnp.int32(0),
        jnp.int32(kpad),
        _sort,
        (
            best_d,
            jnp.zeros((kpad, width), jnp.float32),
            jnp.zeros((kpad, width), jnp.int32),
        ),
    )
    return sd, si


def _write_sorted(out_d_ref, out_i_ref, best_d, best_i, *, width, kpad):
    """Store the sorted top-k as slot-major ``(1, Kp, W)`` blocks."""
    sd, si = _sorted(best_d, best_i, width=width, kpad=kpad)
    out_d_ref[0, :, :] = sd
    out_i_ref[0, :, :] = si


def _scatter_sorted(out_d_ref, out_i_ref, qid, best_d, best_i, *, width, k, kpad, nout):
    """Scatter each query lane's sorted top-k into row ``qid`` of ``(Q * k,)``.

    Lanes with ``qid < 0`` (data, padding) write nothing, and the ``Kp - k``
    padding rows are dropped, so the leading ``Q * k`` entries hold ``(Q, k)``.
    A masked entry still gets its own address in a ``Kp * W`` tail past
    ``nout``: interpret mode evaluates a masked scatter as read-modify-write,
    so masked duplicates of a live address could restore its old value.
    """
    sd, si = _sorted(best_d, best_i, width=width, kpad=kpad)
    iota_k = lax.broadcasted_iota(jnp.int32, (kpad, width), 0)
    iota_l = lax.broadcasted_iota(jnp.int32, (kpad, width), 1)
    col = iota_k - (kpad - k)
    ok = (col >= 0) & (qid >= 0)[None, :]
    addr = jnp.where(ok, qid[None, :] * k + col, nout + iota_k * width + iota_l)
    plgpu.store(out_d_ref.at[addr], sd, mask=ok)
    plgpu.store(out_i_ref.at[addr], si, mask=ok)


def _leaf_topk_kernel(
    x_ref: KernelRef,
    y_ref: KernelRef,
    z_ref: KernelRef,
    dmask_ref: KernelRef,
    qmask_ref: KernelRef,
    slot_ref: KernelRef,
    cx_ref: KernelRef,
    cy_ref: KernelRef,
    cz_ref: KernelRef,
    cr_ref: KernelRef,
    offsets_ref: KernelRef,
    sources_ref: KernelRef,
    out_d_ref: KernelRef,
    out_i_ref: KernelRef,
    *,
    width: int,
    k: int,
    kpad: int,
    exclude_self: bool,
) -> None:
    """Exact top-``k`` of one target leaf's query lanes over its CSR sources.

    Parameters
    ----------
    x_ref
        Leaf-major coordinates, ``(L, W)``, full tables (gathered by row).
    y_ref
        As ``x_ref``.
    z_ref
        As ``x_ref``.
    dmask_ref
        ``(L, W)`` int32: lane is a data point (a candidate) / a query.
    qmask_ref
        As ``dmask_ref``.
    slot_ref
        ``(L, W)`` int32 sorted slot of each lane (self-exclusion and output).
    cx_ref
        ``(L,)`` bounding sphere of each leaf.
    cy_ref
        As ``cx_ref``.
    cz_ref
        As ``cx_ref``.
    cr_ref
        As ``cx_ref``.
    offsets_ref
        ``(L + 1,)`` CSR offsets into ``sources_ref``.
    sources_ref
        ``(E,)`` source leaf rows, nearest first within each target.
    out_d_ref
        **Outputs** ``(1, Kp, W)``: ascending squared distances and source
        slots per lane; the first ``Kp - k`` rows are padding (``-1`` / ``-1``).
    out_i_ref
        As ``out_d_ref``.
    width
        ``W``, a power of two. Static.
    k
        Neighbours. Static.
    kpad
        ``Kp``, a power of two ``>= k``. Static.
    exclude_self
        Skip a candidate in the query's own slot. Static.
    """
    t = pl.program_id(0)
    q, carry = _lane_state(
        x_ref, y_ref, z_ref, qmask_ref, slot_ref, t, width=width, k=k, kpad=kpad
    )
    merge = _make_merge(
        x_ref,
        y_ref,
        z_ref,
        dmask_ref,
        slot_ref,
        q,
        width=width,
        kpad=kpad,
        exclude_self=exclude_self,
    )

    def _source(e: Any, carry: tuple) -> tuple:
        s = sources_ref[e]
        need, any_need = _need(q, cx_ref[s], cy_ref[s], cz_ref[s], cr_ref[s], carry[2])
        return lax.cond(any_need, lambda c: merge(s, need, c), lambda c: c, carry)

    best_d, best_i, _ = lax.fori_loop(
        offsets_ref[t], offsets_ref[t + 1], _source, carry
    )
    _write_sorted(out_d_ref, out_i_ref, best_d, best_i, width=width, kpad=kpad)


def _leaf_topk_traverse_kernel(
    x_ref: KernelRef,
    y_ref: KernelRef,
    z_ref: KernelRef,
    dmask_ref: KernelRef,
    qmask_ref: KernelRef,
    slot_ref: KernelRef,
    ncx_ref: KernelRef,
    ncy_ref: KernelRef,
    ncz_ref: KernelRef,
    nr_ref: KernelRef,
    first_ref: KernelRef,
    skip_ref: KernelRef,
    nlo_ref: KernelRef,
    nhi_ref: KernelRef,
    qid_ref: KernelRef,
    out_d_ref: KernelRef,
    out_i_ref: KernelRef,
    *maybe_stats_ref: KernelRef,
    width: int,
    k: int,
    kpad: int,
    exclude_self: bool,
    num_leaves: int,
    num_internal: int,
    root: int,
    end: int,
    seed: int,
    nout: int,
) -> None:
    """Exact top-``k`` of one target leaf by a stackless packet traversal.

    The leaf's lanes walk the tree together in pre-order: a node is entered when
    any lane's k-th ball reaches its sphere and skipped (``skip_ref``, the next
    pre-order node after its subtree) otherwise; a reached leaf is merged. The
    leaves ``t - seed .. t + seed`` (Morton neighbours) are merged first, so the
    walk starts with finite k-th distances, and are skipped by the walk.

    Parameters
    ----------
    x_ref
        Leaf tables, as :func:`_leaf_topk_kernel`.
    y_ref
        As ``x_ref``.
    z_ref
        As ``x_ref``.
    dmask_ref
        As ``x_ref``.
    qmask_ref
        As ``x_ref``.
    slot_ref
        As ``x_ref``.
    ncx_ref
        ``(nodes,)`` bounding sphere of every node (radius ``-inf`` on empty
        nodes, which are then never entered).
    ncy_ref
        As ``ncx_ref``.
    ncz_ref
        As ``ncx_ref``.
    nr_ref
        As ``ncx_ref``.
    first_ref
        ``(nodes,)`` first child of an internal node, ``-1`` on leaves.
    skip_ref
        ``(nodes,)`` next pre-order node after the subtree (``end`` for none).
    nlo_ref
        ``(nodes,)`` first and last leaf of each node's subtree: a subtree inside
        the seed window is skipped whole.
    nhi_ref
        As ``nlo_ref``.
    qid_ref
        ``(L, W)`` int32 output row of each query lane (``-1``: not a query).
    out_d_ref
        **Outputs** ``(Q * k,)``: each query's ascending squared distances and
        source slots, scattered to its row.
    out_i_ref
        As ``out_d_ref``.
    *maybe_stats_ref
        Optional ``(1, W)`` output of per-program counters (``debug_stats``).
    width
        As :func:`_leaf_topk_kernel`. Static.
    k
        As ``width``.
    kpad
        As ``width``.
    exclude_self
        As ``width``.
    num_leaves
        Tree layout (leaf ``l`` is node ``num_internal + l``) and the seed
        half-width. Static.
    num_internal
        As ``num_leaves``.
    root
        As ``num_leaves``.
    end
        As ``num_leaves``.
    seed
        As ``num_leaves``.
    nout
        ``Q * k``: live output entries; a scratch tail follows. Static.
    """
    t = pl.program_id(0)
    q, carry = _lane_state(
        x_ref, y_ref, z_ref, qmask_ref, slot_ref, t, width=width, k=k, kpad=kpad
    )
    merge = _make_merge(
        x_ref,
        y_ref,
        z_ref,
        dmask_ref,
        slot_ref,
        q,
        width=width,
        kpad=kpad,
        exclude_self=exclude_self,
    )
    lo = jnp.maximum(t - seed, 0)
    hi = jnp.minimum(t + seed + 1, num_leaves)

    def _seed(e: Any, carry: tuple) -> tuple:
        # centre-out: t, t+1, t-1, t+2, ... so the nearest leaves fill first
        half = (e + 1) // 2
        s = jnp.where(e % 2 == 1, t + half, t - half)
        ok = (s >= lo) & (s < hi)
        sc = jnp.clip(s, 0, num_leaves - 1)
        node = num_internal + sc
        need, any_need = _need(
            q, ncx_ref[node], ncy_ref[node], ncz_ref[node], nr_ref[node], carry[2]
        )
        return lax.cond(ok & any_need, lambda c: merge(sc, need, c), lambda c: c, carry)

    carry = lax.fori_loop(jnp.int32(0), jnp.int32(2 * seed + 1), _seed, carry)

    def _cond(st: tuple) -> Any:
        return st[0] < end

    def _body(st: tuple) -> tuple:
        node, bd, bi, kt, nv, nm = st
        need, any_need = _need(
            q, ncx_ref[node], ncy_ref[node], ncz_ref[node], nr_ref[node], kt
        )
        child = first_ref[node]
        leaf = node - num_internal
        # a subtree wholly inside the seed window was merged already
        inside = (nlo_ref[node] >= lo) & (nhi_ref[node] < hi)
        any_need = any_need & ~inside
        do_merge = any_need & (child < 0)
        bd, bi, kt = lax.cond(
            do_merge, lambda c: merge(leaf, need, c), lambda c: c, (bd, bi, kt)
        )
        nxt = jnp.where(any_need & (child >= 0), child, skip_ref[node])
        return nxt, bd, bi, kt, nv + 1, nm + jnp.where(do_merge, 1, 0)

    zero = jnp.int32(0)
    _, best_d, best_i, _, n_visit, n_merge = lax.while_loop(
        _cond, _body, (jnp.int32(root), carry[0], carry[1], carry[2], zero, zero)
    )
    if maybe_stats_ref:
        lane_i = lax.broadcasted_iota(jnp.int32, (width,), 0)
        maybe_stats_ref[0][0, :] = jnp.where(
            lane_i == 0, n_visit, jnp.where(lane_i == 1, n_merge, 0)
        )
    qid = qid_ref[t, pl.dslice(0, width)]
    _scatter_sorted(
        out_d_ref,
        out_i_ref,
        qid,
        best_d,
        best_i,
        width=width,
        k=k,
        kpad=kpad,
        nout=nout,
    )


def _check_inputs(x, sources, k: int) -> tuple[int, int, int]:
    num_leaves, width = int(x.shape[0]), int(x.shape[1])
    if width & (width - 1):
        raise ValueError(f"leaf width must be a power of two, got {width}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return num_leaves, width, next_pow2(k)


def leaf_topk_pallas(
    x: Any,
    y: Any,
    z: Any,
    dmask: Any,
    qmask: Any,
    slot: Any,
    cx: Any,
    cy: Any,
    cz: Any,
    cr: Any,
    offsets: Any,
    sources: Any,
    *,
    k: int,
    exclude_self: bool,
    num_warps: int | None = None,
    num_stages: int = 1,
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Exact per-lane top-``k`` over CSR source leaves, one program per leaf.

    Parameters
    ----------
    x
        ``(L, W)`` float32 leaf-major coordinates; ``W`` a power of two.
    y
        As ``x``.
    z
        As ``x``.
    dmask
        ``(L, W)`` int32: data lane (candidate) / query lane.
    qmask
        As ``dmask``.
    slot
        ``(L, W)`` int32 sorted slot of each lane.
    cx
        ``(L,)`` float32 bounding sphere of each leaf.
    cy
        As ``cx``.
    cz
        As ``cx``.
    cr
        As ``cx``.
    offsets
        ``(L + 1,)`` int32 CSR offsets.
    sources
        ``(E,)`` int32 source leaf rows, nearest first per target.
    k
        Neighbours per query.
    exclude_self
        Skip the query's own slot.
    num_warps
        Triton warps per program (``None``: 1 for ``W <= 32``, else 2).
    num_stages
        Triton pipeline stages.
    interpret
        Run in Pallas interpret mode (CPU semantics).

    Returns
    -------
    tuple
        ``(d2, idx)``, each ``(L, W, k)``: ascending squared distances (``inf``
        where fewer than ``k`` candidates were reached) and source slots
        (``-1`` there). Rows of non-query lanes are meaningless.

    Raises
    ------
    RuntimeError
        If Pallas could not be imported.
    """
    if pl is None:  # pragma: no cover - environment-dependent
        raise RuntimeError("jax.experimental.pallas is not available")
    num_leaves, width, kpad = _check_inputs(x, sources, k)
    if num_warps is None:
        num_warps = 1 if width <= 32 else 2

    def _kernel(*refs: KernelRef) -> None:
        _leaf_topk_kernel(
            *refs, width=width, k=int(k), kpad=kpad, exclude_self=bool(exclude_self)
        )

    full2 = pl.BlockSpec((num_leaves, width), lambda t: (0, 0))
    full1 = pl.BlockSpec((num_leaves,), lambda t: (0,))
    offs = pl.BlockSpec((num_leaves + 1,), lambda t: (0,))
    srcs = pl.BlockSpec((int(sources.shape[0]),), lambda t: (0,))
    out = pl.BlockSpec((1, kpad, width), lambda t: (t, 0, 0))
    call = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_leaves, kpad, width), jnp.float32),
            jax.ShapeDtypeStruct((num_leaves, kpad, width), jnp.int32),
        ),
        in_specs=[full2] * 6 + [full1] * 4 + [offs, srcs],
        out_specs=(out, out),
        grid=(num_leaves,),
        interpret=bool(interpret),
        name=f"leaf_topk_w{width}_k{int(k)}",
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        ),
    )
    d2, idx = call(
        x.astype(jnp.float32),
        y.astype(jnp.float32),
        z.astype(jnp.float32),
        dmask.astype(jnp.int32),
        qmask.astype(jnp.int32),
        slot.astype(jnp.int32),
        cx.astype(jnp.float32),
        cy.astype(jnp.float32),
        cz.astype(jnp.float32),
        cr.astype(jnp.float32),
        offsets.astype(jnp.int32),
        sources.astype(jnp.int32),
    )
    d2 = jnp.swapaxes(d2, 1, 2)
    idx = jnp.swapaxes(idx, 1, 2)
    return d2[..., kpad - k :], idx[..., kpad - k :]


def leaf_topk_traverse_pallas(
    x: Any,
    y: Any,
    z: Any,
    dmask: Any,
    qmask: Any,
    slot: Any,
    ncx: Any,
    ncy: Any,
    ncz: Any,
    nr: Any,
    first: Any,
    skip: Any,
    node_lo: Any,
    node_hi: Any,
    qid: Any,
    *,
    num_queries: int,
    k: int,
    exclude_self: bool,
    num_internal: int,
    root: int,
    seed: int = 16,
    num_warps: int | None = None,
    num_stages: int = 1,
    interpret: bool = False,
    debug_stats: bool = False,
) -> tuple[Any, ...]:
    """Exact per-lane top-``k`` by a stackless packet traversal of the tree.

    Parameters
    ----------
    x
        ``(L, W)`` leaf tables, as :func:`leaf_topk_pallas`.
    y
        As ``x``.
    z
        As ``x``.
    dmask
        As ``x``.
    qmask
        As ``x``.
    slot
        As ``x``.
    ncx
        ``(nodes,)`` float32 node spheres; ``nr = -inf`` marks an empty node.
    ncy
        As ``ncx``.
    ncz
        As ``ncx``.
    nr
        As ``ncx``.
    first
        ``(nodes,)`` int32 first child (``-1`` on leaves) and pre-order skip
        (``nodes`` past the end).
    skip
        As ``first``.
    node_lo
        ``(nodes,)`` int32 first and last leaf of each node's subtree.
    node_hi
        As ``node_lo``.
    qid
        ``(L, W)`` int32 output row of each query lane, ``-1`` elsewhere.
    num_queries
        ``Q``, the number of output rows.
    k
        Neighbours per query.
    exclude_self
        Skip the query's own slot.
    num_internal
        Leaf ``l`` is node ``num_internal + l``; the walk starts at ``root``.
    root
        As ``num_internal``.
    seed
        Morton-neighbour leaves merged first on each side of the target (a
        window of ``2 seed + 1``); the walk then skips subtrees inside it.
    num_warps
        As :func:`leaf_topk_pallas`.
    num_stages
        As ``num_warps``.
    interpret
        As ``num_warps``.
    debug_stats
        Also return ``(L, W)`` int32 per-program counters: lane 0 nodes tested,
        lane 1 leaves merged by the walk (profiling only).

    Returns
    -------
    tuple
        ``(d2, idx)``, each ``(Q, k)`` in query order: ascending squared
        distances (``inf`` where fewer than ``k`` were reached) and source
        slots (``-1`` there); ``(d2, idx, stats)`` with ``debug_stats``.

    Raises
    ------
    RuntimeError
        If Pallas could not be imported.
    """
    if pl is None:  # pragma: no cover - environment-dependent
        raise RuntimeError("jax.experimental.pallas is not available")
    num_leaves, width, kpad = _check_inputs(x, first, k)
    nodes = int(first.shape[0])
    if num_warps is None:
        num_warps = 1 if width <= 32 else 2

    def _kernel(*refs: KernelRef) -> None:
        _leaf_topk_traverse_kernel(
            *refs,
            width=width,
            k=int(k),
            kpad=kpad,
            exclude_self=bool(exclude_self),
            num_leaves=num_leaves,
            num_internal=int(num_internal),
            root=int(root),
            end=nodes,
            seed=int(seed),
            nout=nout,
        )

    full2 = pl.BlockSpec((num_leaves, width), lambda t: (0, 0))
    node1 = pl.BlockSpec((nodes,), lambda t: (0,))
    out = pl.BlockSpec((1, kpad, width), lambda t: (t, 0, 0))
    nout = int(num_queries) * int(k)
    ntot = nout + kpad * width  # scratch tail for masked lanes
    out_shape = [
        jax.ShapeDtypeStruct((ntot,), jnp.float32),
        jax.ShapeDtypeStruct((ntot,), jnp.int32),
    ]
    flat = pl.BlockSpec((ntot,), lambda t: (0,))
    out_specs = [flat, flat]
    if debug_stats:
        out_shape.append(jax.ShapeDtypeStruct((num_leaves, width), jnp.int32))
        out_specs.append(pl.BlockSpec((1, width), lambda t: (t, 0)))
    call = pl.pallas_call(
        _kernel,
        out_shape=tuple(out_shape),
        in_specs=[full2] * 6 + [node1] * 8 + [full2],
        out_specs=tuple(out_specs),
        grid=(num_leaves,),
        interpret=bool(interpret),
        name=f"leaf_topk_traverse_w{width}_k{int(k)}",
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        ),
    )
    res = call(
        x.astype(jnp.float32),
        y.astype(jnp.float32),
        z.astype(jnp.float32),
        dmask.astype(jnp.int32),
        qmask.astype(jnp.int32),
        slot.astype(jnp.int32),
        ncx.astype(jnp.float32),
        ncy.astype(jnp.float32),
        ncz.astype(jnp.float32),
        nr.astype(jnp.float32),
        first.astype(jnp.int32),
        skip.astype(jnp.int32),
        node_lo.astype(jnp.int32),
        node_hi.astype(jnp.int32),
        qid.astype(jnp.int32),
    )
    d2 = res[0][:nout].reshape(int(num_queries), int(k))
    idx = res[1][:nout].reshape(int(num_queries), int(k))
    return (d2, idx, res[2]) if debug_stats else (d2, idx)


def leaf_topk_xla(
    x: Any,
    y: Any,
    z: Any,
    dmask: Any,
    qmask: Any,
    slot: Any,
    cx: Any,
    cy: Any,
    cz: Any,
    cr: Any,
    offsets: Any,
    sources: Any,
    *,
    k: int,
    exclude_self: bool,
    max_sources: int,
    leaf_chunk: int = 32,
) -> tuple[Any, Any]:
    """Pure-XLA twin of :func:`leaf_topk_pallas` (same inputs and outputs).

    Gathers each target leaf's first ``max_sources`` source leaves into a dense
    candidate block and takes ``lax.top_k``, ``leaf_chunk`` targets at a time.
    Exact when no target has more than ``max_sources`` sources (the caller
    sizes it); the sphere-skip of the kernel is not needed here.

    Parameters
    ----------
    x
        As :func:`leaf_topk_pallas`.
    y
        As ``x``.
    z
        As ``x``.
    dmask
        As ``x``.
    qmask
        As ``x``.
    slot
        As ``x``.
    cx
        As ``x``.
    cy
        As ``x``.
    cz
        As ``x``.
    cr
        As ``x``.
    offsets
        As ``x``.
    sources
        As ``x``.
    k
        Neighbours per query.
    exclude_self
        Skip the query's own slot.
    max_sources
        Static bound on any target's source count.
    leaf_chunk
        Targets per ``lax.map`` step (memory bound).

    Returns
    -------
    tuple
        ``(d2, idx)``, each ``(L, W, k)``, as :func:`leaf_topk_pallas`.
    """
    del cx, cy, cz, cr, qmask
    num_leaves, width, _ = _check_inputs(x, sources, k)
    m = int(max_sources)
    pos = jnp.stack([x, y, z], axis=-1).astype(jnp.float32)  # (L, W, 3)
    nsrc = offsets[1:] - offsets[:-1]
    e = offsets[:-1, None] + jnp.arange(m)[None, :]
    src_ok = jnp.arange(m)[None, :] < nsrc[:, None]
    src = jnp.where(src_ok, sources[jnp.clip(e, 0, sources.shape[0] - 1)], 0)

    nchunk = -(-num_leaves // leaf_chunk)
    pad = nchunk * leaf_chunk - num_leaves
    tgt = jnp.concatenate([jnp.arange(num_leaves), jnp.zeros(pad, jnp.int32)])

    def one(tt):
        sl = src[tt]  # (C, m)
        ok_s = src_ok[tt]
        cand = pos[sl].reshape(tt.shape[0], m * width, 3)
        cv = (dmask[sl] != 0) & ok_s[:, :, None]
        cs = slot[sl].reshape(tt.shape[0], m * width)
        cv = cv.reshape(tt.shape[0], m * width)
        q = pos[tt]  # (C, W, 3)
        d2 = jnp.sum((q[:, :, None, :] - cand[:, None, :, :]) ** 2, axis=-1)
        ok = cv[:, None, :]
        if exclude_self:
            ok = ok & (cs[:, None, :] != slot[tt][:, :, None])
        d2 = jnp.where(ok, d2, jnp.inf)
        neg, j = lax.top_k(-d2, k)
        ids = jnp.take_along_axis(
            jnp.broadcast_to(cs[:, None, :], d2.shape), j, axis=-1
        )
        d = -neg
        return d, jnp.where(jnp.isfinite(d), ids, -1)

    d2, idx = lax.map(one, tgt.reshape(nchunk, leaf_chunk))
    d2 = d2.reshape(nchunk * leaf_chunk, width, k)[:num_leaves]
    idx = idx.reshape(nchunk * leaf_chunk, width, k)[:num_leaves]
    return d2, idx
