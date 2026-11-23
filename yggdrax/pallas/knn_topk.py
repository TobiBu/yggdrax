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

from typing import TYPE_CHECKING, Any, Optional

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
    "leaf_topk_preorder_pallas",
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


def _need_box(q, lo, hi, kt, box=None):
    """Lanes whose k-th ball reaches the box ``[lo, hi]``, and whether any do.

    Point-to-box distance, tighter than a bounding sphere around a box. Exact
    as a bound in float32: ``lo`` and ``hi`` are actual point coordinates and
    rounding is monotone, so it never exceeds a member point's distance. An
    empty box (``lo = +inf``, ``hi = -inf``) is never reached.

    With a periodic ``box`` (points, boxes and queries all in ``[0, L)``) the
    gap on an axis is the smallest over the three images of the interval,
    ``min(g, lo + L - q, q - hi + L)``: the minimum-image distance bound.
    """
    zero = jnp.float32(0.0)
    d2 = zero
    for a, (qc, l_, h_) in enumerate(zip(q[:3], lo, hi)):
        g = jnp.maximum(jnp.maximum(l_ - qc, qc - h_), zero)
        if box is not None:
            L = jnp.float32(box[a])
            g = jnp.minimum(g, jnp.minimum(l_ + L - qc, qc - h_ + L))
        d2 = d2 + g * g
    need = d2 < kt
    return need, jnp.max(jnp.where(need, 1, 0)) > 0


def _image(sc: Any, qc: Any, axis: int, box: Any) -> Any:
    """Minimum image of ``sc - qc`` for coordinates in ``[0, L)``.

    The shifted coordinate is formed first -- ``(sc - L) - qc`` when ``sc`` is
    the far one -- because ``sc - L`` is exact for ``sc`` in ``[L/2, L)``
    (Sterbenz), while ``(sc - qc) - L`` cancels: a pair straddling a face
    lost ~1e-4 relative distance in float32 at 1e7 points.
    """
    d = sc - qc
    if box is None:
        return d
    # constants built in Python: the Triton lowering cannot negate a literal
    L = jnp.float32(box[axis])
    hp = jnp.float32(0.5 * box[axis])
    hn = jnp.float32(-0.5 * box[axis])
    return jnp.where(d > hp, (sc - L) - qc, jnp.where(d < hn, sc - (qc - L), d))


def _make_insert(
    x_ref,
    y_ref,
    z_ref,
    dmask_ref,
    slot_ref,
    q,
    *,
    width,
    k,
    exclude_self,
    box=None,
    use_dmask=True,
    unroll=4,
):
    """``merge(s, need, carry)`` on a SORTED top-k held as ``k`` registers.

    The carry is ``(bd, bi, kt)`` with ``bd``/``bi`` tuples of ``k`` ``(W,)``
    vectors in DESCENDING order, so the k-th distance is ``bd[0]``. Candidate
    ``v`` enters by ``new[i] = v < bd[i+1] ? bd[i+1] : (v < bd[i] ? v : bd[i])``:
    about five element ops per slot and no reduction, against an argmin over
    the block, a worst-slot search and a max per insertion of replace-the-
    worst. A candidate no lane wants (one warp vote) costs its distance only.
    """
    qx, qy, qz, qv, qslot = q
    inf = jnp.float32(_INF)

    def insert(v: Any, vi: Any, bd: tuple, bi: tuple) -> tuple:
        h = [v < b for b in bd]
        nd, ni = [], []
        for i in range(k):
            if i + 1 < k:
                nd.append(jnp.where(h[i + 1], bd[i + 1], jnp.where(h[i], v, bd[i])))
                ni.append(jnp.where(h[i + 1], bi[i + 1], jnp.where(h[i], vi, bi[i])))
            else:
                nd.append(jnp.where(h[i], v, bd[i]))
                ni.append(jnp.where(h[i], vi, bi[i]))
        return tuple(nd), tuple(ni)

    def merge(s: Any, need: Any, carry: tuple) -> tuple:
        s0 = slot_ref[s, 0]

        def group(g: Any, c: tuple) -> tuple:
            # ``unroll`` candidates at once: their loads, distances and warp
            # votes are independent, so they overlap instead of forming one
            # serial load -> vote -> branch chain per candidate. The votes use
            # the k-th distance from the start of the group: stale is only
            # larger, so a vote can say yes to a candidate the insert then
            # leaves alone (an insert of v >= bd[0] is the identity).
            bd, bi, kt = c
            js = [g * unroll + u for u in range(unroll)]
            vs, wants = [], []
            for j in js:
                dx = _image(x_ref[s, j], qx, 0, box)
                dy = _image(y_ref[s, j], qy, 1, box)
                dz = _image(z_ref[s, j], qz, 2, box)
                d2 = dx * dx + dy * dy + dz * dz
                ok = (need & (dmask_ref[s, j] != 0)) if use_dmask else need
                if exclude_self:
                    ok = ok & (qslot != s0 + j)
                v = jnp.where(ok, d2, inf)
                vs.append(v)
                wants.append(jnp.max(jnp.where(v < kt, 1, 0)) > 0)
            for j, v, want in zip(js, vs, wants):

                def ins(c2: tuple, v=v, j=j) -> tuple:
                    nd, ni = insert(v, s0 + j, c2[0], c2[1])
                    return nd, ni, nd[0]

                bd, bi, kt = lax.cond(want, ins, lambda c2: c2, (bd, bi, kt))
            return bd, bi, kt

        return lax.fori_loop(jnp.int32(0), jnp.int32(width // unroll), group, carry)

    return merge


def _insert_state(qv: Any, *, width: int, k: int) -> tuple:
    """Empty sorted top-k: ``+inf`` on query lanes, ``-1`` elsewhere (never
    accepts, never asks for a leaf)."""
    fill = jnp.where(qv, jnp.float32(_INF), jnp.float32(-1.0))
    bd = tuple(fill for _ in range(k))
    bi = tuple(jnp.full((width,), -1, jnp.int32) for _ in range(k))
    return bd, bi, fill


def _scatter_insert(out_d_ref, out_i_ref, qid, bd, bi, *, width, k, nout):
    """Store a descending register top-k ascending into rows ``qid`` of
    ``(Q * k,)``; masked lanes go to a scratch tail past ``nout``."""
    lane = lax.broadcasted_iota(jnp.int32, (width,), 0)
    ok = qid >= 0
    for r in range(k):
        d = bd[k - 1 - r]
        i = jnp.where(d < jnp.float32(_INF), bi[k - 1 - r], -1)
        addr = jnp.where(ok, qid * k + r, nout + r * width + lane)
        plgpu.store(out_d_ref.at[addr], d, mask=ok)
        plgpu.store(out_i_ref.at[addr], i, mask=ok)


def _make_merge(
    x_ref,
    y_ref,
    z_ref,
    dmask_ref,
    slot_ref,
    q,
    *,
    width,
    kpad,
    exclude_self,
    box=None,
    use_dmask=True,
):
    """``(merge, fill)``: fold source leaf ``s`` into the running top-k.

    ``merge(s, need, carry)`` inserts, one candidate per lane per step, every
    candidate that beats the lane's current k-th distance (replacing its
    worst). ``fill(s, need, carry)`` is the first merge into an empty top-k
    when ``Kp == W == k``: the block becomes the top-k without a loop.
    """
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
        # a leaf's lanes are consecutive sorted slots: candidate j is s0 + j
        s0 = slot_ref[s, 0]
        dx = _image(sx[:, None], qx[None, :], 0, box)
        dy = _image(sy[:, None], qy[None, :], 1, box)
        dz = _image(sz[:, None], qz[None, :], 2, box)
        d2 = dx * dx + dy * dy + dz * dz  # (Ws, W): candidate x query lane
        ok = jnp.broadcast_to(need[None, :], (width, width))
        if use_dmask:
            ok = ok & (dmask_ref[s, lanes] != 0)[:, None]
        if exclude_self:
            ok = ok & (slot_ref[s, lanes][:, None] != qslot[None, :])
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

    def fill(s: Any, need: Any, carry: tuple) -> tuple:
        """First merge into an EMPTY top-k with ``Kp == W == k``: the block IS
        the top-k (no loop), unsorted, as replace-the-worst keeps it."""
        del carry
        sx = x_ref[s, lanes]
        sy = y_ref[s, lanes]
        sz = z_ref[s, lanes]
        s0 = slot_ref[s, 0]
        dx = _image(sx[:, None], qx[None, :], 0, box)
        dy = _image(sy[:, None], qy[None, :], 1, box)
        dz = _image(sz[:, None], qz[None, :], 2, box)
        d2 = dx * dx + dy * dy + dz * dz
        ok = jnp.broadcast_to(need[None, :], (width, width))
        if use_dmask:
            ok = ok & (dmask_ref[s, lanes] != 0)[:, None]
        if exclude_self:
            ok = ok & (slot_ref[s, lanes][:, None] != qslot[None, :])
        bd = jnp.where(ok, d2, inf)
        bi = jnp.where(ok, s0 + iota_s, -1)
        kt = jnp.where(qv, jnp.max(bd, axis=0), neg)
        return bd, bi, kt

    return merge, fill


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
    merge, _ = _make_merge(
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


def _leaf_topk_preorder_kernel(
    x_ref: KernelRef,
    y_ref: KernelRef,
    z_ref: KernelRef,
    dmask_ref: KernelRef,
    qmask_ref: KernelRef,
    slot_ref: KernelRef,
    tx_ref: KernelRef,
    ty_ref: KernelRef,
    tz_ref: KernelRef,
    tq_ref: KernelRef,
    tslot_ref: KernelRef,
    seedc_ref: KernelRef,
    llox_ref: KernelRef,
    lloy_ref: KernelRef,
    lloz_ref: KernelRef,
    lhix_ref: KernelRef,
    lhiy_ref: KernelRef,
    lhiz_ref: KernelRef,
    plox_ref: KernelRef,
    ploy_ref: KernelRef,
    ploz_ref: KernelRef,
    phix_ref: KernelRef,
    phiy_ref: KernelRef,
    phiz_ref: KernelRef,
    pleaf_ref: KernelRef,
    psize_ref: KernelRef,
    pnlo_ref: KernelRef,
    pnhi_ref: KernelRef,
    qid_ref: KernelRef,
    out_d_ref: KernelRef,
    out_i_ref: KernelRef,
    *,
    width: int,
    k: int,
    kpad: int,
    exclude_self: bool,
    num_leaves: int,
    end: int,
    seed: int,
    nout: int,
    line: int,
    profile: str = "full",
    merge_impl: str = "replace",
    box: Optional[tuple] = None,
    unroll: int = 4,
) -> None:
    """Exact top-``k`` of one target leaf: packet traversal in pre-order lines.

    The nodes are numbered in pre-order, so entering a node is ``p + 1`` and
    skipping its subtree is ``p + size[p]``. Both mostly land a few positions
    ahead, so the program loads a LINE of ``line`` consecutive nodes with one
    vector load and walks inside it, reloading only when a skip leaves the
    line: most steps then cost no memory round trip (the old walk paid one per
    node, ~500 per program at 1e6).

    Parameters
    ----------
    x_ref
        Leaf-major ``(L, W)`` coordinates, as :func:`_leaf_topk_kernel`.
    y_ref
        As ``x_ref``.
    z_ref
        As ``x_ref``.
    dmask_ref
        ``(L, W)`` int32 data lanes.
    qmask_ref
        ``(L, W)`` int32 query lanes.
    slot_ref
        ``(L, W)`` int32 sorted slot of each lane.
    tx_ref
        ``(T, W)`` TARGET (query) block coordinates (x); the source tables
        above again when queries are the data.
    ty_ref
        As ``tx_ref`` (y).
    tz_ref
        As ``tx_ref`` (z).
    tq_ref
        ``(T, W)`` int32: lane is a live query.
    tslot_ref
        ``(T, W)`` int32 source slot of the query itself (self-exclusion).
    seedc_ref
        ``(T,)`` int32 source leaf the seed window is centred on.
    llox_ref
        ``(L,)`` leaf box lower corner (x), for the seed window (``+inf``
        on an empty leaf).
    lloy_ref
        As ``llox_ref`` (y).
    lloz_ref
        As ``llox_ref`` (z).
    lhix_ref
        ``(L,)`` leaf box upper corner (x) (``-inf`` on an empty leaf).
    lhiy_ref
        As ``lhix_ref`` (y).
    lhiz_ref
        As ``lhix_ref`` (z).
    plox_ref
        ``(P,)`` node box lower corner in pre-order, padded by ``line``.
    ploy_ref
        As ``plox_ref``.
    ploz_ref
        As ``plox_ref``.
    phix_ref
        ``(P,)`` node box upper corner in pre-order.
    phiy_ref
        As ``phix_ref``.
    phiz_ref
        As ``phix_ref``.
    pleaf_ref
        ``(P,)`` leaf row of a leaf node, ``-1`` for an internal one.
    psize_ref
        ``(P,)`` subtree size (the skip distance).
    pnlo_ref
        ``(P,)`` first leaf of the subtree.
    pnhi_ref
        ``(P,)`` last leaf of the subtree.
    qid_ref
        ``(L, W)`` int32 output row of each query lane (``-1``: not a query).
    out_d_ref
        **Output** ``(Q * k + tail,)`` squared distances, scattered by query.
    out_i_ref
        **Output**, the matching source slots.
    width
        ``W``. Static.
    k
        Neighbours. Static.
    kpad
        ``Kp``. Static.
    exclude_self
        Skip the query's own slot. Static.
    num_leaves
        ``L``, the SOURCE leaf count. Static.
    end
        Number of nodes (the walk ends there). Static.
    seed
        Seed half-width (Morton-neighbour leaves merged first). Static.
    nout
        ``Q * k``. Static.
    line
        Nodes per line load, a power of two. Static.
    profile
        ``"full"``; ``"seed"`` (seed window only) and ``"nomerge"`` (walk
        without leaf merges) give WRONG results and exist to time the phases.
    merge_impl
        ``"replace"`` (unsorted top-k, replace-the-worst per inserted
        candidate) or ``"insert"`` (sorted register top-k, see
        :func:`_make_insert`). Static.
    box
        Periodic box ``(Lx, Ly, Lz)`` (everything in ``[0, L)``) or ``None``.
        Static.
    unroll
        ``"insert"``: candidates loaded and voted on together (a power of two
        dividing ``W``). Static.
    """
    t = pl.program_id(0)
    q, carry = _lane_state(
        tx_ref, ty_ref, tz_ref, tq_ref, tslot_ref, t, width=width, k=k, kpad=kpad
    )
    if merge_impl == "insert":
        merge = _make_insert(
            x_ref,
            y_ref,
            z_ref,
            dmask_ref,
            slot_ref,
            q,
            width=width,
            k=k,
            exclude_self=exclude_self,
            box=box,
            use_dmask=False,
            unroll=min(int(unroll), width),
        )
        fill = None
        carry = _insert_state(q[3], width=width, k=k)
    else:
        merge, fill = _make_merge(
            x_ref,
            y_ref,
            z_ref,
            dmask_ref,
            slot_ref,
            q,
            width=width,
            kpad=kpad,
            exclude_self=exclude_self,
            box=box,
            use_dmask=False,
        )
    c0 = seedc_ref[t]
    lo = jnp.maximum(c0 - seed, 0)
    hi = jnp.minimum(c0 + seed + 1, num_leaves)
    iota_b = lax.broadcasted_iota(jnp.int32, (line,), 0)
    refs = (
        plox_ref,
        ploy_ref,
        ploz_ref,
        phix_ref,
        phiy_ref,
        phiz_ref,
        pleaf_ref,
        psize_ref,
        pnlo_ref,
        pnhi_ref,
    )

    def _load(n: Any) -> tuple:
        # an index-vector gather (a dynamic-start slice of a 1-D ref does not
        # lower); the arrays are padded by ``line`` so it never runs past
        at = n + iota_b
        return tuple(r[at] for r in refs)

    def _pick(v: Any, sel: Any) -> Any:
        if jnp.issubdtype(v.dtype, jnp.floating):
            return jnp.sum(jnp.where(sel, v, jnp.zeros((), v.dtype)))
        return jnp.sum(jnp.where(sel, v, 0), dtype=jnp.int32)

    def _leaf_need(sc: Any, kt: Any) -> tuple:
        return _need_box(
            q,
            (llox_ref[sc], lloy_ref[sc], lloz_ref[sc]),
            (lhix_ref[sc], lhiy_ref[sc], lhiz_ref[sc]),
            kt,
            box,
        )

    # 1. the seed window, centre-out (c0, c0+1, c0-1, ...): the nearest leaves
    #    first. With Kp == W == k the centre leaf fills the empty top-k
    #    directly, without the per-candidate loop.
    first = jnp.int32(0)
    if fill is not None and kpad == width == k:
        need0, any0 = _leaf_need(c0, carry[2])
        carry = lax.cond(any0, lambda c: fill(c0, need0, c), lambda c: c, carry)
        first = jnp.int32(1)

    def _seed(e: Any, carry: tuple) -> tuple:
        half = (e + 1) // 2
        s = jnp.where(e % 2 == 1, c0 + half, c0 - half)
        ok = (s >= lo) & (s < hi)
        sc = jnp.clip(s, 0, num_leaves - 1)
        need, any_need = _leaf_need(sc, carry[2])
        return lax.cond(ok & any_need, lambda c: merge(sc, need, c), lambda c: c, carry)

    carry = lax.fori_loop(first, jnp.int32(2 * seed + 1), _seed, carry)

    # 2. the packet walk over the pre-order lines
    def _cond(st: tuple) -> Any:
        return st[0] < end

    def _body(st: tuple) -> tuple:
        node, base, lines, bd, bi, kt = st
        reload = (node < base) | (node >= base + line)
        base, lines = lax.cond(
            reload,
            lambda o: (o[0], _load(o[0])),
            lambda o: (o[1], o[2]),
            (node, base, lines),
        )
        sel = iota_b == (node - base)
        lx, ly, lz, hx, hy, hz, leaf, size, nlo, nhi = (_pick(v, sel) for v in lines)
        need, any_need = _need_box(q, (lx, ly, lz), (hx, hy, hz), kt, box)
        # a subtree wholly inside the seed window was merged already
        any_need = any_need & ~((nlo >= lo) & (nhi < hi))
        is_leaf = leaf >= 0
        if profile != "nomerge":
            bd, bi, kt = lax.cond(
                any_need & is_leaf,
                lambda c: merge(jnp.maximum(leaf, 0), need, c),
                lambda c: c,
                (bd, bi, kt),
            )
        nxt = jnp.where(any_need & ~is_leaf, node + 1, node + size)
        return nxt, base, lines, bd, bi, kt

    if profile == "seed":
        best_d, best_i = carry[0], carry[1]
    else:
        zero = jnp.int32(0)
        _, _, _, best_d, best_i, _ = lax.while_loop(
            _cond, _body, (zero, zero, _load(zero), carry[0], carry[1], carry[2])
        )
    qid = qid_ref[t, pl.dslice(0, width)]
    if merge_impl == "insert":
        _scatter_insert(
            out_d_ref, out_i_ref, qid, best_d, best_i, width=width, k=k, nout=nout
        )
        return
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


def leaf_topk_preorder_pallas(
    x: Any,
    y: Any,
    z: Any,
    dmask: Any,
    qmask: Any,
    slot: Any,
    leaf_boxes: tuple,
    preorder: tuple,
    qid: Any,
    *,
    targets: Optional[tuple] = None,
    num_queries: int,
    k: int,
    exclude_self: bool,
    seed: int = 8,
    line: int = 32,
    num_warps: int | None = None,
    num_stages: int = 1,
    interpret: bool = False,
    profile: str = "full",
    merge_impl: str = "replace",
    box: Optional[tuple] = None,
    unroll: int = 4,
) -> tuple[Any, Any]:
    """Exact per-lane top-``k`` by a line-cached pre-order packet traversal.

    Parameters
    ----------
    x
        ``(L, W)`` leaf-major x coordinates (``W`` a power of two).
    y
        As ``x``.
    z
        As ``x``.
    dmask
        ``(L, W)`` int32 data lanes.
    qmask
        ``(L, W)`` int32 query lanes.
    slot
        ``(L, W)`` int32 sorted slot of each lane.
    leaf_boxes
        ``(lox, loy, loz, hix, hiy, hiz)``, each ``(L,)``: leaf AABBs
        (``+inf`` / ``-inf`` on an empty leaf).
    preorder
        ``(lox, loy, loz, hix, hiy, hiz, leaf, size, nlo, nhi)``, each ``(P,)``
        in pre-order (node AABBs, leaf row or ``-1``, subtree size, first and
        last leaf), NOT yet padded.
    qid
        ``(T, W)`` int32 output row of each query lane, ``-1`` elsewhere.
    targets
        ``(tx, ty, tz, tq, tslot, seedc)``: separate query blocks ``(T, W)``
        and the source leaf each one's seed window is centred on ``(T,)``.
        ``None``: the queries are the source lanes with ``qmask`` (one tree).
    num_queries
        ``Q``.
    k
        Neighbours per query.
    exclude_self
        Skip the query's own slot.
    seed
        Morton-neighbour leaves merged first on each side of the target.
    line
        Nodes per line load (a power of two).
    num_warps
        Triton warps per program (``None``: 1 for ``W <= 32``, else 2).
    num_stages
        Triton pipeline stages.
    interpret
        Run in Pallas interpret mode (CPU semantics).
    profile
        Phase timing only (see :func:`_leaf_topk_preorder_kernel`).
    merge_impl
        ``"replace"`` or ``"insert"`` (see :func:`_leaf_topk_preorder_kernel`).
    box
        Periodic box ``(Lx, Ly, Lz)``; every coordinate and box must then lie
        in ``[0, L)``. ``None``: open.
    unroll
        ``"insert"`` only: candidates per group (see
        :func:`_leaf_topk_preorder_kernel`).
    Returns
    -------
    tuple
        ``(d2, idx)``, each ``(Q, k)`` in query order, as
        :func:`leaf_topk_pallas`.

    Raises
    ------
    RuntimeError
        If Pallas could not be imported.
    """
    if pl is None:  # pragma: no cover - environment-dependent
        raise RuntimeError("jax.experimental.pallas is not available")
    num_leaves, width, kpad = _check_inputs(x, qid, k)
    if targets is None:
        targets = (x, y, z, qmask, slot, jnp.arange(num_leaves, dtype=jnp.int32))
    num_targets = int(targets[0].shape[0])
    nodes = int(preorder[0].shape[0])
    if num_warps is None:
        num_warps = 1 if width <= 32 else 2
    line = next_pow2(line)
    nout = int(num_queries) * int(k)
    ntot = nout + max(kpad, int(k)) * width

    def _pad(a: Any, fill: Any, dtype: Any) -> Any:
        return jnp.concatenate([a.astype(dtype), jnp.full((line,), fill, dtype)])

    lox, loy, loz, hix, hiy, hiz, pleaf, psize, pnlo, pnhi = preorder
    padded = (
        _pad(lox, jnp.inf, jnp.float32),
        _pad(loy, jnp.inf, jnp.float32),
        _pad(loz, jnp.inf, jnp.float32),
        _pad(hix, -jnp.inf, jnp.float32),
        _pad(hiy, -jnp.inf, jnp.float32),
        _pad(hiz, -jnp.inf, jnp.float32),
        _pad(pleaf, -1, jnp.int32),
        _pad(psize, 1, jnp.int32),
        _pad(pnlo, -1, jnp.int32),
        _pad(pnhi, -1, jnp.int32),
    )

    def _kernel(*refs: KernelRef) -> None:
        _leaf_topk_preorder_kernel(
            *refs,
            width=width,
            k=int(k),
            kpad=kpad,
            exclude_self=bool(exclude_self),
            num_leaves=num_leaves,
            end=nodes,
            seed=int(seed),
            nout=nout,
            line=line,
            profile=profile,
            merge_impl=merge_impl,
            box=None if box is None else tuple(float(b) for b in box),
            unroll=int(unroll),
        )

    full2 = pl.BlockSpec((num_leaves, width), lambda t: (0, 0))
    tgt2 = pl.BlockSpec((num_targets, width), lambda t: (0, 0))
    tgt1 = pl.BlockSpec((num_targets,), lambda t: (0,))
    leaf1 = pl.BlockSpec((num_leaves,), lambda t: (0,))
    node1 = pl.BlockSpec((nodes + line,), lambda t: (0,))
    flat = pl.BlockSpec((ntot,), lambda t: (0,))
    call = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((ntot,), jnp.float32),
            jax.ShapeDtypeStruct((ntot,), jnp.int32),
        ),
        in_specs=[full2] * 6
        + [tgt2] * 5
        + [tgt1]
        + [leaf1] * 6
        + [node1] * 10
        + [tgt2],
        out_specs=(flat, flat),
        grid=(num_targets,),
        interpret=bool(interpret),
        name=f"leaf_topk_preorder_w{width}_k{int(k)}",
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
        targets[0].astype(jnp.float32),
        targets[1].astype(jnp.float32),
        targets[2].astype(jnp.float32),
        targets[3].astype(jnp.int32),
        targets[4].astype(jnp.int32),
        targets[5].astype(jnp.int32),
        *(a.astype(jnp.float32) for a in leaf_boxes),
        *padded,
        qid.astype(jnp.int32),
    )
    return (
        d2[:nout].reshape(int(num_queries), int(k)),
        idx[:nout].reshape(int(num_queries), int(k)),
    )


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
