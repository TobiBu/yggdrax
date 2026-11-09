"""The mutual dual-tree walk with ONE Pallas launch per wavefront round.

:func:`yggdrax.interactions.dual_tree_walk_mutual` runs each round of the flat
wavefront walk as ~40 XLA kernels (masked MAC over the queue, three prefix-sum
appends, the refinement expansion, a compaction of the next queue). On the
40-level radix tree over Morton-cell leaves at N = 2x10^5 that is 71 rounds
and 11.5 ms -- the launch floor, since the queue is already sized to the peak
wavefront (85.6k pairs).

Here a round is one ``pallas_call``: a program owns a block of ``B`` queue
slots, gathers both nodes' centres, radii and children, evaluates the MAC
(``bh``/``dehnen``: ``(r_a + r_b)^2 <= theta^2 d^2``), and emits

* accepted pairs to the far list, leaf-leaf rejects to the near list,
* the refinement of every other reject (split the larger node; both when
  equal or ``a == b``) -- up to four child pairs -- to the next queue,

each through a per-lane ``atomic_add`` on a counter that hands out slots, so
the lists come out as SETS in a nondeterministic order (the consumers sort
them by target and source, which makes the fused lane deterministic again).
A slot past the buffer's capacity is dropped and recorded in a flag, exactly
as the flat walk's saturation contract requires. Dead nodes (empty padding of
a capacity-padded leaf partition) are masked by ``node_active``.

The pair convention follows the flat walk: canonical pairs ``(lo, hi)``, the
root paired with itself first, near pairs only between two different leaves.
On the same inputs the two walks give the same far and near pair SETS, the same
round count and the same peak wavefront (``tests/test_mutual_walk_pallas.py``).

Native execution needs the Triton lowering on an Ampere-or-later GPU
(:func:`pallas_mutual_walk_supported`); elsewhere pass ``interpret=True`` (CPU
semantics, slow -- for tests) or use the flat walk. ``dual_tree_walk_mutual(...,
backend="auto")`` makes that choice.

Moved here from jaccpot (``jaccpot.pallas.mutual_walk_pallas``, 2026-11), where it
was the fused FMM lane's walk; the pair sets and the compiled program are
unchanged. Its tuning switches, which jaccpot read from the environment, are
keyword arguments here (``node_layout``, ``max_programs``).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from yggdrax._dehnen_accept import WALK_TABLE_POWER0, dehnen_pair_accept
from yggdrax.pallas._compat import (
    KernelRef,
    pallas_backend_kwargs,
    pallas_triton_supported,
)

if TYPE_CHECKING:
    # Under a type checker Pallas is present, so the kernel body's ``pl.*`` and
    # ``plgpu.*`` type-check. At runtime it is optional: without it
    # :func:`pallas_mutual_walk_supported` is False and only the flat walk runs.
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
else:
    try:  # pragma: no cover - import is environment-dependent
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plgpu
    except Exception:  # pragma: no cover - Pallas is optional
        pl = None
        plgpu = None

__all__ = ["PallasWalkResult", "mutual_walk_pallas", "pallas_mutual_walk_supported"]


def pallas_mutual_walk_supported() -> bool:
    """True where the Triton lowering runs (an Ampere-or-later GPU, sm_80+).

    Returns
    -------
    bool
        Whether the kernel can run natively here (otherwise only
        ``interpret=True``).
    """
    return pallas_triton_supported()


class PallasWalkResult(NamedTuple):
    """Far and near pair lists of the walk, padded to their capacities.

    Entries past ``far_count`` / ``near_count`` are NEGATIVE (each buffer has its
    own fill value, so XLA does not merge the identical fills into one buffer
    that it must then copy for the walk's loop carry); read the live prefix.

    Attributes
    ----------
    far_a : Array
        Lower node of each canonical far pair, live prefix ``far_count``.
    far_b : Array
        Upper node of each canonical far pair.
    far_count : Array
        Number of far pairs in the list, at most the capacity.
    near_a : Array
        Lower leaf of each canonical near pair, live prefix ``near_count``.
    near_b : Array
        Upper leaf of each canonical near pair.
    near_count : Array
        Number of near pairs emitted.
    far_overflow : Array
        The far list hit its capacity, so the lists are incomplete.
    near_overflow : Array
        The near list hit its capacity.
    queue_overflow : Array
        The wavefront queue hit its capacity.
    peak_wavefront : Array
        Largest queue occupancy seen.
    rounds : Array
        Rounds executed.
    far_needed : Array
        Far pairs the walk found, stored or not. A far or near overflow does not
        stop the walk, so this is the capacity the list needed -- exact unless
        ``queue_overflow`` (pairs never classified), when it is a lower bound.
    near_needed : Array
        The same for the near list.
    """

    far_a: Array
    far_b: Array
    far_count: Array
    near_a: Array
    near_b: Array
    near_count: Array
    far_overflow: Array
    near_overflow: Array
    queue_overflow: Array
    peak_wavefront: Array
    rounds: Array
    far_needed: Array
    near_needed: Array


# counter slots
#: counter slots: far, near, next-queue, and one overflow flag EACH (an OR of
#: bits under atomic_max would let a later flag erase an earlier one)
_C_FAR, _C_NEAR, _C_NEXT, _C_OVF_FAR, _C_OVF_NEAR, _C_OVF_Q = 0, 1, 2, 3, 4, 5
_NUM_COUNTERS = 6


def _round_kernel(
    qa_ref: KernelRef,
    qb_ref: KernelRef,
    size_ref: KernelRef,
    left_ref: KernelRef,
    right_ref: KernelRef,
    cent_ref: KernelRef,
    rad_ref: KernelRef,
    active_ref: KernelRef,
    theta_ref: KernelRef,
    err_ref: KernelRef,
    far_a_ref: KernelRef,
    far_b_ref: KernelRef,
    near_a_ref: KernelRef,
    near_b_ref: KernelRef,
    next_a_ref: KernelRef,
    next_b_ref: KernelRef,
    counters_ref: KernelRef,
    far_a_out: KernelRef,
    far_b_out: KernelRef,
    near_a_out: KernelRef,
    near_b_out: KernelRef,
    next_a_out: KernelRef,
    next_b_out: KernelRef,
    counters_out: KernelRef,
    *,
    block: int,
    far_cap: int,
    near_cap: int,
    queue_cap: int,
    node_layout: str = "soa",
    use_floor: bool = False,
    error_order: int = 0,
) -> None:
    """One block of the wavefront: MAC, emit, refine.

    Parameters
    ----------
    qa_ref
        Current queue's first node per pair ``[Q]`` (``-1`` past ``size``).
    qb_ref
        Current queue's second node per pair ``[Q]``.
    size_ref
        Live pair count ``[1]``.
    left_ref
        Left child per node ``[nodes]``, ``-1`` for leaves.
    right_ref
        Right child per node ``[nodes]``, ``-1`` for leaves.
    cent_ref
        Centres ``[nodes, 4]`` (padded).
    rad_ref
        MAC radii ``[nodes]``.
    active_ref
        Node activity ``[nodes]`` (``int32`` 0/1).
    theta_ref
        ``[theta^2, separation floor, theta_max, 0]``.
    err_ref
        ``[nodes, W]`` eq (16a) table (layout: :mod:`yggdrax._dehnen_accept`)
        when ``error_order > 0``, ``W`` a power of two; a dummy otherwise.
    far_a_ref
        Far list, first node ``[far_cap]``; aliased to ``far_a_out``.
    far_b_ref
        Far list, second node ``[far_cap]``.
    near_a_ref
        Near list, first leaf ``[near_cap]``.
    near_b_ref
        Near list, second leaf ``[near_cap]``.
    next_a_ref
        Next queue, first node ``[queue_cap]``.
    next_b_ref
        Next queue, second node ``[queue_cap]``.
    counters_ref
        Counters ``[6]``: far, near, next, and one overflow flag each for
        far, near and the queue.
    far_a_out
        Output alias of ``far_a_ref``, written in place.
    far_b_out
        Output alias of ``far_b_ref``.
    near_a_out
        Output alias of ``near_a_ref``.
    near_b_out
        Output alias of ``near_b_ref``.
    next_a_out
        Output alias of ``next_a_ref``.
    next_b_out
        Output alias of ``next_b_ref``.
    counters_out
        Output alias of ``counters_ref``; the atomics target this ref.
    block
        Pairs per program. Static.
    far_cap
        Far list capacity. Static.
    near_cap
        Near list capacity. Static.
    queue_cap
        Next-queue capacity. Static.
    node_layout
        ``"soa"``: the node fields in their own refs. ``"record"``: ``cent_ref`` is
        a ``[nodes, 8]`` float32 record ``(cx, cy, cz, r, left, right, active, 0)``
        with the integers bit-cast in (one 32-byte sector per node instead of
        five), and the other node refs are unused. Static.
    use_floor
        Also require ``d >= r_a + r_b + floor`` (``theta_ref[1]``, the separation
        floor): no particle of an accepted pair closer than the floor to one of the
        other node (the radii are exact centre-of-mass radii). Static.
    error_order
        ``0``: the opening-angle test. ``p > 0``: Dehnen's eq (16a) at order ``p``
        from ``err_ref`` instead (symmetrised; the convergence clause bounds by
        ``theta_ref[2]``); ``theta`` is then unused. Static.

    Returns
    -------
    None
        Writes into the aliased outputs.
    """
    del (
        far_a_ref,
        far_b_ref,
        near_a_ref,
        near_b_ref,
        next_a_ref,
        next_b_ref,
        counters_ref,
    )
    pid = pl.program_id(0)
    size = size_ref[0]
    # Grid-stride over the live blocks: the grid is capped (``num_programs``), so
    # a round over a small queue no longer launches the full queue capacity's
    # programs only for most of them to exit (66 rounds x 131k programs per
    # refresh at 8e6). Which program emits a pair changes its slot, not the set:
    # the deterministic lists sort it away.
    num_programs = pl.num_programs(0)
    live_blocks = (size + (block - 1)) // block
    trips = jnp.maximum(live_blocks - pid + (num_programs - 1), 0) // num_programs

    def _one(it, carry):
        _round_block(
            qa_ref,
            qb_ref,
            size,
            left_ref,
            right_ref,
            cent_ref,
            rad_ref,
            active_ref,
            theta_ref,
            err_ref,
            far_a_out,
            far_b_out,
            near_a_out,
            near_b_out,
            next_a_out,
            next_b_out,
            counters_out,
            pid=pid + it * num_programs,
            block=block,
            far_cap=far_cap,
            near_cap=near_cap,
            queue_cap=queue_cap,
            node_layout=node_layout,
            use_floor=use_floor,
            error_order=error_order,
        )
        return carry

    lax.fori_loop(0, trips, _one, 0)


def _node_rows(ref: KernelRef, nodes: Array) -> list[Array]:
    """Every column of the ``[nodes, W]`` table ``ref`` at rows ``nodes``, one load each.

    ``ref[nodes, k]`` column by column is one scalar gather instruction per column,
    each touching the same 32-byte row of every lane again. ``ref[nodes, :]`` reads
    each node's row as one contiguous vector (Triton issues it as ``W / 4``
    128-bit loads), and the columns are split apart in registers. The values are
    the same loads, so every test on them is bitwise unchanged.

    The walk is bound by these gathers, not by its arithmetic. The walk alone on the
    25M disc+bulge (captured inputs, A100, contended, min of 3 in one process,
    2026-10-17): the geometric MAC with the eq (16a) table merely loaded beside it,
    column by column, 47.6 ms against 22.9 ms without (identical lists); eq (16a)
    52.5 ms with the table read by column, 32.1 ms read by row, same lists. With
    the node record read by row as well: eq (16a) 27.1 ms, the geometric walk
    20.3 ms (22.9 by column), each with its lists unchanged.

    Parameters
    ----------
    ref
        ``[nodes, W]`` table, ``W`` a power of two (Triton's split).
    nodes
        ``[B]`` row per lane (in range).

    Returns
    -------
    list[Array]
        ``W`` arrays of shape ``[B]``, column ``k`` of each lane's row.
    """
    width = int(ref.shape[1])
    rows = ref[nodes, :]
    return [col.reshape(col.shape[0]) for col in jnp.split(rows, width, axis=1)]


def _round_block(
    qa_ref: KernelRef,
    qb_ref: KernelRef,
    size: Array,
    left_ref: KernelRef,
    right_ref: KernelRef,
    cent_ref: KernelRef,
    rad_ref: KernelRef,
    active_ref: KernelRef,
    theta_ref: KernelRef,
    err_ref: KernelRef,
    far_a_out: KernelRef,
    far_b_out: KernelRef,
    near_a_out: KernelRef,
    near_b_out: KernelRef,
    next_a_out: KernelRef,
    next_b_out: KernelRef,
    counters_out: KernelRef,
    *,
    pid: Array,
    block: int,
    far_cap: int,
    near_cap: int,
    queue_cap: int,
    node_layout: str = "soa",
    use_floor: bool = False,
    error_order: int = 0,
) -> None:
    """Body of one non-empty block (see :func:`_round_kernel`).

    A block claims the next-queue slots of all four child pairs with ONE atomic
    (not one per child) and touches the overflow flags only when it overflowed:
    three counter atomics per block instead of nine.

    Parameters
    ----------
    qa_ref
        Current queue's first node per pair ``[Q]``.
    qb_ref
        Current queue's second node per pair ``[Q]``.
    size
        Live pair count, already loaded from the size ref.
    left_ref
        Left child per node ``[nodes]``, ``-1`` for leaves.
    right_ref
        Right child per node ``[nodes]``.
    cent_ref
        Centres ``[nodes, 4]`` (padded).
    rad_ref
        MAC radii ``[nodes]``.
    active_ref
        Node activity ``[nodes]`` (``int32`` 0/1).
    theta_ref
        ``[theta^2, separation floor, theta_max, 0]``.
    err_ref
        ``[nodes, W]`` eq (16a) table when ``error_order > 0``, ``W`` a power of
        two; a dummy otherwise.
    far_a_out
        Far list, first node; written in place.
    far_b_out
        Far list, second node.
    near_a_out
        Near list, first leaf.
    near_b_out
        Near list, second leaf.
    next_a_out
        Next queue, first node.
    next_b_out
        Next queue, second node.
    counters_out
        Counters ``[6]``; the atomics target this ref.
    pid
        This program's index in the grid.
    block
        Pairs per program. Static.
    far_cap
        Far list capacity. Static.
    near_cap
        Near list capacity. Static.
    queue_cap
        Next-queue capacity. Static.
    node_layout
        ``"soa"``: the node fields in their own refs. ``"record"``: ``cent_ref`` is
        a ``[nodes, 8]`` float32 record ``(cx, cy, cz, r, left, right, active, 0)``
        with the integers bit-cast in (one 32-byte sector per node instead of
        five), and the other node refs are unused. Static.
    use_floor
        Also require ``d >= r_a + r_b + floor`` (``theta_ref[1]``, the separation
        floor): no particle of an accepted pair closer than the floor to one of the
        other node (the radii are exact centre-of-mass radii). Static.
    error_order
        ``0``: the opening-angle test; ``p > 0``: eq (16a) from ``err_ref``. Static.

    Returns
    -------
    None
        Writes into the aliased outputs.
    """
    lane = (pid * block + lax.broadcasted_iota(jnp.int32, (block,), 0)).astype(
        jnp.int32
    )
    in_range = lane < size
    lane_safe = jnp.where(in_range, lane, jnp.zeros_like(lane))
    a = qa_ref[lane_safe]
    b = qb_ref[lane_safe]
    live = in_range & (a >= 0) & (b >= 0)
    a_s = jnp.where(live, a, jnp.zeros_like(a))
    b_s = jnp.where(live, b, jnp.zeros_like(b))
    if node_layout == "record":
        # each node's record in one vector load (_node_rows), not seven gathers
        rec_a = _node_rows(cent_ref, a_s)
        rec_b = _node_rows(cent_ref, b_s)

        def _int(col):
            return lax.bitcast_convert_type(col, jnp.int32)

        live = live & (_int(rec_a[6]) > 0) & (_int(rec_b[6]) > 0)
        la = _int(rec_a[4])
        lb = _int(rec_b[4])
        ra = _int(rec_a[5])
        rb = _int(rec_b[5])
        rad_a = rec_a[3]
        rad_b = rec_b[3]
        cen_a = rec_a[:3]
        cen_b = rec_b[:3]
    else:
        live = live & (active_ref[a_s] > 0) & (active_ref[b_s] > 0)
        la = left_ref[a_s]
        lb = left_ref[b_s]
        ra = right_ref[a_s]
        rb = right_ref[b_s]
        rad_a = rad_ref[a_s]
        rad_b = rad_ref[b_s]
        cen_a = [cent_ref[a_s, k] for k in range(3)]
        cen_b = [cent_ref[b_s, k] for k in range(3)]
    dx = cen_b[0] - cen_a[0]
    dy = cen_b[1] - cen_a[1]
    dz = cen_b[2] - cen_a[2]
    d2 = dx * dx + dy * dy + dz * dz
    same = a_s == b_s
    rsum = rad_a + rad_b
    if error_order > 0:
        # eq (16a), both directions, from the two nodes' table rows (one vector
        # load per row, _node_rows); like the policy, no d > 0 guard (the
        # convergence clause fails at d = 0)
        cols = WALK_TABLE_POWER0 + int(error_order)
        accept = (
            live
            & (~same)
            & dehnen_pair_accept(
                row_a=_node_rows(err_ref, a_s)[:cols],
                row_b=_node_rows(err_ref, b_s)[:cols],
                radius_a=rad_a,
                radius_b=rad_b,
                dist_sq=d2,
                order=int(error_order),
                theta_max=theta_ref[2],
            )
        )
    else:
        theta_sq = theta_ref[0]
        accept = live & (~same) & (d2 > 0.0) & (rsum * rsum <= theta_sq * d2)
    if use_floor:
        reach = rsum + theta_ref[1]
        accept = accept & (reach * reach <= d2)
    a_leaf = la < 0
    b_leaf = lb < 0
    both_leaf = a_leaf & b_leaf
    is_near = live & (~accept) & both_leaf & (~same)
    refine = live & (~accept) & (~both_leaf)
    split_a = refine & (~a_leaf) & (same | b_leaf | (rad_a >= rad_b))
    split_b = refine & (~b_leaf) & (same | a_leaf | (rad_b > rad_a))
    both = split_a & split_b
    only_a = split_a & (~split_b)
    only_b = split_b & (~split_a)
    zero = jnp.zeros_like(lane)
    one = jnp.ones_like(lane)
    neg1 = jnp.full_like(lane, -1)

    def emit(
        mask: Array,
        va: Array,
        vb: Array,
        out_a: KernelRef,
        out_b: KernelRef,
        counter: int,
        cap: int,
        base: Optional[Array] = None,
    ) -> Array:
        # One atomic per program: the block claims sum(mask) slots and hands them
        # out by an in-block exclusive prefix sum (no per-lane counter contention).
        # A given ``base``: the slots were claimed already (the four child pairs
        # share one claim).
        inc = jnp.where(mask, one, zero).astype(jnp.int32)
        if base is None:
            total = jnp.sum(inc).astype(jnp.int32)
            # the stubs type the returned old value as optional; it is not
            base = cast(
                Array,
                plgpu.atomic_add(
                    counters_out, (jnp.asarray(counter, jnp.int32),), total
                ),
            )
        slot = (base + jnp.cumsum(inc) - inc).astype(jnp.int32)
        ok = mask & (slot < cap)
        over = mask & (slot >= cap)
        lo = jnp.minimum(va, vb)
        hi = jnp.maximum(va, vb)
        # Masked-out lanes point PAST the buffer (unique, out of range): the
        # Triton store never touches them and the interpreter drops them, whereas
        # a shared in-range dummy index collides with a real write to that slot.
        where = jnp.where(ok, slot, (cap + lane).astype(jnp.int32))
        plgpu.store(out_a.at[where], lo.astype(jnp.int32), mask=ok)
        plgpu.store(out_b.at[where], hi.astype(jnp.int32), mask=ok)
        return over

    over_far = emit(accept, a_s, b_s, far_a_out, far_b_out, _C_FAR, far_cap)
    over_near = emit(is_near, a_s, b_s, near_a_out, near_b_out, _C_NEAR, near_cap)
    # refinement: up to four child pairs per lane (the flat walk's five cases)
    # case split both & same:   (la,la) (la,ra) (ra,ra)
    # case split both & cross:  (la,lb) (la,rb) (ra,lb) (ra,rb)
    # case split a only:        (la,b) (ra,b)
    # case split b only:        (a,lb) (a,rb)
    c0a = jnp.where(both & same, la, jnp.where(both, la, jnp.where(only_a, la, a_s)))
    c0b = jnp.where(both & same, la, jnp.where(both, lb, jnp.where(only_a, b_s, lb)))
    c1a = jnp.where(both & same, la, jnp.where(both, la, jnp.where(only_a, ra, a_s)))
    c1b = jnp.where(both & same, ra, jnp.where(both, rb, jnp.where(only_a, b_s, rb)))
    c2a = jnp.where(both & same, ra, jnp.where(both, ra, neg1))
    c2b = jnp.where(both & same, ra, jnp.where(both, lb, neg1))
    c3a = jnp.where(both & (~same), ra, neg1)
    c3b = jnp.where(both & (~same), rb, neg1)
    over_q = jnp.zeros_like(live)
    children = ((c0a, c0b), (c1a, c1b), (c2a, c2b), (c3a, c3b))
    masks = [refine & (ca >= 0) & (cb >= 0) for ca, cb in children]
    # one claim for all four children: child k's slots follow children < k's
    sums = [jnp.sum(jnp.where(m, one, zero)).astype(jnp.int32) for m in masks]
    base = plgpu.atomic_add(
        counters_out,
        (jnp.asarray(_C_NEXT, jnp.int32),),
        sums[0] + sums[1] + sums[2] + sums[3],
    )
    for (ca, cb), m, n_k in zip(children, masks, sums):
        over_q = over_q | emit(
            m, ca, cb, next_a_out, next_b_out, _C_NEXT, queue_cap, base=base
        )
        base = base + n_k
    for slot_id, over in (
        (_C_OVF_FAR, over_far),
        (_C_OVF_NEAR, over_near),
        (_C_OVF_Q, over_q),
    ):
        flag = jnp.max(over.astype(jnp.int32))

        @pl.when(flag > 0)
        def _raise(slot_id=slot_id, flag=flag):
            plgpu.atomic_max(counters_out, (jnp.asarray(slot_id, jnp.int32),), flag)


def _full(arr: Array) -> Any:
    shp = tuple(arr.shape)
    return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))


def mutual_walk_pallas(
    left_child_full: Array,
    right_child_full: Array,
    centers: Array,
    radii: Array,
    theta: float,
    root: Array,
    *,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
    node_active: Optional[Array] = None,
    block: int = 64,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 2,
    max_rounds: int = 256,
    rounds_per_check: int = 16,
    seed_a: Optional[Array] = None,
    seed_b: Optional[Array] = None,
    seed_count: Optional[Array] = None,
    node_layout: str = "record",
    max_programs: int = 2048,
    separation_floor: float = 0.0,
    error_table: Optional[Array] = None,
    error_order: int = 0,
    theta_max: float = 1.0,
) -> PallasWalkResult:
    """Run the mutual walk, one Pallas launch per round.

    Same contract as ``dual_tree_walk_mutual`` (``mac_type="dehnen"`` / ``"bh"``
    semantics: non-strict ``(r_a + r_b)^2 <= theta^2 d^2``), lists returned as
    sets in atomic emission order.

    Parameters
    ----------
    left_child_full
        ``[nodes]`` left children, ``-1`` for leaves.
    right_child_full
        ``[nodes]`` right children, ``-1`` for leaves.
    centers
        ``[nodes, 3]`` MAC centres.
    radii
        ``[nodes]`` MAC radii.
    theta
        Opening angle.
    root
        Root node id (scalar).
    max_pair_queue
        Queue capacity (also the block-padded grid). Static.
    far_cap
        Far list capacity. Static.
    near_cap
        Near list capacity. Static.
    node_active
        ``[nodes]`` bool; ``None`` = all active.
    block
        Pairs per program (power of two). Static.
    interpret
        Pallas interpret mode.
    backend
        Pallas GPU lowering.
    num_warps
        Warps per program.
    max_rounds
        Safety bound on the round loop.
    rounds_per_check
        Rounds launched per ``while_loop`` iteration: every iteration costs a
        device-to-host copy of the predicate AND a copy of every carried buffer
        (XLA preserves the loop state around the aliased pallas_call: 19 D2D
        memcpys per iteration at N=2e5), so many rounds run between checks; an
        empty round is one early-exiting launch plus a few scalar ops.
    seed_a
        ``[K]`` first nodes of the starting pairs in place of ``(root, root)``,
        ``-1`` for dead slots (the walk filters them). ``K <= max_pair_queue``. This
        is what makes the kernel a ONE-SIDED walk: a node with no children (an
        imported row, a summary cell) is never split, so only the other side refines
        -- the same contract as ``dual_tree_walk_mutual``'s seeds, used by the
        multi-GPU export and receiver walks. Pairs come out as ``(min, max)``, so a
        block indexed above the local nodes always lands in ``*_b``.
    seed_b
        ``[K]`` second nodes of the starting pairs; given together with ``seed_a``.
    seed_count
        Live prefix of the seed (a width, not a filter: dead slots past it are
        never read). ``None`` uses ``K``.
    node_layout
        How a round reads the nodes: ``"soa"`` (left, right, radius, activity
        and centre each from its own array: five 32-byte sectors per node) or
        ``"record"`` (one 32-byte record per node, built once per walk: 12 more
        bytes per node than the padded centres it replaces). The same MAC on the
        same values, so the same pair sets. Float32 centres only (otherwise
        ``"soa"``). Static. ``"record"`` (default since 2026-10-06) together
        with the fused emission (one counter atomic for the four child pairs of
        a block) took the walk alone (A100) from 286 to 96 ms at 1e8 particles
        and from ~31 to 15 ms at 8e6 (record alone 248 ms, fused emission alone
        167 ms at 1e8). A round reads the record as one vector per node (two
        128-bit loads).
    max_programs
        Cap on the programs (grid size) of one round's launch: enough to fill
        the card, and the kernel strides over the rest of the queue's blocks.
        Which program emits a pair changes its slot, not the sets. Static.
    separation_floor
        Accept a pair only if also ``|c_b - c_a| >= r_a + r_b + separation_floor``,
        so no far interaction acts between particles closer than the floor (the
        far field is the unsoftened expansion; the floor keeps it out of the
        softening's reach). ``0`` (default): no floor, the kernel unchanged. Static.
    error_table
        ``[nodes, W]`` table of Dehnen's eq (16a) inputs, traced: ``G M``,
        threshold and normalised multipole powers per node, in the column layout
        of :mod:`yggdrax._dehnen_accept` (``WALK_TABLE_*``,
        :func:`~yggdrax._dehnen_accept.walk_table_width`). Required when
        ``error_order > 0``.
    error_order
        ``0`` (default): the opening-angle test. ``p > 0``: accept by eq (16a) at
        expansion order ``p`` instead -- the per-pair error estimate of eq (15),
        symmetrised, against each node's threshold -- and ``theta`` is unused. The
        near and refinement rules are unchanged. Static.
    theta_max
        The eq (16a) convergence clause, ``rho_a + rho_b < theta_max d`` (the
        paper's 1). Read only with ``error_order > 0``.

    Returns
    -------
    PallasWalkResult
        The lists, counts and flags. ``queue_overflow`` is also raised when the
        walk stops at ``max_rounds`` with pairs still queued -- those pairs were
        never classified, so the lists are incomplete. A far or near overflow does
        not stop the walk: ``far_needed`` / ``near_needed`` then hold the capacity
        the list needed.

    Raises
    ------
    ValueError
        If the seed is longer than ``max_pair_queue`` or only one half is given,
        ``node_layout`` is not ``"soa"`` or ``"record"``, ``max_programs < 1``,
        or ``error_order > 0`` comes without an ``error_table`` wide enough for
        it.
    """
    if (seed_a is None) != (seed_b is None):
        raise ValueError("seed_a and seed_b go together")
    if int(error_order) > 0:
        if error_table is None:
            raise ValueError("error_order > 0 needs an error_table")
        if int(jnp.asarray(error_table).shape[1]) < WALK_TABLE_POWER0 + int(
            error_order
        ):
            raise ValueError(
                f"error_table has {int(jnp.asarray(error_table).shape[1])} columns, "
                f"order {int(error_order)} needs {WALK_TABLE_POWER0 + int(error_order)}"
            )
    if seed_a is not None:
        K = int(jnp.asarray(seed_a).shape[0])
        if K > int(max_pair_queue):
            raise ValueError(
                f"seed of {K} pairs exceeds max_pair_queue={int(max_pair_queue)}"
            )
    if int(max_programs) < 1:
        raise ValueError(f"max_programs must be >= 1, got {max_programs!r}")
    if node_layout not in ("soa", "record"):
        raise ValueError(f"node_layout must be 'soa' or 'record', got {node_layout!r}")
    if jnp.asarray(centers).dtype != jnp.float32:
        node_layout = "soa"
    # One jit around the whole walk, so an EAGER call creates the list and queue
    # buffers inside the program and the loop updates them in place. Called op by
    # op, the initial buffers were arguments of the while loop and stayed alive
    # beside its outputs: twice the walk's memory at the peak. Under a trace the
    # jit simply inlines.
    #
    # Every output is an index, a count or a flag, so no derivative flows through
    # the walk; stopping the float inputs' gradients says so to JAX, which would
    # otherwise try to differentiate the pallas_call (and refuse: it aliases its
    # buffers) whenever the geometry carries a tangent, as it does when the walk
    # runs inside a differentiated function. Exact, and the identity in the
    # compiled program.
    stop = jax.lax.stop_gradient
    return _mutual_walk_jit(
        left_child_full,
        right_child_full,
        stop(centers),
        stop(radii),
        root,
        node_active,
        seed_a,
        seed_b,
        seed_count,
        stop(error_table) if int(error_order) > 0 else None,
        theta=float(theta),
        max_pair_queue=int(max_pair_queue),
        far_cap=int(far_cap),
        near_cap=int(near_cap),
        block=int(block),
        interpret=bool(interpret),
        backend=str(backend),
        num_warps=int(num_warps),
        max_rounds=int(max_rounds),
        rounds_per_check=int(rounds_per_check),
        node_layout=str(node_layout),
        max_programs=int(max_programs),
        separation_floor=float(separation_floor),
        error_order=int(error_order),
        theta_max=float(theta_max),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "theta",
        "max_pair_queue",
        "far_cap",
        "near_cap",
        "block",
        "interpret",
        "backend",
        "num_warps",
        "max_rounds",
        "rounds_per_check",
        "node_layout",
        "max_programs",
        "separation_floor",
        "error_order",
        "theta_max",
    ),
)
@jax.named_scope("mutual_walk_pallas")
def _mutual_walk_jit(
    left_child_full: Array,
    right_child_full: Array,
    centers: Array,
    radii: Array,
    root: Array,
    node_active: Optional[Array],
    seed_a: Optional[Array],
    seed_b: Optional[Array],
    seed_count: Optional[Array],
    error_table: Optional[Array],
    *,
    theta: float,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
    block: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    max_rounds: int,
    rounds_per_check: int,
    node_layout: str = "soa",
    max_programs: int = 2048,
    separation_floor: float = 0.0,
    error_order: int = 0,
    theta_max: float = 1.0,
) -> PallasWalkResult:
    """The body of :func:`mutual_walk_pallas` (validated arguments, static sizes).

    Parameters
    ----------
    left_child_full
        ``[nodes]`` left children.
    right_child_full
        ``[nodes]`` right children.
    centers
        ``[nodes, 3]`` MAC centres.
    radii
        ``[nodes]`` MAC radii.
    root
        Root node id.
    node_active
        ``[nodes]`` liveness mask.
    seed_a
        Seed pairs, first nodes.
    seed_b
        Seed pairs, second nodes.
    seed_count
        Live seed prefix.
    error_table
        The eq (16a) table, or ``None`` with ``error_order == 0``.
    theta
        Opening angle.
    max_pair_queue
        Queue capacity.
    far_cap
        Far list capacity.
    near_cap
        Near list capacity.
    block
        Pairs per program.
    interpret
        Pallas interpret mode.
    backend
        Pallas GPU lowering.
    num_warps
        Warps per program.
    max_rounds
        Safety bound on the round loop.
    rounds_per_check
        Rounds per ``while_loop`` iteration.
    node_layout
        ``"soa"`` or ``"record"`` (see :func:`mutual_walk_pallas`).
    max_programs
        Programs per round (see :func:`mutual_walk_pallas`).
    separation_floor
        See :func:`mutual_walk_pallas`. Static.
    error_order
        See :func:`mutual_walk_pallas`. Static.
    theta_max
        See :func:`mutual_walk_pallas`. Static.

    Returns
    -------
    PallasWalkResult
        As :func:`mutual_walk_pallas`.
    """
    idx = jnp.int32
    nodes = int(left_child_full.shape[0])
    left = jnp.asarray(left_child_full, idx)
    right = jnp.asarray(right_child_full, idx)
    dtype = centers.dtype
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    rad = jnp.asarray(radii, dtype)
    active = (
        jnp.ones((nodes,), idx)
        if node_active is None
        else jnp.asarray(node_active).astype(idx)
    )
    # [theta^2, separation floor, theta_max, 0]: the kernel reads the floor only
    # with use_floor and theta_max only with error_order > 0
    theta_sq = jnp.asarray(
        [float(theta) ** 2, float(separation_floor), float(theta_max), 0.0], dtype
    )
    if error_order > 0:
        assert error_table is not None  # checked by the wrapper
        err = jnp.asarray(error_table, dtype)
        # the kernel reads each row as one vector and splits it (_node_rows):
        # Triton splits power-of-two widths only (dehnen_walk_table's already are)
        width = int(err.shape[1])
        if width & (width - 1):
            err = jnp.pad(err, ((0, 0), (0, (1 << width.bit_length()) - width)))
    else:
        err = jnp.zeros((1, 8), dtype)
    if node_layout == "record":
        # one 32-byte sector per node: (cx, cy, cz, r, left, right, active, 0)
        def _as_f32(x: Array) -> Array:
            return lax.bitcast_convert_type(x.astype(idx), jnp.float32)[:, None]

        cent = jnp.concatenate(
            [
                jnp.asarray(centers, jnp.float32),
                rad.astype(jnp.float32)[:, None],
                _as_f32(left),
                _as_f32(right),
                _as_f32(active),
                jnp.zeros((nodes, 1), jnp.float32),
            ],
            axis=1,
        )
        tiny = jnp.zeros((1,), idx)
        left = right = active = tiny
        rad = jnp.zeros((1,), dtype)
    Q = int(max_pair_queue)
    blk = int(block)
    # programs per round: enough to fill the card, the kernel strides over the rest
    grid = min((Q + blk - 1) // blk, int(max_programs))
    backend_kwargs = pallas_backend_kwargs(
        backend, interpret=bool(interpret), num_warps=int(num_warps)
    )
    kernel = functools.partial(
        _round_kernel,
        block=blk,
        far_cap=int(far_cap),
        near_cap=int(near_cap),
        queue_cap=Q,
        node_layout=node_layout,
        use_floor=float(separation_floor) > 0.0,
        error_order=int(error_order),
    )

    def one_round(
        qa: Array,
        qb: Array,
        size: Array,
        far_a: Array,
        far_b: Array,
        near_a: Array,
        near_b: Array,
        next_a: Array,
        next_b: Array,
        counters: Array,
    ) -> tuple[Array, ...]:  # pallas_call returns a tuple, not a list
        operands = [
            qa,
            qb,
            size,
            left,
            right,
            cent,
            rad,
            active,
            theta_sq,
            err,
            far_a,
            far_b,
            near_a,
            near_b,
            next_a,
            next_b,
            counters,
        ]
        outs = [far_a, far_b, near_a, near_b, next_a, next_b, counters]
        n_in = len(operands)
        alias = {n_in - 7 + k: k for k in range(7)}
        return pl.pallas_call(
            kernel,
            grid=(grid,),
            in_specs=[_full(o) for o in operands],
            out_specs=[_full(o) for o in outs],
            out_shape=[jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outs],
            input_output_aliases=alias,
            interpret=bool(interpret),
            name=f"mutual_walk_round_b{blk}",
            **backend_kwargs,
        )(*operands)

    # Every carried buffer gets its OWN fill (all negative: dead). Identical
    # fills are one broadcast to XLA, which then copies it into each loop-carry
    # slot and keeps it alive for later -1 selects: at 1.28e8 particles a 1.4 GiB
    # far buffer plus two copies of it at the walk's start.
    if seed_a is None:
        qa0 = jnp.full((Q,), -11, idx).at[0].set(jnp.asarray(root, idx))
        qb0 = jnp.full((Q,), -12, idx).at[0].set(jnp.asarray(root, idx))
        size0 = jnp.asarray(1, idx)
    else:
        assert seed_b is not None  # checked above; for the type checker
        K = int(jnp.asarray(seed_a).shape[0])
        qa0 = jnp.full((Q,), -11, idx).at[:K].set(jnp.asarray(seed_a, idx))
        qb0 = jnp.full((Q,), -12, idx).at[:K].set(jnp.asarray(seed_b, idx))
        size0 = (
            jnp.asarray(K, idx)
            if seed_count is None
            else jnp.minimum(jnp.asarray(seed_count, idx), jnp.asarray(K, idx))
        )
    init = (
        qa0,
        qb0,
        # the spare queue pair the next round writes into (double buffering)
        jnp.full((Q,), -13, idx),
        jnp.full((Q,), -14, idx),
        size0,
        jnp.full((int(far_cap),), -15, idx),
        jnp.full((int(far_cap),), -16, idx),
        jnp.full((int(near_cap),), -17, idx),
        jnp.full((int(near_cap),), -18, idx),
        jnp.zeros((_NUM_COUNTERS,), idx),  # far, near, next, overflow far/near/queue
        size0,  # peak
        jnp.asarray(0, idx),  # rounds
    )

    def cond(state: tuple[Array, ...]) -> Array:
        _qa, _qb, _sa, _sb, size, *_rest, counters, _peak, rounds = state
        # Only a QUEUE overflow stops the walk: pairs dropped from the queue are
        # never classified. A full far or near list drops what does not fit and
        # keeps counting, so the walk reports the capacity it needed and an eager
        # caller can size the list in one retry instead of a doubling ladder.
        return (size > 0) & (counters[_C_OVF_Q] == 0) & (rounds < int(max_rounds))

    def one_step(state: tuple[Array, ...]) -> tuple[Array, ...]:
        (
            qa,
            qb,
            spare_a,
            spare_b,
            size,
            far_a,
            far_b,
            near_a,
            near_b,
            counters,
            peak,
            rounds,
        ) = state
        # The next queue goes into the spare pair (the kernel reads only lanes <
        # size, so neither needs a fill) and this round's queue becomes the next
        # spare. A fresh ``jnp.empty`` per round cost a full-queue copy per round
        # (XLA merged the two identical broadcasts into one buffer, and an aliased
        # output cannot share it): 33 MB x ~2.4 per round at 8e6, 6.5 ms per step.
        # After an even number of rounds every carry slot is back in its own
        # buffer, so the while body needs no copy either.
        counters = counters.at[_C_NEXT].set(0)
        far_a, far_b, near_a, near_b, next_a, next_b, counters = one_round(
            qa, qb, size[None], far_a, far_b, near_a, near_b, spare_a, spare_b, counters
        )
        new_size = counters[_C_NEXT]
        peak = jnp.maximum(peak, new_size)
        # an empty round leaves everything untouched, so the count stays honest
        rounds = rounds + jnp.where(size > 0, 1, 0).astype(idx)
        return (
            next_a,
            next_b,
            qa,
            qb,
            jnp.minimum(new_size, Q),
            far_a,
            far_b,
            near_a,
            near_b,
            counters,
            peak,
            rounds,
        )

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        for _ in range(max(int(rounds_per_check), 1)):
            state = one_step(state)
        return state

    (
        qa,
        qb,
        _sa,
        _sb,
        size,
        far_a,
        far_b,
        near_a,
        near_b,
        counters,
        peak,
        rounds,
    ) = lax.while_loop(cond, body, init)
    # Stopping at max_rounds with pairs still queued leaves them unclassified. A
    # queue overflow also stops the loop with pairs queued and is flagged already.
    unfinished = (size > 0) & (counters[_C_OVF_Q] == 0)
    return PallasWalkResult(
        far_a=far_a,
        far_b=far_b,
        far_count=jnp.minimum(counters[_C_FAR], int(far_cap)),
        near_a=near_a,
        near_b=near_b,
        near_count=jnp.minimum(counters[_C_NEAR], int(near_cap)),
        far_overflow=counters[_C_OVF_FAR] > 0,
        near_overflow=counters[_C_OVF_NEAR] > 0,
        queue_overflow=(counters[_C_OVF_Q] > 0) | unfinished,
        peak_wavefront=peak,
        rounds=rounds,
        far_needed=counters[_C_FAR],
        near_needed=counters[_C_NEAR],
    )
