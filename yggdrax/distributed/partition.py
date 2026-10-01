"""Space-filling-curve domain decomposition for Yggdrax multi-GPU (Phase 1).

Redistributes particles across GPUs so that each device ends up owning a
*contiguous Morton-code range* -- a spatial domain -- following jztree's
``distr_zsort``. The core is a distributed **sample sort**: every device
Morton-sorts its shard locally, a small set of samples is gathered to choose
``ndev-1`` splitters, and particles are routed to their owning device with the
ragged all-to-all from :mod:`yggdrax.distributed.comm`.

Two balancing modes:

* the sample sort alone gives contiguous, *approximately* balanced domains and
  can *snap* domain boundaries to coarse Morton cells (``align_level``) so a
  top-level tree node never straddles two GPUs (jztree's
  ``adjust_domain_for_nodesize``);
* ``equalize`` adds an exact rank-based rebalance pass (each device gets
  ``floor``/``ceil`` of ``N/ndev`` particles) at the cost of breaking cell
  alignment.

Everything runs inside ``jax.shard_map`` over the mesh axis and keeps static
buffer shapes (padded to ``output_capacity``) with a dynamic valid ``count``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..morton import morton_encode_impl
from .comm import _COUNT_DTYPE, _exclusive_cumsum, ragged_all_to_all_exchange
from .sharding import AXIS_NAME

# Padding sentinel for Morton codes: sorts padding rows to the tail so the
# valid (leading ``count``) rows stay contiguous after a re-sort. Built as a
# uint64 scalar so it never gets parsed as a (overflowing) weak int64.
_CODE_SENTINEL = jnp.uint64((1 << 64) - 1)


@dataclass
class ShardedDomain:
    """Global (host-side) view of a decomposed particle set.

    Leaves are the concatenation of every device's padded shard:
    ``positions``/``masses``/``codes`` have leading dim ``ndev * capacity``;
    ``counts[g]`` is the number of valid leading rows in device ``g``'s shard.
    """

    positions: Array
    masses: Array
    codes: Array
    counts: Array


def global_bounds(
    positions_local: Array,
    *,
    axis_name: str = AXIS_NAME,
    pad: float = 1e-6,
) -> tuple[Array, Array]:
    """Collective axis-aligned bounding box over all devices' particles."""

    lo = jnp.min(positions_local, axis=0)
    hi = jnp.max(positions_local, axis=0)
    los = jax.lax.all_gather(lo, axis_name, tiled=False)
    his = jax.lax.all_gather(hi, axis_name, tiled=False)
    gmin = jnp.min(los, axis=0)
    gmax = jnp.max(his, axis=0)
    span = jnp.where(gmax > gmin, gmax - gmin, 1.0)
    return gmin - span * pad, gmax + span * pad


def _align_pivots(pivots: Array, align_level: Optional[int]) -> Array:
    """Snap splitter codes down to a level-``align_level`` Morton cell edge."""

    if align_level is None:
        return pivots
    shift = max(0, 63 - 3 * int(align_level))
    if shift == 0:
        return pivots
    mask = (~jnp.uint64(0)) << jnp.uint64(shift)
    return pivots & mask


def _choose_pivots(
    codes_sorted: Array,
    ndev: int,
    num_samples: int,
    axis_name: str,
    align_level: Optional[int],
    count: Optional[Array] = None,
) -> Array:
    """Sample-based splitter selection: returns ``ndev-1`` ascending pivots.

    ``count`` restricts sampling to the live prefix. Sampling a padded shard at even
    ranks over the whole array draws mostly the padding sentinel, which drags every
    pivot to the top of the code range and sends the real particles to one device.
    """

    n = codes_sorted.shape[0] if count is None else count
    idx = (jnp.arange(num_samples) * jnp.asarray(n)) // num_samples
    samples = codes_sorted[idx]
    total = num_samples * ndev
    if count is not None:
        # A device with NO live particle samples its padding sentinel; those sort
        # last, so they are excluded by ranking over the live devices' samples only.
        # Bit-identical to the plain form whenever every device holds particles.
        live_dev = (jnp.asarray(count) > 0).astype(_COUNT_DTYPE)
        samples = jnp.where(live_dev > 0, samples, _CODE_SENTINEL)
        total = num_samples * jnp.maximum(jax.lax.psum(live_dev, axis_name), 1)
    all_samples = jnp.sort(jax.lax.all_gather(samples, axis_name, tiled=True))
    piv_idx = (jnp.arange(1, ndev) * total) // ndev
    return _align_pivots(all_samples[piv_idx], align_level)


def _tree_take(payload: Any, order: Array) -> Any:
    """Apply one row permutation to every leaf of a payload pytree."""
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[order], payload)


def _leaf_fills(payload: Any, payload_fill: Any) -> list:
    """One fill value per payload leaf: ``None`` -> -1 everywhere, a scalar -> that
    value everywhere, else a pytree with the payload's structure."""
    leaves = jax.tree_util.tree_leaves(payload)
    if payload_fill is None:
        return [-1] * len(leaves)
    fills = jax.tree_util.tree_leaves(payload_fill)
    if len(fills) == 1 and len(leaves) != 1:
        return fills * len(leaves)
    if len(fills) != len(leaves):
        raise ValueError(
            f"payload_fill has {len(fills)} leaves, the payload has {len(leaves)}"
        )
    return fills


def _route_payload(
    payload: Any,
    send_sizes: Array,
    output_capacity: int,
    axis_name: str,
    payload_fill: Any = None,
) -> Any:
    """Ragged-exchange a payload PYTREE: one round per (dtype, fill) group.

    Leaves keep their own dtype end to end -- the exchange never casts -- so an int32
    global id is exact however large, which a float32 column packed beside velocities
    is not beyond 2^24. Leaves sharing a dtype and a fill are concatenated into one
    ``(rows, k)`` block, so velocities + ids cost two rounds, not one per leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    fills = _leaf_fills(payload, payload_fill)
    flat = [
        jnp.asarray(leaf).reshape(jnp.asarray(leaf).shape[0], -1) for leaf in leaves
    ]
    groups: dict = {}
    for i, (leaf, fill) in enumerate(zip(flat, fills)):
        groups.setdefault((jnp.dtype(leaf.dtype).name, float(fill)), []).append(i)
    out: list = [None] * len(leaves)
    for (_dtype, fill), members in groups.items():
        block = jnp.concatenate([flat[i] for i in members], axis=1)
        routed, _, _ = ragged_all_to_all_exchange(
            block,
            send_sizes,
            output_capacity=output_capacity,
            axis_name=axis_name,
            fill_value=fill,
        )
        col = 0
        for i in members:
            width = flat[i].shape[1]
            piece = routed[:, col : col + width]
            col += width
            shape = (output_capacity,) + tuple(jnp.asarray(leaves[i]).shape[1:])
            out[i] = piece.reshape(shape).astype(jnp.asarray(leaves[i]).dtype)
    return jax.tree_util.tree_unflatten(treedef, out)


def _resort_by_code(positions, masses, codes, count, payload=None):
    """Re-sort a padded shard by Morton code (padding sentinel -> tail).

    ``payload`` rides the SAME permutation, which is the whole point of routing it
    here rather than reconstructing it afterwards: two particles may share a Morton
    code, so a caller re-deriving the order from codes alone cannot recover which row
    went where.
    """

    order = jnp.argsort(codes)
    positions = positions[order]
    masses = masses[order]
    codes = codes[order]
    cap = codes.shape[0]
    valid = jnp.arange(cap) < count
    code_lo = jnp.min(jnp.where(valid, codes, _CODE_SENTINEL))
    code_hi = jnp.max(jnp.where(valid, codes, jnp.uint64(0)))
    return (
        positions,
        masses,
        codes,
        code_lo,
        code_hi,
        None if payload is None else _tree_take(payload, order),
    )


def _route(
    positions,
    masses,
    codes,
    send_sizes,
    output_capacity,
    axis_name,
    payload=None,
    payload_fill=None,
):
    """Ragged-exchange a shard already grouped by destination device.

    ``payload`` is routed as a fourth round so that whatever identifies a particle
    travels WITH it. Reconstructing it afterwards from a host-side array is the
    documented way to get this wrong: the host order and the shard order coincide
    only while ``capacity == count``, and `docs/distributed_padding_force_defect.md`
    records what that looks like once they stop -- "plausible, smooth, and wrong by
    tens of percent".
    """

    pos_out, recv_sizes, _ = ragged_all_to_all_exchange(
        positions, send_sizes, output_capacity=output_capacity, axis_name=axis_name
    )
    mass_out, _, _ = ragged_all_to_all_exchange(
        masses[:, None],
        send_sizes,
        output_capacity=output_capacity,
        axis_name=axis_name,
    )
    code_out, _, _ = ragged_all_to_all_exchange(
        codes[:, None],
        send_sizes,
        output_capacity=output_capacity,
        axis_name=axis_name,
        fill_value=_CODE_SENTINEL,
    )
    count = jnp.sum(recv_sizes).astype(_COUNT_DTYPE)
    if payload is None:
        return pos_out, mass_out[:, 0], code_out[:, 0], count, None
    pl_out = _route_payload(
        payload, send_sizes, output_capacity, axis_name, payload_fill=payload_fill
    )
    return pos_out, mass_out[:, 0], code_out[:, 0], count, pl_out


def sfc_partition(
    positions_local: Array,
    masses_local: Array,
    ndev: int,
    *,
    output_capacity: int,
    bounds: Optional[tuple[Array, Array]] = None,
    num_samples: int = 8,
    align_level: Optional[int] = None,
    axis_name: str = AXIS_NAME,
    payload: Any = None,
    count: Optional[Array] = None,
    payload_fill: Any = None,
):
    """Sample-sort this device's shard into contiguous Morton domains.

    Returns ``(positions, masses, codes, count)`` for this device: a padded
    shard (leading ``count`` rows valid, Morton-sorted) owning a contiguous
    code range disjoint from every other device's.

    The fifth element is ``payload`` routed along the same exchange and permuted by
    the same sorts, or ``None`` when none was given -- the arity is fixed either way,
    because a return whose LENGTH depends on an argument cannot be type-checked at
    the call sites.

    Pass the global particle ids through it: a particle's identity has to travel WITH
    the particle, because the input order and the shard order coincide only while
    ``capacity == count``, and a host-side id array silently stops matching once they
    diverge.

    Parameters
    ----------
    payload:
        Optional per-particle data to route alongside, e.g. global ids: an array or
        any PYTREE of arrays with a leading row axis. Every leaf keeps its dtype.
    payload_fill:
        Fill for the padding rows of each payload leaf: ``None`` gives -1 (right for
        ids), a scalar applies to every leaf, a pytree matches the payload.
    count:
        Live rows of an already-padded input. ``None`` treats every row as a
        particle, which is right for a fresh decomposition and WRONG for
        re-partitioning a shard that is already capacity-padded: the padding would
        be routed as particles and push real ones out of the capacity. Measured --
        it silently lost half the particles before this argument existed.
    """

    if bounds is None:
        bounds = global_bounds(positions_local, axis_name=axis_name)
    codes = morton_encode_impl(positions_local, bounds)
    if count is not None:
        # dead rows take the sentinel so the sort puts them last, and are then
        # excluded from every destination: they are capacity, not particles
        # cast: three-argument `jnp.where` is always an Array (stubs: `Array | tuple`)
        codes = cast(
            Array,
            jnp.where(
                jnp.arange(codes.shape[0]) < jnp.asarray(count), codes, _CODE_SENTINEL
            ),
        )

    # Local Morton sort -> particles become grouped by destination device
    # automatically, since both codes and pivots are ascending.
    order = jnp.argsort(codes)
    positions = positions_local[order]
    masses = masses_local[order]
    codes = codes[order]

    pivots = _choose_pivots(
        codes, ndev, num_samples, axis_name, align_level, count=count
    )
    dest = jnp.searchsorted(pivots, codes, side="right").astype(_COUNT_DTYPE)
    if count is not None:
        live = jnp.arange(codes.shape[0]) < jnp.asarray(count)
        dest = cast(Array, jnp.where(live, dest, _COUNT_DTYPE(ndev)))
    send_sizes = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)

    if payload is None:
        pos_out, mass_out, code_out, count, _ = _route(
            positions, masses, codes, send_sizes, output_capacity, axis_name
        )
        pos_out, mass_out, code_out, _lo, _hi, _ = _resort_by_code(
            pos_out, mass_out, code_out, count
        )
        return pos_out, mass_out, code_out, count, None

    pl = _tree_take(payload, order)
    pos_out, mass_out, code_out, count, pl_out = _route(
        positions,
        masses,
        codes,
        send_sizes,
        output_capacity,
        axis_name,
        payload=pl,
        payload_fill=payload_fill,
    )
    pos_out, mass_out, code_out, _, _, pl_out = _resort_by_code(
        pos_out, mass_out, code_out, count, payload=pl_out
    )
    return pos_out, mass_out, code_out, count, pl_out


class RepartitionResult(NamedTuple):
    """One device's view of a capacity-checked repartition.

    Attributes
    ----------
    positions, masses, codes:
        The new padded shard, Morton-sorted, live rows first (``live_count`` of them).
    live_count:
        Live rows on this device after the repartition. (Not ``count``: a
        NamedTuple field of that name shadows ``tuple.count``.)
    payload:
        The routed payload pytree (``None`` when none was given).
    declined:
        Replicated bool. ``True`` when routing would have overfilled some device's
        capacity, in which case NOTHING moved: every device kept its own particles
        (re-sorted), which is still a valid partition.
    recv_counts:
        ``(ndev,)`` replicated: what each device WOULD have received. Reported in both
        outcomes, so a decline names the device that did not fit.
    sent_off_device:
        Live rows this device sent to another device (0 when declined). A positive
        mesh total is the proof that a repartition actually moved something.
    max_util:
        Replicated ``max(recv_counts) / capacity``.
    """

    positions: Array
    masses: Array
    codes: Array
    live_count: Array
    payload: Any
    declined: Array
    recv_counts: Array
    sent_off_device: Array
    max_util: Array


def sfc_repartition(
    positions: Array,
    masses: Array,
    count: Array,
    ndev: int,
    *,
    output_capacity: int,
    bounds: tuple[Array, Array],
    num_samples: int = 256,
    axis_name: str = AXIS_NAME,
    payload: Any = None,
    payload_fill: Any = None,
) -> RepartitionResult:
    """Re-partition an already padded shard, checking every receiver's capacity FIRST.

    The send-size matrix is all-gathered before anything moves. If any device would
    receive more than ``output_capacity`` rows the repartition is DECLINED on every
    device at once: each device routes its live rows to itself, so nothing can
    overflow and the old ownership stands. Overflow cannot be detected afterwards --
    the native ``ragged_all_to_all`` writing past the end of its output buffer is not
    a defined truncation.

    ``bounds`` is required: the box must be the one the force builds its tree in
    (dead rows excluded), not :func:`global_bounds`, which would count padding rows.

    Parameters
    ----------
    positions, masses:
        This device's padded shard.
    count:
        Its live rows (leading).
    ndev:
        Mesh size. Static.
    output_capacity:
        The shard capacity every device shares. Static.
    bounds:
        The global Morton box, identical on every device.
    num_samples:
        Samples per device for the pivots. Balance overshoot is about
        ``2 * ndev / num_samples`` of a shard; 8 gives ~50 % at ndev 2. Static.
    axis_name:
        Mesh axis; must match the enclosing ``shard_map``.
    payload:
        Optional pytree routed with the particles (velocities, global ids, ...).
    payload_fill:
        As :func:`sfc_partition`.

    Returns
    -------
    RepartitionResult
        The new shard and the replicated outcome diagnostics.
    """
    cap = int(output_capacity)
    live_in = jnp.arange(positions.shape[0]) < jnp.asarray(count)
    codes = morton_encode_impl(positions, bounds)
    codes = cast(Array, jnp.where(live_in, codes, _CODE_SENTINEL))
    order = jnp.argsort(codes)
    positions = positions[order]
    masses = masses[order]
    codes = codes[order]
    live = jnp.arange(codes.shape[0]) < jnp.asarray(count)

    pivots = _choose_pivots(codes, ndev, num_samples, axis_name, None, count=count)
    dest = jnp.searchsorted(pivots, codes, side="right").astype(_COUNT_DTYPE)
    dest = cast(Array, jnp.where(live, dest, _COUNT_DTYPE(ndev)))
    proposed = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)
    # full[s, r] = rows device s would send to device r -- replicated
    full = jax.lax.all_gather(proposed, axis_name, tiled=False)
    recv_counts = jnp.sum(full, axis=0)
    declined = jnp.any(recv_counts > cap)
    me = jax.lax.axis_index(axis_name).astype(_COUNT_DTYPE)
    dest = cast(Array, jnp.where(declined & live, me, dest))
    send_sizes = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)
    sent_off = jnp.sum((live & (dest != me)).astype(_COUNT_DTYPE))

    pl = None if payload is None else _tree_take(payload, order)
    pos_out, mass_out, code_out, new_count, pl_out = _route(
        positions,
        masses,
        codes,
        send_sizes,
        cap,
        axis_name,
        payload=pl,
        payload_fill=payload_fill,
    )
    pos_out, mass_out, code_out, _lo, _hi, pl_out = _resort_by_code(
        pos_out, mass_out, code_out, new_count, payload=pl_out
    )
    return RepartitionResult(
        positions=pos_out,
        masses=mass_out,
        codes=code_out,
        live_count=new_count,
        payload=pl_out,
        declined=declined,
        recv_counts=recv_counts,
        sent_off_device=sent_off,
        max_util=jnp.max(recv_counts).astype(jnp.float32) / jnp.float32(max(cap, 1)),
    )


def equalize_domain(
    positions,
    masses,
    codes,
    count,
    ndev: int,
    *,
    output_capacity: int,
    axis_name: str = AXIS_NAME,
):
    """Exact rank-based rebalance: each device gets floor/ceil of ``N/ndev``.

    Assumes the input is the output of :func:`sfc_partition` (globally
    Morton-ordered across devices in device order). Preserves ordering and
    contiguity while equalising counts.
    """

    counts = jax.lax.all_gather(
        count.astype(_COUNT_DTYPE), axis_name, tiled=False
    )  # [ndev]
    me = jax.lax.axis_index(axis_name)
    total = jnp.sum(counts)
    global_offset = _exclusive_cumsum(counts)[me]

    cap = codes.shape[0]
    j = jnp.arange(cap, dtype=_COUNT_DTYPE)
    global_rank = global_offset + j

    base = total // ndev
    rem = total - base * ndev
    target_sizes = base + (jnp.arange(ndev, dtype=_COUNT_DTYPE) < rem).astype(
        _COUNT_DTYPE
    )
    target_ends = jnp.cumsum(target_sizes)

    valid = j < count
    dest = jnp.searchsorted(target_ends, global_rank, side="right").astype(_COUNT_DTYPE)
    dest = jnp.minimum(dest, ndev - 1)
    dest = jnp.where(valid, dest, ndev)  # drop padding rows from routing
    send_sizes = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)

    pos_out, mass_out, code_out, new_count, _ = _route(
        positions, masses, codes, send_sizes, output_capacity, axis_name
    )
    pos_out, mass_out, code_out, _lo, _hi, _ = _resort_by_code(
        pos_out, mass_out, code_out, new_count
    )
    return pos_out, mass_out, code_out, new_count


def sfc_decompose(
    mesh,
    positions: Array,
    masses: Array,
    *,
    output_capacity: int,
    num_samples: int = 8,
    align_level: Optional[int] = None,
    equalize: bool = True,
    axis_name: str = AXIS_NAME,
) -> ShardedDomain:
    """Decompose a global particle set into per-GPU Morton domains.

    ``positions`` (``[N, 3]``) and ``masses`` (``[N]``) are sharded evenly along
    axis 0 over the mesh (``N`` must be divisible by ``mesh.size``). Returns a
    :class:`ShardedDomain` global view. ``equalize`` and ``align_level`` are
    mutually-exclusive goals (rank rebalancing ignores cell edges); pass
    ``equalize=False`` when using ``align_level``.
    """

    try:  # stable across recent JAX versions
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    ndev = mesh.size

    def fn(pos, mass):
        p, m, c, cnt, _ = sfc_partition(
            pos,
            mass,
            ndev,
            output_capacity=output_capacity,
            num_samples=num_samples,
            align_level=align_level,
            axis_name=axis_name,
        )
        if equalize:
            p, m, c, cnt = equalize_domain(
                p, m, c, cnt, ndev, output_capacity=output_capacity, axis_name=axis_name
            )
        return p, m, c, cnt[None]

    p, m, c, cnt = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name)),
    )(positions, masses)
    return ShardedDomain(positions=p, masses=m, codes=c, counts=cnt)


__all__ = [
    "RepartitionResult",
    "ShardedDomain",
    "equalize_domain",
    "global_bounds",
    "maybe_repartition",
    "repartition_due",
    "sfc_decompose",
    "sfc_partition",
    "sfc_repartition",
]


def repartition_due(
    count: Array,
    capacity: int,
    step: Array,
    *,
    interval: int = 16,
    headroom: float = 0.9,
    axis_name: str = AXIS_NAME,
) -> Array:
    """Whether to re-run :func:`sfc_partition` this step -- the SAME answer everywhere.

    **The verdict must be mesh-uniform, and that is the whole point of this function.**
    ``sfc_partition`` contains collectives: two ``all_gather`` rounds and a ragged
    all-to-all. If one device repartitions and another does not, the first blocks on a
    collective the second never enters and the mesh DEADLOCKS. So the drift test cannot
    be the local ``count``; it is an all-reduced maximum over every device's occupancy.

    Two triggers, both uniform by construction:

    * **scheduled** -- every ``interval`` steps. Particles move, so device ownership
      goes stale, but only on the dynamical time: the fused lane re-Morton-sorts and
      rebuilds its local tree every step anyway, so drift WITHIN a domain is absorbed
      for free and only the assignment needs refreshing. ``step`` is replicated, so
      this needs no reduction.
    * **drift guard** -- any device's ``count`` passing ``headroom`` of ``capacity``.
      ``capacity`` is a compile-time constant that every shape depends on, so a count
      reaching it is not a slowdown but an overflow.

    Parameters
    ----------
    count:
        This device's live particle count.
    capacity:
        The static shard capacity. Static.
    step:
        Step index, replicated across the mesh.
    interval:
        Scheduled cadence in steps. Static.
    headroom:
        Fraction of ``capacity`` above which the guard fires, regardless of schedule.
    axis_name:
        Mesh axis; must match the enclosing ``shard_map``.

    Returns
    -------
    Array
        Scalar bool, **identical on every device**.
    """
    frac = jnp.asarray(count, jnp.float32) / jnp.float32(max(int(capacity), 1))
    worst = jax.lax.pmax(frac, axis_name)
    scheduled = (jnp.asarray(step).astype(jnp.int32) % jnp.int32(int(interval))) == 0
    return jnp.logical_or(scheduled, worst > jnp.float32(headroom))


def maybe_repartition(
    positions: Array,
    masses: Array,
    count: Array,
    should: Array,
    ndev: int,
    *,
    output_capacity: int,
    bounds: Optional[tuple[Array, Array]] = None,
    num_samples: int = 8,
    axis_name: str = AXIS_NAME,
    payload: Optional[Array] = None,
):
    """Repartition under ``should``, with both branches the same shapes.

    The branches must agree in shape AND in manual-axis variance, and the second is
    the one that bites. ``sfc_partition`` ends in collectives, and under the NATIVE
    ``ragged_all_to_all`` JAX infers its result as axis-INVARIANT, while the identity
    branch passes through a sharded input and is ``{V:gpus}`` varying -- so the
    ``cond`` is rejected for "manual axis types do not match" even though every shape
    and dtype agrees. Both branches are therefore pushed to varying with
    :func:`jax.lax.pcast`.

    **This does not reproduce on forced CPU devices**, because
    :func:`~yggdrax.distributed.comm.resolve_ragged_method` picks the ``buf``
    all-gather fallback there and ``native`` on GPU, and the two infer variance
    differently. Testing distributed code on forced CPU devices does not cover the
    GPU tracing path.

    ``should`` must come from :func:`repartition_due`, or from something else that is
    mesh-uniform. A locally-computed predicate deadlocks.

    Parameters
    ----------
    positions, masses, count:
        The current padded shard and its live count.
    should:
        Mesh-uniform verdict.
    ndev, output_capacity, bounds, num_samples, axis_name, payload:
        As :func:`sfc_partition`.

    Returns
    -------
    tuple
        ``(positions, masses, codes, count, payload)``, repartitioned or not.
    """
    if bounds is None:
        bounds = global_bounds(positions, axis_name=axis_name)

    def _to_varying(x):
        """Make ``x`` axis-varying, whichever it already is.

        ``pcast`` converts invariant -> varying and RAISES on an input that is
        already varying, and the variance is not exposed as an attribute to test.
        Which case applies depends on the backend -- the native
        ``ragged_all_to_all`` leaves `sfc_partition`'s result invariant, the ``buf``
        fallback leaves it varying -- so the branch is decided by trying it. This is
        trace time; nothing is attempted at runtime.
        """
        if not hasattr(x, "shape"):
            return x
        try:
            return jax.lax.pcast(x, axis_name, to="varying")
        except (ValueError, TypeError):
            return x

    def _varying(tree):
        return jax.tree.map(_to_varying, tree)

    def _do(_):
        return _varying(
            sfc_partition(
                positions,
                masses,
                ndev,
                output_capacity=output_capacity,
                bounds=bounds,
                num_samples=num_samples,
                align_level=None,
                axis_name=axis_name,
                payload=payload,
                count=count,
            )
        )

    def _keep(_):
        codes = morton_encode_impl(positions, bounds)
        valid = jnp.arange(codes.shape[0]) < count
        return _varying(
            (
                positions,
                masses,
                jnp.where(valid, codes, _CODE_SENTINEL),
                count,
                payload,
            )
        )

    return jax.lax.cond(jnp.asarray(should), _do, _keep, operand=None)
