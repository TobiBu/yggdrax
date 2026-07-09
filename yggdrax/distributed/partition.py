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
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .comm import _COUNT_DTYPE, _exclusive_cumsum, ragged_all_to_all_exchange
from .sharding import AXIS_NAME
from ..morton import morton_encode_impl

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
) -> Array:
    """Sample-based splitter selection: returns ``ndev-1`` ascending pivots."""

    n = codes_sorted.shape[0]
    idx = (jnp.arange(num_samples) * n) // num_samples
    samples = codes_sorted[idx]
    all_samples = jnp.sort(jax.lax.all_gather(samples, axis_name, tiled=True))
    total = all_samples.shape[0]
    piv_idx = (jnp.arange(1, ndev) * total) // ndev
    return _align_pivots(all_samples[piv_idx], align_level)


def _resort_by_code(positions, masses, codes, count):
    """Re-sort a padded shard by Morton code (padding sentinel -> tail)."""

    order = jnp.argsort(codes)
    positions = positions[order]
    masses = masses[order]
    codes = codes[order]
    cap = codes.shape[0]
    valid = jnp.arange(cap) < count
    code_lo = jnp.min(jnp.where(valid, codes, _CODE_SENTINEL))
    code_hi = jnp.max(jnp.where(valid, codes, jnp.uint64(0)))
    return positions, masses, codes, code_lo, code_hi


def _route(positions, masses, codes, send_sizes, output_capacity, axis_name):
    """Ragged-exchange a shard already grouped by destination device."""

    pos_out, recv_sizes, _ = ragged_all_to_all_exchange(
        positions, send_sizes, output_capacity=output_capacity, axis_name=axis_name
    )
    mass_out, _, _ = ragged_all_to_all_exchange(
        masses[:, None], send_sizes, output_capacity=output_capacity, axis_name=axis_name
    )
    code_out, _, _ = ragged_all_to_all_exchange(
        codes[:, None],
        send_sizes,
        output_capacity=output_capacity,
        axis_name=axis_name,
        fill_value=_CODE_SENTINEL,
    )
    count = jnp.sum(recv_sizes).astype(_COUNT_DTYPE)
    return pos_out, mass_out[:, 0], code_out[:, 0], count


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
):
    """Sample-sort this device's shard into contiguous Morton domains.

    Returns ``(positions, masses, codes, count)`` for this device: a padded
    shard (leading ``count`` rows valid, Morton-sorted) owning a contiguous
    code range disjoint from every other device's.
    """

    if bounds is None:
        bounds = global_bounds(positions_local, axis_name=axis_name)
    codes = morton_encode_impl(positions_local, bounds)

    # Local Morton sort -> particles become grouped by destination device
    # automatically, since both codes and pivots are ascending.
    order = jnp.argsort(codes)
    positions = positions_local[order]
    masses = masses_local[order]
    codes = codes[order]

    pivots = _choose_pivots(codes, ndev, num_samples, axis_name, align_level)
    dest = jnp.searchsorted(pivots, codes, side="right").astype(_COUNT_DTYPE)
    send_sizes = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)

    pos_out, mass_out, code_out, count = _route(
        positions, masses, codes, send_sizes, output_capacity, axis_name
    )
    pos_out, mass_out, code_out, _, _ = _resort_by_code(
        pos_out, mass_out, code_out, count
    )
    return pos_out, mass_out, code_out, count


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
    dest = jnp.searchsorted(target_ends, global_rank, side="right").astype(
        _COUNT_DTYPE
    )
    dest = jnp.minimum(dest, ndev - 1)
    dest = jnp.where(valid, dest, ndev)  # drop padding rows from routing
    send_sizes = jnp.bincount(dest, length=ndev).astype(_COUNT_DTYPE)

    pos_out, mass_out, code_out, new_count = _route(
        positions, masses, codes, send_sizes, output_capacity, axis_name
    )
    pos_out, mass_out, code_out, _, _ = _resort_by_code(
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
        p, m, c, cnt = sfc_partition(
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
    "ShardedDomain",
    "equalize_domain",
    "global_bounds",
    "sfc_decompose",
    "sfc_partition",
]
