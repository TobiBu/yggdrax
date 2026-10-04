"""Adaptive Morton-cell leaf partitions with static shapes.

A ``static_radix`` tree cuts the Morton-sorted particles into buckets of exactly
``leaf_size`` consecutive particles. In a centrally concentrated distribution
the buckets in the low-density shell straddle several Morton cells and become
huge: at N = 2x10^5 (Plummer, leaf 64) 341 of 3125 leaves had a radius above
4x the median and sat in 56 % of the near-field volume, because the mutual MAC
makes a huge leaf a neighbour of most of the tree. Morton CELLS -- each leaf the
coarsest cell that holds at most ``leaf_size`` particles -- are bounded in
extent and sparse where the density is low; the same walk on cell leaves summed
0.0052 N directly instead of 0.1225 N (jaccpot ``probe_tree_volume.py``,
2026-09-11).

:func:`adaptive_cell_leaf_partition` computes that partition on device with
static shapes: the leaf count is data dependent, so the leaf arrays are padded
to a caller-chosen ``capacity`` (empty leaves ``start = end = n``) and an
overflow flag says when the capacity did not hold. A particle's leaf depth is
the smallest Morton level at which its cell holds at most ``leaf_size``
particles (cells at the deepest level are leaves whatever they hold, so
coincident particles cannot recurse forever); the leaf key is the cell at that
depth, and leaves are the runs of equal keys -- consistent because a cell's
occupancy is a property of the cell, so every particle of a qualifying cell
picks the same depth.

Cost: no pass per Morton level. A particle's cell at level ``d`` overflows
exactly when a window of ``leaf_size + 1`` consecutive particles containing it
shares the level-``d`` prefix, so the depth follows from one xor/clz between
particles ``j`` and ``j + leaf_size`` and one sliding maximum (the numpy
reference keeps the per-level prefix-max / suffix-min loop it replaced).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from .dtypes import INDEX_DTYPE

__all__ = [
    "CellLeafPartition",
    "MORTON_LEVELS",
    "adaptive_cell_leaf_partition",
    "adaptive_cell_leaf_partition_numpy",
]

#: 21 bits per axis, 63 bits used; level ``d`` cell id = ``code >> (63 - 3 d)``.
MORTON_LEVELS = 21


class CellLeafPartition(NamedTuple):
    """Leaf partition of Morton-sorted particles, padded to a static capacity.

    Attributes
    ----------
    leaf_starts : Array
        First particle index of each leaf, ``(capacity,)``; padding = ``n``.
    leaf_ends : Array
        Exclusive end of each leaf, ``(capacity,)``; padding = ``n``.
    leaf_depths : Array
        Morton level of each leaf's cell, ``(capacity,)``; padding = ``-1``.
    num_leaves : Array
        Number of live leaves (scalar).
    overflow : Array
        ``True`` when more than ``capacity`` leaves were needed; the arrays then
        hold the first ``capacity`` leaves and the rest of the particles are
        NOT covered -- the caller must treat this as fatal.
    """

    leaf_starts: Array
    leaf_ends: Array
    leaf_depths: Array
    num_leaves: Array
    overflow: Array


def adaptive_cell_leaf_partition(
    sorted_codes: Array,
    *,
    leaf_size: int,
    capacity: int,
    max_level: int = MORTON_LEVELS,
    num_valid: Array | int | None = None,
    min_level: int = 0,
) -> CellLeafPartition:
    """Adaptive Morton-cell leaves of Morton-sorted codes, padded to ``capacity``.

    Parameters
    ----------
    sorted_codes
        Morton codes in nondecreasing order, ``(n,)``, ``uint64`` (21 bits per
        axis).
    leaf_size
        Maximum particles per leaf. Static.
    capacity
        Static leaf capacity the arrays are padded to.
    max_level
        Deepest level examined; cells at this level are leaves whatever they
        hold. Static, at most :data:`MORTON_LEVELS`.
    num_valid
        Number of leading live codes, for a capacity-padded input whose trailing
        rows are dead (a distributed shard). May be traced. ``None`` means every
        row is live, and the result is then bit-identical to omitting it.

        Dead rows must already sort last -- give them the maximal Morton code --
        but that alone is not enough, and the failure is not obvious: every dead
        row then carries the *same* code, so their common-prefix depth is total,
        no level ever satisfies ``occ <= leaf_size``, and they collapse into one
        leaf of ``n - num_valid`` rows. Wider than the leaf table, that leaf
        either trips the eager leaf-size check or, under trace, truncates into a
        live leaf of mass 0 and radius 0 sitting at a real position -- which
        fails the MAC against everything near it and floods the near list. So
        ``num_valid`` cuts the run boundaries, the leaf starts, the leaf count
        and the last leaf's end; the dead rows then belong to no leaf at all.

    min_level
        No leaf coarser than this Morton level: a cell shallower than it is split
        even when it already holds at most ``leaf_size`` particles. Static. A
        sparse outskirt cell (a few far-apart outliers) otherwise stays ONE leaf
        whose bounding sphere spans much of the box, fails the MAC against nearly
        everything and pulls a whole remote domain into the cross-domain near
        export. ``0`` (default) is the unconstrained partition, bit for bit.

    Returns
    -------
    CellLeafPartition
        See the class; every array has static shape.

    Raises
    ------
    ValueError
        If ``leaf_size`` or ``capacity`` is not positive, or ``max_level`` or
        ``min_level`` is out of range.
    """
    if int(leaf_size) < 1:
        raise ValueError("leaf_size must be >= 1")
    if int(capacity) < 1:
        raise ValueError("capacity must be >= 1")
    if not 0 <= int(max_level) <= MORTON_LEVELS:
        raise ValueError(f"max_level must be in [0, {MORTON_LEVELS}]")
    if not 0 <= int(min_level) <= int(max_level):
        raise ValueError(f"min_level must be in [0, max_level={int(max_level)}]")
    codes = jnp.asarray(sorted_codes).astype(jnp.uint64)
    n = int(codes.shape[0])
    capacity = int(capacity)
    idx = jnp.arange(n, dtype=INDEX_DTYPE)
    # Live/dead cut. ``None`` -> every row live, and every use below degenerates
    # to the unpadded expression, so the result is unchanged bit for bit.
    padded = num_valid is not None
    n_valid = jnp.asarray(n if num_valid is None else num_valid, INDEX_DTYPE)
    is_dead = idx >= n_valid if padded else jnp.zeros((n,), dtype=bool)
    # A particle's level-d cell holds more than S = leaf_size particles exactly
    # when some window of S + 1 consecutive (live) particles containing it shares
    # the level-d prefix (sorted codes: both ends share it iff all between do).
    # With g_j the levels particles j and j + S share (codes use 63 bits, 3 per
    # level: floor((clz64(xor) - 1) / 3)) and G(i) the max of g over the windows
    # [i - S, i], the cell fits from level G(i) + 1 on, so the depth -- the
    # shallowest fitting level, at least min_level, at most max_level -- is
    # max(min_level, G(i) + 1) capped at max_level. One xor/clz and one sliding
    # max instead of a prefix-max and a suffix-min scan per level (26 scans over
    # N per refresh at min_level 8); the numpy reference keeps the level loop.
    S = int(leaf_size)
    minus1 = jnp.asarray(-1, INDEX_DTYPE)
    if n > S:
        x = jnp.bitwise_xor(codes[S:], codes[:-S])
        clz = lax.clz(x).astype(INDEX_DTYPE)  # 64 when equal
        g = jnp.minimum((clz - 1) // 3, MORTON_LEVELS)
        # a window must lie in the live rows (dead rows are boundaries at every
        # level), so one ending at or past n_valid shares nothing
        g = jnp.where(jnp.arange(n - S, dtype=INDEX_DTYPE) + S < n_valid, g, minus1)
        g = jnp.concatenate(
            [jnp.full((S,), -1, INDEX_DTYPE), g, jnp.full((S,), -1, INDEX_DTYPE)]
        )
        shared = lax.reduce_window(g, minus1, lax.max, (S + 1,), (1,), "VALID")
    else:
        shared = jnp.full((n,), -1, INDEX_DTYPE)
    fit_level = jnp.maximum(jnp.asarray(int(min_level), INDEX_DTYPE), shared + 1)
    depth = jnp.where(
        fit_level < int(max_level), fit_level, jnp.asarray(int(max_level), INDEX_DTYPE)
    )
    shift_p = (3 * (MORTON_LEVELS - depth)).astype(jnp.uint64)
    key = jnp.right_shift(codes, shift_p)
    first = jnp.concatenate(
        [jnp.ones((1,), bool), (key[1:] != key[:-1]) | (depth[1:] != depth[:-1])]
    )
    first = first & ~is_dead
    slot = jnp.cumsum(first.astype(INDEX_DTYPE)) - 1  # leaf index of each particle
    # Live-leaf count. Equal to ``slot[-1] + 1`` once ``first`` is masked, but
    # stated directly so it cannot be read as counting padding.
    num_leaves = (
        jnp.sum(first.astype(INDEX_DTYPE)) if n > 0 else jnp.asarray(0, INDEX_DTYPE)
    )
    overflow = num_leaves > capacity
    # One writer per leaf (its first particle); every other lane points PAST the
    # table at its own index (out of range and distinct) and is dropped. The old
    # min/max scatter onto one sentinel row serialised ~N atomics on one address.
    # (``capacity + idx`` must stay positive: a wrapped index would be normalised
    # back INTO the table.)
    if int(capacity) + int(n) >= int(np.iinfo(np.dtype(INDEX_DTYPE)).max):
        raise ValueError(
            f"capacity + n = {int(capacity) + int(n)} overflows {INDEX_DTYPE}"
        )
    target = jnp.where(
        first & (slot < capacity), slot, jnp.asarray(capacity, INDEX_DTYPE) + idx
    )
    starts = (
        jnp.full((capacity,), n, INDEX_DTYPE)
        .at[target]
        .set(idx, mode="drop", unique_indices=True)
    )
    depths_out = (
        jnp.full((capacity,), -1, INDEX_DTYPE)
        .at[target]
        .set(depth, mode="drop", unique_indices=True)
    )
    live = jnp.arange(capacity, dtype=INDEX_DTYPE) < jnp.minimum(num_leaves, capacity)
    next_start = jnp.concatenate([starts[1:], jnp.asarray([n], INDEX_DTYPE)])
    ends = jnp.where(
        live,
        jnp.where(
            jnp.arange(capacity) + 1 < jnp.minimum(num_leaves, capacity),
            next_start,
            n_valid,  # the LAST live leaf stops at the cut, not at the array end
        ),
        n,  # padding leaves keep start == end == n, which reads as an empty range
    )
    starts = jnp.where(live, starts, n)
    depths_out = jnp.where(live, depths_out, -1)
    return CellLeafPartition(
        leaf_starts=starts.astype(INDEX_DTYPE),
        leaf_ends=ends.astype(INDEX_DTYPE),
        leaf_depths=depths_out.astype(INDEX_DTYPE),
        num_leaves=num_leaves.astype(INDEX_DTYPE),
        overflow=overflow,
    )


def adaptive_cell_leaf_partition_numpy(
    sorted_codes: np.ndarray,
    *,
    leaf_size: int,
    max_level: int = MORTON_LEVELS,
    min_level: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference of :func:`adaptive_cell_leaf_partition` (unpadded).

    Parameters
    ----------
    sorted_codes
        Morton codes in nondecreasing order, ``uint64``.
    leaf_size
        Maximum particles per leaf.
    max_level
        Deepest level examined.
    min_level
        No leaf coarser than this level (see the device version).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(leaf_starts, leaf_ends_exclusive, leaf_depths)`` of the live leaves.
    """
    codes = np.asarray(sorted_codes).astype(np.uint64)
    n = codes.shape[0]
    depth = np.full(n, int(max_level), np.int64)
    assigned = np.zeros(n, bool)
    for d in range(int(min_level), int(max_level)):
        cell = codes >> np.uint64(3 * (MORTON_LEVELS - d))
        change = np.concatenate([[True], cell[1:] != cell[:-1]])
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], [n]])
        run_len = np.repeat(ends - starts, ends - starts)
        fit = (run_len <= leaf_size) & ~assigned
        depth[fit] = d
        assigned |= fit
        if assigned.all():
            break
    key = codes >> (3 * (MORTON_LEVELS - depth)).astype(np.uint64)
    change = np.concatenate([[True], (key[1:] != key[:-1]) | (depth[1:] != depth[:-1])])
    starts = np.flatnonzero(change)
    ends = np.concatenate([starts[1:], [n]])
    return starts, ends, depth[starts]
