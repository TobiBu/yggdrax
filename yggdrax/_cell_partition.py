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

Cost: one pass per Morton level -- a prefix-max and a suffix-min scan over the
per-boundary common-prefix depth (no ``searchsorted``: 14 -> ~1 ms at N = 2e5),
21 levels at most; ``max_level`` can stop earlier.
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
    leaf_size_i = jnp.asarray(int(leaf_size), INDEX_DTYPE)
    idx = jnp.arange(n, dtype=INDEX_DTYPE)
    # Live/dead cut. ``None`` -> every row live, and every use below degenerates
    # to the unpadded expression, so the result is unchanged bit for bit.
    padded = num_valid is not None
    n_valid = jnp.asarray(n if num_valid is None else num_valid, INDEX_DTYPE)
    is_dead = idx >= n_valid if padded else jnp.zeros((n,), dtype=bool)
    # Common Morton depth of each consecutive pair (the boundary BEFORE particle
    # i): codes use 63 bits, 3 per level, so the pair agrees through
    # floor((clz64(xor) - 1) / 3) whole levels. Boundary 0 agrees through nothing.
    if n > 1:
        x = jnp.bitwise_xor(codes[1:], codes[:-1])
        clz = lax.clz(x).astype(INDEX_DTYPE)  # 64 when equal
        agree = jnp.minimum((clz - 1) // 3, MORTON_LEVELS)
        agree = jnp.concatenate([jnp.asarray([-1], INDEX_DTYPE), agree])
    else:
        agree = jnp.asarray([-1], INDEX_DTYPE)
    depth = jnp.full((n,), int(max_level), INDEX_DTYPE)
    assigned = jnp.zeros((n,), dtype=bool)
    n_i = jnp.asarray(n, INDEX_DTYPE)
    # Static level loop: the trip count is the Morton depth, not the data. A cell
    # at depth d is a maximal run whose interior boundaries all agree through at
    # least d levels; its start is the last boundary with agree < d at or before
    # i (prefix max) and its end the first such boundary after i (suffix min).
    for d in range(int(min_level), int(max_level)):
        # A dead row is a boundary at every level: without this the last live
        # cell's occupancy would count the padding, so a shard whose live part
        # fits in one cell would be split into several -- a legal partition, but
        # not the one the same particles get unpadded.
        is_boundary = (agree < d) | is_dead
        start_idx = lax.cummax(jnp.where(is_boundary, idx, jnp.asarray(0, INDEX_DTYPE)))
        nxt = jnp.where(is_boundary, idx, n_i)
        end_idx = jnp.concatenate(
            [lax.cummin(nxt[1:], reverse=True), jnp.asarray([n], INDEX_DTYPE)]
        )
        occ = end_idx - start_idx
        fit = (occ <= leaf_size_i) & ~assigned
        depth = jnp.where(fit, jnp.asarray(d, INDEX_DTYPE), depth)
        assigned = assigned | fit
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
    target = jnp.where(
        first & (slot < capacity), slot, jnp.asarray(capacity, INDEX_DTYPE)
    )
    starts = jnp.full((capacity + 1,), n, INDEX_DTYPE).at[target].min(idx)[:capacity]
    depths_out = (
        jnp.full((capacity + 1,), -1, INDEX_DTYPE).at[target].max(depth)[:capacity]
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
