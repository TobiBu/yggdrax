"""Geometry helpers for radix trees.

This module derives per-node bounding geometry from Morton-sorted
particles so the Fast Multipole Method can reason about node extents
when deciding whether a multipole expansion is admissible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import core as jax_core
from jax import lax
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index, require_x64
from .morton import _compact3_u64
from .tree import require_morton_topology


class TreeGeometry(NamedTuple):
    """Bounding information for every node in a radix tree."""

    center: Array
    half_extent: Array
    radius: Array
    max_extent: Array


def _validate_inputs(tree: object, positions_sorted: Array) -> None:
    total_nodes = tree.parent.shape[0]
    if tree.node_ranges.shape[0] != total_nodes:
        raise ValueError("tree.node_ranges must align with tree.parent shape")
    num_particles = jnp.asarray(tree.num_particles)
    if not isinstance(num_particles, jax_core.Tracer):
        if positions_sorted.shape[0] != int(num_particles):
            raise ValueError("positions_sorted must match tree.num_particles")
    if positions_sorted.ndim != 2 or positions_sorted.shape[1] < 1:
        raise ValueError("positions must have shape (N, D) with D >= 1")
    # The Morton-geometry path is 3-D specific (it decodes 3-D Morton codes);
    # arbitrary D is only reached via the range-based path used by the KD-tree
    # (use_morton_geometry=False). Morton backends always supply 3-D positions.


@jaxtyped(typechecker=beartype)
def _compute_leaf_bounds(
    positions_sorted: Array,
    leaf_starts: Array,
    leaf_counts: Array,
    *,
    max_leaf_size: int | None = None,
    particle_radius: Array | None = None,
) -> tuple[Array, Array]:
    dtype = positions_sorted.dtype
    num_leaves = leaf_starts.shape[0]
    num_particles = positions_sorted.shape[0]

    # Upper bound on particles per leaf.
    #
    # This bound drives the temporary gather shape ``(num_leaves, max_count, 3)``.
    # Falling back to ``num_particles`` is functionally safe but disastrous for
    # large JIT-compiled trees because it inflates geometry staging buffers from
    # "leaf-sized" to "whole-problem-sized". Callers that already know the tree's
    # leaf cap should pass it explicitly to keep geometry memory proportional to
    # leaf occupancy rather than total particle count.
    if max_leaf_size is not None:
        max_count = int(max_leaf_size)
    elif isinstance(leaf_counts, jax_core.Tracer):
        max_count = num_particles
    else:
        max_count = int(jnp.max(leaf_counts)) if num_leaves > 0 else 0

    dim = positions_sorted.shape[1]
    if max_count == 0:
        # All leaves are empty – return neutral bounds.
        inf_mins = jnp.full((num_leaves, dim), jnp.inf, dtype=dtype)
        neg_inf_maxs = jnp.full((num_leaves, dim), -jnp.inf, dtype=dtype)
        return inf_mins, neg_inf_maxs
    if num_leaves * max_count > num_particles:
        # The padded gather below would hold more rows than there are particles
        # (a capacity-padded tree's leaves are mostly part-full: about 5x at leaf
        # 64), times four arrays. Reduce over the particles by leaf instead.
        return _segment_leaf_bounds(
            positions_sorted,
            leaf_starts,
            jnp.minimum(leaf_counts, as_index(max_count)),
            particle_radius=particle_radius,
        )
    return _padded_leaf_bounds(
        positions_sorted,
        leaf_starts,
        leaf_counts,
        max_count,
        particle_radius=particle_radius,
    )


def _padded_leaf_bounds(
    positions_sorted: Array,
    leaf_starts: Array,
    leaf_counts: Array,
    max_count: int,
    *,
    particle_radius: Array | None = None,
) -> tuple[Array, Array]:
    """``_compute_leaf_bounds`` by a ``(leaves, max_count)`` padded gather."""
    dtype = positions_sorted.dtype
    # Build a (num_leaves, max_count) index array and a validity mask.
    offsets = jnp.arange(max_count, dtype=INDEX_DTYPE)[None, :]  # (1, M)
    indices = leaf_starts[:, None] + offsets  # (L, M)
    valid = offsets < leaf_counts[:, None]  # (L, M)

    # Clamp out-of-range indices so the gather is safe.
    safe_indices = jnp.where(valid, indices, as_index(0))
    points = positions_sorted[safe_indices]  # (L, M, 3)

    # A "particle" may stand for an aggregate -- an LET coarse particle is a whole
    # remote leaf reduced to its centre of mass -- in which case the bound has to
    # cover the ball it represents, not the point it was reduced to. Bounding the
    # points alone is what makes a coarse tree's MAC extent understate the source
    # region it is standing in for, so the MAC accepts pairs whose true separation
    # is smaller than the source's own radius. Growing each point into a cube here
    # (rather than padding ``radius`` afterwards) keeps the box centre, the
    # half-extent, the sphere radius and the L-infinity extent mutually consistent
    # and lets the upward pass carry the inflation to every ancestor for free.
    lows = points
    highs = points
    if particle_radius is not None:
        radius = jnp.asarray(particle_radius, dtype=dtype)[safe_indices][..., None]
        lows = points - radius
        highs = points + radius

    # Masked min / max reduction along axis=1.
    big = jnp.finfo(dtype).max
    mins = jnp.where(valid[:, :, None], lows, big)
    maxs = jnp.where(valid[:, :, None], highs, -big)
    mins = jnp.min(mins, axis=1)  # (L, 3)
    maxs = jnp.max(maxs, axis=1)  # (L, 3)
    return mins, maxs


def _segment_leaf_bounds(
    positions_sorted: Array,
    leaf_starts: Array,
    leaf_counts: Array,
    *,
    particle_radius: Array | None = None,
) -> tuple[Array, Array]:
    """``_compute_leaf_bounds`` by segment min/max over the particles.

    Each non-empty leaf is the run ``[start, start + count)`` of sorted rows; a
    row's leaf is found by marking each run's first row and carrying the mark
    forward with a running maximum. Rows outside every run are dropped. Memory
    is O(particles + leaves) where the padded gather is O(leaves x leaf cap).
    The values are the same min/max over the same particles, and so are the
    derivatives (``segment_min`` splits a tie like ``jnp.min``). Empty leaves
    get the padded path's ``+/- finfo.max``.
    """
    dtype = positions_sorted.dtype
    num_leaves = int(leaf_starts.shape[0])
    n = int(positions_sorted.shape[0])
    idx = INDEX_DTYPE
    starts = jnp.asarray(leaf_starts, idx)
    counts = jnp.maximum(jnp.asarray(leaf_counts, idx), as_index(0))
    has = counts > 0
    first = jnp.where(has & (starts >= 0), starts, as_index(n))
    mark = jnp.full((n,), -1, idx).at[first].set(first, mode="drop")
    owner = (
        jnp.full((n,), num_leaves, idx)
        .at[first]
        .set(jnp.arange(num_leaves, dtype=idx), mode="drop")
    )
    at = lax.cummax(mark)
    leaf = jnp.where(at >= 0, owner[jnp.maximum(at, 0)], as_index(num_leaves))
    offset = jnp.arange(n, dtype=idx) - at
    inside = (at >= 0) & (offset < counts[jnp.clip(leaf, 0, max(num_leaves - 1, 0))])
    seg = jnp.where(inside, leaf, as_index(num_leaves))

    lows = positions_sorted
    highs = positions_sorted
    if particle_radius is not None:
        radius = jnp.asarray(particle_radius, dtype=dtype)[:, None]
        lows = positions_sorted - radius
        highs = positions_sorted + radius
    big = jnp.finfo(dtype).max
    mins = jax.ops.segment_min(lows, seg, num_segments=num_leaves)
    maxs = jax.ops.segment_max(highs, seg, num_segments=num_leaves)
    mins = jnp.where(has[:, None], mins, big)
    maxs = jnp.where(has[:, None], maxs, -big)
    return mins, maxs


_MAX_MORTON_LEVEL = 21


def _compute_leaf_bounds_from_morton(tree: object) -> tuple[Array, Array]:
    require_x64("63-bit Morton codes")
    require_morton_topology(tree)

    bounds_min = jnp.asarray(tree.bounds_min)
    bounds_max = jnp.asarray(tree.bounds_max)
    domain = bounds_max - bounds_min

    leaf_codes = jnp.asarray(tree.leaf_codes, dtype=jnp.uint64)
    leaf_depths = jnp.asarray(tree.leaf_depths, dtype=INDEX_DTYPE)

    depth_clamped = jnp.clip(
        leaf_depths,
        as_index(0),
        as_index(_MAX_MORTON_LEVEL),
    )
    shift = jnp.asarray(_MAX_MORTON_LEVEL, dtype=jnp.uint64) - depth_clamped.astype(
        jnp.uint64
    )

    x_coords = _compact3_u64(leaf_codes)
    y_coords = _compact3_u64(leaf_codes >> jnp.uint64(1))
    z_coords = _compact3_u64(leaf_codes >> jnp.uint64(2))

    x_idx = x_coords >> shift
    y_idx = y_coords >> shift
    z_idx = z_coords >> shift

    indices = jnp.stack([x_idx, y_idx, z_idx], axis=1).astype(bounds_min.dtype)

    counts = jnp.left_shift(
        jnp.ones_like(depth_clamped, dtype=jnp.uint64),
        depth_clamped.astype(jnp.uint64),
    )
    counts = jnp.maximum(counts, jnp.uint64(1))
    counts = counts.astype(bounds_min.dtype)

    cell_sizes = domain[None, :] / counts[:, None]
    mins = bounds_min[None, :] + cell_sizes * indices
    maxs = mins + cell_sizes
    return mins, maxs


def _same_bits(a: Array, b: Array) -> Array:
    """Elementwise: ``a`` and ``b`` have the same bit pattern (``-0.0 != 0.0``)."""
    uint = jnp.dtype(f"uint{8 * jnp.dtype(a.dtype).itemsize}")
    return lax.bitcast_convert_type(a, uint) == lax.bitcast_convert_type(b, uint)


def _gather_inherited_bounds(
    stopped: Array, source_leaf: Array, leaf_bounds: Array, num_internal: int
) -> Array:
    """Internal-node bounds as the live bounds of the leaves they came from.

    ``stopped`` holds the upward pass's merged values (no tangents) and
    ``source_leaf`` the leaf each one was taken from, per axis (``-1`` for an
    internal node no level reached, which keeps its stopped value: zero). The
    gather returns the same values bit for bit, and its derivative is the
    subgradient of the min/max: the extremal leaf's own bound gets it all (the
    first child on an exact tie).
    """
    # the node arrays' dtype, as the old ``.at[...].set`` cast it (the Morton
    # shortcut computes leaf bounds in the bounds' dtype)
    leaf_bounds = jnp.asarray(leaf_bounds).astype(stopped.dtype)
    internal_src = source_leaf[:num_internal]
    gathered = jnp.take_along_axis(leaf_bounds, jnp.maximum(internal_src, 0), axis=0)
    internal = jnp.where(internal_src >= 0, gathered, stopped[:num_internal])
    return jnp.concatenate([internal, leaf_bounds], axis=0)


@jaxtyped(typechecker=beartype)
def compute_tree_geometry(
    tree: object,
    positions_sorted: Array,
    *,
    max_leaf_size: int | None = None,
    particle_radius: Array | None = None,
) -> TreeGeometry:
    """Compute per-node bounding boxes and helper radii.

    Parameters
    ----------
    tree
        Topology object exposing tree-structure arrays.
    positions_sorted
        Particle positions reordered into Morton order. This should be the
        ``positions`` output from ``build_tree(..., return_reordered=True)``.
    max_leaf_size
        Optional explicit upper bound for particles per leaf. Passing the known
        tree leaf cap keeps JIT-staged leaf gathers bounded by leaf occupancy
        instead of the full particle count.
    particle_radius
        Optional per-particle radius ``(n_particles,)``. Each particle is then
        bounded as a ball of that radius rather than as a point, so every node's
        box covers the aggregates its particles stand for. Required for trees
        built over *proxies* -- an LET coarse tree's particles are whole remote
        leaves reduced to their centres of mass, and without their radii the
        resulting MAC extent understates the source region it represents.

    Returns
    -------
    TreeGeometry
        Named tuple containing node centers, half-extents, and bounding radii.

    Notes
    -----
    Differentiable in ``positions_sorted`` (and ``particle_radius``) in forward
    and reverse mode, eagerly and under ``jax.jit``, for a fixed topology. Each
    internal node's bound is the bound of one of its leaves, per axis; the
    upward pass picks that leaf without gradients and gathers its live bound,
    so the derivative is the min/max subgradient (all of it to the extremal
    leaf; on an exact tie, to the first child).
    """

    _validate_inputs(tree, positions_sorted)

    ranges = tree.node_ranges.astype(INDEX_DTYPE)
    num_internal = tree.left_child.shape[0]
    num_nodes = ranges.shape[0]

    use_morton_raw = getattr(tree, "use_morton_geometry", False)
    use_morton_bounds = jnp.asarray(use_morton_raw, dtype=jnp.bool_)
    has_morton_fields = all(
        hasattr(tree, name)
        for name in ("bounds_min", "bounds_max", "leaf_codes", "leaf_depths")
    )
    if (not has_morton_fields) and (not isinstance(use_morton_raw, jax_core.Tracer)):
        if bool(jnp.asarray(use_morton_raw)):
            require_morton_topology(tree)

    def _leaf_bounds_from_ranges(_):
        leaf_ranges = ranges[num_internal:]
        leaf_starts = leaf_ranges[:, 0]
        leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)
        return _compute_leaf_bounds(
            positions_sorted,
            leaf_starts,
            leaf_counts,
            max_leaf_size=max_leaf_size,
            particle_radius=particle_radius,
        )

    # The Morton shortcut derives leaf bounds from the cell geometry rather than
    # from the particles, so it cannot honour a per-particle radius -- taking it
    # would silently drop the inflation and hand back the understated extents this
    # argument exists to prevent. Use the positional path whenever radii are given.
    if has_morton_fields and particle_radius is None:
        leaf_min, leaf_max = lax.cond(
            use_morton_bounds,
            lambda _: _compute_leaf_bounds_from_morton(tree),
            _leaf_bounds_from_ranges,
            operand=None,
        )
    else:
        leaf_min, leaf_max = _leaf_bounds_from_ranges(None)

    dim = positions_sorted.shape[1]
    num_leaves = num_nodes - num_internal
    # The upward pass decides WHICH leaf bound each internal node inherits (a
    # combinatorial choice, made on stopped values) and then gathers that bound
    # from the live leaf bounds -- see _gather_inherited_bounds. The loop below
    # therefore carries no tangents, which is what lets reverse mode through it:
    # its trip count is traced under jit, and JAX cannot transpose such a loop.
    # One array per node row: the ``dim`` minima, then the ``dim`` maxima, so a
    # level reads each child row once.
    dtype = positions_sorted.dtype
    leaf_bounds = jnp.concatenate(
        [jnp.asarray(leaf_min).astype(dtype), jnp.asarray(leaf_max).astype(dtype)],
        axis=1,
    )
    bounds = jnp.zeros((num_nodes, 2 * dim), dtype=dtype)
    bounds = bounds.at[num_internal:].set(lax.stop_gradient(leaf_bounds))
    # per node and column: the leaf whose bound the node holds (-1: none yet)
    leaf_ids = jnp.broadcast_to(
        jnp.arange(num_leaves, dtype=jnp.int32)[:, None], (num_leaves, 2 * dim)
    )
    source = jnp.full((num_nodes, 2 * dim), -1, dtype=jnp.int32)
    source = source.at[num_internal:].set(leaf_ids)
    is_min_column = jnp.arange(2 * dim) < dim

    left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
    right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)

    if num_internal > 0:
        # Level-parallel upward pass: process one tree level at a time from
        # the deepest internal level up to the root.  All internal nodes at
        # the same level are independent, so the merge within each level is
        # fully vectorised.
        #
        # We derive correct node depths from the parent array instead of
        # relying on tree.node_level (which may contain stale values when
        # the tree build propagated levels in index order rather than BFS
        # order).
        parent = jnp.asarray(tree.parent, dtype=INDEX_DTYPE)
        parent_safe = jnp.where(parent >= 0, parent, as_index(0))

        # Compute depths via pointer doubling: each node maintains a
        # shortcut pointer and an accumulated distance. On each round
        # the shortcut doubles its reach, converging in O(log depth)
        # rounds instead of O(depth).
        is_root = parent < 0
        init_dist = jnp.where(is_root, as_index(0), as_index(1))
        init_shortcut = jnp.where(
            is_root,
            jnp.arange(num_nodes, dtype=INDEX_DTYPE),
            parent_safe,
        )

        def _depth_cond(state):
            _sc, _d, changed = state
            return changed

        def _depth_body(state):
            sc, d, _changed = state
            new_d = d + d[sc]
            new_sc = sc[sc]
            changed = jnp.any(new_sc != sc)
            return new_sc, new_d, changed

        _, node_depth, _ = lax.while_loop(
            _depth_cond,
            _depth_body,
            (init_shortcut, init_dist, jnp.bool_(True)),
        )

        max_depth = jnp.max(node_depth)
        internal_depth = node_depth[:num_internal]

        def _upward_body(rev_idx, state):
            bounds_s, source_s = state
            # rev_idx 0 -> deepest internal level, …
            level = max_depth - as_index(1) - as_index(rev_idx)

            # Mask: which internal nodes sit at this level?
            at_level = (internal_depth == level)[:, None]

            # Merge children bounds (computed for ALL internal nodes but
            # only applied where at_level is True).
            left_b, right_b = bounds_s[left_child], bounds_s[right_child]
            merged = jnp.where(
                is_min_column,
                jnp.minimum(left_b, right_b),
                jnp.maximum(left_b, right_b),
            )
            # The merged value IS one of the two children's (bit for bit, signed
            # zeros included): record which, by its bits.
            merged_source = jnp.where(
                _same_bits(merged, left_b),
                source_s[left_child],
                source_s[right_child],
            )

            def _apply(full, new):
                return full.at[:num_internal].set(
                    jnp.where(at_level, new, full[:num_internal])
                )

            return _apply(bounds_s, merged), _apply(source_s, merged_source)

        # fori_loop accepts traced bounds; extra iterations where no node
        # matches a negative level are harmless no-ops.
        num_internal_levels = jnp.maximum(max_depth, as_index(0))
        bounds, source = lax.fori_loop(
            0, num_internal_levels, _upward_body, (bounds, source)
        )

    bounds = _gather_inherited_bounds(bounds, source, leaf_bounds, num_internal)
    mins, maxs = bounds[:, :dim], bounds[:, dim:]

    centers = 0.5 * (mins + maxs)
    half_extents = 0.5 * (maxs - mins)
    min_extent = jnp.asarray(1e-6, dtype=half_extents.dtype)
    half_extents = jnp.maximum(half_extents, min_extent)
    radii = jnp.linalg.norm(half_extents, axis=1)
    max_extents = jnp.max(half_extents, axis=1)

    return TreeGeometry(centers, half_extents, radii, max_extents)


__all__ = [
    "TreeGeometry",
    "compute_tree_geometry",
]
