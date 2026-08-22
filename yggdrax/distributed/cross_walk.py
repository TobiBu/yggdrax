"""Cross-tree dual walk for Yggdrax multi-GPU (Phase 3 core).

The single-device walk (``_interactions_impl._dual_tree_walk_impl``) is a *self*
traversal: it seeds root-vs-root of one tree, exploits symmetry (forward +
backward emission, canonicalised ``min/max`` node pairs, ``same``-node handling)
and lives in one node index space. Distributed FMM needs a *cross* walk: local
**target** nodes against imported remote **source** nodes -- two distinct trees /
index spaces, no symmetry.

This module implements that cross walk additively (it does not touch the
production self-walk). It reuses the self-walk's MAC, action, extent and
prefix helpers, but:

* seeds ``(target_root, source_root)`` and refines target/source children in
  their own index spaces (ordered ``(target, source)`` pairs -- never swapped);
* emits **forward only** -- ``target_node <- source_node`` for the far list and
  ``target_leaf <- source_leaf`` for the near list;
* drops all ``same``-node / self-exclusion logic (the trees are disjoint).

Returns the same :class:`DualTreeWalkResult` contract, so downstream
interaction/neighbour consumers work unchanged -- with the understanding that
``interaction_targets``/``neighbour rows`` index the *target* tree and
``interaction_sources``/``neighbor_indices`` index the *source* tree.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._interactions_impl import (
    _ACTION_ACCEPT,
    _ACTION_NEAR,
    _ACTION_REFINE,
    _DEFAULT_PAIR_BATCH,
    DualTreeWalkResult,
    MACType,
    _build_mac_extents,
    _compute_mac_ok,
    _default_pair_actions_only,
    _per_key_prefix,
    _resolve_leaf_ordering,
)
from ..dtypes import INDEX_DTYPE, as_index
from ..geometry import TreeGeometry

# One target and one source child at most, expanded per split case.
_MAX_CROSS_REFINEMENT_PAIRS = 4


def _children_full(tree, total_nodes, num_internal):
    """Combined-index child arrays padded so leaves index in range (-1 child)."""

    leaf_fill = jnp.full((total_nodes - num_internal,), -1, dtype=INDEX_DTYPE)
    left = jnp.concatenate([tree.left_child, leaf_fill], axis=0)
    right = jnp.concatenate([tree.right_child, leaf_fill], axis=0)
    return left, right


def dual_tree_walk_cross_impl(
    target_tree: object,
    target_geometry: TreeGeometry,
    source_tree: object,
    source_geometry: TreeGeometry,
    theta: float,
    *,
    mac_type: MACType = "bh",
    dehnen_radius_scale: float = 1.0,
    max_interactions_per_node: int,
    max_neighbors_per_leaf: int,
    max_pair_queue: int,
    collect_far: bool = True,
    collect_near: bool = True,
) -> DualTreeWalkResult:
    """Dual walk of target-tree nodes against source-tree nodes (un-jitted impl).

    Call this raw form inside another transform (e.g. ``shard_map``); use the
    jitted :func:`dual_tree_walk_cross` wrapper standalone.

    Far list is keyed by target node (``target_node <- source_node``); near list
    is keyed by target leaf (``target_leaf <- source_leaf``). Fixed-capacity,
    static output shapes, overflow flags returned -- safe to call under
    ``shard_map`` with capacities chosen as static args.
    """

    t_parent = target_tree.parent
    t_total = t_parent.shape[0]
    t_internal = target_tree.left_child.shape[0]
    t_leaves = t_total - t_internal

    s_parent = source_tree.parent
    s_total = s_parent.shape[0]
    s_internal = source_tree.left_child.shape[0]
    s_leaves = s_total - s_internal

    t_leaf_indices, t_leaf_position, _a, _b = _resolve_leaf_ordering(
        target_tree, total_nodes=t_total, num_internal=t_internal
    )
    _s_leaf_indices, s_leaf_position, _c, _d = _resolve_leaf_ordering(
        source_tree, total_nodes=s_total, num_internal=s_internal
    )

    # Degenerate trees (single leaf, no internal nodes): nothing to refine.
    if t_internal == 0 or s_internal == 0:
        return DualTreeWalkResult(
            interaction_offsets=jnp.zeros((t_total + 1,), dtype=INDEX_DTYPE),
            interaction_sources=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_targets=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_tags=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_counts=jnp.zeros((t_total,), dtype=INDEX_DTYPE),
            neighbor_offsets=jnp.zeros((t_leaves + 1,), dtype=INDEX_DTYPE),
            neighbor_indices=jnp.zeros((0,), dtype=INDEX_DTYPE),
            neighbor_counts=jnp.zeros((t_leaves,), dtype=INDEX_DTYPE),
            leaf_indices=t_leaf_indices,
            far_pair_count=as_index(0),
            near_pair_count=as_index(0),
            queue_overflow=jnp.bool_(False),
            far_overflow=jnp.bool_(False),
            near_overflow=jnp.bool_(False),
            accept_decisions=as_index(0),
            near_decisions=as_index(0),
            refine_decisions=as_index(0),
        )

    t_centers = jnp.asarray(target_geometry.center)
    s_centers = jnp.asarray(source_geometry.center)
    t_extents, _ = _build_mac_extents(
        t_parent, target_geometry, t_internal, mac_type, dehnen_radius_scale
    )
    s_extents, _ = _build_mac_extents(
        s_parent, source_geometry, s_internal, mac_type, dehnen_radius_scale
    )
    theta_sq = jnp.asarray(theta, dtype=t_centers.dtype) ** 2

    t_left, t_right = _children_full(target_tree, t_total, t_internal)
    s_left, s_right = _children_full(source_tree, s_total, s_internal)

    t_root = as_index(jnp.argmin(t_parent))
    s_root = as_index(jnp.argmin(s_parent))

    cap = max(int(max_pair_queue), 4)
    wf_indices = jnp.arange(cap, dtype=INDEX_DTYPE)
    wf_t = jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[0].set(t_root)
    wf_s = jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[0].set(s_root)

    Kf = max_interactions_per_node
    Kn = max_neighbors_per_leaf
    far_buffer = jnp.full((t_total, Kf), -1, dtype=INDEX_DTYPE)
    far_counts = jnp.zeros((t_total,), dtype=INDEX_DTYPE)
    nbr_buffer = jnp.full((t_leaves, Kn), -1, dtype=INDEX_DTYPE)
    near_counts = jnp.zeros((t_leaves,), dtype=INDEX_DTYPE)

    t_internal_v = as_index(t_internal)
    s_internal_v = as_index(s_internal)

    filler = jnp.asarray([-1, -1], dtype=INDEX_DTYPE)

    def _ordered(t, s):
        return jnp.stack([t, s], axis=0)

    def _refine(tgt, src, sb, st, ss, tl, tr, sl, sr):
        # ordered (target, source); never swapped (distinct index spaces)
        both = jnp.stack(
            [_ordered(tl, sl), _ordered(tl, sr), _ordered(tr, sl), _ordered(tr, sr)],
            axis=0,
        )
        only_t = jnp.stack(
            [_ordered(tl, src), _ordered(tr, src), filler, filler], axis=0
        )
        only_s = jnp.stack(
            [_ordered(tgt, sl), _ordered(tgt, sr), filler, filler], axis=0
        )
        empty = jnp.tile(filler[None, :], (_MAX_CROSS_REFINEMENT_PAIRS, 1))
        result = empty
        result = jnp.where(ss, only_s, result)
        result = jnp.where(st, only_t, result)
        result = jnp.where(sb, both, result)
        return result

    refine_vm = jax.vmap(_refine, in_axes=(0,) * 9)

    def cond_fun(state):
        size, over_wf, over_far, over_near = state[2], state[8], state[9], state[10]
        return (size > 0) & (~over_wf) & (~over_far) & (~over_near)

    def body_fun(state):
        (
            wf_t,
            wf_s,
            wf_size,
            far_buffer,
            far_counts,
            nbr_buffer,
            near_counts,
            far_total,
            over_wf,
            over_far,
            over_near,
            n_accept,
            n_near,
            n_refine,
        ) = state

        valid = (wf_indices < wf_size) & (wf_t >= 0) & (wf_s >= 0)
        vb = valid.astype(jnp.bool_)
        st_t = jnp.where(valid, wf_t, as_index(0))
        st_s = jnp.where(valid, wf_s, as_index(0))

        ct = t_centers[st_t]
        cs = s_centers[st_s]
        delta = (ct - cs) * valid[:, None].astype(ct.dtype)
        dist_sq = jnp.sum(delta * delta, axis=1)

        et = t_extents[st_t]
        es = s_extents[st_s]
        mac_ok = _compute_mac_ok(
            mac_type=mac_type,
            theta_sq=theta_sq,
            dist_sq=dist_sq,
            extent_target=et,
            extent_source=es,
            valid_pairs=vb,
            different_nodes=vb,  # disjoint trees: always "different"
        )

        t_int = vb & (wf_t < t_internal_v)
        s_int = vb & (wf_s < s_internal_v)
        t_leaf = vb & (~t_int)
        s_leaf = vb & (~s_int)

        actions = _default_pair_actions_only(
            mac_ok=mac_ok,
            valid_pairs=vb,
            different_nodes=vb,
            target_leaf=t_leaf,
            source_leaf=s_leaf,
        )
        accept = vb & (actions == as_index(_ACTION_ACCEPT))
        near = vb & (actions == as_index(_ACTION_NEAR))
        refine = vb & (actions == as_index(_ACTION_REFINE))

        split_t = refine & t_int & ((~s_int) | (et >= es))
        split_s = refine & s_int & ((~t_int) | (es > et))
        split_both = split_t & split_s

        tl = t_left[st_t]
        tr = t_right[st_t]
        sl = s_left[st_s]
        sr = s_right[st_s]

        # ---- far update (forward only: target_node <- source_node) ----
        if collect_far:

            def _far(carry):
                buf, cnts, tot, ofl = carry
                prefix = _per_key_prefix(
                    jnp.where(accept, st_t, as_index(-1)), accept, t_total
                )
                slot = cnts[st_t] + prefix
                ok = accept & (slot < as_index(Kf))
                ofl = ofl | jnp.any(accept & (slot >= as_index(Kf)))
                row = jnp.where(ok, st_t, as_index(t_total))
                col = jnp.where(ok, slot, as_index(Kf - 1))
                buf = buf.at[row, col].set(
                    jnp.where(ok, st_s, as_index(-1)), mode="drop"
                )
                cnts = cnts + jax.ops.segment_sum(
                    ok.astype(INDEX_DTYPE), st_t, num_segments=t_total
                )
                tot = tot + jnp.sum(ok.astype(INDEX_DTYPE), dtype=INDEX_DTYPE)
                return buf, cnts, tot, ofl

            far_buffer, far_counts, far_total, over_far = lax.cond(
                jnp.any(accept),
                _far,
                lambda c: c,
                (far_buffer, far_counts, far_total, over_far),
            )

        # ---- near update (forward only: target_leaf <- source_leaf node) ----
        if collect_near:

            def _near(carry):
                buf, cnts, ofl = carry
                lt = jnp.where(near, t_leaf_position[st_t], as_index(0))
                prefix = _per_key_prefix(
                    jnp.where(near, lt, as_index(-1)), near, t_leaves
                )
                slot = cnts[lt] + prefix
                ok = near & (slot < as_index(Kn))
                ofl = ofl | jnp.any(near & (slot >= as_index(Kn)))
                row = jnp.where(ok, lt, as_index(t_leaves))
                col = jnp.where(ok, slot, as_index(Kn - 1))
                buf = buf.at[row, col].set(
                    jnp.where(ok, st_s, as_index(-1)), mode="drop"
                )
                cnts = cnts + jax.ops.segment_sum(
                    ok.astype(INDEX_DTYPE), lt, num_segments=t_leaves
                )
                return buf, cnts, ofl

            nbr_buffer, near_counts, over_near = lax.cond(
                jnp.any(near),
                _near,
                lambda c: c,
                (nbr_buffer, near_counts, over_near),
            )

        # ---- refine -> next wavefront ----
        pairs = refine_vm(
            st_t,
            st_s,
            split_both.astype(jnp.bool_),
            split_t.astype(jnp.bool_),
            split_s.astype(jnp.bool_),
            tl,
            tr,
            sl,
            sr,
        )
        rt = pairs[..., 0].reshape((cap * _MAX_CROSS_REFINEMENT_PAIRS,))
        rs = pairs[..., 1].reshape((cap * _MAX_CROSS_REFINEMENT_PAIRS,))
        push = (rt >= 0) & (rs >= 0)
        pos = jnp.cumsum(push.astype(INDEX_DTYPE), dtype=INDEX_DTYPE) - push.astype(
            INDEX_DTYPE
        )
        push_ok = push & (pos < as_index(cap))
        over_wf = over_wf | jnp.any(push & (pos >= as_index(cap)))
        slot = jnp.where(push_ok, pos, as_index(cap))
        new_t = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE)
            .at[slot]
            .set(jnp.where(push_ok, rt, as_index(-1)), mode="drop")
        )
        new_s = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE)
            .at[slot]
            .set(jnp.where(push_ok, rs, as_index(-1)), mode="drop")
        )
        new_size = jnp.sum(push_ok.astype(INDEX_DTYPE), dtype=INDEX_DTYPE)

        return (
            new_t,
            new_s,
            new_size,
            far_buffer,
            far_counts,
            nbr_buffer,
            near_counts,
            far_total,
            over_wf,
            over_far,
            over_near,
            n_accept + jnp.sum(accept.astype(INDEX_DTYPE), dtype=INDEX_DTYPE),
            n_near + jnp.sum(near.astype(INDEX_DTYPE), dtype=INDEX_DTYPE),
            n_refine + jnp.sum(refine.astype(INDEX_DTYPE), dtype=INDEX_DTYPE),
        )

    init = (
        wf_t,
        wf_s,
        as_index(1),
        far_buffer,
        far_counts,
        nbr_buffer,
        near_counts,
        as_index(0),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        as_index(0),
        as_index(0),
        as_index(0),
    )
    (
        _wt,
        _ws,
        _sz,
        far_buffer,
        far_counts,
        nbr_buffer,
        near_counts,
        _far_total,
        over_wf,
        over_far,
        over_near,
        n_accept,
        n_near,
        n_refine,
    ) = lax.while_loop(cond_fun, body_fun, init)

    # ---- compact far_buffer -> flat (target-node level order) ----
    # Sources are laid out in *level* order, so offsets must be scattered by
    # node (offsets[node] = that node's level-major start) to stay consistent --
    # exactly as yggdrax's _result_to_interactions does, so this feeds jaccpot's
    # accumulate_m2l_contributions directly.
    nbl = jnp.asarray(target_tree.nodes_by_level, dtype=INDEX_DTYPE)
    num_nbl = nbl.shape[0]
    max_far = max(t_total * Kf, 1)
    if collect_far:
        level_counts = far_counts[nbl]
        write_off = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=INDEX_DTYPE),
                jnp.cumsum(level_counts, dtype=INDEX_DTYPE),
            ]
        )
        node_rep = jnp.repeat(jnp.arange(num_nbl, dtype=INDEX_DTYPE), Kf)
        slot_rep = jnp.tile(jnp.arange(Kf, dtype=INDEX_DTYPE), num_nbl)
        node_ids = nbl[node_rep]
        valid_s = slot_rep < far_counts[node_ids]
        write_pos = write_off[node_rep] + slot_rep
        src_vals = far_buffer[node_ids, slot_rep]
        safe = jnp.where(valid_s, write_pos, as_index(max_far))
        interaction_sources = (
            jnp.full((max_far,), -1, dtype=INDEX_DTYPE)
            .at[safe]
            .set(src_vals, mode="drop")
        )
        interaction_targets = (
            jnp.full((max_far,), -1, dtype=INDEX_DTYPE)
            .at[safe]
            .set(node_ids, mode="drop")
        )
        far_node_offsets = (
            jnp.zeros((t_total,), dtype=INDEX_DTYPE).at[nbl].set(write_off[:-1])
        )
        interaction_offsets = jnp.concatenate(
            [far_node_offsets, jnp.sum(far_counts, dtype=INDEX_DTYPE)[None]]
        )
    else:
        interaction_sources = jnp.zeros((0,), dtype=INDEX_DTYPE)
        interaction_targets = jnp.zeros((0,), dtype=INDEX_DTYPE)
        interaction_offsets = jnp.zeros((t_total + 1,), dtype=INDEX_DTYPE)

    # ---- compact nbr_buffer -> flat (target-leaf order) ----
    # Neighbours are laid out in leaf-row order, so CSR offsets = prefix sum of
    # near_counts (length t_leaves+1).
    max_near = max(t_leaves * Kn, 1)
    if collect_near:
        n_node_rep = jnp.repeat(jnp.arange(t_leaves, dtype=INDEX_DTYPE), Kn)
        n_slot_rep = jnp.tile(jnp.arange(Kn, dtype=INDEX_DTYPE), t_leaves)
        n_valid = n_slot_rep < near_counts[n_node_rep]
        n_write_off = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=INDEX_DTYPE),
                jnp.cumsum(near_counts, dtype=INDEX_DTYPE),
            ]
        )
        n_write_pos = n_write_off[n_node_rep] + n_slot_rep
        n_vals = nbr_buffer[n_node_rep, n_slot_rep]
        n_safe = jnp.where(n_valid, n_write_pos, as_index(max_near))
        neighbor_indices = (
            jnp.full((max_near,), -1, dtype=INDEX_DTYPE)
            .at[n_safe]
            .set(n_vals, mode="drop")
        )
        neighbor_offsets = n_write_off
    else:
        neighbor_indices = jnp.full((0,), -1, dtype=INDEX_DTYPE)
        neighbor_offsets = jnp.zeros((t_leaves + 1,), dtype=INDEX_DTYPE)

    return DualTreeWalkResult(
        interaction_offsets=interaction_offsets,
        interaction_sources=interaction_sources,
        interaction_targets=interaction_targets,
        interaction_tags=jnp.full(
            (interaction_sources.shape[0],), -1, dtype=INDEX_DTYPE
        ),
        interaction_counts=far_counts,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        neighbor_counts=near_counts,
        leaf_indices=t_leaf_indices,
        far_pair_count=jnp.sum(far_counts, dtype=INDEX_DTYPE),
        near_pair_count=jnp.sum(near_counts, dtype=INDEX_DTYPE),
        queue_overflow=over_wf,
        far_overflow=over_far,
        near_overflow=over_near,
        accept_decisions=n_accept,
        near_decisions=n_near,
        refine_decisions=n_refine,
    )


dual_tree_walk_cross = partial(
    jax.jit,
    static_argnames=(
        "max_interactions_per_node",
        "max_neighbors_per_leaf",
        "max_pair_queue",
        "mac_type",
        "collect_far",
        "collect_near",
    ),
)(dual_tree_walk_cross_impl)


# ---------------------------------------------------------------------------
# Cross-domain canonical ownership
# ---------------------------------------------------------------------------
#
# A mutual (momentum-conserving) FMM evaluates each unordered pair ONCE and
# applies +f/-f. Within one device the self-walk gets that for free by emitting a
# canonical `a < b` from a single tree (see `dual_tree_walk_mutual`). Across
# devices it does not: the cross walk runs on BOTH sides of a boundary -- device p
# walks its tree against q's imported nodes, and q walks its tree against p's --
# so both discover the *same* geometric pair. Exactly one of them must emit it.
#
# Get this wrong in either direction and the failure is quiet. Emit on both and the
# pair is double-counted; emit on neither and it is dropped. In BOTH cases every
# device's local momentum sum still looks perfect, because +f/-f cancel within
# whatever each device did do -- the error shows up only in the force, at the
# percent level, and only in a global sum. That is why this predicate is a named,
# tested function rather than an inline comparison.


def cross_pair_owner(
    device_a: Array,
    index_a: Array,
    device_b: Array,
    index_b: Array,
) -> Array:
    """Which device owns the cross-domain pair ``(a, b)``.

    Both endpoints are identified globally by ``(device, local node index)``, which
    is the ordering the single-device walk did not need: local indices alone are
    ambiguous across devices, since every device numbers its own nodes from zero.

    The rule is symmetric under swapping ``a`` and ``b`` -- it has to be, because
    the two devices see the pair in opposite orders and must still agree. It orders
    the two endpoints by their global key and then picks by the parity of the index
    sum:

    * ``(index_a + index_b)`` even -> the lower key's device owns it;
    * odd -> the higher key's device owns it.

    Ordering alone (always "lower key wins") would also be consistent, but it gives
    every cross pair between two domains to the same device, so with an SFC
    partition device 0 accumulates a boundary with each of its neighbours while the
    last device gets almost none. The parity term splits each domain-pair's work
    roughly 50/50 while staying a pure function of data both sides already have --
    so it needs no agreement round.

    Parameters
    ----------
    device_a:
        Owning device of the ``a`` endpoint.
    index_a:
        Node index of ``a`` within its own device's tree.
    device_b:
        Owning device of the ``b`` endpoint.
    index_b:
        Node index of ``b`` within its own device's tree.

    Returns
    -------
    Array
        The owning device id, broadcast over the inputs.
    """
    a_first = (device_a < device_b) | ((device_a == device_b) & (index_a <= index_b))
    lo_dev = jnp.where(a_first, device_a, device_b)
    hi_dev = jnp.where(a_first, device_b, device_a)
    even = ((index_a + index_b) % 2) == 0
    return jnp.where(even, lo_dev, hi_dev)


def cross_pair_is_owned(
    this_device: Array,
    index_local: Array,
    source_device: Array,
    index_remote: Array,
) -> Array:
    """Whether *this* device should emit the pair, from its own point of view.

    Thin wrapper over :func:`cross_pair_owner` in the argument order a walk
    actually has them: its own device and local node, and the source device and the
    remote node. Both sides of a boundary calling this on the same geometric pair
    get exactly one ``True`` between them.

    Parameters
    ----------
    this_device:
        The device running the walk.
    index_local:
        Node index on this device.
    source_device:
        Device the remote node came from.
    index_remote:
        Node index on the source device.

    Returns
    -------
    Array
        True where this device owns the pair.
    """
    return (
        cross_pair_owner(this_device, index_local, source_device, index_remote)
        == this_device
    )


def single_owner_domain(
    left_child_full: Array,
    right_child_full: Array,
    tag_domain: Array,
    *,
    max_depth: int,
) -> Array:
    """Per-node owning domain, or ``-1`` where a node straddles several.

    A merged remote coarse tree (:func:`~yggdrax.distributed.let.build_remote_coarse_tree`)
    holds *other* domains' frontiers in ONE tree, so its leaves each carry a single
    origin domain but its internal nodes generally aggregate several. That matters
    for a mutual force: the ``-f`` half of an accepted far pair has to be sent
    somewhere, and an internal node spanning three domains has no single somewhere.

    Propagates bottom-up: a leaf owns its own tag; an internal node owns a domain
    only if both children resolve to the *same* one. Runs a fixed number of rounds
    rather than a convergence test so it stays a fixed-shape traceable program;
    ``max_depth`` rounds suffice, since information travels one level per round.

    Leaves are never straddling by construction, which is what makes refinement a
    terminating strategy: a walk that refuses to accept a straddling node and
    descends instead always reaches single-owner nodes.

    Parameters
    ----------
    left_child_full:
        ``(total_nodes,)`` left children, -1 for leaves.
    right_child_full:
        ``(total_nodes,)`` right children, -1 for leaves.
    tag_domain:
        ``(total_nodes,)`` origin domain per node; only leaf entries are read.
    max_depth:
        Rounds to propagate, at least the tree depth. Static.

    Returns
    -------
    Array
        ``(total_nodes,)`` owning domain, ``-1`` where the node spans more than one.
    """
    is_leaf = left_child_full < 0
    unknown = as_index(-2)
    owner = jnp.where(is_leaf, tag_domain.astype(INDEX_DTYPE), unknown)

    def round_fn(_: Array, own: Array) -> Array:
        # Narrowed for the type checker: `fori_loop` widens the carry to
        # `Array | tuple[Array, ...]` inside the body, and this carry is one array.
        own = jnp.asarray(own)
        lo = own[jnp.maximum(left_child_full, 0)]
        ro = own[jnp.maximum(right_child_full, 0)]
        resolved = (lo != unknown) & (ro != unknown)
        agree = resolved & (lo == ro) & (lo >= as_index(0))
        internal = jnp.where(resolved, jnp.where(agree, lo, as_index(-1)), unknown)
        return jnp.where(is_leaf, own, internal)

    # `jnp.asarray`: `fori_loop` is typed as returning the carry's union type,
    # `Array | tuple[Array, ...]`, so the narrowing is for the type checker rather
    # than the runtime -- the carry here is a single array.
    owner = jnp.asarray(lax.fori_loop(0, int(max_depth), round_fn, owner))
    # Anything still unresolved is treated as straddling: refuse to accept it and
    # refine instead, which is the safe direction.
    return jnp.where(owner == unknown, as_index(-1), owner)


class CrossMutualWalkResult(NamedTuple):
    """Flat canonical cross-domain pair lists, one entry per owned pair.

    Mirrors ``MutualWalkResult`` from the single-device walk: flat COO rather than
    per-target CSR, because a mutual pair belongs to neither endpoint in
    particular. ``*_local`` indexes this device's tree and ``*_remote`` the source
    device's, so a consumer knows which side to apply ``+f`` to locally and which
    side owes a ``-f`` back to ``source_device``.
    """

    far_local: Array
    far_remote: Array
    far_owner: Array
    far_count: Array
    near_local: Array
    near_remote: Array
    near_owner: Array
    near_count: Array
    far_overflow: Array
    near_overflow: Array
    queue_overflow: Array


def dual_tree_walk_cross_mutual(
    local_left_child_full: Array,
    local_right_child_full: Array,
    local_centers: Array,
    local_radii: Array,
    local_root: Array,
    remote_left_child_full: Array,
    remote_right_child_full: Array,
    remote_centers: Array,
    remote_radii: Array,
    remote_root: Array,
    theta: float,
    *,
    this_device: Array,
    remote_owner: Array,
    remote_index_in_owner: Optional[Array] = None,
    accept_only_remote_leaves: bool = False,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
) -> CrossMutualWalkResult:
    """Cross-domain dual walk emitting each owned pair ONCE, flat.

    The mutual analogue of :func:`dual_tree_walk_cross_impl`, and a separate
    function rather than a flag on it: the production cross walk is target-centric
    and its per-target CSR layout is what the target-centric FMM consumes, so it
    stays untouched. This differs in three ways that cannot be bolted on.

    **Flat, not CSR.** A mutual pair is not owned by its target -- both endpoints
    receive force -- so the output is a flat pair list, which is the shape the
    mutual near and far kernels already take.

    **Filtered by ownership.** Both sides of a boundary run this walk and discover
    the same geometric pairs, so each emits only what :func:`cross_pair_is_owned`
    gives it. Without that filter every cross pair is evaluated twice, and the
    resulting force error is invisible to a per-device momentum check -- see the
    note above :func:`cross_pair_owner`.

    That filter only partitions if both devices key the pair on the *same* two
    numbers, which is what ``remote_index_in_owner`` exists for. The rule is a
    function of ``(device, node index)`` per endpoint, and a node index is only
    canonical in the tree its owner built. When the remote tree IS the source
    device's own tree -- an ``all_gather`` of its node arrays -- the remote node
    index already is that, and the default is right. When the remote tree is a
    merged LET coarse tree
    (:func:`~yggdrax.distributed.let.build_remote_coarse_tree`), it is NOT: that
    tree is built by the importer over every *other* domain's frontier, so its
    numbering is local to the importer and two devices assign different indices to
    the same remote node. Left at the default the two sides then disagree about who
    owns a pair, which drops some pairs and duplicates others -- and both
    failure modes leave every per-device momentum sum exact, so only a global force
    comparison notices.

    **Centres and radii are caller-supplied**, not derived by
    ``_build_mac_extents``, for the same reason the single-device
    ``dual_tree_walk_mutual`` takes them: the mutual MAC is defined on centres of
    MASS and exact max centre-of-mass-to-particle radii, recomputed from the live
    positions on every evaluation, not on the bounding-sphere proxies the
    target-centric walk selects. Different extents accept a different pair set, so
    this stays agnostic rather than silently re-baselining accuracy.

    The MAC is ``theta * |c_b - c_a| > r_a + r_b``, strict, and symmetric in the
    two nodes by construction -- which is what lets one decision serve both
    directions of the pair.

    Parameters
    ----------
    local_left_child_full:
        ``(total_nodes,)`` left-child indices for this device's tree, **-1 for
        leaves** (the -1 is what marks a leaf, so no separate mask is needed).
    local_right_child_full:
        ``(total_nodes,)`` right children, likewise -1 for leaves.
    local_centers:
        ``(total_nodes, 3)`` centres of mass for this device's nodes.
    local_radii:
        ``(total_nodes,)`` max centre-of-mass-to-particle radii.
    local_root:
        Root index in this device's tree.
    remote_left_child_full:
        As above, for the imported source tree.
    remote_right_child_full:
        As above, for the imported source tree.
    remote_centers:
        As above, for the imported source tree.
    remote_radii:
        As above, for the imported source tree.
    remote_root:
        Root index in the source tree.
    theta:
        Opening angle; acceptance is strict.
    this_device:
        Device id running this walk.
    remote_owner:
        ``(remote_total_nodes,)`` owning domain per remote node, ``-1`` where the
        node straddles several -- as produced by :func:`single_owner_domain`. Per
        NODE, not a single scalar, because the imported remote tree is a MERGE of
        every other domain's frontier, so different nodes belong to different
        devices.
    accept_only_remote_leaves:
        Refuse to accept a far pair against an INTERNAL remote node, refining it
        instead. Set this when the remote tree is a merged LET coarse tree: an
        accepted pair's ``-f`` half is a local expansion that has to be addressed to a
        node in its owner's own tree, and only a coarse LEAF corresponds to one such
        node (it is exactly one frontier leaf). Leave it off when the remote tree IS
        the source device's own tree, where every node is addressable by its own index
        and accepting high up prunes more.

        The pruning this costs is bounded and recoverable: the remote side ends up
        represented at frontier-leaf granularity, which is the resolution the frontier
        publishes anyway, and a sender that later wants to accept higher can push its
        coarse local expansions down to the leaves itself without changing what
        crosses the wire.
    remote_index_in_owner:
        ``(remote_total_nodes,)`` each remote node's index **in its owning domain's
        own tree**, which is the only numbering both sides of a boundary agree on.
        ``None`` means "the remote index already is that", which holds exactly when
        the remote tree is the source device's own tree. For a merged coarse tree
        pass ``tag_node_id`` (via each leaf's ``node_ranges`` start), and see the
        paragraph on ownership above for what going without costs.
    max_pair_queue:
        Wavefront capacity in node pairs.
    far_cap:
        Output capacity for the far list.
    near_cap:
        Output capacity for the near list.

    Returns
    -------
    CrossMutualWalkResult
        Flat owned pair lists plus overflow flags. Overflow is reported, never
        silently truncated: a dropped cross pair loses both halves, so momentum
        stays exact and only the force is wrong.
    """
    from yggdrax._interactions_impl import _flat_append

    cap = int(max_pair_queue)
    theta_sq = jnp.asarray(theta, dtype=local_centers.dtype) ** 2
    wf_indices = jnp.arange(cap, dtype=INDEX_DTYPE)
    # `None` stays None rather than becoming an identity table: the check below is a
    # trace-time Python branch, so the default path emits no gather at all.
    r_key = (
        None
        if remote_index_in_owner is None
        else jnp.asarray(remote_index_in_owner).astype(INDEX_DTYPE)
    )

    init = (
        jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[0].set(local_root),
        jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[0].set(remote_root),
        as_index(1),
        jnp.full((far_cap,), -1, dtype=INDEX_DTYPE),
        jnp.full((far_cap,), -1, dtype=INDEX_DTYPE),
        jnp.full((far_cap,), -1, dtype=INDEX_DTYPE),
        as_index(0),
        jnp.full((near_cap,), -1, dtype=INDEX_DTYPE),
        jnp.full((near_cap,), -1, dtype=INDEX_DTYPE),
        jnp.full((near_cap,), -1, dtype=INDEX_DTYPE),
        as_index(0),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
    )

    def cond_fun(state):
        return state[2] > as_index(0)

    def body_fun(state):
        (
            wf_l,
            wf_r,
            wf_size,
            far_l,
            far_r,
            far_o,
            far_n,
            near_l,
            near_r,
            near_o,
            near_n,
            of_far,
            of_near,
            of_wf,
        ) = state
        live = (wf_indices < wf_size) & (wf_l >= 0) & (wf_r >= 0)
        sl = jnp.asarray(jnp.where(live, wf_l, as_index(0)))
        sr = jnp.asarray(jnp.where(live, wf_r, as_index(0)))

        delta = remote_centers[sr] - local_centers[sl]
        dist_sq = jnp.sum(delta * delta, axis=1)
        radius_sum = local_radii[sl] + remote_radii[sr]
        mac_ok = live & (theta_sq * dist_sq > radius_sum * radius_sum)

        l_leaf = local_left_child_full[sl] < 0
        r_leaf = remote_left_child_full[sr] < 0
        both_leaf = l_leaf & r_leaf

        # A far pair may only be ACCEPTED against a single-owner remote node. An
        # internal node of the merged remote tree can aggregate several domains, and
        # the `-f` half of an accepted pair has to be sent somewhere -- a node
        # spanning three domains has no single somewhere. So a straddling node is
        # refined instead, which terminates because coarse leaves each carry exactly
        # one origin domain.
        r_dom = remote_owner[sr]
        single_owner = r_dom >= as_index(0)
        accept_geom = mac_ok & single_owner
        if accept_only_remote_leaves:
            # Same argument one step further: a single OWNER is not yet a single
            # ADDRESS. An internal coarse node spans several of its owner's frontier
            # leaves, so there is no one node in the owner's own tree to send an
            # accepted pair's local expansion to -- and picking the first would be a
            # valid-looking wrong answer. Refining instead terminates for exactly the
            # reason straddling does: coarse leaves are addressable by construction.
            accept_geom = accept_geom & r_leaf
        near_geom = live & (~mac_ok) & both_leaf

        # The ownership rule is only a partition if both devices key the pair
        # identically, and a merged coarse tree renumbers remote nodes -- hence the
        # translation to the owner's own numbering when one was given.
        owned = cross_pair_is_owned(
            this_device, sl, r_dom, sr if r_key is None else r_key[sr]
        )
        emit_far = accept_geom & owned
        emit_near = near_geom & owned
        # The remote index and its owning domain are appended together, so the
        # reverse exchange never has to re-derive a destination from an index.
        far_l, far_r, far_n_next, ofl = _flat_append(
            far_l, far_r, far_n, emit_far, sl, sr, far_cap
        )
        far_o, _unused_a, _unused_n, _unused_o = _flat_append(
            far_o, far_o, far_n, emit_far, r_dom, r_dom, far_cap
        )
        far_n = far_n_next
        near_l, near_r, near_n_next, ofn = _flat_append(
            near_l, near_r, near_n, emit_near, sl, sr, near_cap
        )
        near_o, _unused_b, _unused_m, _unused_p = _flat_append(
            near_o, near_o, near_n, emit_near, r_dom, r_dom, near_cap
        )
        near_n = near_n_next

        # Refinement is deliberately NOT filtered by ownership: a pair this device
        # does not own may still need splitting to reach descendants it does.
        # Filtering here would prune away owned pairs deeper in the tree.
        # Refine anything not emitted that still has structure left: the MAC
        # rejected it, or the MAC accepted it but the remote node straddles domains.
        refine = live & (~accept_geom) & (~near_geom) & (~both_leaf)
        split_l = refine & (~l_leaf) & (r_leaf | (local_radii[sl] >= remote_radii[sr]))
        split_r = refine & (~r_leaf) & (l_leaf | (remote_radii[sr] > local_radii[sl]))
        both = split_l & split_r

        ll = local_left_child_full[sl]
        lr = local_right_child_full[sl]
        rl = remote_left_child_full[sr]
        rr = remote_right_child_full[sr]

        def _cand(sel, a, b):
            return jnp.where(sel, a, as_index(-1)), jnp.where(sel, b, as_index(-1))

        cands = [
            _cand(both, ll, rl),
            _cand(both, ll, rr),
            _cand(both, lr, rl),
            _cand(both, lr, rr),
            _cand(split_l & (~both), ll, sr),
            _cand(split_l & (~both), lr, sr),
            _cand(split_r & (~both), sl, rl),
            _cand(split_r & (~both), sl, rr),
        ]
        cand_l = jnp.concatenate([jnp.asarray(c[0]) for c in cands])
        cand_r = jnp.concatenate([jnp.asarray(c[1]) for c in cands])
        push = (cand_l >= 0) & (cand_r >= 0)
        pos = jnp.cumsum(push.astype(INDEX_DTYPE), dtype=INDEX_DTYPE) - push.astype(
            INDEX_DTYPE
        )
        push_ok = push & (pos < as_index(cap))
        of_wf = of_wf | jnp.any(push & (pos >= as_index(cap)))
        slot = jnp.where(push_ok, pos, as_index(cap))
        new_l = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE)
            .at[slot]
            .set(jnp.where(push_ok, cand_l, as_index(-1)), mode="drop")
        )
        new_r = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE)
            .at[slot]
            .set(jnp.where(push_ok, cand_r, as_index(-1)), mode="drop")
        )
        return (
            new_l,
            new_r,
            jnp.sum(push_ok.astype(INDEX_DTYPE), dtype=INDEX_DTYPE),
            far_l,
            far_r,
            far_o,
            far_n,
            near_l,
            near_r,
            near_o,
            near_n,
            of_far | ofl,
            of_near | ofn,
            of_wf,
        )

    (
        _l,
        _r,
        _sz,
        far_l,
        far_r,
        far_o,
        far_n,
        near_l,
        near_r,
        near_o,
        near_n,
        of_far,
        of_near,
        of_wf,
    ) = lax.while_loop(cond_fun, body_fun, init)
    return CrossMutualWalkResult(
        far_local=far_l,
        far_remote=far_r,
        far_owner=far_o,
        far_count=far_n,
        near_local=near_l,
        near_remote=near_r,
        near_owner=near_o,
        near_count=near_n,
        far_overflow=of_far,
        near_overflow=of_near,
        queue_overflow=of_wf,
    )


__all__ = [
    "CrossMutualWalkResult",
    "single_owner_domain",
    "cross_pair_is_owned",
    "cross_pair_owner",
    "dual_tree_walk_cross",
    "dual_tree_walk_cross_impl",
    "dual_tree_walk_cross_mutual",
]
