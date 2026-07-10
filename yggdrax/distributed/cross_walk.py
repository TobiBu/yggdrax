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
from typing import Optional

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
    _compute_effective_extents,
    _compute_leaf_effective_extents,
    _compute_mac_ok,
    _default_pair_actions_only,
    _per_key_prefix,
    _resolve_leaf_ordering,
)
from ..dtypes import INDEX_DTYPE, as_index
from ..geometry import TreeGeometry

# One target and one source child at most, expanded per split case.
_MAX_CROSS_REFINEMENT_PAIRS = 4


def _build_mac_extents(parent, geometry, num_internal, mac_type, dehnen_radius_scale):
    """Per-node MAC extent proxy (box or sphere) for one tree.

    Mirrors the extent selection in ``_dual_tree_walk_impl`` (internal nodes use
    the far proxy, leaves the leaf proxy; dehnen scales the radius).
    """

    extents_box = jnp.asarray(geometry.max_extent)
    extents_sphere = jnp.asarray(geometry.radius)
    eff_box_far = _compute_effective_extents(parent, extents_box)
    eff_box_leaf = _compute_leaf_effective_extents(parent, extents_box, num_internal)
    eff_sph_far = _compute_effective_extents(parent, extents_sphere)
    eff_sph_leaf = _compute_leaf_effective_extents(parent, extents_sphere, num_internal)

    use_sphere = (mac_type == "dehnen") | (mac_type == "engblom")
    extents_far = jnp.where(use_sphere, eff_sph_far, eff_box_far)
    extents_leaf = jnp.where(use_sphere, eff_sph_leaf, eff_box_leaf)

    scale = jnp.asarray(dehnen_radius_scale, dtype=extents_box.dtype)
    is_dehnen = jnp.asarray(mac_type == "dehnen")
    extents_far = jnp.where(is_dehnen, scale * extents_far, extents_far)
    extents_leaf = jnp.where(is_dehnen, scale * extents_leaf, extents_leaf)

    total = parent.shape[0]
    node_idx = jnp.arange(total, dtype=INDEX_DTYPE)
    return jnp.where(node_idx >= as_index(num_internal), extents_leaf, extents_far)


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
    t_extents = _build_mac_extents(
        t_parent, target_geometry, t_internal, mac_type, dehnen_radius_scale
    )
    s_extents = _build_mac_extents(
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


__all__ = ["dual_tree_walk_cross", "dual_tree_walk_cross_impl"]
