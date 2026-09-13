"""``dual_tree_walk_mutual(node_active=...)``: dead nodes emit nothing and change nothing else."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition
from yggdrax._tree_impl import (
    RadixTree,
    _build_balanced_bucket_structure,
    reorder_particles_by_indices,
)
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import dual_tree_walk_mutual
from yggdrax.morton import morton_encode
from yggdrax.tree_moments import compute_tree_mass_moments


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _cells_tree(n=3000, leaf_size=16, capacity=None, seed=0):
    """Cell leaves in a balanced bucket structure padded to ``capacity`` (empty leaves at the end)."""
    P = jnp.asarray(_plummer(n, seed), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    bounds = infer_bounds(P)
    codes = morton_encode(P, bounds)
    order = jnp.argsort(codes, stable=True)
    sorted_codes = codes[order]
    probe = adaptive_cell_leaf_partition(sorted_codes, leaf_size=leaf_size, capacity=n)
    k = int(probe.num_leaves)
    cap = int(capacity or (1 << int(np.ceil(np.log2(1.5 * k)))))
    part = adaptive_cell_leaf_partition(sorted_codes, leaf_size=leaf_size, capacity=cap)
    starts = np.asarray(part.leaf_starts).astype(np.int64)
    ends = np.asarray(part.leaf_ends).astype(np.int64)
    (
        parent,
        left,
        right,
        lil,
        ril,
        node_ranges,
        node_level,
        level_offsets,
        nodes_by_level,
        num_levels,
    ) = _build_balanced_bucket_structure(starts, ends)
    ps, ms, _inv = reorder_particles_by_indices(P, M, order)
    I = jnp.int32
    topo = RadixTree(
        parent=jnp.asarray(parent, I),
        left_child=jnp.asarray(left, I),
        right_child=jnp.asarray(right, I),
        left_is_leaf=jnp.asarray(lil),
        right_is_leaf=jnp.asarray(ril),
        particle_indices=jnp.asarray(order, I),
        morton_codes=sorted_codes,
        node_ranges=jnp.asarray(node_ranges, I),
        num_particles=n,
        num_internal_nodes=cap - 1,
        node_level=jnp.asarray(node_level, I),
        level_offsets=jnp.asarray(level_offsets, I),
        nodes_by_level=jnp.asarray(nodes_by_level, I),
        num_levels=jnp.asarray(num_levels, I),
        bounds_min=jnp.asarray(bounds[0], P.dtype),
        bounds_max=jnp.asarray(bounds[1], P.dtype),
        leaf_codes=sorted_codes[jnp.asarray(np.minimum(starts, n - 1), I)],
        leaf_depths=jnp.asarray(part.leaf_depths, I),
        use_morton_geometry=jnp.asarray(False),
        leaf_size=leaf_size,
    )
    return topo, ps, ms, k, cap


def _walk(
    topo,
    ps,
    ms,
    node_active,
    theta=0.7,
    queue=1 << 17,
    cap=1 << 20,
    allow_overflow=False,
):
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    num_leaves = total - num_internal
    idx = topo.parent.dtype
    left_full = jnp.concatenate([topo.left_child, jnp.full((num_leaves,), -1, idx)])
    right_full = jnp.concatenate([topo.right_child, jnp.full((num_leaves,), -1, idx)])
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    ranges = np.asarray(topo.node_ranges)
    pos = np.asarray(ps, np.float64)
    c = np.asarray(com, np.float64)
    radii = np.zeros(total)
    for i, (a, b) in enumerate(ranges):
        if b >= a:
            radii[i] = np.sqrt(np.max(np.sum((pos[a : b + 1] - c[i]) ** 2, axis=1)))
    res = dual_tree_walk_mutual(
        left_full,
        right_full,
        com,
        jnp.asarray(radii, ps.dtype),
        theta,
        jnp.argmin(topo.parent).astype(idx),
        max_pair_queue=queue,
        far_cap=cap,
        near_cap=cap,
        mac_type="dehnen",
        node_active=node_active,
    )
    if not allow_overflow:
        assert not (
            bool(res.far_overflow)
            or bool(res.near_overflow)
            or bool(res.queue_overflow)
        )
    fa, fb = (
        np.asarray(res.far_a)[: int(res.far_count)],
        np.asarray(res.far_b)[: int(res.far_count)],
    )
    na, nb = (
        np.asarray(res.near_a)[: int(res.near_count)],
        np.asarray(res.near_b)[: int(res.near_count)],
    )
    return (
        set(zip(fa.tolist(), fb.tolist())),
        set(zip(na.tolist(), nb.tolist())),
        num_internal,
    )


def test_padding_nodes_are_dead_with_the_mask_and_flood_without_it():
    topo, ps, ms, k, cap = _cells_tree()
    ranges = np.asarray(topo.node_ranges)
    active = jnp.asarray(ranges[:, 1] >= ranges[:, 0])
    assert int(cap - k) > 0 and not bool(jnp.all(active))
    far_m, near_m, num_internal = _walk(topo, ps, ms, active)
    # the unmasked walk floods on the empty leaves (one centre, radius zero):
    # give it a wide queue and wide lists so the comparison below is on full lists
    far_0, near_0, _ = _walk(topo, ps, ms, None, queue=1 << 18, cap=1 << 22)
    live_leaves = set(range(num_internal, num_internal + k))
    empty_nodes = set(np.flatnonzero(~np.asarray(active)).tolist())
    # with the mask: no pair touches an empty node, and there ARE pairs
    assert far_m and near_m
    assert not any(a in empty_nodes or b in empty_nodes for a, b in far_m | near_m)
    assert all(a in live_leaves and b in live_leaves for a, b in near_m)
    # without it the empty leaves (one centre, radius 0) pair up as near pairs
    assert any(a in empty_nodes or b in empty_nodes for a, b in near_0)
    # and the live-only part of the unmasked lists is what the mask keeps
    far_0_live = {
        p for p in far_0 if p[0] not in empty_nodes and p[1] not in empty_nodes
    }
    near_0_live = {
        p for p in near_0 if p[0] not in empty_nodes and p[1] not in empty_nodes
    }
    assert near_m == near_0_live
    assert (
        far_m <= far_0_live
    )  # a dead ancestor is never split, so no descendants' pairs either


def test_all_active_mask_is_identical_to_no_mask():
    topo, ps, ms, k, cap = _cells_tree(capacity=None)
    # a tree with no padding: capacity == live leaves
    topo2, ps2, ms2, k2, cap2 = (
        _cells_tree(capacity=int(k)) if (k & (k - 1)) == 0 else (None,) * 5
    )
    if topo2 is None:
        pytest.skip(
            "live leaf count is not a power of two; padded tree exercised by the other test"
        )
    ones = jnp.ones((int(topo2.parent.shape[0]),), dtype=bool)
    assert _walk(topo2, ps2, ms2, ones)[:2] == _walk(topo2, ps2, ms2, None)[:2]
