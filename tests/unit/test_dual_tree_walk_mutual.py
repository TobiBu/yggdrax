"""``dual_tree_walk_mutual`` against the dual-tree walk: same lists, as sets.

The flat-emission walk is the fused single-GPU lane's traversal candidate (plan
"tree walk", 2026-09-10). It emits each unordered pair once with ``a < b``; the
dual-tree walk emits per-node rows in both directions. Fed the dual walk's own
``mac_extents`` and ``mac_type`` (the non-strict acceptance test), the two must
agree exactly as sets -- far pairs after un-mutualising, near pairs per leaf --
with the self pair absent from both. The strict default rule may differ on exact
equality only, which the symmetric sample tree below actually hits.

Also pinned: the walk traces under an outer ``jax.jit``; ``peak_wavefront`` and
``rounds`` are reported and bounded; each overflow flag fires when its capacity
is too small; the index dtype follows ``left_child_full``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig, Tree, compute_tree_geometry
from yggdrax._interactions_impl import (
    MutualWalkResult,
    _build_mac_extents,
    build_interactions_and_neighbors,
    dual_tree_walk_mutual,
)


def _disc(n: int, seed: int = 9):
    rng = np.random.default_rng(seed)
    r = 10.0 * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack(
        [r * np.cos(th), r * np.sin(th), rng.normal(scale=0.2, size=n)], axis=1
    )
    return pos.astype(np.float64), rng.uniform(0.8, 1.2, n).astype(np.float64)


def _tree(n=3000, leaf=16, seed=9):
    pos, mass = _disc(n, seed)
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.asarray(mass), leaf_size=leaf, tree_type="radix"
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=leaf)
    return tree.topology, geometry


def _mutual_inputs(topology, geometry, mac_type, scale=1.0, idx=None):
    idx = topology.parent.dtype if idx is None else idx
    num_internal = int(topology.left_child.shape[0])
    total_nodes = int(topology.parent.shape[0])
    num_leaves = total_nodes - num_internal
    left = jnp.concatenate(
        [jnp.asarray(topology.left_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    right = jnp.concatenate(
        [jnp.asarray(topology.right_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    root = jnp.argmin(topology.parent).astype(idx)
    centers = jnp.asarray(geometry.center)
    extents = _build_mac_extents(
        topology.parent, geometry, num_internal, mac_type, scale
    )[0]
    return left, right, centers, jnp.asarray(extents, centers.dtype), root, num_internal


def _dual_sets(topology, geometry, theta, mac_type, scale=1.0):
    config = DualTreeTraversalConfig(
        max_pair_queue=1 << 16,
        process_block=64,
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
    )
    interactions, neighbors, result = build_interactions_and_neighbors(
        topology,
        geometry,
        theta=theta,
        traversal_config=config,
        mac_type=mac_type,
        dehnen_radius_scale=scale,
        return_result=True,
    )
    assert not (
        bool(result.queue_overflow)
        or bool(result.far_overflow)
        or bool(result.near_overflow)
    )
    src = np.asarray(result.interaction_sources)
    tgt = np.asarray(result.interaction_targets)
    live = (src >= 0) & (tgt >= 0)
    far = {
        (min(a, b), max(a, b)) for a, b in zip(tgt[live].tolist(), src[live].tolist())
    }
    # every accepted pair appears in both directions
    directed = set(zip(tgt[live].tolist(), src[live].tolist()))
    assert all((b, a) in directed for a, b in directed)
    offsets = np.asarray(neighbors.offsets)
    counts = np.asarray(neighbors.counts)
    nbrs = np.asarray(neighbors.neighbors)
    leaf_nodes = np.asarray(neighbors.leaf_indices)
    near = set()
    for row, leaf_node in enumerate(leaf_nodes.tolist()):
        for k in range(int(counts[row])):
            other = int(nbrs[int(offsets[row]) + k])
            assert other != leaf_node, "dual walk emitted a self neighbour"
            near.add((min(leaf_node, other), max(leaf_node, other)))
    return far, near, int(live.sum()), int(counts.sum())


def _mutual_sets(res):
    fa, fb = np.asarray(res.far_a), np.asarray(res.far_b)
    na, nb = np.asarray(res.near_a), np.asarray(res.near_b)
    nf, nn = int(res.far_count), int(res.near_count)
    assert np.all(fa[:nf] < fb[:nf]) and np.all(na[:nn] < nb[:nn]), "canonical a < b"
    assert np.all(fa[nf:] == -1) and np.all(
        na[nn:] == -1
    ), "-1 tail after the live prefix"
    return set(zip(fa[:nf].tolist(), fb[:nf].tolist())), set(
        zip(na[:nn].tolist(), nb[:nn].tolist())
    )


@pytest.mark.parametrize("mac_type", ["bh", "dehnen"])
@pytest.mark.parametrize("theta", [0.3, 0.5, 0.9])
def test_mutual_walk_matches_the_dual_walk_as_sets(mac_type, theta):
    topology, geometry = _tree()
    far_ref, near_ref, n_far_directed, n_near_directed = _dual_sets(
        topology, geometry, theta, mac_type
    )
    left, right, centers, extents, root, _ = _mutual_inputs(
        topology, geometry, mac_type
    )
    res = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        theta,
        root,
        max_pair_queue=1 << 16,
        far_cap=1 << 17,
        near_cap=1 << 17,
        mac_type=mac_type,
    )
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )
    far, near = _mutual_sets(res)
    assert far == far_ref
    assert near == near_ref
    assert 2 * len(far) == n_far_directed and 2 * len(near) == n_near_directed
    assert len(far) > 0 and len(near) > 0


def test_dehnen_radius_scale_reaches_the_acceptance():
    topology, geometry = _tree(seed=3)
    far_ref, near_ref, _, _ = _dual_sets(topology, geometry, 0.5, "dehnen", scale=1.3)
    left, right, centers, extents, root, _ = _mutual_inputs(
        topology, geometry, "dehnen", scale=1.3
    )
    res = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=1 << 16,
        far_cap=1 << 17,
        near_cap=1 << 17,
        mac_type="dehnen",
    )
    far, near = _mutual_sets(res)
    assert far == far_ref and near == near_ref
    # the scale changes the lists (otherwise this test would prove nothing)
    far_unscaled, _, _, _ = _dual_sets(topology, geometry, 0.5, "dehnen", scale=1.0)
    assert far_unscaled != far_ref


def test_strict_default_differs_from_the_dual_walk_at_most_on_equality():
    """The strict rule may only move pairs sitting exactly on the MAC boundary."""
    topology, geometry = _tree(seed=5)
    far_ref, near_ref, _, _ = _dual_sets(topology, geometry, 0.5, "bh")
    left, right, centers, extents, root, _ = _mutual_inputs(topology, geometry, "bh")
    res = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=1 << 16,
        far_cap=1 << 17,
        near_cap=1 << 17,
    )
    far, near = _mutual_sets(res)
    theta_sq = 0.5**2
    c = np.asarray(centers)
    e = np.asarray(extents)
    for a, b in far ^ far_ref:
        d2 = float(np.sum((c[a] - c[b]) ** 2))
        assert np.isclose((e[a] + e[b]) ** 2, theta_sq * d2, rtol=1e-6), (a, b)


def test_traces_under_jit_and_reports_peak_and_rounds():
    topology, geometry = _tree(seed=7)
    left, right, centers, extents, root, _ = _mutual_inputs(
        topology, geometry, "dehnen"
    )
    q = 1 << 15

    @jax.jit
    def run(lf, rf, c, r, rt):
        return dual_tree_walk_mutual(
            lf,
            rf,
            c,
            r,
            0.5,
            rt,
            max_pair_queue=q,
            far_cap=1 << 17,
            near_cap=1 << 17,
            mac_type="dehnen",
        )

    res = run(left, right, centers, extents, root)
    assert not bool(res.queue_overflow)
    peak, rounds = int(res.peak_wavefront), int(res.rounds)
    assert 1 <= peak <= q and rounds > 0
    # a queue exactly at the peak does not overflow; below it does
    ok = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=peak,
        far_cap=1 << 17,
        near_cap=1 << 17,
        mac_type="dehnen",
    )
    assert not bool(ok.queue_overflow) and int(ok.far_count) == int(res.far_count)
    if peak > 4:
        short = dual_tree_walk_mutual(
            left,
            right,
            centers,
            extents,
            0.5,
            root,
            max_pair_queue=peak - 1,
            far_cap=1 << 17,
            near_cap=1 << 17,
            mac_type="dehnen",
        )
        assert bool(short.queue_overflow)


def test_far_and_near_overflow_flags_fire_and_counts_report_the_need():
    topology, geometry = _tree(seed=11)
    left, right, centers, extents, root, _ = _mutual_inputs(
        topology, geometry, "dehnen"
    )
    full = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=1 << 16,
        far_cap=1 << 17,
        near_cap=1 << 17,
        mac_type="dehnen",
    )
    nf, nn = int(full.far_count), int(full.near_count)
    tight_far = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=1 << 16,
        far_cap=max(4, nf // 2),
        near_cap=1 << 17,
        mac_type="dehnen",
    )
    assert bool(tight_far.far_overflow) and int(tight_far.far_count) > max(4, nf // 2)
    tight_near = dual_tree_walk_mutual(
        left,
        right,
        centers,
        extents,
        0.5,
        root,
        max_pair_queue=1 << 16,
        far_cap=1 << 17,
        near_cap=max(4, nn // 2),
        mac_type="dehnen",
    )
    assert bool(tight_near.near_overflow) and int(tight_near.near_count) > max(
        4, nn // 2
    )


def test_index_dtype_follows_the_child_arrays():
    topology, geometry = _tree(seed=13)
    for idx in (jnp.int32, jnp.int64):
        left, right, centers, extents, root, _ = _mutual_inputs(
            topology, geometry, "dehnen", idx=idx
        )
        res = dual_tree_walk_mutual(
            left,
            right,
            centers,
            extents,
            0.5,
            root,
            max_pair_queue=1 << 16,
            far_cap=1 << 17,
            near_cap=1 << 17,
            mac_type="dehnen",
        )
        for leaf in (
            res.far_a,
            res.far_b,
            res.near_a,
            res.near_b,
            res.far_count,
            res.near_count,
            res.peak_wavefront,
            res.rounds,
        ):
            assert leaf.dtype == jnp.dtype(idx), (idx, leaf.dtype)


def test_wavefront_ladder_is_bit_identical_to_the_full_width_body(monkeypatch):
    """Every rung of the ladder yields the same pairs in the same order.

    The 3000-particle tree peaks at a few thousand live pairs, so the floor is
    lowered to exercise five rungs (64, 256, 1024, 4096, 16384) in one walk.
    """
    from yggdrax import _interactions_impl as impl

    topology, geometry = _tree()
    left, right, centers, radii, root, _ = _mutual_inputs(topology, geometry, "dehnen")
    monkeypatch.setattr(impl, "_WAVEFRONT_LADDER_FLOOR", 64)
    assert impl._wavefront_ladder(1 << 14) == (64, 256, 1024, 4096, 1 << 14)
    kw = dict(
        max_pair_queue=1 << 14, far_cap=1 << 16, near_cap=1 << 16, mac_type="dehnen"
    )
    args = (left, right, centers, radii, 0.5, root)
    ladder = dual_tree_walk_mutual(*args, **kw, wavefront_ladder=True)
    full = dual_tree_walk_mutual(*args, **kw, wavefront_ladder=False)
    assert int(ladder.far_count) > 0 and int(ladder.near_count) > 0
    for name in MutualWalkResult._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(ladder, name)),
            np.asarray(getattr(full, name)),
            err_msg=name,
        )
    assert int(ladder.rounds) == int(full.rounds)
    assert int(ladder.peak_wavefront) > 64  # more than one rung was needed


def test_wavefront_ladder_reports_queue_overflow_like_the_full_width_body():
    topology, geometry = _tree()
    left, right, centers, radii, root, _ = _mutual_inputs(topology, geometry, "dehnen")
    kw = dict(max_pair_queue=256, far_cap=1 << 16, near_cap=1 << 16, mac_type="dehnen")
    args = (left, right, centers, radii, 0.5, root)
    ladder = dual_tree_walk_mutual(*args, **kw, wavefront_ladder=True)
    full = dual_tree_walk_mutual(*args, **kw, wavefront_ladder=False)
    assert bool(ladder.queue_overflow) and bool(full.queue_overflow)
    assert int(ladder.peak_wavefront) == int(full.peak_wavefront) > 256


def test_env_switch_sets_the_ladder_default(monkeypatch):
    import importlib
    import subprocess
    import sys

    code = (
        "from yggdrax import _interactions_impl as m; "
        "print(int(m._WAVEFRONT_LADDER_DEFAULT))"
    )
    for value, expect in (("0", "0"), ("1", "1"), (None, "1")):
        env = dict(__import__("os").environ)
        env.pop("YGGDRAX_MUTUAL_WALK_LADDER", None)
        if value is not None:
            env["YGGDRAX_MUTUAL_WALK_LADDER"] = value
        env["JAX_PLATFORMS"] = "cpu"
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expect, (value, out.stdout)
    del importlib, monkeypatch
