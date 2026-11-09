"""``dual_tree_walk_mutual(pair_discard=...)``: the pruning hook C1 relies on.

* a test that never fires leaves the far and near lists bit-identical;
* one that always fires prunes at the root, so nothing is emitted;
* a range cut (prune when the two bounding spheres are farther apart than ``R``)
  under a periodic box keeps every particle pair closer than ``R`` covered exactly
  once by the near list or a leaf's own block;
* the Pallas backend refuses the hook rather than ignoring it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import Tree, compute_tree_geometry
from yggdrax._interactions_impl import dual_tree_walk_mutual, mutual_walk_inputs

_L = 1.0


def _setup(n=1200, leaf=16, box=None, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, _L, (n, 3))
    kw = dict(box_size=box) if box is not None else {}
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.ones(n), leaf_size=leaf, tree_type="radix", **kw
    )
    geom = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=leaf)
    w = mutual_walk_inputs(tree.topology, geom, mac_type="dehnen", box_size=box)
    return tree, geom, w


def _never(data, a, b, dist_sq, ra, rb):
    del data, a, b, ra, rb
    return jnp.zeros(dist_sq.shape, bool)


def _always(data, a, b, dist_sq, ra, rb):
    del data, a, b, ra, rb
    return jnp.ones(dist_sq.shape, bool)


def _beyond(data, a, b, dist_sq, ra, rb):
    del a, b
    s = ra + rb + data
    return dist_sq > s * s


def _walk(w, **kw):
    return dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        0.5,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=1 << 15,
        near_cap=1 << 15,
        mac_type="dehnen",
        box_size=w.box_size,
        periodic_radii=w.periodic_radii,
        **kw,
    )


def test_a_test_that_never_fires_changes_nothing():
    _, _, w = _setup()
    base = _walk(w)
    hooked = _walk(w, pair_discard=_never)
    for f in ("far_a", "far_b", "near_a", "near_b", "far_count", "near_count"):
        np.testing.assert_array_equal(getattr(base, f), getattr(hooked, f))


def test_a_test_that_always_fires_prunes_the_root():
    _, _, w = _setup()
    res = _walk(w, pair_discard=_always)
    assert int(res.far_count) == 0 and int(res.near_count) == 0


def test_a_periodic_range_cut_covers_every_close_pair_once():
    R = 0.08
    tree, geom, w = _setup(n=800, box=_L)
    res = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        jnp.asarray(geom.center),
        jnp.asarray(geom.radius),
        0.0,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=8,
        near_cap=1 << 15,
        node_active=w.node_active,
        box_size=w.box_size,
        periodic_radii=w.periodic_radii,
        pair_accept=_never,
        pair_discard=_beyond,
        pair_accept_data=jnp.asarray(R),
    )
    assert not bool(res.near_overflow | res.queue_overflow)
    assert int(res.far_count) == 0
    ranges = np.asarray(tree.node_ranges)
    pos = np.asarray(tree.positions_sorted)
    n = len(pos)
    leaf_of = np.empty(n, np.int64)
    ni = int(tree.topology.left_child.shape[0])
    for leaf in range(ni, len(ranges)):
        s, e = ranges[leaf]
        leaf_of[s : e + 1] = leaf
    seen = np.zeros((n, n), np.int64)
    for a, b in zip(np.asarray(res.near_a), np.asarray(res.near_b)):
        if a < 0:
            continue
        sa, ea = ranges[a]
        sb, eb = ranges[b]
        seen[sa : ea + 1, sb : eb + 1] += 1
        seen[sb : eb + 1, sa : ea + 1] += 1
    same = leaf_of[:, None] == leaf_of[None, :]
    d = pos[:, None] - pos[None]
    d -= _L * np.round(d / _L)
    close = np.sqrt((d * d).sum(-1)) < R
    off = ~np.eye(n, dtype=bool)
    # every close pair in different leaves is in exactly one near pair
    assert np.all(seen[close & ~same & off] == 1)
    # no pair is emitted twice
    assert seen.max() <= 1


def test_pallas_refuses_the_hook():
    _, _, w = _setup(n=200)
    with pytest.raises(ValueError, match="pair_discard"):
        dual_tree_walk_mutual(
            w.left_child_full,
            w.right_child_full,
            w.centers,
            w.radii,
            0.5,
            w.root,
            max_pair_queue=1024,
            far_cap=1024,
            near_cap=1024,
            mac_type="dehnen",
            pair_discard=_never,
            backend="pallas",
            pallas_interpret=True,
        )
