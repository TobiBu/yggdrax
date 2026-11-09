"""``dual_tree_walk_mutual(pair_band=...)``: the blend-band hook of the smooth rule.

* a band test that never fires leaves the far and near lists bit-identical and
  logs nothing;
* with a band that fires on part of the accepted pairs, weighting every band
  pair's far term by ``sigma`` and everything below it by ``1 - sigma`` (the
  product along ``far_parent`` / ``near_parent`` / ``band_parent``) covers every
  cross-leaf particle pair with total weight exactly 1 -- the partition of
  unity the smooth acceptance rule is built on;
* a too-small ``band_cap`` raises ``band_overflow``;
* the Pallas backend refuses the hook.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import Tree, compute_tree_geometry
from yggdrax._interactions_impl import dual_tree_walk_mutual, mutual_walk_inputs

_THETA = 0.6


def _setup(n=900, leaf=16, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, (n, 3))
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.ones(n), leaf_size=leaf, tree_type="radix"
    )
    geom = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=leaf)
    w = mutual_walk_inputs(tree.topology, geom, mac_type="dehnen")
    return tree, w


def _never(data, a, b, dist_sq, ra, rb):
    del data, a, b, ra, rb
    return jnp.zeros(dist_sq.shape, bool)


def _outer_annulus(data, a, b, dist_sq, ra, rb):
    """Band: pairs with ``(r_a + r_b) / d`` in ``(data * theta, theta]``."""
    del a, b
    s = ra + rb
    lo = data * _THETA
    return (s * s > lo * lo * dist_sq) & (s * s <= _THETA * _THETA * dist_sq)


def _walk(w, **kw):
    return dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        _THETA,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=1 << 15,
        near_cap=1 << 15,
        mac_type="dehnen",
        **kw,
    )


def test_a_band_that_never_fires_changes_nothing():
    _, w = _setup()
    base = _walk(w)
    hooked = _walk(w, pair_band=_never, band_cap=16)
    for f in ("far_a", "far_b", "near_a", "near_b", "far_count", "near_count"):
        np.testing.assert_array_equal(getattr(base, f), getattr(hooked, f))
    assert int(hooked.band_count) == 0 and not bool(hooked.band_overflow)
    assert np.all(np.asarray(hooked.far_parent) == -1)
    assert np.all(np.asarray(hooked.near_parent) == -1)
    assert base.band_a is None  # the hook-free result is unchanged


@pytest.mark.parametrize("sigma", [0.5, 0.2])
def test_band_weights_are_a_partition_of_unity(sigma):
    tree, w = _setup()
    res = _walk(
        w, pair_band=_outer_annulus, pair_accept_data=jnp.asarray(0.7), band_cap=1 << 14
    )
    assert not bool(
        res.far_overflow | res.near_overflow | res.queue_overflow | res.band_overflow
    )
    nb = int(res.band_count)
    assert nb > 0
    parent = np.asarray(res.band_parent)[:nb]
    accepted = np.asarray(res.band_accepted)[:nb]
    assert np.all(parent < np.arange(nb))  # ancestors are logged first
    factor = np.where(accepted, 1.0 - sigma, 1.0)
    prod = np.ones(nb)
    for j in range(nb):  # slot order is a topological order
        prod[j] = factor[j] * (prod[parent[j]] if parent[j] >= 0 else 1.0)

    def hang(p):
        return prod[p] if p >= 0 else 1.0

    ranges = np.asarray(tree.node_ranges)
    n = int(tree.positions_sorted.shape[0])
    cover = np.zeros((n, n))
    band_set = {
        (int(a), int(b))
        for a, b in zip(np.asarray(res.band_a)[:nb], np.asarray(res.band_b)[:nb])
    }
    fa, fb, fp = (np.asarray(x) for x in (res.far_a, res.far_b, res.far_parent))
    for i in range(int(res.far_count)):
        a, b = int(fa[i]), int(fb[i])
        own = sigma if (a, b) in band_set else 1.0
        sa, ea = ranges[a]
        sb, eb = ranges[b]
        cover[sa : ea + 1, sb : eb + 1] += own * hang(int(fp[i]))
    na, nb_, npar = (np.asarray(x) for x in (res.near_a, res.near_b, res.near_parent))
    for i in range(int(res.near_count)):
        sa, ea = ranges[int(na[i])]
        sb, eb = ranges[int(nb_[i])]
        cover[sa : ea + 1, sb : eb + 1] += hang(int(npar[i]))
    cover = cover + cover.T
    ni = int(tree.topology.left_child.shape[0])
    leaf_of = np.empty(n, np.int64)
    for leaf in range(ni, len(ranges)):
        s, e = ranges[leaf]
        leaf_of[s : e + 1] = leaf
    cross = leaf_of[:, None] != leaf_of[None, :]
    np.testing.assert_allclose(cover[cross], 1.0, rtol=0, atol=1e-12)
    assert np.all(cover[~cross] == 0)


def test_a_small_band_cap_overflows():
    _, w = _setup()
    res = _walk(
        w, pair_band=_outer_annulus, pair_accept_data=jnp.asarray(0.7), band_cap=2
    )
    assert bool(res.band_overflow)
    assert int(res.band_count) > 2


def test_pallas_refuses_the_hook():
    _, w = _setup(n=200)
    with pytest.raises(ValueError, match="pair_band"):
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
            pair_band=_never,
            band_cap=4,
            backend="pallas",
            pallas_interpret=True,
        )
