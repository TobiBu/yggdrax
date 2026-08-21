"""The two sides of a boundary must partition the cross pairs between them.

`dual_tree_walk_cross_mutual` runs on both devices of a boundary and each emits
only what it owns. The property to pin is a PARTITION: every geometric cross pair
appears exactly once across the two runs -- never twice, never zero times.

Both failure modes are silent downstream. Double-counting and dropping each leave
per-device momentum exact, because +f/-f still cancel within whatever each device
did do; only a global force comparison would notice. So the combinatorics are
checked here directly rather than trusted to a downstream assertion.

Trees are synthetic: a small random binary tree given explicitly as child/centre/
radius arrays, which is what the walk takes. That keeps the test about the walk's
partitioning rather than about any tree builder.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed.cross_walk import dual_tree_walk_cross_mutual


def _random_tree(n_leaves: int, seed: int, offset):
    """A complete binary tree over ``n_leaves`` leaves, as explicit arrays.

    Layout: internal nodes ``0 .. n_leaves-2``, leaves ``n_leaves-1 .. 2*n_leaves-2``,
    root at 0, children -1 for leaves.
    """
    assert n_leaves & (n_leaves - 1) == 0, "use a power of two"
    total = 2 * n_leaves - 1
    left = np.full(total, -1, dtype=np.int32)
    right = np.full(total, -1, dtype=np.int32)
    for i in range(n_leaves - 1):
        left[i] = 2 * i + 1
        right[i] = 2 * i + 2
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=0.4, size=(total, 3)) + np.asarray(offset)
    radii = np.abs(rng.normal(scale=0.15, size=(total,))) + 0.02
    # internal radii must cover their children, or the MAC is inconsistent
    for i in range(n_leaves - 2, -1, -1):
        for c in (left[i], right[i]):
            d = np.linalg.norm(centers[i] - centers[c])
            radii[i] = max(radii[i], d + radii[c])
    return (
        jnp.asarray(left),
        jnp.asarray(right),
        jnp.asarray(centers),
        jnp.asarray(radii),
        jnp.asarray(0),
    )


CAPS = dict(max_pair_queue=8192, far_cap=16384, near_cap=16384)


def _walk(a, b, this_dev, src_dev, theta=0.35, **caps):
    return dual_tree_walk_cross_mutual(
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        theta,
        this_device=jnp.asarray(this_dev),
        source_device=jnp.asarray(src_dev),
        **(caps or CAPS),
    )


def _sets(res, swap):
    n, m = int(res.far_count), int(res.near_count)
    fa = np.asarray(res.far_local)[:n].tolist()
    fb = np.asarray(res.far_remote)[:n].tolist()
    na = np.asarray(res.near_local)[:m].tolist()
    nb = np.asarray(res.near_remote)[:m].tolist()
    if swap:  # put both runs in the same (device-0 node, device-1 node) frame
        return set(zip(fb, fa)), set(zip(nb, na))
    return set(zip(fa, fb)), set(zip(na, nb))


def test_the_two_devices_partition_the_cross_pairs():
    a = _random_tree(8, 0, (0.0, 0.0, 0.0))
    b = _random_tree(8, 1, (0.9, 0.0, 0.0))
    r0 = _walk(a, b, 0, 1)
    r1 = _walk(b, a, 1, 0)
    for r in (r0, r1):
        assert not bool(r.queue_overflow), "wavefront overflowed"
        assert not bool(r.far_overflow or r.near_overflow), "output overflowed"

    far0, near0 = _sets(r0, swap=False)
    far1, near1 = _sets(r1, swap=True)
    assert not (far0 & far1), f"{len(far0 & far1)} far pairs emitted by BOTH devices"
    assert not (near0 & near1), f"{len(near0 & near1)} near pairs emitted by BOTH"
    # a vacuous pass is the trap: both lists empty would satisfy the above
    assert far0 or far1, "no far pairs at all -- test would be vacuous"
    assert near0 or near1, "no near pairs at all -- test would be vacuous"


def test_union_equals_the_unfiltered_pair_set():
    """Nothing is DROPPED: the partition covers the whole set.

    The reference is the same walk with both device ids equal, which makes
    `cross_pair_is_owned` true for every pair.
    """
    a = _random_tree(8, 5, (0.0, 0.0, 0.0))
    b = _random_tree(8, 6, (0.8, 0.2, 0.0))
    ref_far, ref_near = _sets(_walk(a, b, 0, 0), swap=False)
    far0, near0 = _sets(_walk(a, b, 0, 1), swap=False)
    far1, near1 = _sets(_walk(b, a, 1, 0), swap=True)

    assert far0 | far1 == ref_far, (
        f"far: missing {len(ref_far - (far0 | far1))}, "
        f"extra {len((far0 | far1) - ref_far)}"
    )
    assert near0 | near1 == ref_near, (
        f"near: missing {len(ref_near - (near0 | near1))}, "
        f"extra {len((near0 | near1) - ref_near)}"
    )
    assert ref_far and ref_near, "reference empty -- test would be vacuous"


def test_both_devices_get_a_share_of_the_work():
    """Ownership must balance, not hand a whole boundary to one device."""
    a = _random_tree(16, 21, (0.0, 0.0, 0.0))
    b = _random_tree(16, 22, (0.9, 0.0, 0.0))
    n0 = int(_walk(a, b, 0, 1).near_count) + int(_walk(a, b, 0, 1).far_count)
    n1 = int(_walk(b, a, 1, 0).near_count) + int(_walk(b, a, 1, 0).far_count)
    total = n0 + n1
    assert total > 0
    share = n0 / total
    assert 0.3 < share < 0.7, f"lopsided split {share:.2f}/{1 - share:.2f}"


def test_overflow_is_reported_not_truncated():
    """A dropped cross pair loses BOTH halves, so momentum stays exact and only the
    force is wrong -- which is precisely why this must be a loud flag."""
    a = _random_tree(8, 11, (0.0, 0.0, 0.0))
    b = _random_tree(8, 12, (0.9, 0.0, 0.0))
    res = _walk(a, b, 0, 1, max_pair_queue=8192, far_cap=2, near_cap=2)
    assert bool(
        res.far_overflow or res.near_overflow
    ), "undersized caps did not raise an overflow flag"
