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


def _all_owned_by(tree, dev):
    """Tag every node of ``tree`` with one owning domain.

    The simple boundary case. A real remote tree merges several domains and its
    internal nodes straddle; that is covered by
    `test_straddling_nodes_are_never_accepted`.
    """
    total = tree[0].shape[0]
    return jnp.full((total,), dev, dtype=jnp.int32)


def _walk(a, b, this_dev, src_dev, theta=0.35, remote_owner=None, **caps):
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
        remote_owner=(
            _all_owned_by(b, src_dev) if remote_owner is None else remote_owner
        ),
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


def test_single_owner_domain_marks_straddling_internal_nodes():
    """A merged remote tree's internal nodes generally span several domains."""
    from yggdrax.distributed.cross_walk import single_owner_domain

    # 4 leaves (nodes 3..6) under internals 0..2; leaves belong to domains 7,7,9,9
    left = jnp.asarray([1, 3, 5, -1, -1, -1, -1], dtype=jnp.int32)
    right = jnp.asarray([2, 4, 6, -1, -1, -1, -1], dtype=jnp.int32)
    tag = jnp.asarray([0, 0, 0, 7, 7, 9, 9], dtype=jnp.int32)
    own = np.asarray(single_owner_domain(left, right, tag, max_depth=8))

    assert own[3] == own[4] == 7, "leaves keep their own tag"
    assert own[5] == own[6] == 9
    assert own[1] == 7, "both children in domain 7 -> single owner"
    assert own[2] == 9
    assert own[0] == -1, "root spans domains 7 and 9 -> must be marked straddling"


def test_straddling_nodes_are_never_accepted_as_far_pairs():
    """Option 1 of the design: refine a straddling node instead of accepting it.

    An accepted far pair owes a `-f` to the remote endpoint's domain. An internal
    node aggregating three domains has no single destination, so accepting it would
    leave the reverse exchange undefined. Refining instead terminates, because coarse
    leaves each carry exactly one origin domain.
    """
    from yggdrax.distributed.cross_walk import single_owner_domain

    a = _random_tree(8, 31, (0.0, 0.0, 0.0))
    b = _random_tree(8, 32, (0.9, 0.0, 0.0))
    total_b = b[0].shape[0]
    # give b's 8 leaves two different domains, so every internal node straddles
    tag = np.zeros(total_b, dtype=np.int32)
    leaves = np.where(np.asarray(b[0]) < 0)[0]
    tag[leaves[: len(leaves) // 2]] = 3
    tag[leaves[len(leaves) // 2 :]] = 5
    owner = single_owner_domain(b[0], b[1], jnp.asarray(tag), max_depth=16)
    own_np = np.asarray(owner)

    res = _walk(a, b, 0, 0, remote_owner=owner)
    n = int(res.far_count)
    accepted_remote = np.asarray(res.far_remote)[:n]
    assert n > 0, "no far pairs accepted -- test would be vacuous"
    assert np.all(
        own_np[accepted_remote] >= 0
    ), "a straddling remote node was accepted as a far pair"
    # and the recorded owner matches the node's actual domain, so a reverse
    # exchange can trust it
    assert np.array_equal(
        np.asarray(res.far_owner)[:n], own_np[accepted_remote]
    ), "far_owner disagrees with the remote node's domain"


def test_near_pairs_are_always_single_owner():
    """Near pairs are leaf-leaf, and a coarse leaf carries exactly one domain."""
    from yggdrax.distributed.cross_walk import single_owner_domain

    a = _random_tree(8, 41, (0.0, 0.0, 0.0))
    b = _random_tree(8, 42, (0.9, 0.0, 0.0))
    total_b = b[0].shape[0]
    tag = np.zeros(total_b, dtype=np.int32)
    leaves = np.where(np.asarray(b[0]) < 0)[0]
    tag[leaves[::2]] = 2
    tag[leaves[1::2]] = 6
    owner = single_owner_domain(b[0], b[1], jnp.asarray(tag), max_depth=16)
    res = _walk(a, b, 0, 0, remote_owner=owner)
    m = int(res.near_count)
    assert m > 0, "no near pairs -- test would be vacuous"
    assert np.all(np.asarray(res.near_owner)[:m] >= 0)
