"""``dual_tree_walk_mutual(seed_a=, seed_b=)``: many seed pairs instead of one root pair.

The single-device walk starts from ``(root, root)``. A ONE-SIDED cross-domain
evaluation needs it to start from ``(local_root, imported_i)`` for every imported
source node, which is what makes that evaluation need no new kernel: concatenate the
imported set onto the local nodes at indices ``[n_local, n_local + n_remote)`` and the
walk's existing ``(min, max)`` canonicalisation orders every emitted pair as
(local target, imported source) exactly, for free.

That ordering is load-bearing and silent if wrong -- put the imports first instead and
the M2L expands the wrong way round with a plausible-looking result -- so it is
asserted directly here, not inferred from the index arithmetic.

    JAX_PLATFORMS=cpu pytest tests/test_mutual_walk_seed.py -q
"""

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

_THETA = 0.6


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _tree(points, bounds, leaf_size=16):
    """Cell leaves in a balanced bucket structure, capacity == the live leaf count."""
    P = jnp.asarray(points, jnp.float32)
    n = P.shape[0]
    M = jnp.ones((n,), jnp.float32)
    codes = morton_encode(P, bounds)
    order = jnp.argsort(codes, stable=True)
    sc = codes[order]
    part = adaptive_cell_leaf_partition(sc, leaf_size=leaf_size, capacity=n)
    k = int(part.num_leaves)
    part = adaptive_cell_leaf_partition(sc, leaf_size=leaf_size, capacity=k)
    starts = np.asarray(part.leaf_starts).astype(np.int64)[:k]
    ends = np.asarray(part.leaf_ends).astype(np.int64)[:k]
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
        morton_codes=sc,
        node_ranges=jnp.asarray(node_ranges, I),
        num_particles=n,
        num_internal_nodes=k - 1,
        node_level=jnp.asarray(node_level, I),
        level_offsets=jnp.asarray(level_offsets, I),
        nodes_by_level=jnp.asarray(nodes_by_level, I),
        num_levels=jnp.asarray(num_levels, I),
        bounds_min=jnp.asarray(bounds[0], P.dtype),
        bounds_max=jnp.asarray(bounds[1], P.dtype),
        leaf_codes=sc[jnp.asarray(np.minimum(starts, n - 1), I)],
        leaf_depths=jnp.asarray(part.leaf_depths, I)[:k],
        use_morton_geometry=jnp.asarray(False),
        leaf_size=leaf_size,
    )
    return topo, ps, ms


def _com_radii(topo, ps, ms):
    """Centres of mass and exact max COM-to-particle radii, as the mutual MAC wants."""
    total = int(topo.parent.shape[0])
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    ranges = np.asarray(topo.node_ranges)
    pos = np.asarray(ps, np.float64)
    c = np.asarray(com, np.float64)
    radii = np.zeros(total)
    for i, (a, b) in enumerate(ranges):
        if b >= a:
            radii[i] = np.sqrt(np.max(np.sum((pos[a : b + 1] - c[i]) ** 2, axis=1)))
    return com, jnp.asarray(radii, ps.dtype)


def _children_full(topo):
    total = int(topo.parent.shape[0])
    nint = int(topo.left_child.shape[0])
    idx = topo.parent.dtype
    fill = jnp.full((total - nint,), -1, idx)
    return (
        jnp.concatenate([topo.left_child, fill]),
        jnp.concatenate([topo.right_child, fill]),
    )


def _walk(left, right, centers, radii, root, *, queue=1 << 14, cap=1 << 18, **kw):
    return dual_tree_walk_mutual(
        left,
        right,
        centers,
        radii,
        _THETA,
        root,
        max_pair_queue=queue,
        far_cap=cap,
        near_cap=cap,
        mac_type="dehnen",
        **kw,
    )


def _pairs(res):
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )
    fa = np.asarray(res.far_a)[: int(res.far_count)]
    fb = np.asarray(res.far_b)[: int(res.far_count)]
    na = np.asarray(res.near_a)[: int(res.near_count)]
    nb = np.asarray(res.near_b)[: int(res.near_count)]
    return set(zip(fa.tolist(), fb.tolist())), set(zip(na.tolist(), nb.tolist()))


def _one_tree(n=1500, seed=0):
    P = _plummer(n, seed)
    bounds = infer_bounds(jnp.asarray(P, jnp.float32))
    topo, ps, ms = _tree(P, bounds)
    left, right = _children_full(topo)
    com, radii = _com_radii(topo, ps, ms)
    root = jnp.argmin(topo.parent).astype(topo.parent.dtype)
    return topo, left, right, com, radii, root


def test_root_seed_is_identical_to_no_seed():
    """The control: the one-pair seed must reproduce the default, field by field."""
    _t, left, right, com, radii, root = _one_tree()
    a = _walk(left, right, com, radii, root)
    b = _walk(
        left,
        right,
        com,
        radii,
        root,
        seed_a=jnp.asarray([root]),
        seed_b=jnp.asarray([root]),
    )
    for f in a._fields:
        assert np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))), f


def test_seeding_the_roots_children_covers_the_same_pairs():
    """Seeding one level down must emit exactly the pairs the root walk does.

    The root pair itself is never emitted (a node is never far from or near to
    itself), so refining it by hand into its three child pairs is a partition of
    everything below it -- an equality, not a subset.
    """
    topo, left, right, com, radii, root = _one_tree()
    r = int(root)
    lc, rc = int(np.asarray(topo.left_child)[r]), int(np.asarray(topo.right_child)[r])
    far0, near0 = _pairs(_walk(left, right, com, radii, root))
    seeded = _walk(
        left,
        right,
        com,
        radii,
        root,
        seed_a=jnp.asarray([lc, lc, rc]),
        seed_b=jnp.asarray([lc, rc, rc]),
    )
    far1, near1 = _pairs(seeded)
    assert far0 and near0
    assert far1 == far0
    assert near1 == near0


def test_seed_order_does_not_matter():
    """Seeds are canonicalised, so a reversed pair seeds the same traversal."""
    topo, left, right, com, radii, root = _one_tree()
    r = int(root)
    lc, rc = int(np.asarray(topo.left_child)[r]), int(np.asarray(topo.right_child)[r])
    fwd = _pairs(
        _walk(
            left,
            right,
            com,
            radii,
            root,
            seed_a=jnp.asarray([lc]),
            seed_b=jnp.asarray([rc]),
        )
    )
    rev = _pairs(
        _walk(
            left,
            right,
            com,
            radii,
            root,
            seed_a=jnp.asarray([rc]),
            seed_b=jnp.asarray([lc]),
        )
    )
    assert fwd == rev


def test_seed_count_masks_the_tail():
    """A capacity-padded seed buffer plus a live count == the exact-length seed."""
    topo, left, right, com, radii, root = _one_tree()
    r = int(root)
    lc, rc = int(np.asarray(topo.left_child)[r]), int(np.asarray(topo.right_child)[r])
    exact = _walk(
        left,
        right,
        com,
        radii,
        root,
        seed_a=jnp.asarray([lc, lc, rc]),
        seed_b=jnp.asarray([lc, rc, rc]),
    )
    padded = _walk(
        left,
        right,
        com,
        radii,
        root,
        seed_a=jnp.asarray([lc, lc, rc, -1, -1]),
        seed_b=jnp.asarray([lc, rc, rc, -1, -1]),
        seed_count=jnp.asarray(3),
    )
    assert _pairs(exact) == _pairs(padded)


def test_one_sided_ordering_local_target_before_imported_source():
    """THE load-bearing claim: every emitted pair is (local target, imported source).

    Build the combined index space the one-sided lane will build -- local tree nodes
    first, an imported source set at ``[n_local, n_local + n_remote)`` carrying no
    children -- seed ``(local_root, imported_i)``, and check the orientation directly.
    Putting the imports first instead is the silent failure this guards.
    """
    bounds = infer_bounds(
        jnp.asarray(
            np.concatenate([_plummer(1200, 0) - 1.5, _plummer(1200, 1) + 1.5]),
            jnp.float32,
        )
    )
    loc_t, loc_p, loc_m = _tree(_plummer(1200, 0) - 1.5, bounds)
    rem_t, rem_p, rem_m = _tree(_plummer(1200, 1) + 1.5, bounds)
    l_left, l_right = _children_full(loc_t)
    l_com, l_rad = _com_radii(loc_t, loc_p, loc_m)
    r_com, r_rad = _com_radii(rem_t, rem_p, rem_m)

    n_local = int(loc_t.parent.shape[0])
    n_rint = int(rem_t.left_child.shape[0])
    rem_leaves = np.arange(n_rint, int(rem_t.parent.shape[0]))
    n_remote = rem_leaves.size

    idx = loc_t.parent.dtype
    left = jnp.concatenate([l_left, jnp.full((n_remote,), -1, idx)])
    right = jnp.concatenate([l_right, jnp.full((n_remote,), -1, idx)])
    centers = jnp.concatenate([jnp.asarray(l_com), jnp.asarray(r_com)[rem_leaves]])
    radii = jnp.concatenate([jnp.asarray(l_rad), jnp.asarray(r_rad)[rem_leaves]])

    local_root = int(np.argmin(np.asarray(loc_t.parent)))
    imported = np.arange(n_local, n_local + n_remote)
    res = _walk(
        left,
        right,
        centers,
        radii,
        jnp.asarray(local_root, idx),
        queue=1 << 16,
        seed_a=jnp.asarray(np.full(n_remote, local_root), idx),
        seed_b=jnp.asarray(imported, idx),
    )
    far, near = _pairs(res)
    assert far and near
    for a, b in far | near:
        assert (
            a < n_local <= b
        ), f"pair ({a}, {b}) is not (local target, imported source)"
    # the walk refined only on the local side: every imported node stayed whole
    assert {b for _a, b in far | near} <= set(imported.tolist())


def test_seed_validation():
    _t, left, right, com, radii, root = _one_tree()
    with pytest.raises(ValueError, match="together"):
        _walk(left, right, com, radii, root, seed_a=jnp.asarray([root]))
    with pytest.raises(ValueError, match="matching 1-D"):
        _walk(
            left,
            right,
            com,
            radii,
            root,
            seed_a=jnp.asarray([root, root]),
            seed_b=jnp.asarray([root]),
        )
    with pytest.raises(ValueError, match="cannot hold its own seed"):
        _walk(
            left,
            right,
            com,
            radii,
            root,
            queue=2,
            seed_a=jnp.asarray([root] * 4),
            seed_b=jnp.asarray([root] * 4),
        )
