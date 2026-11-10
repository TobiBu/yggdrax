"""Periodic boundaries (``box_size``) against brute-force minimum image.

What a periodic walk promises (``docs/periodic_boundaries.md``):

* **coverage** -- every unordered particle pair lies in exactly one near leaf pair
  (or one leaf's own block) or in exactly one accepted far node pair, as in the
  non-periodic walk;
* **unambiguity** -- for an accepted far pair, the image that is the minimum
  image of the node centres is the minimum image of *every* particle pair inside
  it (no other image is closer), so a far-field evaluation through that single
  image is exact;
* **separation floor** -- with a floor ``s`` no accepted far pair holds a particle
  pair closer than ``s`` (minimum image), so a pair count below ``s`` needs only
  the near list.

All three are checked particle by particle against brute force, on small boxes
with particles crowded onto faces, edges and corners and ``theta`` up to 0.9, for
the mutual walk (also on ``mutual_walk_inputs`` and the capacity-padded static
cells tree under ``jit``) and for every route of the dual walk. The kNN query and
the radius count are checked on all their backends.

    JAX_PLATFORMS=cpu pytest tests/unit/test_periodic.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_prepared_tree_artifacts,
    compute_tree_geometry,
    minimum_image,
    pairwise_periodic_distance_sq,
    periodic_distance_sq,
    wrap_positions,
)
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.interactions import (
    build_compact_far_pairs,
    build_compact_far_pairs_and_leaf_neighbor_lists,
    build_interactions_and_neighbors,
    build_interactions_and_neighbors_split,
    build_leaf_neighbor_lists,
    build_well_separated_interactions,
    dual_tree_walk_mutual,
    mutual_walk_inputs,
)
from yggdrax.kdtree import build_kdtree, count_neighbors, query_neighbors
from yggdrax.periodic import (
    far_pair_is_unambiguous,
    normalize_box_size,
    periodic_bounds,
)
from yggdrax.tree import resolve_tree_topology

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _box_points(n: int, box, seed: int = 0) -> np.ndarray:
    """Uniform points plus crowds on the faces, edges and corners of the box.

    A share of the points sits within 1e-3 L of a face (on both sides of it,
    before wrapping), at exactly 0, and outside the box, so wrapping and the
    across-the-face pairs are exercised.
    """
    rng = np.random.default_rng(seed)
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    n_uniform = n // 2
    pts = [rng.uniform(0.0, 1.0, (n_uniform, 3)) * L]
    # corners: each point near one of the 8 corners, on either side of the faces
    n_corner = n // 6
    corners = rng.integers(0, 2, (n_corner, 3)).astype(np.float64) * L
    pts.append(corners + rng.uniform(-1e-3, 1e-3, (n_corner, 3)) * L)
    # faces/edges: one or two coordinates pinned near 0 or L
    n_face = n // 6
    face = rng.uniform(0.0, 1.0, (n_face, 3)) * L
    axis = rng.integers(0, 3, n_face)
    side = rng.integers(0, 2, n_face).astype(np.float64)
    face[np.arange(n_face), axis] = (side + rng.uniform(-1e-3, 1e-3, n_face)) * L[axis]
    axis2 = (axis + 1) % 3
    edge = rng.uniform(size=n_face) < 0.5
    face[edge, axis2[edge]] = 0.0
    pts.append(face)
    # far outside the box: wrapping must bring them in
    n_out = n - n_uniform - n_corner - n_face
    pts.append(rng.uniform(-2.0, 3.0, (n_out, 3)) * L)
    return np.concatenate(pts, axis=0)


def _brute_min_image_d2(pos: np.ndarray, box) -> np.ndarray:
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    d = pos[None, :, :] - pos[:, None, :]
    d -= L * np.round(d / L)
    return np.sum(d * d, axis=-1)


def _build(pos, box, *, tree_type="radix", leaf=8):
    tree = Tree.from_particles(
        jnp.asarray(pos),
        jnp.ones((pos.shape[0],)),
        tree_type=tree_type,
        leaf_size=leaf,
        box_size=box,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    return tree, geometry


def _node_ranges(tree) -> np.ndarray:
    return np.asarray(tree.topology.node_ranges)


def _members(ranges, node) -> np.ndarray:
    lo, hi = ranges[node]
    return np.arange(int(lo), int(hi) + 1)


_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=1 << 15,
    process_block=64,
    max_interactions_per_node=2048,
    max_neighbors_per_leaf=2048,
)


# ---------------------------------------------------------------------------
# checks shared by the walks
# ---------------------------------------------------------------------------


def _check_coverage(n, ranges, leaf_nodes, far_pairs, near_pairs):
    """Every unordered pair i < j is covered exactly once (sorted-order indices)."""
    hits = np.zeros((n, n), dtype=np.int64)
    for leaf in leaf_nodes:
        m = _members(ranges, leaf)
        hits[np.ix_(m, m)] += 1
    for a, b in list(far_pairs) + list(near_pairs):
        ma, mb = _members(ranges, a), _members(ranges, b)
        hits[np.ix_(ma, mb)] += 1
        hits[np.ix_(mb, ma)] += 1
    np.fill_diagonal(hits, 1)
    bad = np.argwhere(hits != 1)
    assert bad.size == 0, f"{len(bad)} pair entries covered != 1 time, e.g. {bad[:4]}"


def _check_far_unambiguous(pos_sorted, centers, ranges, far_pairs, box):
    """One image -- the centres' minimum image -- serves every pair of the node pair."""
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    half = 0.5 * float(np.min(L))
    for a, b in far_pairs:
        ma, mb = _members(ranges, a), _members(ranges, b)
        dc = centers[b] - centers[a]
        shift = -L * np.round(dc / L)  # the node pair's image
        raw = pos_sorted[mb][None, :, :] - pos_sorted[ma][:, None, :]
        through_node_image = raw + shift
        mi = raw - L * np.round(raw / L)
        np.testing.assert_allclose(through_node_image, mi, rtol=0, atol=1e-9)
        assert np.all(np.linalg.norm(through_node_image, axis=-1) < half)


def _binned_counts_from_walk(
    pos_sorted, centers, ranges, leaf_nodes, far, near, box, edges
):
    """Hard-binned pair counts assembled from the walk's lists.

    Near and self-leaf blocks use particle minimum image; far pairs use the
    node pair's single image, which is exact by unambiguity.
    """
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    def add(d):
        counts[:] += np.histogram(np.sqrt(np.sum(d * d, axis=-1)), bins=edges)[0]

    for leaf in leaf_nodes:
        m = _members(ranges, leaf)
        raw = pos_sorted[m][None, :, :] - pos_sorted[m][:, None, :]
        iu = np.triu_indices(len(m), k=1)
        add((raw - L * np.round(raw / L))[iu])
    for a, b in near:
        ma, mb = _members(ranges, a), _members(ranges, b)
        raw = pos_sorted[mb][None, :, :] - pos_sorted[ma][:, None, :]
        add((raw - L * np.round(raw / L)).reshape(-1, 3))
    for a, b in far:
        ma, mb = _members(ranges, a), _members(ranges, b)
        shift = -L * np.round((centers[b] - centers[a]) / L)
        raw = pos_sorted[mb][None, :, :] - pos_sorted[ma][:, None, :]
        add((raw + shift).reshape(-1, 3))
    return counts


def _brute_binned(pos_sorted, box, edges):
    d2 = _brute_min_image_d2(pos_sorted, box)
    iu = np.triu_indices(pos_sorted.shape[0], k=1)
    return np.histogram(np.sqrt(d2[iu]), bins=edges)[0]


def _leaf_nodes(tree):
    total = int(tree.topology.parent.shape[0])
    num_internal = int(tree.topology.left_child.shape[0])
    return list(range(num_internal, total))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_minimum_image_and_wrap_edge_cases():
    L = 2.0
    d = jnp.asarray([[0.9, -0.9, 1.1], [-1.1, 3.0, -3.0], [0.0, 1.0, -1.0]])
    mi = np.asarray(minimum_image(d, L))
    assert np.all(np.abs(mi) <= L / 2 + 1e-12)
    np.testing.assert_allclose(
        np.round((np.asarray(d) - mi) / L) * L, np.asarray(d) - mi
    )
    assert minimum_image(d, None) is d

    x = jnp.asarray([[-1e-17, L, 2 * L + 0.25], [-0.5, 1e300 * 0 + 1.0, -2 * L]])
    w = np.asarray(wrap_positions(x, L))
    assert np.all(w >= 0.0) and np.all(w < L)
    np.testing.assert_allclose(w[0, 2], 0.25)
    np.testing.assert_allclose(w[1, 0], 1.5)

    a = jnp.asarray([[0.05, 0.05, 0.05]])
    b = jnp.asarray([[1.95, 1.95, 1.95]])
    np.testing.assert_allclose(periodic_distance_sq(a, b, L), [3 * 0.1**2])
    np.testing.assert_allclose(
        pairwise_periodic_distance_sq(a, b, (L, L, 10.0)), [[2 * 0.1**2 + 1.9**2]]
    )


def test_normalize_box_size():
    assert normalize_box_size(None) is None
    assert normalize_box_size(2) == (2.0,)
    assert normalize_box_size(2.0, 3) == (2.0, 2.0, 2.0)
    assert normalize_box_size((1, 2, 3), 3) == (1.0, 2.0, 3.0)
    assert normalize_box_size(np.asarray([1.0, 2.0, 3.0])) == (1.0, 2.0, 3.0)
    for bad in (0.0, -1.0, float("inf"), (1.0, float("nan"), 1.0), ()):
        with pytest.raises(ValueError):
            normalize_box_size(bad)
    with pytest.raises(ValueError):
        normalize_box_size((1.0, 2.0), 3)
    with pytest.raises(TypeError):
        jax.jit(lambda b: normalize_box_size(b))(jnp.asarray(1.0))
    lo, hi = periodic_bounds(3.0, 3)
    np.testing.assert_array_equal(np.asarray(lo), 0.0)
    np.testing.assert_array_equal(np.asarray(hi), 3.0)


def test_far_pair_guard_is_strict():
    ok = far_pair_is_unambiguous(
        jnp.asarray([0.25, 0.2499]), jnp.asarray([0.125, 0.125]), 0.0, 2.0
    )
    # reach = 0.5 + 0.125 = 0.625 vs half box 1.0 ... both fine
    assert bool(jnp.all(ok))
    ok = far_pair_is_unambiguous(jnp.asarray([0.25]), jnp.asarray([0.25]), 0.25, 2.0)
    assert not bool(ok[0])  # reach == L/2 exactly is refused


# ---------------------------------------------------------------------------
# tree build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tree_type", ["radix", "octree", "kdtree"])
def test_periodic_build_wraps_into_fixed_box(tree_type):
    box = 5.0
    pos = _box_points(300, box, seed=1)
    tree = Tree.from_particles(
        jnp.asarray(pos),
        jnp.ones((pos.shape[0],)),
        tree_type=tree_type,
        leaf_size=8,
        box_size=box,
    )
    ps = np.asarray(tree.positions_sorted)
    assert np.all(ps >= 0.0) and np.all(ps < box)
    # the reordering is a permutation of the wrapped input
    inv = np.asarray(tree.inverse_permutation)
    np.testing.assert_allclose(
        ps[inv], np.asarray(wrap_positions(jnp.asarray(pos), box))
    )
    if tree_type != "kdtree":
        np.testing.assert_array_equal(np.asarray(tree.topology.bounds_min), 0.0)
        np.testing.assert_array_equal(np.asarray(tree.topology.bounds_max), box)
    with pytest.raises(ValueError, match="either bounds or box_size"):
        Tree.from_particles(
            jnp.asarray(pos),
            jnp.ones((pos.shape[0],)),
            tree_type=tree_type,
            bounds=(jnp.zeros(3), jnp.full(3, box)),
            box_size=box,
        )


# ---------------------------------------------------------------------------
# mutual walk
# ---------------------------------------------------------------------------


def _mutual(tree, geometry, theta, box, separation_floor=0.0, mac_type=None):
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    fill = jnp.full((total - num_internal,), -1, idx)
    left = jnp.concatenate([jnp.asarray(topo.left_child, idx), fill])
    right = jnp.concatenate([jnp.asarray(topo.right_child, idx), fill])
    res = dual_tree_walk_mutual(
        left,
        right,
        jnp.asarray(geometry.center),
        jnp.asarray(geometry.radius),
        theta,
        jnp.argmin(topo.parent).astype(idx),
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
        mac_type=mac_type,
        separation_floor=separation_floor,
        box_size=box,
    )
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )
    nf, nn = int(res.far_count), int(res.near_count)
    far = list(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    near = list(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    return far, near


@pytest.mark.parametrize("theta", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("box", [1.0, (1.0, 1.5, 0.8)])
def test_mutual_walk_periodic_against_brute_force(theta, box):
    pos = _box_points(600, box, seed=2)
    tree, geometry = _build(pos, box, leaf=4)
    ps = np.asarray(tree.positions_sorted)
    centers = np.asarray(geometry.center)
    ranges = _node_ranges(tree)
    leaves = _leaf_nodes(tree)
    far, near = _mutual(tree, geometry, theta, box)
    assert len(far) > 0, "the test should exercise far acceptance"

    _check_coverage(ps.shape[0], ranges, leaves, far, near)
    _check_far_unambiguous(ps, centers, ranges, far, box)

    half_diag = 0.5 * float(np.linalg.norm(np.broadcast_to(np.asarray(box), (3,))))
    edges = np.linspace(0.0, half_diag * 1.0001, 41)
    np.testing.assert_array_equal(
        _binned_counts_from_walk(ps, centers, ranges, leaves, far, near, box, edges),
        _brute_binned(ps, box, edges),
    )


def test_mutual_walk_periodic_finds_pairs_across_faces():
    """Pairs that are close only through the box face are near (or same-leaf) pairs.

    With a separation floor ``s`` an accepted far pair holds no particle pair
    closer than ``s`` (minimum image), so the pairs within ``s`` must be exactly
    those of the near list and the leaves' own blocks -- including every pair that
    is close only across a face, which a non-periodic walk would accept as far.
    """
    box = 1.0
    r_max = 0.06
    pos = _box_points(480, box, seed=3)
    tree, geometry = _build(pos, box)
    ps = np.asarray(tree.positions_sorted)
    ranges = _node_ranges(tree)
    far, near = _mutual(tree, geometry, 0.7, box, separation_floor=r_max)

    close = set()
    for leaf in _leaf_nodes(tree):
        m = _members(ranges, leaf)
        close.update((int(i), int(j)) for i in m for j in m if i < j)
    for a, b in near:
        for i in _members(ranges, a):
            for j in _members(ranges, b):
                close.add((min(int(i), int(j)), max(int(i), int(j))))
    d2 = _brute_min_image_d2(ps, box)
    want = {(int(i), int(j)) for i, j in np.argwhere(d2 < r_max**2) if i < j}
    raw = ps[None, :, :] - ps[:, None, :]
    across = {
        (int(i), int(j))
        for i, j in np.argwhere(np.sum(raw * raw, axis=-1) >= r_max**2)
        if i < j and (int(i), int(j)) in want
    }
    assert len(across) > 10, "the fixture should have pairs close only across a face"
    assert want <= close


def test_mutual_walk_box_none_is_unchanged():
    pos = _box_points(400, 1.0, seed=4)
    tree, geometry = _build(pos, 1.0)
    far0, near0 = _mutual(tree, geometry, 0.6, None)
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    fill = jnp.full((total - num_internal,), -1, idx)
    res = dual_tree_walk_mutual(
        jnp.concatenate([jnp.asarray(topo.left_child, idx), fill]),
        jnp.concatenate([jnp.asarray(topo.right_child, idx), fill]),
        jnp.asarray(geometry.center),
        jnp.asarray(geometry.radius),
        0.6,
        jnp.argmin(topo.parent).astype(idx),
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
    )
    nf, nn = int(res.far_count), int(res.near_count)
    assert far0 == list(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    assert near0 == list(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    # and a periodic walk of the same tree differs: it is not a no-op
    far1, near1 = _mutual(tree, geometry, 0.6, 1.0)
    assert (far1, near1) != (far0, near0)


def test_mutual_walk_periodic_pair_accept_is_guarded():
    """A ``pair_accept`` that accepts everything still cannot bypass the guard."""
    box = 1.0
    pos = _box_points(300, box, seed=5)
    tree, geometry = _build(pos, box)
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    fill = jnp.full((total - num_internal,), -1, idx)
    res = dual_tree_walk_mutual(
        jnp.concatenate([jnp.asarray(topo.left_child, idx), fill]),
        jnp.concatenate([jnp.asarray(topo.right_child, idx), fill]),
        jnp.asarray(geometry.center),
        jnp.asarray(geometry.radius),
        0.5,
        jnp.argmin(topo.parent).astype(idx),
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
        box_size=box,
        pair_accept=_accept_all,
    )
    nf, nn = int(res.far_count), int(res.near_count)
    far = list(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    near = list(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    ps = np.asarray(tree.positions_sorted)
    ranges = _node_ranges(tree)
    _check_coverage(ps.shape[0], ranges, _leaf_nodes(tree), far, near)
    _check_far_unambiguous(ps, np.asarray(geometry.center), ranges, far, box)


def _accept_all(data, a, b, dist_sq, ra, rb):
    del data, a, b, ra, rb
    return jnp.ones(dist_sq.shape, bool)


# ---------------------------------------------------------------------------
# dual walk, every route
# ---------------------------------------------------------------------------


def _far_set_from_csr(interactions) -> set:
    offsets = np.asarray(interactions.offsets)
    sources = np.asarray(interactions.sources)
    counts = np.asarray(interactions.counts)
    out = set()
    for t in range(counts.shape[0]):
        for k in range(int(counts[t])):
            s = int(sources[int(offsets[t]) + k])
            out.add((min(t, s), max(t, s)))
    return out


def _far_set_from_compact(compact) -> set:
    nc = int(compact.far_pair_count)
    targets = np.asarray(compact.targets)[:nc].tolist()
    sources = np.asarray(compact.sources)[:nc].tolist()
    return {(min(t, s), max(t, s)) for t, s in zip(targets, sources)}


def _near_set(neighbors) -> set:
    offsets = np.asarray(neighbors.offsets)
    counts = np.asarray(neighbors.counts)
    nbrs = np.asarray(neighbors.neighbors)
    leaf_nodes = np.asarray(neighbors.leaf_indices)
    out = set()
    for row, leaf in enumerate(leaf_nodes.tolist()):
        for k in range(int(counts[row])):
            other = int(nbrs[int(offsets[row]) + k])
            if other != leaf:
                out.add((min(leaf, other), max(leaf, other)))
    return out


@pytest.mark.parametrize("tree_type", ["radix", "kdtree"])
@pytest.mark.parametrize("mac_type", ["bh", "dehnen", "engblom"])
@pytest.mark.parametrize("theta", [0.5, 0.9])
def test_dual_walk_periodic_against_brute_force(tree_type, mac_type, theta):
    box = (1.0, 1.0, 1.25)
    pos = _box_points(700, box, seed=6)
    tree, geometry = _build(pos, box, tree_type=tree_type, leaf=4)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type=mac_type,
        traversal_config=_CONFIG,
        box_size=box,
    )
    far = _far_set_from_csr(interactions)
    near = _near_set(neighbors)
    assert far, "the test should exercise far acceptance"
    ps = np.asarray(tree.positions_sorted)
    ranges = _node_ranges(tree)
    leaves = np.asarray(neighbors.leaf_indices).tolist()
    _check_coverage(ps.shape[0], ranges, leaves, far, near)
    _check_far_unambiguous(ps, np.asarray(geometry.center), ranges, far, box)
    edges = np.linspace(0.0, 1.0, 33)
    np.testing.assert_array_equal(
        _binned_counts_from_walk(
            ps, np.asarray(geometry.center), ranges, leaves, far, near, box, edges
        ),
        _brute_binned(ps, box, edges),
    )


@pytest.mark.parametrize("mac_type", ["dehnen", "engblom"])
def test_dual_walk_periodic_separation_floor(mac_type):
    """Pairs within the floor come from the near list only (sphere MAC extents)."""
    box = 1.0
    r_max = 0.05
    pos = _box_points(400, box, seed=7)
    tree, geometry = _build(pos, box)
    _, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.8,
        mac_type=mac_type,
        traversal_config=_CONFIG,
        separation_floor=r_max,
        box_size=box,
    )
    ps = np.asarray(tree.positions_sorted)
    ranges = _node_ranges(tree)
    close = set()
    for leaf in np.asarray(neighbors.leaf_indices).tolist():
        m = _members(ranges, leaf)
        close.update((int(i), int(j)) for i in m for j in m if i < j)
    for a, b in _near_set(neighbors):
        for i in _members(ranges, a):
            for j in _members(ranges, b):
                close.add((min(int(i), int(j)), max(int(i), int(j))))
    d2 = _brute_min_image_d2(ps, box)
    want = {(int(i), int(j)) for i, j in np.argwhere(d2 < r_max**2) if i < j}
    assert want and want <= close


def test_dual_walk_routes_agree_periodic():
    """Every route of the dual walk applies the same periodic rule.

    The joint walk, the bounded count-pass routes (taken with a traversal config
    and no explicit per-node capacity) and the split walk must give the same far
    and near sets -- the drift ``_resolve_dehnen_radius_scale`` once had between
    routes is what this guards against for ``box_size``.
    """
    box = 1.0
    pos = _box_points(360, box, seed=8)
    for tree_type, mac_type in (("radix", "bh"), ("kdtree", "dehnen")):
        tree, geometry = _build(pos, box, tree_type=tree_type)
        kw = dict(theta=0.7, mac_type=mac_type, box_size=box)
        inter, nbrs = build_interactions_and_neighbors(
            tree, geometry, traversal_config=_CONFIG, **kw
        )
        far, near = _far_set_from_csr(inter), _near_set(nbrs)

        inter0, nbrs0 = build_interactions_and_neighbors(tree, geometry, **kw)
        assert _far_set_from_csr(inter0) == far
        assert _near_set(nbrs0) == near

        bounded = _CONFIG  # with no per-node capacity argument: count-pass routes
        assert (
            _far_set_from_csr(
                build_well_separated_interactions(
                    tree, geometry, traversal_config=bounded, **kw
                )
            )
            == far
        )
        assert (
            _near_set(
                build_leaf_neighbor_lists(
                    tree, geometry, traversal_config=bounded, **kw
                )
            )
            == near
        )
        compact, nb2 = build_compact_far_pairs_and_leaf_neighbor_lists(
            tree, geometry, traversal_config=bounded, **kw
        )
        assert _far_set_from_compact(compact) == far
        assert _near_set(nb2) == near
        assert (
            _far_set_from_compact(
                build_compact_far_pairs(tree, geometry, traversal_config=bounded, **kw)
            )
            == far
        )

        inter_s, nbrs_s = build_interactions_and_neighbors_split(
            tree, geometry, traversal_config=_CONFIG, **kw
        )
        assert _far_set_from_csr(inter_s) == far
        assert _near_set(nbrs_s) == near

        # and the periodic lists really differ from the non-periodic ones
        inter_np, nbrs_np = build_interactions_and_neighbors(
            tree, geometry, traversal_config=_CONFIG, theta=0.7, mac_type=mac_type
        )
        assert (_far_set_from_csr(inter_np), _near_set(nbrs_np)) != (far, near)


def test_dual_walk_pair_policy_cannot_bypass_guard():
    box = 1.0
    pos = _box_points(300, box, seed=9)
    tree, geometry = _build(pos, box)

    from yggdrax._interactions_impl import _ACTION_ACCEPT, _ACTION_REFINE

    def policy(state, **kw):
        # accepts every pair of two different nodes, ignoring mac_ok and so the
        # periodic guard; a node against itself refines
        del state
        accept = kw["valid_pairs"] & kw["different_nodes"]
        return (
            jnp.where(accept, _ACTION_ACCEPT, _ACTION_REFINE),
            jnp.zeros(accept.shape, jnp.int64),
        )

    inter, nbrs = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.5,
        traversal_config=_CONFIG,
        pair_policy=policy,
        box_size=box,
    )
    far, near = _far_set_from_csr(inter), _near_set(nbrs)
    ps = np.asarray(tree.positions_sorted)
    ranges = _node_ranges(tree)
    _check_coverage(
        ps.shape[0], ranges, np.asarray(nbrs.leaf_indices).tolist(), far, near
    )
    _check_far_unambiguous(ps, np.asarray(geometry.center), ranges, far, box)


def test_prepared_artifacts_periodic():
    box = 2.0
    pos = _box_points(300, box, seed=10)
    art = build_prepared_tree_artifacts(
        jnp.asarray(pos),
        jnp.ones((pos.shape[0],)),
        leaf_size=8,
        theta=0.6,
        box_size=box,
    )
    ps = np.asarray(art.positions_sorted)
    assert np.all(ps >= 0.0) and np.all(ps < box)
    far, near = _far_set_from_csr(art.interactions), _near_set(art.neighbors)
    ranges = np.asarray(art.tree.topology.node_ranges)
    _check_coverage(
        ps.shape[0], ranges, np.asarray(art.neighbors.leaf_indices).tolist(), far, near
    )
    _check_far_unambiguous(ps, np.asarray(art.geometry.center), ranges, far, box)


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["dense", "tiled", "tree"])
@pytest.mark.parametrize("box", [1.0, (1.0, 2.0, 0.5)])
def test_query_neighbors_periodic(backend, box):
    pts = np.asarray(wrap_positions(jnp.asarray(_box_points(400, box, seed=11)), box))
    tree = build_kdtree(jnp.asarray(pts), leaf_size=8)
    rng = np.random.default_rng(12)
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    queries = np.concatenate(
        [
            pts[:40],
            rng.uniform(-0.002, 0.002, (40, 3)) * L,  # around the corner, both sides
            rng.uniform(-1.0, 2.0, (40, 3)) * L,  # outside: wrapped by the query
        ]
    )
    k = 6
    idx, dist = query_neighbors(
        tree,
        jnp.asarray(queries),
        k=k,
        backend=backend,
        point_block_size=64,
        box_size=box,
    )
    q = np.asarray(wrap_positions(jnp.asarray(queries), box))
    d = q[:, None, :] - pts[None, :, :]
    d -= L * np.round(d / L)
    d2 = np.sum(d * d, axis=-1)
    want = np.sort(np.sqrt(d2), axis=1)[:, :k]
    np.testing.assert_allclose(np.sort(np.asarray(dist), axis=1), want, atol=1e-12)
    got = np.take_along_axis(np.sqrt(d2), np.asarray(idx), axis=1)
    np.testing.assert_allclose(np.sort(got, axis=1), want, atol=1e-12)


def _pinned_queries(box, rng) -> np.ndarray:
    """Queries exactly on the corners, edges and faces of the box, and just inside."""
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    top = np.nextafter(L, 0.0)  # the largest coordinate inside [0, L)
    corners = np.array(np.meshgrid(*[[0.0, t] for t in top])).reshape(3, -1).T
    edges, faces = [], []
    for axis in range(3):
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                e = rng.uniform(0.0, 1.0, (3, 3)) * L
                others = [i for i in range(3) if i != axis]
                e[:, others[0]] = a * top[others[0]]
                e[:, others[1]] = b * top[others[1]]
                edges.append(e)
            f = rng.uniform(0.0, 1.0, (4, 3)) * L
            f[:, axis] = a * top[axis]
            faces.append(f)
    near = rng.uniform(0.0, 1e-4, (16, 3)) * L
    near[8:] = top - near[8:]
    return np.concatenate([corners, *edges, *faces, near])


@pytest.mark.parametrize("box", [1.0, (1.0, 2.0, 0.5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("leaf_size,k", [(8, 1), (8, 16), (32, 8), (32, 32)])
def test_query_neighbors_periodic_tree_pinned_queries(box, dtype, leaf_size, k):
    """The pruned periodic walk (box-ordered, box-pruned) is exact at the faces.

    Larger N than test_query_neighbors_periodic, so the tree has many levels and
    the far-child ordering and the minimum-image box bound decide what is
    visited; queries sit exactly on corners, edges and faces.
    """
    rng = np.random.default_rng(31)
    pts = np.asarray(
        wrap_positions(jnp.asarray(_box_points(3000, box, seed=30)), box)
    ).astype(dtype)
    queries = np.concatenate([_pinned_queries(box, rng), pts[:64]]).astype(dtype)
    tree = build_kdtree(jnp.asarray(pts), leaf_size=leaf_size)
    idx, dist = query_neighbors(
        tree, jnp.asarray(queries), k=k, backend="tree", box_size=box
    )
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    p64, q64 = pts.astype(np.float64), queries.astype(np.float64)
    d = q64[:, None, :] - p64[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.sum(d * d, axis=-1))
    want = np.sort(r, axis=1)[:, :k]
    tol = 1e-12 if dtype == np.float64 else 2e-6
    np.testing.assert_allclose(np.asarray(dist, np.float64), want, atol=tol)
    got = np.take_along_axis(r, np.asarray(idx), axis=1)
    np.testing.assert_allclose(np.sort(got, axis=1), want, atol=tol)
    # the dense backend is the brute force on the same tree: identical sets
    idx_d, _ = query_neighbors(
        tree, jnp.asarray(queries), k=k, backend="dense", box_size=box
    )
    if dtype == np.float64:
        np.testing.assert_array_equal(
            np.sort(np.asarray(idx), axis=1), np.sort(np.asarray(idx_d), axis=1)
        )


def test_query_neighbors_periodic_tree_exclude_self_at_corners():
    box = (1.0, 1.5, 0.75)
    pts = np.asarray(wrap_positions(jnp.asarray(_box_points(2000, box, seed=32)), box))
    tree = build_kdtree(jnp.asarray(pts), leaf_size=16)
    idx, dist = query_neighbors(
        tree, jnp.asarray(pts), k=8, exclude_self=True, backend="tree", box_size=box
    )
    L = np.asarray(box)
    d = pts[:, None, :] - pts[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(r, np.inf)
    np.testing.assert_allclose(np.asarray(dist), np.sort(r, axis=1)[:, :8], atol=1e-12)
    assert not np.any(np.asarray(idx) == np.arange(len(pts))[:, None])


@pytest.mark.parametrize("box", [1.0, (1.0, 2.0, 0.5)])
def test_count_neighbors_periodic_tree_pinned_queries(box):
    rng = np.random.default_rng(33)
    pts = np.asarray(wrap_positions(jnp.asarray(_box_points(3000, box, seed=34)), box))
    queries = np.concatenate([_pinned_queries(box, rng), pts[:64]])
    tree = build_kdtree(jnp.asarray(pts), leaf_size=16)
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    radii = np.asarray([0.005, 0.02, 0.08, 0.2]) * float(np.min(L))
    got = count_neighbors(
        tree,
        jnp.asarray(queries),
        radius=jnp.asarray(radii),
        backend="tree",
        box_size=box,
    )
    d = queries[:, None, :] - pts[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.sum(d * d, axis=-1))
    want = (r[:, :, None] <= radii[None, None, :]).sum(axis=1)
    np.testing.assert_array_equal(np.asarray(got), want)


def test_query_neighbors_periodic_rejects_points_outside_box():
    tree = build_kdtree(jnp.asarray(_box_points(64, 1.0, seed=13)), leaf_size=8)
    with pytest.raises(ValueError, match="inside"):
        query_neighbors(tree, jnp.zeros((2, 3)), k=2, box_size=1.0)


def test_query_neighbors_box_none_unchanged():
    pts = np.asarray(wrap_positions(jnp.asarray(_box_points(200, 1.0, seed=14)), 1.0))
    tree = build_kdtree(jnp.asarray(pts), leaf_size=8)
    for backend in ("dense", "tiled", "tree"):
        a = query_neighbors(tree, jnp.asarray(pts[:30]), k=4, backend=backend)
        b = query_neighbors(
            tree, jnp.asarray(pts[:30]), k=4, backend=backend, box_size=None
        )
        np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
        np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))


# ---------------------------------------------------------------------------
# mutual_walk_inputs, the Pallas backend and the static cells tree
# ---------------------------------------------------------------------------


def _mutual_from_inputs(w, theta, mac_type, **kwargs):
    res = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        theta,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
        mac_type=mac_type,
        node_active=w.node_active,
        box_size=w.box_size,
        periodic_radii=w.periodic_radii,
        **kwargs,
    )
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )
    nf, nn = int(res.far_count), int(res.near_count)
    far = set(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    near = set(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    return far, near


@pytest.mark.parametrize("tree_type", ["radix", "octree", "kdtree"])
@pytest.mark.parametrize("mac_type", ["bh", "dehnen"])
def test_mutual_walk_on_periodic_inputs_is_the_periodic_dual_walk(tree_type, mac_type):
    """``mutual_walk_inputs(box_size=)`` reproduces the periodic dual walk as sets.

    ``bh`` radii are box half-extents, which do not bound particle distances, so
    this also pins that the helper hands the walk the bounding radii for the guard.
    """
    box = (1.0, 1.0, 1.25)
    theta = 0.7
    pos = _box_points(700, box, seed=21)
    tree, geometry = _build(pos, box, tree_type=tree_type, leaf=4)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type=mac_type,
        traversal_config=_CONFIG,
        box_size=box,
    )
    far_ref = _far_set_from_csr(interactions)
    near_ref = _near_set(neighbors)
    w = mutual_walk_inputs(tree, geometry, mac_type=mac_type, box_size=box)
    assert w.box_size == box
    np.testing.assert_array_equal(
        np.asarray(w.periodic_radii), np.asarray(geometry.radius)
    )
    far, near = _mutual_from_inputs(w, theta, mac_type)
    assert far_ref, "the test should exercise far acceptance"
    assert far == far_ref
    assert near == near_ref
    # without a box the helper carries nothing periodic
    plain = mutual_walk_inputs(tree, geometry, mac_type=mac_type)
    assert plain.box_size is None and plain.periodic_radii is None


def test_mutual_walk_pallas_backend_refuses_a_box():
    box = 1.0
    pos = _box_points(300, box, seed=22)
    tree, geometry = _build(pos, box, leaf=4)
    w = mutual_walk_inputs(tree, geometry, mac_type="dehnen", box_size=box)
    with pytest.raises(ValueError, match="periodic box_size"):
        _mutual_from_inputs(w, 0.6, "dehnen", backend="pallas", pallas_interpret=True)
    # "auto" (even where it would pick the kernel) walks flat with a box
    flat = _mutual_from_inputs(w, 0.6, "dehnen")
    auto = _mutual_from_inputs(w, 0.6, "dehnen", backend="auto", pallas_interpret=True)
    assert auto == flat


def test_static_cells_tree_periodic_under_jit():
    box = (1.0, 1.0, 1.25)
    pos = _box_points(1500, box, seed=23)
    # one particle in the top corner cell on every axis: the padding code's cell
    pos[0] = np.nextafter(np.asarray(box), 0.0)
    pos_j = jnp.asarray(pos)
    mass = jnp.ones((pos.shape[0],))
    theta = 0.7

    @jax.jit
    def step(p, m):
        topo, ps, _ms, _inv, overflow = build_static_cells_tree(
            p,
            m,
            leaf_size=8,
            leaf_capacity=1024,
            return_reordered=True,
            return_overflow=True,
            box_size=box,
        )
        geometry = compute_tree_geometry(topo, ps)
        w = mutual_walk_inputs(
            topo, geometry, mac_type="dehnen", mask_empty_nodes=True, box_size=box
        )
        res = dual_tree_walk_mutual(
            w.left_child_full,
            w.right_child_full,
            w.centers,
            w.radii,
            theta,
            w.root,
            max_pair_queue=1 << 16,
            far_cap=1 << 17,
            near_cap=1 << 17,
            mac_type="dehnen",
            node_active=w.node_active,
            box_size=w.box_size,
            periodic_radii=w.periodic_radii,
        )
        topology = resolve_tree_topology(topo)
        return (
            res,
            ps,
            geometry.center,
            topology.node_ranges,
            overflow,
        )

    res, ps, centers, ranges, overflow = step(pos_j, mass)
    assert not bool(overflow)
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )
    ps = np.asarray(ps)
    L = np.broadcast_to(np.asarray(box), (3,))
    assert np.all(ps >= 0.0) and np.all(ps < L)
    ranges = np.asarray(ranges)
    num_internal = (ranges.shape[0] - 1) // 2  # a full binary tree
    leaves = list(range(num_internal, ranges.shape[0]))
    nf, nn = int(res.far_count), int(res.near_count)
    far = list(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    near = list(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    assert far, "the test should exercise far acceptance"
    centers = np.asarray(centers)
    _check_coverage(ps.shape[0], ranges, leaves, far, near)
    _check_far_unambiguous(ps, centers, ranges, far, box)
    edges = np.linspace(0.0, 1.0, 33)
    np.testing.assert_array_equal(
        _binned_counts_from_walk(ps, centers, ranges, leaves, far, near, box, edges),
        _brute_binned(ps, box, edges),
    )


def test_static_cells_tree_box_argument_errors():
    pos = jnp.asarray(_box_points(64, 1.0, seed=24))
    mass = jnp.ones((64,))
    with pytest.raises(ValueError, match="either bounds or box_size"):
        build_static_cells_tree(
            pos,
            mass,
            (jnp.zeros(3), jnp.ones(3)),
            leaf_size=8,
            leaf_capacity=64,
            box_size=1.0,
        )
    with pytest.raises(ValueError, match="bounds or box_size"):
        build_static_cells_tree(pos, mass, leaf_size=8, leaf_capacity=64)


# ---------------------------------------------------------------------------
# radius counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["tiled", "tree"])
@pytest.mark.parametrize("box", [1.0, (1.0, 2.0, 0.5)])
@pytest.mark.parametrize("include_self", [True, False])
def test_count_neighbors_periodic(backend, box, include_self):
    pts = np.asarray(wrap_positions(jnp.asarray(_box_points(400, box, seed=25)), box))
    tree = build_kdtree(jnp.asarray(pts), leaf_size=8)
    L = np.broadcast_to(np.asarray(box, np.float64), (3,))
    radii = np.asarray([0.01, 0.05, 0.1, 0.2]) * float(np.min(L))
    d = pts[:, None, :] - pts[None, :, :]
    d -= L * np.round(d / L)
    dist = np.sqrt(np.sum(d * d, axis=-1))
    if not include_self:
        np.fill_diagonal(dist, np.inf)
    want = (dist[:, :, None] <= radii[None, None, :]).sum(axis=1)
    got = count_neighbors(
        tree,
        jnp.asarray(pts),
        radius=jnp.asarray(radii),
        include_self=include_self,
        backend=backend,
        point_block_size=64,
        box_size=box,
    )
    np.testing.assert_array_equal(np.asarray(got), want)
    # queries outside the box are wrapped; a scalar radius gives one column
    rng = np.random.default_rng(26)
    queries = rng.uniform(-1.0, 2.0, (50, 3)) * L
    q = np.asarray(wrap_positions(jnp.asarray(queries), box))
    dq = q[:, None, :] - pts[None, :, :]
    dq -= L * np.round(dq / L)
    want_q = (np.sum(dq * dq, axis=-1) <= radii[2] ** 2).sum(axis=1)
    got_q = count_neighbors(
        tree,
        jnp.asarray(queries),
        radius=float(radii[2]),
        backend=backend,
        point_block_size=64,
        box_size=box,
    )
    np.testing.assert_array_equal(np.asarray(got_q), want_q)


def test_count_neighbors_periodic_rejects_points_outside_box_and_none_unchanged():
    raw = _box_points(200, 1.0, seed=27)
    with pytest.raises(ValueError, match="inside"):
        count_neighbors(
            build_kdtree(jnp.asarray(raw), leaf_size=8),
            jnp.zeros((2, 3)),
            radius=0.1,
            box_size=1.0,
        )
    pts = jnp.asarray(wrap_positions(jnp.asarray(raw), 1.0))
    tree = build_kdtree(pts, leaf_size=8)
    for backend in ("tiled", "tree"):
        a = count_neighbors(tree, pts[:30], radius=0.2, backend=backend)
        b = count_neighbors(tree, pts[:30], radius=0.2, backend=backend, box_size=None)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# optional cross-check against Corrfunc
# ---------------------------------------------------------------------------


def test_pair_counts_match_corrfunc():
    Corrfunc = pytest.importorskip("Corrfunc")
    from Corrfunc.theory.DD import DD

    box = 1.0
    pos = np.asarray(wrap_positions(jnp.asarray(_box_points(480, box, seed=15)), box))
    tree, geometry = _build(pos, box)
    far, near = _mutual(tree, geometry, 0.7, box)
    ps = np.asarray(tree.positions_sorted)
    edges = np.linspace(0.01, 0.45, 12)
    ours = _binned_counts_from_walk(
        ps,
        np.asarray(geometry.center),
        _node_ranges(tree),
        _leaf_nodes(tree),
        far,
        near,
        box,
        edges,
    )
    res = DD(
        1,
        1,
        edges,
        ps[:, 0].copy(),
        ps[:, 1].copy(),
        ps[:, 2].copy(),
        periodic=True,
        boxsize=box,
    )
    # Corrfunc counts ordered pairs in an auto-correlation
    np.testing.assert_array_equal(2 * ours, np.asarray(res["npairs"], np.int64))
    del Corrfunc
