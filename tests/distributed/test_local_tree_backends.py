"""Multi-GPU distributed tree build + cross-walk across backends.

`distributed_tree_moments` / the LET drivers gained a `tree_type` selector
(``radix``, ``octree``, ``kdtree``). The build correctness anchor is
backend-independent: the union of the per-domain coarse moments must reproduce
the *global* total mass and centre of mass a single-device build would give.

For ``kdtree`` the distributed path uses the leaf-only bucket build
(:func:`yggdrax.kdtree.build_leaf_kdtree`), which stores all points in leaves and
so satisfies the same topology contract as radix/octree -- unlike the heap
KD-tree used by the KNN kernels. The cross-walk test asserts the actual thing
that was previously broken for KD: a complete, disjoint source partition.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_local_tree_backends.py -o addopts="" -q
"""

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.bounds import infer_bounds
from yggdrax.distributed import (
    build_distributed_coarse_tree,
    device_count,
    distributed_tree_moments,
    dual_tree_walk_cross,
    make_mesh,
)
from yggdrax.distributed.local_tree import (
    DISTRIBUTED_TREE_TYPES,
    _build_local_tree,
    _validate_distributed_tree_type,
)
from yggdrax.geometry import compute_tree_geometry

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 16
_N = _PER_DEV * _NDEV
_CAP = 4 * _PER_DEV

needs_devices = pytest.mark.skipif(
    device_count() < 2, reason="distributed tree build needs >= 2 devices"
)


# ---- validation (no devices needed) ----
def test_distributed_tree_types_advertised():
    assert DISTRIBUTED_TREE_TYPES == ("radix", "octree", "kdtree")


def test_unknown_tree_type_rejected():
    with pytest.raises(ValueError, match="Unsupported tree_type"):
        _validate_distributed_tree_type("bogus")


# ---- distributed build reconstruction: radix baseline + octree + kdtree ----
@pytest.fixture(scope="module")
def built():
    """Run distributed_tree_moments once per backend; reuse across assertions."""
    mesh = make_mesh(_NDEV)
    rng = np.random.default_rng(10)
    pts = rng.uniform(-1.0, 1.0, size=(_N, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(_N,)).astype(np.float32)
    out = {}
    for tt in ("octree", "kdtree"):
        out[tt] = distributed_tree_moments(
            mesh,
            jnp.asarray(pts),
            jnp.asarray(mass),
            leaf_size=_LEAF,
            output_capacity=_CAP,
            equalize=True,
            tree_type=tt,
        )
    return pts, mass, out


@needs_devices
@pytest.mark.parametrize("tree_type", ["octree", "kdtree"])
def test_domain_moments_reconstruct_global(built, tree_type):
    pts, mass, out = built
    res = out[tree_type]
    domain_mass = np.asarray(res.domain_mass)
    domain_com = np.asarray(res.domain_com)
    counts = np.asarray(res.counts)

    assert int(counts.sum()) == _N
    np.testing.assert_allclose(domain_mass.sum(), mass.sum(), rtol=1e-4)
    global_com = (mass[:, None] * pts).sum(0) / mass.sum()
    recon_com = (domain_mass[:, None] * domain_com).sum(0) / domain_mass.sum()
    np.testing.assert_allclose(recon_com, global_com, rtol=1e-3, atol=1e-4)


@needs_devices
@pytest.mark.parametrize("tree_type", ["octree", "kdtree"])
def test_shared_top_tree_replicated(built, tree_type):
    _, _, out = built
    res = out[tree_type]
    top_mass = np.asarray(res.top_mass).reshape(_NDEV, _NDEV)
    domain_mass = np.asarray(res.domain_mass)
    for g in range(_NDEV):
        np.testing.assert_allclose(top_mass[g], top_mass[0], rtol=1e-5)
    np.testing.assert_allclose(top_mass[0], domain_mass, rtol=1e-5)


@needs_devices
@pytest.mark.parametrize("tree_type", ["octree", "kdtree"])
def test_coarse_tree_root_is_global(built, tree_type):
    """build_distributed_coarse_tree: coarse root mass == global (any local backend)."""
    pts, mass, _ = built
    mesh = make_mesh(_NDEV)
    metrics = build_distributed_coarse_tree(
        mesh,
        jnp.asarray(pts),
        jnp.asarray(mass),
        leaf_size=_LEAF,
        output_capacity=_CAP,
        equalize=True,
        tree_type=tree_type,
    )
    np.testing.assert_allclose(np.asarray(metrics.coarse_root_mass), mass.sum(), rtol=1e-3)


# ---- cross-tree walk over leaf-only KD trees: complete disjoint source partition ----
def _kd_build(points, leaf_size=_LEAF):
    pts = jnp.asarray(points)
    tree, ps, _ms = _build_local_tree(
        pts, jnp.ones(pts.shape[0], jnp.float32), infer_bounds(pts),
        tree_type="kdtree", leaf_size=leaf_size,
    )
    geom = compute_tree_geometry(tree, ps, max_leaf_size=leaf_size)
    return tree, geom


def test_kdtree_cross_walk_partitions_sources():
    """Every target leaf's far ancestors + near leaves tile ALL source particles once.

    This is the invariant the heap KD-tree violated (internal-node pivots absent
    from the near-list); the leaf-only build must satisfy it exactly.
    """
    rng = np.random.default_rng(3)
    tgt = rng.uniform(-1.0, 0.2, size=(80, 3)).astype(np.float32)
    src = rng.uniform(-0.2, 1.0, size=(80, 3)).astype(np.float32)
    # shared frame
    allpts = jnp.asarray(np.concatenate([tgt, src], 0))
    bounds = infer_bounds(allpts)

    def build(points):
        pts = jnp.asarray(points)
        tree, ps, _ = _build_local_tree(
            pts, jnp.ones(pts.shape[0], jnp.float32), bounds,
            tree_type="kdtree", leaf_size=_LEAF,
        )
        return tree, compute_tree_geometry(tree, ps, max_leaf_size=_LEAF)

    t_tree, t_geom = build(tgt)
    s_tree, s_geom = build(src)
    res = dual_tree_walk_cross(
        t_tree, t_geom, s_tree, s_geom, 0.5, mac_type="bh",
        max_interactions_per_node=512, max_neighbors_per_leaf=512, max_pair_queue=16384,
    )
    assert not bool(res.queue_overflow | res.far_overflow | res.near_overflow)

    parent = np.asarray(t_tree.parent)
    s_ranges = np.asarray(s_tree.node_ranges)
    tt = np.asarray(res.interaction_targets)
    ss = np.asarray(res.interaction_sources)
    m = tt >= 0
    far_map = {}
    for t_node, s_node in zip(tt[m].tolist(), ss[m].tolist()):
        far_map.setdefault(t_node, []).append(s_node)
    near_off = np.asarray(res.neighbor_offsets)
    near_idx = np.asarray(res.neighbor_indices)
    near_cnt = np.asarray(res.neighbor_counts)
    leaf_indices = np.asarray(res.leaf_indices)
    n_source = int(src.shape[0])
    root = int(np.argmin(parent))

    def src_particles(node):
        lo, hi = int(s_ranges[node, 0]), int(s_ranges[node, 1])
        return list(range(lo, hi + 1))

    for row, leaf_node in enumerate(leaf_indices.tolist()):
        far_sources, node = [], leaf_node
        while True:
            far_sources.extend(far_map.get(node, []))
            if node == root:
                break
            node = int(parent[node])
        o, c = int(near_off[row]), int(near_cnt[row])
        covered = []
        for sn in far_sources + near_idx[o:o + c].tolist():
            covered.extend(src_particles(sn))
        assert set(covered) == set(range(n_source)), (
            f"leaf_row {row}: covered {len(set(covered))}/{n_source}"
        )
        assert len(covered) == n_source, f"leaf_row {row}: overlap {len(covered)}"
