"""``export_walk``: what a device owes every other device, decided from summaries alone.

The load-bearing claims, each asserted rather than inferred:

* walking every receiver AT ONCE gives exactly what walking them one at a time gives;
* every emitted pair is ``(cell, local node)`` -- the orientation the receiver's
  evaluation depends on, and silent if reversed;
* a device never exports to ITSELF, which would count its particles twice while
  leaving momentum exact;
* the export is CONSERVATIVE against the receiver's real targets: nothing a finer
  summary would have asked for is missing, once covered-by-an-ancestor is allowed for.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_export_walk.py -q
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
from yggdrax.distributed.export import export_walk
from yggdrax.distributed.summary import occupancy_cut
from yggdrax.geometry import compute_tree_geometry
from yggdrax.morton import morton_encode

_THETA = 0.6
_MAC = "dehnen"
_LEAF = 16


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _tree(points, bounds):
    P = jnp.asarray(points, jnp.float32)
    n = P.shape[0]
    M = jnp.ones((n,), jnp.float32)
    codes = morton_encode(P, bounds)
    order = jnp.argsort(codes, stable=True)
    sc = codes[order]
    k = int(adaptive_cell_leaf_partition(sc, leaf_size=_LEAF, capacity=n).num_leaves)
    part = adaptive_cell_leaf_partition(sc, leaf_size=_LEAF, capacity=k)
    starts = np.asarray(part.leaf_starts).astype(np.int64)[:k]
    ends = np.asarray(part.leaf_ends).astype(np.int64)[:k]
    parent, left, right, lil, ril, ranges, lvl, loff, nbl, nlv = (
        _build_balanced_bucket_structure(starts, ends)
    )
    ps, ms, _ = reorder_particles_by_indices(P, M, order)
    I = jnp.int32
    topo = RadixTree(
        parent=jnp.asarray(parent, I),
        left_child=jnp.asarray(left, I),
        right_child=jnp.asarray(right, I),
        left_is_leaf=jnp.asarray(lil),
        right_is_leaf=jnp.asarray(ril),
        particle_indices=jnp.asarray(order, I),
        morton_codes=sc,
        node_ranges=jnp.asarray(ranges, I),
        num_particles=n,
        num_internal_nodes=k - 1,
        node_level=jnp.asarray(lvl, I),
        level_offsets=jnp.asarray(loff, I),
        nodes_by_level=jnp.asarray(nbl, I),
        num_levels=jnp.asarray(nlv, I),
        bounds_min=jnp.asarray(bounds[0], P.dtype),
        bounds_max=jnp.asarray(bounds[1], P.dtype),
        leaf_codes=sc[jnp.asarray(np.minimum(starts, n - 1), I)],
        leaf_depths=jnp.asarray(part.leaf_depths, I)[:k],
        use_morton_geometry=jnp.asarray(False),
        leaf_size=_LEAF,
    )
    geom = compute_tree_geometry(topo, ps, max_leaf_size=_LEAF)
    return topo, geom


def _children_full(topo):
    total = int(topo.parent.shape[0])
    nint = int(topo.left_child.shape[0])
    fill = jnp.full((total - nint,), -1, topo.parent.dtype)
    return (
        jnp.concatenate([topo.left_child, fill]),
        jnp.concatenate([topo.right_child, fill]),
    )


def _domains(ndev, per=700, sep=2.6):
    """Separated clouds, one per device, in a shared frame."""
    offs = np.array([[0, 0, 0], [sep, 0, 0], [0, sep, 0], [0, 0, sep]], np.float64)
    pts = [_plummer(per, seed=d) + offs[d] for d in range(ndev)]
    bounds = infer_bounds(jnp.asarray(np.concatenate(pts), jnp.float32))
    return [_tree(p, bounds) for p in pts]


def _summaries(doms, max_leaves=4):
    """Every device's occupancy-cut summary, padded to a common capacity."""
    cuts = []
    for topo, geom in doms:
        nint = int(topo.left_child.shape[0])
        cuts.append(
            occupancy_cut(
                topo.parent,
                topo.node_ranges,
                nint,
                max_leaves=max_leaves,
                capacity=int(topo.parent.shape[0]),
            )
        )
    cap = max(int(c.num_cells) for c in cuts)
    cen = np.zeros((len(doms), cap, 3), np.float32)
    rad = np.zeros((len(doms), cap), np.float32)
    act = np.zeros((len(doms), cap), bool)
    for d, ((topo, geom), c) in enumerate(zip(doms, cuts)):
        n = int(c.num_cells)
        cells = np.asarray(c.cells)[:n]
        cen[d, :n] = np.asarray(geom.center)[cells]
        rad[d, :n] = np.asarray(geom.radius)[cells]
        act[d, :n] = True
    return jnp.asarray(cen), jnp.asarray(rad), jnp.asarray(act), cap, cuts


def _run(doms, me, cen, rad, act, **kw):
    topo, geom = doms[me]
    left, right = _children_full(topo)
    return export_walk(
        left,
        right,
        jnp.asarray(geom.center),
        jnp.asarray(geom.radius),
        jnp.argmin(topo.parent).astype(topo.parent.dtype),
        cen,
        rad,
        act,
        _THETA,
        jnp.asarray(me),
        max_pair_queue=kw.get("queue", 1 << 16),
        far_cap=kw.get("cap", 1 << 18),
        near_cap=kw.get("cap", 1 << 18),
        mac_type=_MAC,
    )


def _pairs(res):
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )
    f = set(
        zip(
            np.asarray(res.far_cell)[: int(res.far_count)].tolist(),
            np.asarray(res.far_node)[: int(res.far_count)].tolist(),
        )
    )
    n = set(
        zip(
            np.asarray(res.near_cell)[: int(res.near_count)].tolist(),
            np.asarray(res.near_node)[: int(res.near_count)].tolist(),
        )
    )
    return f, n


@pytest.mark.parametrize("ndev", [2, 3])
def test_all_receivers_at_once_equals_one_at_a_time(ndev):
    """The reason the driver walks every receiver in ONE walk."""
    doms = _domains(ndev)
    cen, rad, act, cap, _ = _summaries(doms)
    me = 0
    together = _pairs(_run(doms, me, cen, rad, act))
    apart_f, apart_n = set(), set()
    for r in range(ndev):
        if r == me:
            continue
        one = np.zeros_like(np.asarray(act))
        one[r] = np.asarray(act)[r]
        f, n = _pairs(_run(doms, me, cen, rad, jnp.asarray(one)))
        apart_f |= f
        apart_n |= n
    assert together[0] == apart_f
    assert together[1] == apart_n
    assert apart_f or apart_n


def test_a_device_never_exports_to_itself():
    """Its particles are already local; exporting them counts them twice.

    A momentum check cannot see that, so it is asserted directly: no emitted pair
    names a cell of the sender's own device, even though those cells are handed in
    ACTIVE and the function is the only thing masking them.
    """
    doms = _domains(3)
    cen, rad, act, cap, _ = _summaries(doms)
    assert bool(np.asarray(act)[1].any()), "device 1's own cells must start active"
    f, n = _pairs(_run(doms, 1, cen, rad, act))
    own = range(1 * cap, 2 * cap)
    assert not any(c in own for c, _ in f | n)
    assert f or n


def test_pairs_are_cell_then_local_node_and_the_owner_is_the_block():
    doms = _domains(3)
    cen, rad, act, cap, _ = _summaries(doms)
    me = 2
    f, n = _pairs(_run(doms, me, cen, rad, act))
    n_local = int(doms[me][0].parent.shape[0])
    for c, node in f | n:
        assert 0 <= c < 3 * cap, "cell index out of the summary block"
        assert 0 <= node < n_local, "node index is not in the sender's own space"
        assert c // cap != me
    # near pairs name LEAVES of the sender, far pairs may name any node
    nint = int(doms[me][0].left_child.shape[0])
    assert all(node >= nint for _c, node in n)


def test_every_cells_list_tiles_the_senders_particles_exactly_once():
    """THE correctness property of the export, and the one the receiver relies on.

    For each receiver cell, the nodes exported for it must partition the sender's
    particles: each covered exactly once, none missed. That is what makes the
    receiver's per-cell evaluation a valid sum -- a gap loses force, an overlap
    double-counts it, and a conservation check sees neither.

    It is checked per CELL, not over the union of all cells. The union is
    deliberately not a cut: different cells accept at different depths, so the
    pooled set holds nodes together with their own descendants. That is exactly why
    the payload has to carry provenance.
    """
    for ndev in (2, 3):
        doms = _domains(ndev)
        cen, rad, act, cap, _ = _summaries(doms, max_leaves=4)
        me = 0
        f, n = _pairs(_run(doms, me, cen, rad, act))
        ranges = np.asarray(doms[me][0].node_ranges)
        n_src = int(doms[me][0].num_particles)
        by_cell = {}
        for c, node in f | n:
            by_cell.setdefault(c, []).append(node)
        assert by_cell, "nothing was exported at all"
        for c, nodes in by_cell.items():
            cov = np.zeros(n_src, np.int32)
            for node in nodes:
                lo, hi = int(ranges[node, 0]), int(ranges[node, 1])
                if hi >= lo:
                    cov[lo : hi + 1] += 1
            assert cov.max() <= 1, f"cell {c}: a source particle exported twice"
            assert cov.min() >= 1, f"cell {c}: a source particle never exported"


def test_a_coarser_summary_ships_finer_nodes():
    """The direction of the conservatism, stated the right way round.

    A cell is BIGGER than the leaves inside it, so the MAC is harder to satisfy
    against it and the walk descends FURTHER. A coarse summary therefore ships
    DESCENDANTS of what a per-leaf summary ships, never ancestors -- which is why
    comparing the two as node sets says nothing useful, and why the test above
    compares coverage instead.
    """
    doms = _domains(2)
    coarse = _summaries(doms, max_leaves=8)
    fine = _summaries(doms, max_leaves=1)
    cf, cn = _pairs(_run(doms, 0, *coarse[:3]))
    ff, fn = _pairs(_run(doms, 0, *fine[:3]))
    depth = np.asarray(doms[0][0].node_level)
    coarse_depth = np.mean([depth[n] for _c, n in cf | cn])
    fine_depth = np.mean([depth[n] for _c, n in ff | fn])
    assert coarse_depth > fine_depth, (coarse_depth, fine_depth)


def test_capacity_overflow_is_reported():
    doms = _domains(2)
    cen, rad, act, cap, _ = _summaries(doms)
    tight = _run(doms, 0, cen, rad, act, cap=8)
    assert bool(tight.far_overflow) or bool(tight.near_overflow)


def test_shape_validation():
    doms = _domains(2)
    cen, rad, act, cap, _ = _summaries(doms)
    topo, geom = doms[0]
    left, right = _children_full(topo)
    with pytest.raises(ValueError, match="cell_radii and cell_active"):
        export_walk(
            left,
            right,
            jnp.asarray(geom.center),
            jnp.asarray(geom.radius),
            jnp.argmin(topo.parent).astype(topo.parent.dtype),
            cen,
            rad[:, :-1],
            act,
            _THETA,
            jnp.asarray(0),
            max_pair_queue=1 << 12,
            far_cap=1 << 12,
            near_cap=1 << 12,
        )
