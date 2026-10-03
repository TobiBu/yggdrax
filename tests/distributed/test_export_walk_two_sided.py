"""``export_walk_two_sided``: the export walk refines the RECEIVER's side too.

The claims, each asserted rather than inferred:

* the export lists COVER every (receiver particle, sender particle) pair exactly once
  -- counted on a particle matrix, because momentum cannot see a coverage error;
* the same count holds for the one-sided :func:`export_walk` (the control);
* against a summary that is a single cell, the two walks emit the same pairs;
* near pairs only ever name cut cells (the receiver ships particles against them);
* far pairs are fewer, because big sender nodes pair with big receiver nodes;
* a device never exports to itself.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_export_walk_two_sided.py -q
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
from yggdrax.distributed.export import export_walk, export_walk_two_sided
from yggdrax.distributed.summary import occupancy_cut, summary_tree
from yggdrax.geometry import compute_tree_geometry
from yggdrax.morton import morton_encode

_MAC = "dehnen"
_LEAF = 16
_CAP = 1 << 17


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 30.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _domain(points, bounds, max_leaves):
    """One device: tree, box geometry, full child arrays, cut and summary tree."""
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
    ps, _, _ = reorder_particles_by_indices(P, M, order)
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
    nint = k - 1
    total = int(topo.parent.shape[0])
    fill = jnp.full((total - nint,), -1, I)
    cut = occupancy_cut(
        topo.parent, topo.node_ranges, nint, max_leaves=max_leaves, capacity=total
    )
    st = summary_tree(
        topo.parent,
        topo.left_child,
        topo.right_child,
        topo.node_ranges,
        nint,
        cut,
        capacity=total,
    )
    assert not (bool(cut.overflow) or bool(st.overflow))
    return dict(
        n=int(n),
        ranges=np.asarray(ranges),
        left=jnp.concatenate([topo.left_child, fill]),
        right=jnp.concatenate([topo.right_child, fill]),
        center=jnp.asarray(geom.center),
        radius=jnp.asarray(geom.radius),
        root=jnp.argmin(topo.parent).astype(I),
        cut=cut,
        st=st,
    )


def _halves(n=4000, seed=3, max_leaves=4):
    """Two ADJACENT domains: a Plummer sphere split at x = median."""
    pts = _plummer(n, seed)
    order = np.argsort(pts[:, 0])
    bounds = infer_bounds(jnp.asarray(pts, jnp.float32))
    h = n // 2
    return [
        _domain(pts[order[:h]], bounds, max_leaves),
        _domain(pts[order[h:]], bounds, max_leaves),
    ]


def _quads(n=4000, seed=5, max_leaves=4):
    """Three domains (ndev = 3), split along x into thirds."""
    pts = _plummer(n, seed)
    order = np.argsort(pts[:, 0])
    bounds = infer_bounds(jnp.asarray(pts, jnp.float32))
    parts = np.array_split(order, 3)
    return [_domain(pts[p], bounds, max_leaves) for p in parts]


def _pad(rows, S, fill):
    out = np.full((S,) + rows.shape[1:], fill, rows.dtype)
    out[: rows.shape[0]] = rows
    return out


def _published_trees(doms):
    """Every device's summary tree, padded to a common S (what the gather delivers)."""
    S = max(int(d["st"].num_nodes) for d in doms)
    cen, rad, lft, rgt, act, nodes = [], [], [], [], [], []
    for d in doms:
        st = d["st"]
        m = int(st.num_nodes)
        nd = np.asarray(st.nodes)[:m]
        nodes.append(_pad(nd, S, -1))
        cen.append(_pad(np.asarray(d["center"])[nd], S, 0.0))
        rad.append(_pad(np.asarray(d["radius"])[nd], S, 0.0))
        lft.append(_pad(np.asarray(st.left)[:m], S, -1))
        rgt.append(_pad(np.asarray(st.right)[:m], S, -1))
        act.append(_pad(np.asarray(st.active)[:m], S, False))
    arr = [jnp.asarray(np.stack(a)) for a in (cen, rad, lft, rgt, act)]
    return arr, np.stack(nodes), S


def _published_cells(doms):
    """Every device's cut cells, padded to a common capacity (the one-sided summary)."""
    S = max(int(d["cut"].num_cells) for d in doms)
    cen, rad, act, nodes = [], [], [], []
    for d in doms:
        m = int(d["cut"].num_cells)
        cells = np.asarray(d["cut"].cells)[:m]
        nodes.append(_pad(cells, S, -1))
        cen.append(_pad(np.asarray(d["center"])[cells], S, 0.0))
        rad.append(_pad(np.asarray(d["radius"])[cells], S, 0.0))
        act.append(_pad(np.ones(m, bool), S, False))
    arr = [jnp.asarray(np.stack(a)) for a in (cen, rad, act)]
    return arr, np.stack(nodes), S


def _two_sided(doms, me, theta, **kw):
    (cen, rad, lft, rgt, act), nodes, S = _published_trees(doms)
    d = doms[me]
    ex = export_walk_two_sided(
        d["left"],
        d["right"],
        d["center"],
        d["radius"],
        d["root"],
        cen,
        rad,
        lft,
        rgt,
        act,
        float(theta),
        jnp.asarray(me),
        max_pair_queue=_CAP,
        far_cap=_CAP,
        near_cap=_CAP,
        mac_type=_MAC,
        **kw,
    )
    return ex, nodes, S


def _one_sided(doms, me, theta):
    (cen, rad, act), nodes, S = _published_cells(doms)
    d = doms[me]
    ex = export_walk(
        d["left"],
        d["right"],
        d["center"],
        d["radius"],
        d["root"],
        cen,
        rad,
        act,
        float(theta),
        jnp.asarray(me),
        max_pair_queue=_CAP,
        far_cap=_CAP,
        near_cap=_CAP,
        mac_type=_MAC,
    )
    return ex, nodes, S


def _lists(ex):
    assert not (
        bool(ex.far_overflow) or bool(ex.near_overflow) or bool(ex.queue_overflow)
    )
    nf, nn = int(ex.far_count), int(ex.near_count)
    return (
        np.asarray(ex.far_cell)[:nf],
        np.asarray(ex.far_node)[:nf],
        np.asarray(ex.near_cell)[:nn],
        np.asarray(ex.near_node)[:nn],
    )


def _coverage(doms, sender, ex, nodes, S):
    """Count matrices (one per receiver) over (receiver particle, sender particle)."""
    fc, fn, nc, nn = _lists(ex)
    send = doms[sender]
    C = {r: np.zeros((doms[r]["n"], send["n"]), np.int32) for r in range(len(doms))}
    for cells, snodes in ((fc, fn), (nc, nn)):
        for g, s in zip(cells.tolist(), snodes.tolist()):
            r, i = divmod(g, S)
            t = int(nodes[r, i])
            t0, t1 = doms[r]["ranges"][t]
            s0, s1 = send["ranges"][s]
            if t0 <= t1 and s0 <= s1:
                C[r][t0 : t1 + 1, s0 : s1 + 1] += 1
    return C


@pytest.mark.parametrize("walk", ["two_sided", "one_sided"])
@pytest.mark.parametrize("theta", [0.8, 0.5])
def test_the_export_lists_cover_every_cross_pair_once(walk, theta):
    doms = _halves()
    for me in (0, 1):
        ex, nodes, S = (_two_sided if walk == "two_sided" else _one_sided)(
            doms, me, theta
        )
        assert int(ex.far_count) > 0 and int(ex.near_count) > 0, "vacuous"
        C = _coverage(doms, me, ex, nodes, S)
        other = 1 - me
        assert np.all(C[other] == 1), (
            f"uncovered {int(np.sum(C[other] == 0))}, "
            f"multiply covered {int(np.sum(C[other] > 1))}"
        )
        assert not np.any(C[me]), "a device exported to itself"


def test_three_devices_cover_every_cross_pair_once():
    doms = _quads()
    for me in range(3):
        ex, nodes, S = _two_sided(doms, me, 0.7)
        C = _coverage(doms, me, ex, nodes, S)
        for r in range(3):
            if r == me:
                assert not np.any(C[r]), "a device exported to itself"
            else:
                assert np.all(C[r] == 1), f"{me} -> {r}"


def test_near_pairs_name_cut_cells_only():
    doms = _halves()
    ex, nodes, S = _two_sided(doms, 0, 0.8)
    _, _, nc, _ = _lists(ex)
    r, i = np.divmod(nc, S)
    assert np.all(r == 1)
    is_cell = np.asarray(doms[1]["st"].is_cell)
    assert np.all(is_cell[i]), "a near pair landed on a summary node that is no cell"
    # and far pairs DO land above the cells -- otherwise the walk was one-sided
    fc, _, _, _ = _lists(ex)
    assert np.any(~is_cell[fc % S]), "vacuous: no far pair above the cut"


def test_a_single_cell_summary_gives_the_one_sided_pairs():
    """When the receiver's top tree is one cell (the root), there is nothing to split
    on its side, and the two walks must agree pair for pair."""
    doms = _halves(max_leaves=1 << 20)
    for d in doms:
        assert int(d["cut"].num_cells) == 1 and int(d["st"].num_nodes) == 1
    two, _, S2 = _two_sided(doms, 0, 0.8)
    one, _, S1 = _one_sided(doms, 0, 0.8)
    assert S1 == S2 == 1
    a, b = _lists(two), _lists(one)
    assert set(zip(a[0].tolist(), a[1].tolist())) == set(
        zip(b[0].tolist(), b[1].tolist())
    )
    assert set(zip(a[2].tolist(), a[3].tolist())) == set(
        zip(b[2].tolist(), b[3].tolist())
    )
    assert len(a[0]) > 0 and len(a[2]) > 0, "vacuous"


def test_two_sided_emits_fewer_far_pairs():
    doms = _halves(n=8000)
    two, _, _ = _two_sided(doms, 0, 0.8)
    one, _, _ = _one_sided(doms, 0, 0.8)
    f2, f1 = int(two.far_count), int(one.far_count)
    n2, n1 = int(two.near_count), int(one.near_count)
    print(f"far pairs: one-sided {f1}, two-sided {f2} ({f1 / max(f2, 1):.2f}x fewer)")
    print(f"near pairs: one-sided {n1}, two-sided {n2}")
    assert f2 < f1
    # the near list is (cell, leaf) in both and must not grow
    assert n2 <= n1


def test_my_own_block_is_never_walked_even_when_active():
    """The caller does not have to mask its own row: the walk does."""
    doms = _halves()
    ex, _, S = _two_sided(doms, 1, 0.8)
    fc, _, nc, _ = _lists(ex)
    assert np.all(fc // S == 0) and np.all(nc // S == 0)


def test_shape_mismatch_is_rejected():
    doms = _halves(n=400)
    (cen, rad, lft, rgt, act), _, _ = _published_trees(doms)
    d = doms[0]
    with pytest.raises(ValueError, match="summary_left"):
        export_walk_two_sided(
            d["left"],
            d["right"],
            d["center"],
            d["radius"],
            d["root"],
            cen,
            rad,
            lft[:, :-1],
            rgt,
            act,
            0.8,
            jnp.asarray(0),
            max_pair_queue=64,
            far_cap=64,
            near_cap=64,
        )
