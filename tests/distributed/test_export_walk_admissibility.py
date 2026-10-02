"""Every pair `export_walk` emits as FAR must actually satisfy the MAC.

This is the sender half of a question raised by the jaccpot cross-field probe: at
ndev = 2 the receiver re-tested the pairs the sender sent and rejected about 17 %
of them, which should be impossible if both ends apply the same criterion to the
same pair. The receiver's rejection is measured over the wire; this test asks the
cheaper half of the question -- is the sender's own list admissible by its own
geometry? -- and needs no mesh, no exchange and no GPU.

The predicate is IMPORTED rather than reimplemented, so the test cannot drift from
what the walk actually does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition
from yggdrax._interactions_impl import _compute_mac_ok
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
_MAX_CELLS = 256


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
    return topo, compute_tree_geometry(topo, ps, max_leaf_size=_LEAF)


def _two_domains(n=4000, seed=3):
    """Split a Plummer sphere in Morton order, exactly as the mesh probe does."""
    pts = _plummer(n, seed)
    bounds = infer_bounds(jnp.asarray(pts, jnp.float32))
    codes = np.asarray(morton_encode(jnp.asarray(pts, jnp.float32), bounds))
    shards = np.array_split(np.argsort(codes), 2)
    out = []
    for sel in shards:
        topo, geom = _tree(pts[sel], bounds)
        out.append((topo, geom))
    return out


def _summary(topo, geom):
    n_int = int(topo.left_child.shape[0])
    s = occupancy_cut(
        jnp.asarray(topo.parent),
        jnp.asarray(topo.node_ranges),
        n_int,
        max_leaves=4,
        capacity=_MAX_CELLS,
    )
    live = jnp.arange(_MAX_CELLS) < s.num_cells
    safe = jnp.where(live, s.cells, 0)
    return (
        jnp.where(live[:, None], jnp.asarray(geom.center)[safe], 0.0),
        jnp.where(live, jnp.asarray(geom.radius)[safe], 0.0),
        live,
        s,
    )


def _mac_fail_fraction(cell_idx, node_idx, count, all_cen, all_rad, geom):
    """Recompute the real MAC over (global cell, local node) pairs."""
    ci = jnp.asarray(cell_idx)
    ni = jnp.asarray(node_idx)
    live = (jnp.arange(ci.shape[0]) < count) & (ci >= 0)
    dev = jnp.where(live, ci // _MAX_CELLS, 0)
    slot = jnp.where(live, ci % _MAX_CELLS, 0)
    safe = jnp.where(live, ni, 0)
    d = all_cen[dev, slot] - jnp.asarray(geom.center)[safe]
    d2 = jnp.sum(d * d, axis=-1)
    ok = _compute_mac_ok(
        mac_type=_MAC,
        theta_sq=jnp.asarray(_THETA**2, d2.dtype),
        dist_sq=d2,
        extent_target=all_rad[dev, slot],
        extent_source=jnp.asarray(geom.radius)[safe],
        valid_pairs=live,
        different_nodes=jnp.ones_like(live),
    )
    n_live = int(jnp.sum(live))
    return n_live, int(jnp.sum(live & ~ok))


@pytest.fixture(scope="module")
def _walked():
    doms = _two_domains()
    cen0, rad0, act0, _ = _summary(*doms[0])
    cen1, rad1, act1, _ = _summary(*doms[1])
    all_cen = jnp.stack([cen0, cen1])
    all_rad = jnp.stack([rad0, rad1])
    all_act = jnp.stack([act0, act1])

    topo, geom = doms[0]
    total = int(topo.parent.shape[0])
    n_int = int(topo.left_child.shape[0])
    fill = jnp.full((total - n_int,), -1, topo.parent.dtype)
    ex = export_walk(
        jnp.concatenate([topo.left_child, fill]),
        jnp.concatenate([topo.right_child, fill]),
        jnp.asarray(geom.center),
        jnp.asarray(geom.radius),
        jnp.argmin(jnp.asarray(topo.parent)).astype(topo.parent.dtype),
        all_cen,
        all_rad,
        all_act,
        _THETA,
        jnp.asarray(0),
        max_pair_queue=1 << 16,
        far_cap=1 << 16,
        near_cap=1 << 16,
        mac_type=_MAC,
    )
    return ex, all_cen, all_rad, geom


def test_the_control_rejects_the_near_list(_walked):
    """The recomputation must DISCRIMINATE, or the far-list result means nothing.

    Near pairs bottomed out precisely because they fail the MAC, so most of them
    must be rejected here. If this passed trivially the instrument would be broken.
    """
    ex, all_cen, all_rad, geom = _walked
    n_live, n_fail = _mac_fail_fraction(
        ex.near_cell, ex.near_node, ex.near_count, all_cen, all_rad, geom
    )
    assert n_live > 0, "no near pairs, so the control proves nothing"
    assert n_fail > 0.5 * n_live, (
        f"the recomputation accepted {n_live - n_fail}/{n_live} NEAR pairs; it is "
        "not discriminating, so any verdict on the far list is meaningless"
    )


def test_every_exported_far_pair_satisfies_the_mac(_walked):
    ex, all_cen, all_rad, geom = _walked
    n_live, n_fail = _mac_fail_fraction(
        ex.far_cell, ex.far_node, ex.far_count, all_cen, all_rad, geom
    )
    assert n_live > 0, "no far pairs were exported, so this proves nothing"
    assert n_fail == 0, (
        f"{n_fail} of {n_live} exported FAR pairs fail the MAC by the sender's own "
        "geometry, so the receiver is right to reject them and the export list is "
        "the defect"
    )
