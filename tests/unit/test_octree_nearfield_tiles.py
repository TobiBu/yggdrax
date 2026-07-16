"""Compact near-field W-tiles (:mod:`yggdrax.octree_nearfield_tiles`).

The builder re-packs an adaptive octree's occupancy-bounded near CSR into full uniform
W-tiles + a densely-compacted per-tile source-tile list. The gates here certify the three
correctness contracts the compact near-field relies on:

* (COVERAGE / CELL-RESPECTING) every particle lands in exactly one tile, and every tile's
  particles all belong to a single leaf cell (tiles never span cells -> tile-pairs map 1:1
  onto cell-pairs, so the near/far split is preserved with no double-counting).
* (SOURCE COMPLETENESS) for every target tile the compacted source-tile set equals the
  reference expansion of the leaf-level near set (own cell's OTHER tiles + all tiles of the
  cell's U-neighbour leaves), self-tile excluded.
* (OVERFLOW) the eager-only capacity gate trips exactly when a cap is under-sized.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from yggdrax.octree_nearfield_tiles import build_octree_nearfield_tiles
from yggdrax.octree_uvwx import build_adaptive_octree_execution_view_device


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":  # concentrated Plummer-like cloud
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / (1.0 - 0.85 * r)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = r[:, None] * d
    else:
        pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    return pos.astype(np.float64)


def _build_view(pos, max_depth, leaf_size, u_cap=256):
    return build_adaptive_octree_execution_view_device(
        pos,
        max_depth,
        leaf_size,
        node_capacity=12000,
        leaf_capacity=10000,
        interactions=True,
        u_capacity=u_cap,
    )


def _probe_caps(view, n, w):
    """Eager reference: exact num_tiles, per-leaf tile counts, and max source tiles."""
    nr = np.asarray(view.node_ranges)
    leaf_idx = np.asarray(view.leaf_indices)
    u_nbr = np.asarray(view.u_neighbors)
    node_cap = int(view.parent.shape[0])
    leaf_cap = int(leaf_idx.shape[0])
    u_width = u_nbr.shape[0] // leaf_cap

    occ = np.zeros(leaf_cap, np.int64)
    for r in range(leaf_cap):
        ln = int(leaf_idx[r])
        if ln < node_cap:
            occ[r] = int(nr[ln, 1] - nr[ln, 0] + 1)
    tiles_per_leaf = (occ + w - 1) // w
    tile_base = np.concatenate([[0], np.cumsum(tiles_per_leaf)])[:-1]
    total_tiles = int(tiles_per_leaf.sum())

    # max source tiles over target tiles (self slot INCLUDED, matching the builder's counts)
    max_src = 0
    for r in range(leaf_cap):
        ln = int(leaf_idx[r])
        if ln >= node_cap or occ[r] <= 0:
            continue
        nbr_rows = [r]
        for e in u_nbr[r * u_width : (r + 1) * u_width].tolist():
            if e != node_cap and e < leaf_cap and e != r:
                nbr_rows.append(int(e))
        src_count = int(sum(int(tiles_per_leaf[x]) for x in nbr_rows))
        max_src = max(max_src, src_count)
    return dict(
        node_cap=node_cap,
        leaf_cap=leaf_cap,
        u_width=u_width,
        occ=occ,
        tiles_per_leaf=tiles_per_leaf,
        tile_base=tile_base,
        total_tiles=total_tiles,
        max_src=max_src,
        max_occ=int(occ.max()),
    )


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist", "w"),
    [
        (2000, 6, 64, "uniform", 32),
        (2000, 6, 64, "clustered", 32),
        (4000, 7, 96, "clustered", 64),
        (8000, 7, 128, "uniform", 64),
        (8000, 8, 128, "clustered", 64),
    ],
)
def test_tiles_coverage_and_cell_respecting(n, max_depth, leaf_size, dist, w):
    """Every particle in exactly one tile; every tile confined to one leaf cell."""
    pos = _points(n, 7, dist)
    view = _build_view(pos, max_depth, leaf_size)
    ref = _probe_caps(view, n, w)

    tiles = build_octree_nearfield_tiles(
        view,
        tile_width=w,
        num_tiles_cap=(n + w - 1) // w + ref["leaf_cap"],
        max_source_tiles=ref["max_src"] + 8,
        max_leaf_occupancy=ref["max_occ"],
    )
    assert not bool(np.asarray(tiles.overflow)), "unexpected overflow with ample caps"
    assert int(np.asarray(tiles.num_tiles_used)) == ref["total_tiles"]

    pidx = np.asarray(tiles.tile_particle_idx)
    tmask = np.asarray(tiles.tile_mask)
    tleaf = np.asarray(tiles.tile_leaf_row)
    nr = np.asarray(view.node_ranges)
    leaf_idx = np.asarray(view.leaf_indices)

    # (COVERAGE) union of all tiles' particles == [0, N), each exactly once
    cov = np.zeros(n, np.int64)
    n_tiles = int(np.asarray(tiles.num_tiles_used))
    for t in range(n_tiles):
        parts = pidx[t][tmask[t]]
        cov[parts] += 1
        # (CELL-RESPECTING) all this tile's particles lie in its owning leaf's Morton span
        r = int(tleaf[t])
        assert r < ref["leaf_cap"], f"tile {t} has sentinel leaf row"
        ln = int(leaf_idx[r])
        assert np.all(parts >= nr[ln, 0]) and np.all(parts <= nr[ln, 1])
    assert int(cov.min()) == 1 and int(cov.max()) == 1, (
        f"not an exact partition: missing={int((cov==0).sum())} "
        f"double={int((cov>1).sum())}"
    )


@pytest.mark.parametrize(
    ("n", "max_depth", "leaf_size", "dist", "w"),
    [
        (2000, 6, 64, "clustered", 32),
        (4000, 7, 96, "clustered", 64),
        (8000, 8, 128, "clustered", 64),
    ],
)
def test_tile_source_completeness(n, max_depth, leaf_size, dist, w):
    """Each target tile's compacted source set == reference leaf-near expansion, minus self."""
    pos = _points(n, 11, dist)
    view = _build_view(pos, max_depth, leaf_size)
    ref = _probe_caps(view, n, w)

    tiles = build_octree_nearfield_tiles(
        view,
        tile_width=w,
        num_tiles_cap=(n + w - 1) // w + ref["leaf_cap"],
        max_source_tiles=ref["max_src"] + 8,
        max_leaf_occupancy=ref["max_occ"],
    )
    assert not bool(np.asarray(tiles.overflow))

    tleaf = np.asarray(tiles.tile_leaf_row)
    sids = np.asarray(tiles.tile_source_ids)
    svalid = np.asarray(tiles.tile_source_valid)
    leaf_idx = np.asarray(view.leaf_indices)
    u_nbr = np.asarray(view.u_neighbors)
    node_cap = ref["node_cap"]
    leaf_cap = ref["leaf_cap"]
    u_width = ref["u_width"]
    tiles_per_leaf = ref["tiles_per_leaf"]
    tile_base = ref["tile_base"]
    n_tiles = int(np.asarray(tiles.num_tiles_used))

    def tiles_of_leaf(x):
        return set(range(int(tile_base[x]), int(tile_base[x]) + int(tiles_per_leaf[x])))

    n_bad = 0
    for t in range(n_tiles):
        r = int(tleaf[t])
        ln = int(leaf_idx[r])
        if ln >= node_cap:
            continue
        expected = set(tiles_of_leaf(r))  # own cell's tiles
        for e in u_nbr[r * u_width : (r + 1) * u_width].tolist():
            if e != node_cap and e < leaf_cap and e != r:
                expected |= tiles_of_leaf(int(e))
        expected.discard(t)  # self tile handled by the self-block path
        got = set(int(x) for x in sids[t][svalid[t]].tolist())
        assert len(got) == int(svalid[t].sum()), f"tile {t}: duplicate source ids"
        if got != expected:
            n_bad += 1
            if n_bad <= 3:
                print(
                    f"tile {t} (leaf {r}): missing={sorted(expected-got)[:5]} "
                    f"extra={sorted(got-expected)[:5]}"
                )
    assert n_bad == 0, f"{n_bad} tiles have an incorrect source set"


def test_overflow_flag_trips_on_undersized_caps():
    """The eager capacity gate trips on each under-sized cap and clears with ample caps."""
    pos = _points(4000, 3, "clustered")
    view = _build_view(pos, 7, 96)
    w = 64
    ref = _probe_caps(view, 4000, w)
    ample = dict(
        tile_width=w,
        num_tiles_cap=(4000 + w - 1) // w + ref["leaf_cap"],
        max_source_tiles=ref["max_src"] + 8,
        max_leaf_occupancy=ref["max_occ"],
    )
    assert not bool(np.asarray(build_octree_nearfield_tiles(view, **ample).overflow))

    # under-size the source-tile capacity
    bad_src = dict(ample)
    bad_src["max_source_tiles"] = max(1, ref["max_src"] // 2)
    assert bool(np.asarray(build_octree_nearfield_tiles(view, **bad_src).overflow))

    # under-size the per-leaf occupancy bound (fewer tiles per leaf than needed)
    bad_occ = dict(ample)
    bad_occ["max_leaf_occupancy"] = w  # forces mbpl=1 while dense leaves need more
    assert bool(np.asarray(build_octree_nearfield_tiles(view, **bad_occ).overflow))

    # under-size the global tile cap
    bad_tiles = dict(ample)
    bad_tiles["num_tiles_cap"] = max(1, ref["total_tiles"] // 2)
    assert bool(np.asarray(build_octree_nearfield_tiles(view, **bad_tiles).overflow))
