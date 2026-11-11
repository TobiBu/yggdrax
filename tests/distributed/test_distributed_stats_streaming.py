"""The streamed walk of the distributed statistics (``walk_chunks``).

With ``walk_chunks = K > 1`` the walk over ``[local ; halo]`` stops at a
frontier (``dual_tree_walk_mutual(stop_wavefront=)``), the frontier is dealt out
to ``K`` seeded sub-walks, and each sub-walk's leaf pairs are evaluated before
the next one runs, so the walk's memory is ``1 / K`` of the single walk's. The
leaf pairs are exactly the single walk's. What is asserted, on 1, 2, 4 and 8
devices where they exist:

* ξ counts equal the single walk (``walk_chunks = 1``) to 1e-12 and O(N^2) brute
  force to 1e-10, culled (``sub_block``) and dense, for periodic points hugging
  faces and corners and for clustered points wrapping across faces; the near
  pairs summed over the sub-walks equal the single walk's;
* gradients with respect to positions and weights equal the single walk's to
  1e-12 and brute force to 1e-10, with coincident particles too (the ``r = 0``
  guard);
* with far pairs (``geometric_acceptance``) the streamed and single walks give
  the same counts and the same number of far pairs;
* kNN distances and CDFs are identical, with and without the adaptive reach;
* ``for_box`` streams large walks, its per-walk capacities hold, and a short
  per-walk capacity is flagged.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_distributed_stats_streaming.py -q -o addopts=""
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.stats import (
    DistributedStatsConfig,
    distributed_knn,
    distributed_pair_counts,
    geometric_acceptance,
    soft_window,
    soft_window_cutoffs,
)

_NDEVS = [d for d in (1, 2, 4, 8) if d <= device_count()]
_MULTI = max(_NDEVS)
_L = 1.0
_EDGES = np.geomspace(0.01, 0.06, 7)
_SHARP = 60.0
_EPS = 1e-8
_TOL = 1e-10
# (walk_chunks, walk_frontier): small frontiers, so that these small problems
# (200 points per device on 8 devices) stream at all; 16 sub-walks of one to
# four seeds include some with none
_STREAMS = [(4, 4), (16, 16)]


def _mesh(ndev):
    if ndev > device_count():
        pytest.skip(f"needs {ndev} devices, have {device_count()}")
    return make_mesh(ndev)


def _faces_and_corners(n, seed=0, L=_L):
    """Uniform points, ~20 % within 1e-3 L of a face and ~5 % of a corner."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, L, (n, 3))
    m = n // 5
    axis = rng.integers(0, 3, m)
    side = rng.integers(0, 2, m)
    offs = rng.uniform(0.0, 1e-3 * L, m)
    pts[np.arange(m), axis] = np.where(side == 1, L - offs, offs)
    c = n // 20
    corners = rng.integers(0, 2, (c, 3)) * L + rng.uniform(-1e-3, 1e-3, (c, 3)) * L
    pts[m : m + c] = np.mod(corners, L)
    return pts


def _clustered(n, seed=1, L=_L):
    """Gaussian clusters, one centred on the corner, wrapped into the box."""
    rng = np.random.default_rng(seed)
    centres = np.concatenate(
        [np.zeros((1, 3)), np.array([[0.5, 0.02, 0.98]]), rng.uniform(0, L, (4, 3))]
    )
    which = rng.integers(0, len(centres), n)
    pts = centres[which] + rng.normal(0.0, 0.03 * L, (n, 3))
    bg = n // 5
    pts[:bg] = rng.uniform(0, L, (bg, 3))
    return np.mod(pts, L)


_PTS = {"faces_corners": _faces_and_corners, "clustered": _clustered}


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


def _cfg(n, ndev, leaf, sub, chunks=1, frontier=None, **kw):
    base = DistributedStatsConfig.for_problem(
        max(1, n // ndev), halo_fraction=8.0, leaf_size=leaf
    )
    fields = {
        "sub_block": sub,
        "sub_pair_cap": None,
        "walk_chunks": chunks,
        "walk_frontier": frontier,
        **kw,
    }
    return dataclasses.replace(base, **fields).resolved()


def _xi_brute(p, w):
    """``sum_{i<j} w_i w_j W_b(r_ij)`` over the cut, minimum image, in jax."""
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, _EPS)
    n = p.shape[0]
    d = p[:, None, :] - p[None, :, :]
    d = d - _L * jnp.round(d / _L)
    r2 = jnp.sum(d * d, axis=-1)
    iu = jnp.triu(jnp.ones((n, n), bool), k=1)
    pos = iu & (r2 > 0)
    r = jnp.where(pos, jnp.sqrt(jnp.where(pos, r2, 1.0)), jnp.where(iu, 0.0, 1.0))
    inside = iu & (r >= lo) & (r < hi)
    win = soft_window(r, jnp.asarray(_EDGES), _SHARP)
    ww = jnp.where(inside, w[:, None] * w[None, :], 0.0)
    return jnp.einsum("ij,ijb->b", ww, win)


def _xi(p, w, ndev, leaf, sub, chunks=1, frontier=None, **kw):
    acceptance = kw.pop("acceptance", None)
    return distributed_pair_counts(
        p,
        _EDGES,
        box_size=_L,
        config=_cfg(p.shape[0], ndev, leaf, sub, chunks, frontier, **kw),
        weights=w,
        sharpness=_SHARP,
        eps=_EPS,
        acceptance=acceptance,
        mesh=make_mesh(ndev),
    )


_BIN_COEF = np.linspace(1.0, 2.0, len(_EDGES) - 1)


def _loss_of(counts):
    """A nonlinear loss so the cotangent differs per bin."""
    return jnp.sum(_BIN_COEF * jnp.log1p(counts))


def _flags(res):
    return {k: bool(v) for k, v in res.flags.items()}


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

_N_VAL = 1600
_VAL_REF: dict = {}


def _value_reference(name):
    if name not in _VAL_REF:
        pts = _PTS[name](_N_VAL)
        w = np.random.default_rng(5).uniform(0.5, 2.0, _N_VAL)
        brute = np.asarray(_xi_brute(jnp.asarray(pts), jnp.asarray(w)))
        _VAL_REF[name] = (pts, w, brute)
    return _VAL_REF[name]


@pytest.mark.parametrize("leaf,sub", [(32, 8), (16, 0)], ids=["L32B8", "L16dense"])
@pytest.mark.parametrize("chunks,frontier", _STREAMS, ids=["K4F4", "K16F16"])
@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_streamed_counts_equal_the_single_walk(ndev, name, chunks, frontier, leaf, sub):
    _mesh(ndev)
    pts, w, brute = _value_reference(name)
    one = _xi(pts, w, ndev, leaf, sub)
    res = _xi(pts, w, ndev, leaf, sub, chunks, frontier)
    assert not bool(one.overflow), _flags(one)
    assert not bool(res.overflow), _flags(res)
    counts = np.asarray(res.counts)
    np.testing.assert_allclose(counts, np.asarray(one.counts), rtol=1e-12)
    np.testing.assert_allclose(counts, brute, rtol=_TOL)
    # the same leaf pairs, in several walks: each one smaller than the whole
    total = np.asarray(res.stats["near_pairs_total"])
    np.testing.assert_array_equal(total, np.asarray(one.stats["near_pairs"]))
    np.testing.assert_array_equal(total, np.asarray(one.stats["near_pairs_total"]))
    assert np.all(np.asarray(res.stats["walk_frontier"]) > 0)
    assert np.all(np.asarray(one.stats["walk_frontier"]) == 0)
    assert np.all(np.asarray(res.stats["near_pairs"]) < total)
    if sub:
        assert np.all(
            np.asarray(res.stats["sub_pairs"]) < np.asarray(one.stats["sub_pairs"])
        )


@pytest.mark.parametrize("ndev", sorted({1, _MULTI}))
def test_streamed_far_pairs_equal_the_single_walk(ndev):
    """With far pairs accepted, the sub-walks accept exactly the single walk's."""
    _mesh(ndev)
    pts, w, _ = _value_reference("clustered")
    acc = geometric_acceptance(0.3)
    one = _xi(pts, w, ndev, 16, 0, acceptance=acc)
    res = _xi(pts, w, ndev, 16, 0, 8, 16, acceptance=acc)
    assert not bool(one.overflow) and not bool(res.overflow)
    assert int(np.sum(np.asarray(one.stats["far_pairs"]))) > 0
    np.testing.assert_array_equal(
        np.asarray(res.stats["far_pairs"]), np.asarray(one.stats["far_pairs"])
    )
    np.testing.assert_allclose(
        np.asarray(res.counts), np.asarray(one.counts), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

_N_GRAD = 640
_GRAD_REF: dict = {}


def _grad_reference(name):
    """Points, weights and the brute-force value and gradient."""
    if name not in _GRAD_REF:
        pts = jnp.asarray(_PTS[name](_N_GRAD))
        w = jnp.asarray(np.random.default_rng(7).uniform(0.5, 2.0, _N_GRAD))
        brute = jax.value_and_grad(
            lambda p, ww: _loss_of(_xi_brute(p, ww)), argnums=(0, 1)
        )(pts, w)
        _GRAD_REF[name] = (pts, w, brute)
    return _GRAD_REF[name]


def _value_and_grad(pts, w, ndev, leaf, sub, chunks=1, frontier=None):
    return jax.value_and_grad(
        lambda p, ww: _loss_of(_xi(p, ww, ndev, leaf, sub, chunks, frontier).counts),
        argnums=(0, 1),
    )(pts, w)


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_streamed_gradient_equals_the_single_walk(ndev, name):
    _mesh(ndev)
    leaf, sub, chunks, frontier = 32, 8, 8, 16
    pts, w, (v_b, (gp_b, gw_b)) = _grad_reference(name)
    assert not bool(_xi(pts, w, ndev, leaf, sub, chunks, frontier).overflow)
    v, (gp, gw) = _value_and_grad(pts, w, ndev, leaf, sub, chunks, frontier)
    assert np.all(np.isfinite(gp)) and np.all(np.isfinite(gw))
    assert float(jnp.max(jnp.abs(gp))) > 0
    v1, (gp1, gw1) = _value_and_grad(pts, w, ndev, leaf, sub)
    np.testing.assert_allclose(float(v), float(v1), rtol=1e-12)
    assert _rel(gp, gp1) < 1e-12 and _rel(gw, gw1) < 1e-12
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert _rel(gp, gp_b) < _TOL and _rel(gw, gw_b) < _TOL


@pytest.mark.parametrize("sub", [4, 0], ids=["B4", "dense"])
@pytest.mark.parametrize("ndev", sorted({1, _MULTI}))
def test_streamed_gradient_with_coincident_particles_is_finite(ndev, sub):
    """Exact duplicates: valid pairs at r = 0, inside a block or across two."""
    _mesh(ndev)
    pts = np.asarray(_faces_and_corners(_N_GRAD))
    pts[1:_N_GRAD:50] = pts[0:_N_GRAD:50][: len(pts[1:_N_GRAD:50])]
    p, w = jnp.asarray(pts), jnp.ones((_N_GRAD,))
    v_b, (g_b, _) = jax.value_and_grad(
        lambda x, ww: _loss_of(_xi_brute(x, ww)), argnums=(0, 1)
    )(p, w)
    v, (g, _) = _value_and_grad(p, w, ndev, 16, sub, 8, 16)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert _rel(g, g_b) < _TOL


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adaptive", [True, False], ids=["adaptive", "fixed"])
@pytest.mark.parametrize("ndev", _NDEVS)
def test_streamed_knn_equals_the_single_walk(ndev, adaptive):
    _mesh(ndev)
    n, k, r_max = 3000, 8, 0.06
    pts = _clustered(n, seed=3)
    r_grid = np.geomspace(0.005, 0.05, 9)
    out = {}
    for chunks, frontier in [(1, None), (8, 16)]:
        base = DistributedStatsConfig.for_problem(
            max(1, n // ndev), halo_fraction=8.0, leaf_size=16
        )
        cfg = dataclasses.replace(
            base, walk_chunks=chunks, walk_frontier=frontier
        ).resolved()
        res = distributed_knn(
            pts,
            k=k,
            r_max=r_max,
            box_size=_L,
            config=cfg,
            r_grid=r_grid,
            sharpness=20.0,
            adaptive_reach=adaptive,
            mesh=make_mesh(ndev),
        )
        assert not bool(res.overflow), _flags(res)
        out[chunks] = res
    one, res = out[1], out[8]
    np.testing.assert_array_equal(np.asarray(res.distances), np.asarray(one.distances))
    np.testing.assert_array_equal(np.asarray(res.gid), np.asarray(one.gid))
    np.testing.assert_allclose(np.asarray(res.cdf), np.asarray(one.cdf), rtol=1e-14)
    np.testing.assert_array_equal(
        np.asarray(res.stats["near_pairs_total"]),
        np.asarray(one.stats["near_pairs"]),
    )
    assert np.all(np.asarray(res.stats["walk_frontier"]) > 0)


# ---------------------------------------------------------------------------
# configuration, overflow and sizing
# ---------------------------------------------------------------------------


def test_walk_chunk_defaults_and_validation():
    cfg = DistributedStatsConfig.for_problem(1000, leaf_size=32)
    assert cfg.walk_chunks == 1 and cfg.walk_frontier == 0
    on = dataclasses.replace(cfg, walk_chunks=8).resolved()
    assert on.walk_frontier == 2048
    # resolving twice changes nothing
    assert on.resolved() == on
    with pytest.raises(ValueError, match="walk_chunks"):
        dataclasses.replace(cfg, walk_chunks=0).resolved()
    with pytest.raises(ValueError, match="walk_frontier"):
        dataclasses.replace(cfg, walk_chunks=8, walk_frontier=4).resolved()
    with pytest.raises(ValueError, match="max_pair_queue"):
        dataclasses.replace(
            cfg, walk_chunks=2, walk_frontier=1024, max_pair_queue=64
        ).resolved()


def test_a_short_sub_walk_is_flagged():
    ndev = min(2, device_count())
    _mesh(ndev)
    pts, w, _ = _value_reference("clustered")
    res = _xi(pts, w, ndev, 32, 8, 4, None, near_cap=64)
    assert bool(res.flags["walk_overflow"]) and bool(res.overflow)
    assert int(np.max(np.asarray(res.stats["near_pairs"]))) > 64
    res = _xi(pts, w, ndev, 32, 8, 4, None, sub_pair_cap=64)
    assert bool(res.flags["sub_overflow"]) and bool(res.overflow)


def _blobs(n, rng, L=_L):
    """60 % of the points in blobs of one per 512 points, 40 % uniform."""
    nc = max(4, n // 512)
    centres = rng.uniform(0.0, L, (nc, 3))
    width = 0.1 * (L**3 / nc) ** (1.0 / 3.0)
    pts = centres[rng.integers(0, nc, n)] + rng.normal(0.0, width, (n, 3))
    bg = int(0.4 * n)
    pts[:bg] = rng.uniform(0.0, L, (bg, 3))
    return np.mod(pts, L)


@pytest.mark.parametrize("name", ["uniform", "clustered"])
def test_for_box_streams_the_walk_and_its_capacities_hold(name):
    """``walk_pairs`` small enough to stream: per-walk lists fit, values unchanged.

    16384 per device at leaf 64 (the C3/C4 shape). The halo and export rows of
    the model are asserted only where they hold (see the sub-block tests).
    """
    ndev = _MULTI
    _mesh(ndev)
    n = 16384 * ndev
    rng = np.random.default_rng(11)
    if name == "uniform":
        pts, clustering = rng.uniform(0.0, _L, (n, 3)), 1.0
    else:
        pts, clustering = _blobs(n, rng), 4.0
    spacing = (_L**3 / n) ** (1.0 / 3.0)
    edges = np.geomspace(0.4, 2.5, 7) * spacing
    lo, hi = soft_window_cutoffs(edges, _SHARP, _EPS)
    kw = dict(ndev=ndev, box_size=_L, reach=hi, leaf_size=64, clustering=clustering)
    single = DistributedStatsConfig.for_box(n // ndev, **kw)
    streamed = DistributedStatsConfig.for_box(n // ndev, walk_pairs=1 << 12, **kw)
    assert single.walk_chunks == 1 and streamed.walk_chunks > 1
    # an aggressive walk_pairs is held back: the frontier stays below the peak
    capped = DistributedStatsConfig.for_box(n // ndev, walk_pairs=1 << 6, **kw)
    assert 4 * capped.walk_frontier <= single.near_cap / clustering / 16
    assert streamed.near_cap < single.near_cap
    assert streamed.sub_pair_cap < single.sub_pair_cap
    out = {}
    for cfg in (single, streamed):
        res = distributed_pair_counts(
            pts,
            edges,
            box_size=_L,
            config=cfg,
            sharpness=_SHARP,
            eps=_EPS,
            mesh=make_mesh(ndev),
        )
        flags = _flags(res)
        assert not flags["walk_overflow"] and not flags["sub_overflow"], flags
        if ndev != 2:
            assert not bool(res.overflow), flags
        assert int(np.max(np.asarray(res.stats["near_pairs"]))) <= cfg.near_cap
        assert int(np.max(np.asarray(res.stats["sub_pairs"]))) <= cfg.sub_pair_cap
        assert (
            int(np.max(np.asarray(res.stats["peak_wavefront"]))) <= cfg.max_pair_queue
        )
        out[cfg.walk_chunks] = res
    one, res = out[1], out[streamed.walk_chunks]
    np.testing.assert_allclose(
        np.asarray(res.counts), np.asarray(one.counts), rtol=1e-12
    )
