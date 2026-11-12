"""The culled near field of the distributed ξ (``sub_block``).

Each leaf is cut into runs of ``sub_block`` sorted slots; the walk's leaf pairs
(and every leaf with itself) become sub-block pairs, the pairs whose bounding
boxes are at least ``r_hi`` apart are dropped, and the rest are summed as dense
blocks. The cut is hard, so nothing inside it is lost. What is asserted, on 1, 2,
4 and 8 devices where they exist:

* counts equal O(N^2) brute force to 1e-10 and the dense path (``sub_block=0``)
  to 1e-12, for two leaf/sub-block shapes, periodic points hugging faces and
  corners, and clustered points whose clusters wrap across faces;
* gradients with respect to positions and weights equal brute-force autodiff
  and the dense path's to 1e-10, including coincident particles (the ``r = 0``
  guard);
* ``sub_overflow`` fires when ``sub_pair_cap`` is short and the ``for_box``
  sizing holds for uniform and clustered points.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_distributed_stats_subblock.py -q -o addopts=""
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
    distributed_pair_counts,
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
# (leaf_size, sub_block): 2 x 2 runs of 8, and 8 x 8 runs of 4
_SHAPES = [(16, 8), (32, 4)]


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


def _cfg(n, ndev, leaf, sub, **kw):
    base = DistributedStatsConfig.for_problem(
        max(1, n // ndev), halo_fraction=8.0, leaf_size=leaf
    )
    fields = {"sub_block": sub, "sub_pair_cap": None, **kw}
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


def _xi(p, w, ndev, leaf, sub, **kw):
    return distributed_pair_counts(
        p,
        _EDGES,
        box_size=_L,
        config=_cfg(p.shape[0], ndev, leaf, sub, **kw),
        weights=w,
        sharpness=_SHARP,
        eps=_EPS,
        mesh=make_mesh(ndev),
    )


_BIN_COEF = np.linspace(1.0, 2.0, len(_EDGES) - 1)


def _loss_of(counts):
    """A nonlinear loss so the cotangent differs per bin."""
    return jnp.sum(_BIN_COEF * jnp.log1p(counts))


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


@pytest.mark.parametrize("leaf,sub", _SHAPES, ids=[f"L{a}B{b}" for a, b in _SHAPES])
@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_culled_counts_match_brute_force_and_the_dense_path(ndev, name, leaf, sub):
    _mesh(ndev)
    pts, w, brute = _value_reference(name)
    res = _xi(pts, w, ndev, leaf, sub)
    assert not bool(res.overflow), {k: bool(v) for k, v in res.flags.items()}
    counts = np.asarray(res.counts)
    np.testing.assert_allclose(counts, brute, rtol=_TOL)
    dense = _xi(pts, w, ndev, leaf, 0)
    assert not bool(dense.overflow)
    np.testing.assert_allclose(counts, np.asarray(dense.counts), rtol=1e-12)
    # the culling did something: fewer B x B blocks than the dense leaf blocks
    k = leaf // sub
    sub_pairs = np.asarray(res.stats["sub_pairs"])
    leaves = np.asarray(res.stats["leaves"])
    near = np.asarray(dense.stats["near_pairs"])
    assert np.all(sub_pairs > 0)
    assert np.all(sub_pairs < (near + leaves) * k * k)
    assert np.all(np.asarray(dense.stats["sub_pairs"]) == 0)


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


def _value_and_grad(pts, w, ndev, leaf, sub):
    return jax.value_and_grad(
        lambda p, ww: _loss_of(_xi(p, ww, ndev, leaf, sub).counts), argnums=(0, 1)
    )(pts, w)


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_culled_gradient_matches_brute_force_and_the_dense_path(ndev, name):
    _mesh(ndev)
    leaf, sub = 32, 8
    pts, w, (v_b, (gp_b, gw_b)) = _grad_reference(name)
    assert not bool(_xi(pts, w, ndev, leaf, sub).overflow)
    v, (gp, gw) = _value_and_grad(pts, w, ndev, leaf, sub)
    assert np.all(np.isfinite(gp)) and np.all(np.isfinite(gw))
    assert float(jnp.max(jnp.abs(gp))) > 0
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert _rel(gp, gp_b) < _TOL and _rel(gw, gw_b) < _TOL
    v0, (gp0, gw0) = _value_and_grad(pts, w, ndev, leaf, 0)
    np.testing.assert_allclose(float(v), float(v0), rtol=1e-12)
    assert _rel(gp, gp0) < _TOL and _rel(gw, gw0) < _TOL


@pytest.mark.parametrize("ndev", sorted({1, _MULTI}))
def test_culled_gradient_with_coincident_particles_is_finite(ndev):
    """Exact duplicates: valid pairs at r = 0 inside one sub-block or across two."""
    _mesh(ndev)
    pts = np.asarray(_faces_and_corners(_N_GRAD))
    pts[1:_N_GRAD:50] = pts[0:_N_GRAD:50][: len(pts[1:_N_GRAD:50])]
    p, w = jnp.asarray(pts), jnp.ones((_N_GRAD,))
    v_b, (g_b, _) = jax.value_and_grad(
        lambda x, ww: _loss_of(_xi_brute(x, ww)), argnums=(0, 1)
    )(p, w)
    v, (g, _) = _value_and_grad(p, w, ndev, 16, 4)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert _rel(g, g_b) < _TOL


# ---------------------------------------------------------------------------
# configuration, overflow and sizing
# ---------------------------------------------------------------------------


def test_sub_block_defaults_and_validation():
    cfg = DistributedStatsConfig.for_problem(1000, leaf_size=32)
    assert cfg.sub_block == 8 and cfg.sub_pair_cap > 0
    assert DistributedStatsConfig.for_problem(1000, leaf_size=8).sub_block == 0
    off = DistributedStatsConfig.for_problem(1000, leaf_size=32, sub_block=0)
    assert off.sub_block == 0 and off.sub_pair_cap == 0
    # resolving twice changes nothing
    assert cfg.resolved() == cfg
    with pytest.raises(ValueError, match="sub_block"):
        DistributedStatsConfig.for_problem(1000, leaf_size=32, sub_block=12)


def test_sub_overflow_fires_and_is_reported():
    ndev = min(2, device_count())
    _mesh(ndev)
    pts, w, _ = _value_reference("clustered")
    res = _xi(pts, w, ndev, 32, 8, sub_pair_cap=64)
    assert bool(res.flags["sub_overflow"]) and bool(res.overflow)
    assert int(np.max(np.asarray(res.stats["sub_pairs"]))) > 64
    ok = _xi(pts, w, ndev, 32, 8)
    assert not bool(ok.flags["sub_overflow"]) and not bool(ok.overflow)


def _blobs(n, rng, L=_L):
    """60 % of the points in blobs of one per 512 points, 40 % uniform.

    Clustered at about the level ``clustering = 4`` is meant for (``_clustered``
    above is far denser: 80 % of the points in 6 blobs).
    """
    nc = max(4, n // 512)
    centres = rng.uniform(0.0, L, (nc, 3))
    width = 0.1 * (L**3 / nc) ** (1.0 / 3.0)
    pts = centres[rng.integers(0, nc, n)] + rng.normal(0.0, width, (n, 3))
    bg = int(0.4 * n)
    pts[:bg] = rng.uniform(0.0, L, (bg, 3))
    return np.mod(pts, L)


@pytest.mark.parametrize("name", ["uniform", "clustered"])
def test_for_box_sizes_the_culled_list(name):
    """The geometric sizing covers the walk and the culled list, at leaf 16 and 32.

    16384 per device: below a few thousand the halo model (calibrated at leaf 16
    and unchanged here) can overflow on small, surface-dominated domains, and
    on exactly 2 devices the export list can at leaf 32. Those two are
    asserted only where they hold.
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
    for leaf in (16, 32):
        cfg = DistributedStatsConfig.for_box(
            n // ndev,
            ndev=ndev,
            box_size=_L,
            reach=hi,
            leaf_size=leaf,
            clustering=clustering,
        )
        assert cfg.sub_block == 8
        res = distributed_pair_counts(
            pts,
            edges,
            box_size=_L,
            config=cfg,
            sharpness=_SHARP,
            eps=_EPS,
            mesh=make_mesh(ndev),
        )
        flags = {k: bool(v) for k, v in res.flags.items()}
        assert not flags["walk_overflow"] and not flags["sub_overflow"], flags
        if ndev != 2:
            assert not bool(res.overflow), flags
        assert int(np.max(np.asarray(res.stats["sub_pairs"]))) <= cfg.sub_pair_cap
        assert int(np.max(np.asarray(res.stats["near_pairs"]))) <= cfg.near_cap
