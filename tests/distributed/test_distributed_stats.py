"""Distributed clustering statistics (C1): soft pair counts and kNN in a periodic box.

What is asserted, each against brute force on the same points:

* the soft pair counts on 1, 2, 4 and 8 devices equal the O(N^2) minimum-image
  sum to round-off, and equal each other, for uniform points with many hugging
  the faces and corners, for clustered points whose clusters wrap across faces,
  with weights, and with linear bins;
* a far-field (geometric) acceptance is used and stays within its monopole error;
* the kNN distances (data as queries, self excluded, and random queries) equal
  brute force exactly, with and without the adaptive first pass, and the CDF
  reduction equals the brute-force CDF;
* every capacity's overflow flag fires when that capacity is too small, and only
  then.

Runs on however many devices exist; the multi-device cases skip below their count.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_distributed_stats.py -q --no-cov
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.binning import soft_bin_weights
from yggdrax.distributed import AXIS_NAME, device_count, make_mesh
from yggdrax.distributed.stats import (
    DistributedStatsConfig,
    PairAcceptance,
    distributed_knn,
    distributed_pair_counts,
    geometric_acceptance,
    pair_counts_shard,
    soft_window,
    soft_window_cutoffs,
)

# Only the meshes that exist are parametrized: CI's distributed tier fails a shard
# on any device-related skip, and it runs 4 devices.
_NDEVS = [d for d in (1, 2, 4, 8) if d <= device_count()]
_L = 1.0
_EDGES = np.geomspace(0.01, 0.06, 7)
_SHARP = 60.0


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


def _min_image_r(a, b, L=_L):
    d = a[:, None, :] - b[None, :, :]
    d -= L * np.round(d / L)
    return np.sqrt(np.sum(d * d, axis=-1))


def _brute_counts(pts, edges, sharp, lo, hi, *, weights=None, log=True, L=_L):
    n = len(pts)
    w = np.ones(n) if weights is None else np.asarray(weights)
    r = _min_image_r(pts, pts, L)
    iu = np.triu_indices(n, 1)
    r, ww = r[iu], (w[:, None] * w[None, :])[iu]
    keep = (r >= lo) & (r < hi)
    win = np.asarray(soft_window(jnp.asarray(r[keep]), jnp.asarray(edges), sharp, log))
    return np.sum(ww[keep][:, None] * win, axis=0)


def _brute_knn(data, queries, k, r_max, *, self_exclude, L=_L):
    r = _min_image_r(queries, data, L)
    if self_exclude:
        np.fill_diagonal(r, np.inf)
    r[r > r_max] = np.inf
    return np.sort(r, axis=1)[:, :k]


def _cfg(n, ndev, **kw):
    # a few hundred particles per device: the halo is several times the shard
    kw.setdefault("halo_fraction", 8.0)
    return DistributedStatsConfig.for_problem(max(1, n // ndev), **kw)


# ---------------------------------------------------------------------------
# soft pair counts
# ---------------------------------------------------------------------------


def test_soft_window_is_the_corrfunc_window():
    r = jnp.geomspace(0.005, 0.2, 50)
    np.testing.assert_allclose(
        soft_window(r, jnp.asarray(_EDGES), _SHARP, True),
        soft_bin_weights(r, jnp.asarray(_EDGES), _SHARP, log=True),
        rtol=1e-14,
        atol=1e-15,
    )
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, 1e-8)
    out = soft_window(
        jnp.asarray([lo * 0.999, hi * 1.001]), jnp.asarray(_EDGES), _SHARP
    )
    assert float(jnp.max(out)) <= 1e-8


_PTS = {
    "faces_corners": lambda: _faces_and_corners(1600, seed=0),
    "clustered": lambda: _clustered(1600, seed=1),
}
_REF: dict = {}


def _reference(name):
    if name not in _REF:
        pts = _PTS[name]()
        lo, hi = soft_window_cutoffs(_EDGES, _SHARP, 1e-8)
        _REF[name] = (pts, _brute_counts(pts, _EDGES, _SHARP, lo, hi))
    return _REF[name]


_ONE_DEVICE: dict = {}


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_pair_counts_match_brute_force_on_every_mesh(name, ndev):
    mesh = _mesh(ndev)
    pts, ref = _reference(name)
    res = distributed_pair_counts(
        pts,
        _EDGES,
        box_size=_L,
        config=_cfg(len(pts), ndev),
        sharpness=_SHARP,
        mesh=mesh,
    )
    assert not bool(res.overflow), {k: bool(v) for k, v in res.flags.items()}
    counts = np.asarray(res.counts)
    np.testing.assert_allclose(counts, ref, rtol=1e-10)
    # every particle is partitioned exactly once
    assert int(np.sum(np.asarray(res.stats["particles"]))) == len(pts)
    if ndev > 1:
        assert int(np.min(np.asarray(res.stats["halo"]))) > 0
    if ndev == 1:
        _ONE_DEVICE[name] = counts
    elif name in _ONE_DEVICE:
        np.testing.assert_allclose(counts, _ONE_DEVICE[name], rtol=1e-12)


@pytest.mark.parametrize("ndev", [2, 4])
def test_weighted_linear_bins(ndev):
    mesh = _mesh(ndev)
    rng = np.random.default_rng(5)
    pts = _clustered(1200, seed=5)
    w = rng.uniform(0.2, 2.0, len(pts))
    edges = np.linspace(0.005, 0.05, 6)
    sharp = 400.0  # linear windows: sharpness in 1/length
    lo, hi = soft_window_cutoffs(edges, sharp, 1e-8, log=False)
    ref = _brute_counts(pts, edges, sharp, lo, hi, weights=w, log=False)
    res = distributed_pair_counts(
        pts,
        edges,
        box_size=_L,
        config=_cfg(len(pts), ndev),
        weights=w,
        sharpness=sharp,
        log=False,
        mesh=mesh,
    )
    assert not bool(res.overflow)
    np.testing.assert_allclose(np.asarray(res.counts), ref, rtol=1e-10)


@pytest.mark.parametrize("ndev", [1, 4])
def test_geometric_acceptance_uses_far_pairs_within_monopole_error(ndev):
    mesh = _mesh(ndev)
    pts, ref = _reference("clustered")
    res = distributed_pair_counts(
        pts,
        _EDGES,
        box_size=_L,
        config=_cfg(len(pts), ndev),
        sharpness=_SHARP,
        acceptance=geometric_acceptance(0.3),
        mesh=mesh,
    )
    assert not bool(res.overflow)
    assert int(np.sum(np.asarray(res.stats["far_pairs"]))) > 0
    np.testing.assert_allclose(np.asarray(res.counts), ref, rtol=5e-2)


def _never_discard(data, a, b, dist_sq, ra, rb):
    del data, a, b, ra, rb
    return jnp.zeros(dist_sq.shape, bool)


def test_a_pluggable_rule_is_called_and_a_neutral_one_changes_nothing():
    mesh = _mesh(min(2, device_count()))
    pts, ref = _reference("faces_corners")
    res = distributed_pair_counts(
        pts,
        _EDGES,
        box_size=_L,
        config=_cfg(len(pts), mesh.size),
        sharpness=_SHARP,
        acceptance=PairAcceptance(discard=_never_discard),
        mesh=mesh,
    )
    np.testing.assert_allclose(np.asarray(res.counts), ref, rtol=1e-10)


def test_non_periodic_with_bounds():
    mesh = _mesh(min(4, device_count()))
    rng = np.random.default_rng(9)
    pts = rng.uniform(0.0, 1.0, (1000, 3))
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, 1e-8)
    n = len(pts)
    r = np.sqrt(np.sum((pts[:, None] - pts[None]) ** 2, -1))[np.triu_indices(n, 1)]
    keep = (r >= lo) & (r < hi)
    ref = np.asarray(soft_window(jnp.asarray(r[keep]), jnp.asarray(_EDGES), _SHARP))
    res = distributed_pair_counts(
        pts,
        _EDGES,
        box_size=None,
        bounds=(np.zeros(3), np.full(3, 1.0 + 1e-6)),
        config=_cfg(n, mesh.size),
        sharpness=_SHARP,
        mesh=mesh,
    )
    assert not bool(res.overflow)
    np.testing.assert_allclose(np.asarray(res.counts), ref.sum(0), rtol=1e-10)


def test_the_shard_body_runs_in_a_callers_own_shard_map():
    """Pre-sharded input (what a large run loads per host) through the public body."""
    from jax import jit, shard_map
    from jax.sharding import PartitionSpec as P

    ndev = min(4, device_count())
    mesh = _mesh(ndev)
    pts, ref = _reference("clustered")
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, 1e-8)
    per = -(-len(pts) // ndev)
    pos = np.zeros((ndev * per, 3))
    pos[: len(pts)] = pts
    counts = np.array([min(per, len(pts) - d * per) for d in range(ndev)])
    cfg = _cfg(len(pts), ndev)

    def body(x, c):
        res = pair_counts_shard(
            x,
            jnp.ones(x.shape[0]),
            c[0],
            jnp.asarray(_EDGES),
            ndev=ndev,
            config=cfg,
            box_size=_L,
            r_lo=lo,
            r_hi=hi,
            sharpness=_SHARP,
        )
        return res.counts, res.overflow

    counts_out, overflow = jit(
        shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
            out_specs=(P(), P()),
            check_vma=False,
        )
    )(jnp.asarray(pos), jnp.asarray(counts))
    assert not bool(overflow)
    np.testing.assert_allclose(np.asarray(counts_out), ref, rtol=1e-10)


def test_reach_beyond_half_box_is_refused():
    with pytest.raises(ValueError, match="min\\(box_size\\)/2"):
        distributed_pair_counts(
            np.random.default_rng(0).uniform(0, 1, (64, 3)),
            np.geomspace(0.1, 0.6, 4),
            box_size=1.0,
            config=DistributedStatsConfig.for_problem(64),
            mesh=make_mesh(1),
        )


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------

_K = 6
_RMAX = 0.12
_KNN_PTS = {
    "faces_corners": lambda: _faces_and_corners(1500, seed=3),
    "clustered": lambda: _clustered(1500, seed=4),
}
_KNN_REF: dict = {}


def _knn_reference(name):
    if name not in _KNN_REF:
        pts = _KNN_PTS[name]()
        _KNN_REF[name] = (pts, _brute_knn(pts, pts, _K, _RMAX, self_exclude=True))
    return _KNN_REF[name]


@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("name", sorted(_KNN_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_knn_matches_brute_force_on_every_mesh(name, ndev, adaptive):
    mesh = _mesh(ndev)
    pts, ref = _knn_reference(name)
    res = distributed_knn(
        pts,
        k=_K,
        r_max=_RMAX,
        box_size=_L,
        config=_cfg(len(pts), ndev),
        adaptive_reach=adaptive,
        mesh=mesh,
    )
    assert not bool(res.overflow), {k: bool(v) for k, v in res.flags.items()}
    d = np.asarray(res.distances)
    assert np.array_equal(np.isinf(d), np.isinf(ref))
    np.testing.assert_allclose(d[np.isfinite(d)], ref[np.isfinite(ref)], rtol=1e-12)


def test_adaptive_first_pass_thins_the_halo():
    mesh = _mesh(min(4, device_count()))
    if mesh.size < 2:
        pytest.skip("needs a halo, i.e. >= 2 devices")
    pts, _ = _knn_reference("clustered")
    halos = {}
    for adaptive in (True, False):
        res = distributed_knn(
            pts,
            k=_K,
            r_max=_RMAX,
            box_size=_L,
            config=_cfg(len(pts), mesh.size),
            adaptive_reach=adaptive,
            mesh=mesh,
        )
        halos[adaptive] = int(np.sum(np.asarray(res.stats["halo"])))
    assert halos[True] < halos[False]


@pytest.mark.parametrize("ndev", [1, 4])
def test_random_queries_and_cdf(ndev):
    mesh = _mesh(ndev)
    pts, _ = _knn_reference("clustered")
    rng = np.random.default_rng(11)
    queries = _faces_and_corners(700, seed=12)
    ref = _brute_knn(pts, queries, _K, _RMAX, self_exclude=False)
    grid = np.geomspace(0.005, 0.1, 9)
    res = distributed_knn(
        pts,
        k=_K,
        r_max=_RMAX,
        box_size=_L,
        config=_cfg(len(pts) + len(queries), ndev),
        queries=queries,
        r_grid=grid,
        mesh=mesh,
    )
    del rng
    assert not bool(res.overflow)
    d = np.asarray(res.distances)
    assert d.shape == (len(queries), _K)
    assert np.array_equal(np.isinf(d), np.isinf(ref))
    np.testing.assert_allclose(d[np.isfinite(d)], ref[np.isfinite(ref)], rtol=1e-12)
    cdf_ref = np.mean(ref.T[:, :, None] <= grid[None, None, :], axis=1)
    np.testing.assert_allclose(np.asarray(res.cdf), cdf_ref, atol=1e-12)

    soft = distributed_knn(
        pts,
        k=_K,
        r_max=_RMAX,
        box_size=_L,
        config=_cfg(len(pts) + len(queries), ndev),
        queries=queries,
        r_grid=grid,
        sharpness=20.0,
        mesh=mesh,
    )
    safe = np.where(np.isfinite(ref), ref, 2 * grid.max() + 1)
    step = 1 / (
        1 + np.exp(-20.0 * (np.log(grid)[None, None] - np.log(safe.T)[..., None]))
    )
    np.testing.assert_allclose(np.asarray(soft.cdf), step.mean(1), rtol=1e-10)


def test_knn_reports_inf_beyond_r_max():
    mesh = _mesh(min(2, device_count()))
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, (300, 3))  # sparse: r_8 ~ 0.13 > r_max
    ref = _brute_knn(pts, pts, 8, 0.08, self_exclude=True)
    res = distributed_knn(
        pts, k=8, r_max=0.08, box_size=1.0, config=_cfg(300, mesh.size), mesh=mesh
    )
    d = np.asarray(res.distances)
    assert np.isinf(d).any() and np.isfinite(d).any()
    assert np.array_equal(np.isinf(d), np.isinf(ref))
    np.testing.assert_allclose(d[np.isfinite(d)], ref[np.isfinite(ref)], rtol=1e-12)


# ---------------------------------------------------------------------------
# overflow flags
# ---------------------------------------------------------------------------

_SMALL = [
    ("partition_overflow", dict(capacity=64)),
    ("halo_overflow", dict(halo_capacity=32)),
    ("send_overflow", dict(send_capacity=16)),
    ("export_overflow", dict(export_near_cap=8)),
    ("summary_overflow", dict(summary_capacity=4)),
    ("walk_overflow", dict(near_cap=16)),
    ("leaf_overflow", dict(halo_leaf_capacity=8)),
]


@pytest.mark.parametrize("flag,small", _SMALL, ids=[s[0] for s in _SMALL])
def test_each_overflow_flag_fires(flag, small):
    mesh = _mesh(min(2, device_count()))
    if mesh.size < 2:
        pytest.skip("the halo flags need >= 2 devices")
    pts, _ = _reference("faces_corners")
    base = _cfg(len(pts), mesh.size)
    fields = {**base.__dict__, **small}
    if flag == "partition_overflow":
        # the derived per-tree capacities follow ``capacity`` down, so pin them
        fields.update(leaf_capacity=base.leaf_capacity)
    cfg = DistributedStatsConfig(**fields)
    res = distributed_pair_counts(
        pts, _EDGES, box_size=_L, config=cfg, sharpness=_SHARP, mesh=mesh
    )
    fired = {k for k, v in res.flags.items() if bool(v)}
    assert flag in fired, fired
    assert bool(res.overflow)
    # a generous config raises nothing
    ok = distributed_pair_counts(
        pts, _EDGES, box_size=_L, config=base, sharpness=_SHARP, mesh=mesh
    )
    assert not bool(ok.overflow)


def test_knn_overflow_flag_fires():
    mesh = _mesh(min(2, device_count()))
    if mesh.size < 2:
        pytest.skip("the halo flag needs >= 2 devices")
    pts, _ = _knn_reference("faces_corners")
    cfg = DistributedStatsConfig(
        **{**_cfg(len(pts), mesh.size).__dict__, "halo_capacity": 16}
    )
    res = distributed_knn(pts, k=_K, r_max=_RMAX, box_size=_L, config=cfg, mesh=mesh)
    assert bool(res.flags["halo_overflow"]) and bool(res.overflow)
