"""Gradients through the distributed clustering statistics (C2).

``jax.grad`` with respect to particle positions (and weights) runs through
:func:`distributed_pair_counts` and :func:`distributed_knn` at fixed topology. What
is asserted, on 1, 2, 4 and 8 devices where they exist:

* the gradient equals O(N^2) brute-force autodiff of the same statistic, and the
  1-device gradient, to 1e-10 (measured ~1e-15): soft ξ counts (positions and
  weights), kNN distances, and the soft kNN-CDF of random queries (data and
  query positions); periodic points hugging faces and corners and clustered
  points whose clusters wrap across faces;
* central finite differences agree, at a step where no pair or neighbour set
  changes;
* ``jit(value_and_grad)`` equals the eager gradient, and a ``lax.scan`` of
  gradient steps equals the same steps taken one by one;
* jaccpot's audit (``docs/differentiable_fmm_distributed_audit.md`` there):
  - the TRACE PROBE: ``make_jaxpr(grad(loss))`` traces, and a liveness sweep of
    the forward jaxpr finds no ``while``/``sort``/``top_k`` reached by a live
    float (every topology loop sits behind ``stop_gradient``), while the control
    sweep without the seam finds the walks;
  - the DEPENDENCY SWEEP: perturbing a particle near a face or corner changes
    exactly the gradients brute force says it should, on other devices too;
  - the RAGGED SIDE-EFFECT CHECK: a forward evaluated after a gradient is
    bit-identical to one before it. On GPU, ``YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1``
    repeats it through the native ragged all-to-all.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_distributed_stats_grad.py -q --no-cov
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed import AXIS_NAME, device_count, make_mesh
from yggdrax.distributed.stats import (
    DistributedStatsConfig,
    distributed_knn,
    distributed_pair_counts,
    pair_counts_shard,
    soft_window,
    soft_window_cutoffs,
)

from ._grad_audit import live_while_loops

_NDEVS = [d for d in (1, 2, 4, 8) if d <= device_count()]
_MULTI = max(_NDEVS)  # the largest mesh available (4 in CI's tier)
_L = 1.0
_EDGES = np.geomspace(0.01, 0.06, 7)
_SHARP = 60.0
_EPS = 1e-8
_N = 640
_K = 4
_RMAX = 0.15
_R_GRID = np.geomspace(0.02, 0.12, 8)
_CDF_SHARP = 20.0
_TOL = 1e-10


def _mesh(ndev):
    if ndev > device_count():
        pytest.skip(f"needs {ndev} devices, have {device_count()}")
    return make_mesh(ndev)


def _cfg(n, ndev, **kw):
    kw.setdefault("halo_fraction", 8.0)
    return DistributedStatsConfig.for_problem(max(1, n // ndev), **kw)


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


# ---------------------------------------------------------------------------
# losses: the distributed statistic and its O(N^2) brute-force twin
# ---------------------------------------------------------------------------

_BIN_COEF = np.linspace(1.0, 2.0, len(_EDGES) - 1)


def _xi_loss_of(counts):
    """A nonlinear loss so the cotangent differs per bin."""
    return jnp.sum(_BIN_COEF * jnp.log1p(counts))


def _xi_brute(p, w, eps=_EPS):
    """``sum_{i<j} w_i w_j W_b(r_ij)`` over the cut, minimum image, in jax."""
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, eps)
    n = p.shape[0]
    d = p[:, None, :] - p[None, :, :]
    d = d - _L * jnp.round(d / _L)
    r2 = jnp.sum(d * d, axis=-1)
    iu = jnp.triu(jnp.ones((n, n), bool), k=1)
    r = jnp.sqrt(jnp.where(iu, r2, 1.0))
    inside = iu & (r >= lo) & (r < hi)
    win = soft_window(r, jnp.asarray(_EDGES), _SHARP)
    ww = jnp.where(inside, w[:, None] * w[None, :], 0.0)
    return jnp.einsum("ij,ijb->b", ww, win)


def _xi_dist(p, w, ndev, eps=_EPS, exchange="auto"):
    cfg = dataclasses.replace(_cfg(p.shape[0], ndev), exchange=exchange)
    return distributed_pair_counts(
        p,
        _EDGES,
        box_size=_L,
        config=cfg,
        weights=w,
        sharpness=_SHARP,
        eps=eps,
        mesh=make_mesh(ndev),
    )


def _brute_nbrs(data, queries, *, self_exclude):
    """Brute-force kNN indices (topology) and their within-r_max mask, in numpy."""
    data, queries = np.asarray(data), np.asarray(queries)
    d = queries[:, None, :] - data[None, :, :]
    d -= _L * np.round(d / _L)
    r = np.sqrt(np.sum(d * d, axis=-1))
    if self_exclude:
        np.fill_diagonal(r, np.inf)
    r[r > _RMAX] = np.inf
    idx = np.argsort(r, axis=1, kind="stable")[:, :_K]
    return idx, np.isfinite(np.take_along_axis(r, idx, axis=1))


def _knn_dist_brute(data, queries, idx, ok):
    """Distances at fixed neighbour indices, differentiable in both point sets."""
    d = data[idx] - queries[:, None, :]
    d = d - _L * jnp.round(d / _L)
    return jnp.sqrt(jnp.where(ok, jnp.sum(d * d, axis=-1), 1.0))


def _dist_loss(r, coef):
    return jnp.sum(jnp.where(jnp.isfinite(r), r * coef, 0.0))


def _soft_cdf(r, ok):
    """The driver's soft CDF on ``_R_GRID`` from ``(Q, k)`` distances."""
    rg = jnp.asarray(_R_GRID)
    safe = jnp.where(ok, r, 2.0 * float(np.max(_R_GRID)) + 1.0)
    step = jax.nn.sigmoid(
        _CDF_SHARP * (jnp.log(rg)[None, None, :] - jnp.log(safe)[:, :, None])
    )
    return jnp.sum(step, axis=0) / r.shape[0]


def _cdf_loss_of(cdf):
    return jnp.sum(jnp.linspace(1.0, 2.0, _K)[:, None] * cdf**2)


def _knn_dist(p, ndev, *, queries=None, cdf=False, exchange="auto"):
    n = p.shape[0] + (0 if queries is None else queries.shape[0])
    cfg = dataclasses.replace(_cfg(n, ndev), exchange=exchange)
    return distributed_knn(
        p,
        k=_K,
        r_max=_RMAX,
        box_size=_L,
        config=cfg,
        queries=queries,
        r_grid=_R_GRID if cdf else None,
        sharpness=_CDF_SHARP if cdf else None,
        mesh=make_mesh(ndev),
    )


# ---------------------------------------------------------------------------
# soft pair counts
# ---------------------------------------------------------------------------

_XI_REF: dict = {}


def _xi_reference(name):
    """Points, weights, brute-force gradient and the 1-device gradient."""
    if name not in _XI_REF:
        pts = jnp.asarray(_PTS[name](_N))
        w = jnp.asarray(np.random.default_rng(7).uniform(0.5, 2.0, _N))
        vg = jax.value_and_grad(
            lambda p, ww: _xi_loss_of(_xi_brute(p, ww)), argnums=(0, 1)
        )
        brute = vg(pts, w)
        one = jax.value_and_grad(
            lambda p, ww: _xi_loss_of(_xi_dist(p, ww, 1).counts), argnums=(0, 1)
        )(pts, w)
        _XI_REF[name] = (pts, w, brute, one)
    return _XI_REF[name]


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_pair_count_gradient_is_the_single_device_gradient(name, ndev):
    _mesh(ndev)
    pts, w, (v_b, (gp_b, gw_b)), (v_1, (gp_1, gw_1)) = _xi_reference(name)
    assert not bool(_xi_dist(pts, w, ndev).overflow)
    v, (gp, gw) = jax.value_and_grad(
        lambda p, ww: _xi_loss_of(_xi_dist(p, ww, ndev).counts), argnums=(0, 1)
    )(pts, w)
    assert np.all(np.isfinite(gp)) and np.all(np.isfinite(gw))
    assert float(jnp.max(jnp.abs(gp))) > 0
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    # exactly the single-device gradient: not half of it, not double
    assert _rel(gp, gp_b) < _TOL and _rel(gw, gw_b) < _TOL
    assert _rel(gp, gp_1) < _TOL and _rel(gw, gw_1) < _TOL


# ---------------------------------------------------------------------------
# kNN distances and the soft kNN-CDF
# ---------------------------------------------------------------------------

_KNN_REF: dict = {}


def _knn_reference(name):
    if name not in _KNN_REF:
        pts = _PTS[name](_N)
        idx, ok = _brute_nbrs(pts, pts, self_exclude=True)
        coef = jnp.asarray(np.random.default_rng(3).normal(size=(_N, _K)))
        p = jnp.asarray(pts)

        def brute(x):
            return _dist_loss(
                jnp.where(ok, _knn_dist_brute(x, x, idx, ok), jnp.inf), coef
            )

        one = jax.grad(lambda x: _dist_loss(_knn_dist(x, 1).distances, coef))(p)
        _KNN_REF[name] = (p, coef, jax.value_and_grad(brute)(p), one)
    return _KNN_REF[name]


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_knn_distance_gradient_is_the_single_device_gradient(name, ndev):
    _mesh(ndev)
    p, coef, (v_b, g_b), g_1 = _knn_reference(name)
    assert not bool(_knn_dist(p, ndev).overflow)
    v, g = jax.value_and_grad(lambda x: _dist_loss(_knn_dist(x, ndev).distances, coef))(
        p
    )
    assert float(jnp.max(jnp.abs(g))) > 0
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert _rel(g, g_b) < _TOL
    assert _rel(g, g_1) < _TOL


def _cdf_case():
    rng = np.random.default_rng(11)
    data = _clustered(_N, seed=5)
    queries = rng.uniform(0.0, _L, (_N // 2, 3))
    queries[:20] = np.mod(rng.uniform(-2e-3, 2e-3, (20, 3)), _L)  # on the corner
    return jnp.asarray(data), jnp.asarray(queries)


@pytest.mark.parametrize("ndev", _NDEVS)
def test_soft_knn_cdf_gradient_in_data_and_queries(ndev):
    """kNN-CDF of random queries: d/d(data) and d/d(queries) against brute force."""
    _mesh(ndev)
    data, queries = _cdf_case()
    idx, ok = _brute_nbrs(data, queries, self_exclude=False)

    def brute(d, q):
        return _cdf_loss_of(_soft_cdf(_knn_dist_brute(d, q, idx, ok), ok))

    def dist(d, q):
        return _cdf_loss_of(_knn_dist(d, ndev, queries=q, cdf=True).cdf)

    v_b, (gd_b, gq_b) = jax.value_and_grad(brute, argnums=(0, 1))(data, queries)
    v, (gd, gq) = jax.value_and_grad(dist, argnums=(0, 1))(data, queries)
    np.testing.assert_allclose(float(v), float(v_b), rtol=1e-12)
    assert float(jnp.max(jnp.abs(gd))) > 0 and float(jnp.max(jnp.abs(gq))) > 0
    assert _rel(gd, gd_b) < _TOL
    assert _rel(gq, gq_b) < _TOL


# ---------------------------------------------------------------------------
# finite differences
# ---------------------------------------------------------------------------


def _directional_fd(f, x, v, h):
    return (float(f(x + h * v)) - float(f(x - h * v))) / (2.0 * h)


def test_pair_count_central_differences():
    """FD along a random direction, positions and weights.

    The statistic does not depend on the tree, so a partition or walk that
    changes inside the step does not matter. What could: a pair crossing the hard
    cut. With ``eps = 1e-14`` such a crossing moves the count by at most 1e-14.
    """
    ndev = _MULTI
    rng = np.random.default_rng(21)
    pts = jnp.asarray(_faces_and_corners(400, seed=4))
    w = jnp.asarray(rng.uniform(0.5, 2.0, 400))

    def loss(p, ww):
        return _xi_loss_of(_xi_dist(p, ww, ndev, eps=1e-14).counts)

    gp, gw = jax.grad(loss, argnums=(0, 1))(pts, w)
    vp = jnp.asarray(rng.normal(size=pts.shape))
    vw = jnp.asarray(rng.normal(size=w.shape))
    ad_p = float(jnp.sum(gp * vp))
    ad_w = float(jnp.sum(gw * vw))
    # the windows are sharp, so the central difference's h^2 truncation is
    # visible: measured 5.7e-7 at h = 1e-6, 5.6e-9 at 1e-7 (and identical for
    # FD of the brute-force statistic). Assert the h^2 convergence onto AD.
    err = [
        abs(_directional_fd(lambda p: loss(p, w), pts, vp, h) - ad_p) / abs(ad_p)
        for h in (1e-6, 1e-7)
    ]
    assert err[1] < 2e-8 and err[0] / err[1] > 30.0, err
    fd_w = _directional_fd(lambda ww: loss(pts, ww), w, vw, 1e-5)
    assert abs(fd_w - ad_w) < 1e-9 * abs(ad_w), (fd_w, ad_w)


def test_soft_knn_cdf_central_differences():
    """FD of the soft kNN-CDF loss, at a step that keeps every neighbour set."""
    ndev = _MULTI
    data, queries = _cdf_case()
    rng = np.random.default_rng(22)
    vd = jnp.asarray(rng.normal(size=data.shape))
    vq = jnp.asarray(rng.normal(size=queries.shape))
    h = 1e-7
    # the precondition: the topology is the same at both ends of the step
    ref = _brute_nbrs(data, queries, self_exclude=False)
    for s in (-1.0, 1.0):
        idx, ok = _brute_nbrs(
            data + s * h * vd, queries + s * h * vq, self_exclude=False
        )
        assert np.array_equal(idx, ref[0]) and np.array_equal(ok, ref[1])

    def loss(d, q):
        return _cdf_loss_of(_knn_dist(d, ndev, queries=q, cdf=True).cdf)

    gd, gq = jax.grad(loss, argnums=(0, 1))(data, queries)
    fd = (
        float(loss(data + h * vd, queries + h * vq))
        - float(loss(data - h * vd, queries - h * vq))
    ) / (2.0 * h)
    ad = float(jnp.sum(gd * vd) + jnp.sum(gq * vq))
    assert abs(fd - ad) < 1e-6 * abs(ad), (fd, ad)


# ---------------------------------------------------------------------------
# jit, value_and_grad, lax.scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["xi", "knn"])
def test_jit_value_and_grad_and_a_scan_of_gradient_steps(payload):
    ndev = _MULTI
    pts = jnp.asarray(_clustered(400, seed=8))
    coef = jnp.asarray(np.random.default_rng(9).normal(size=(400, _K)))
    w = jnp.ones((400,))

    if payload == "xi":

        def loss(p):
            return _xi_loss_of(_xi_dist(p, w, ndev).counts)

    else:

        def loss(p):
            return _dist_loss(_knn_dist(p, ndev).distances, coef)

    eager = jax.value_and_grad(loss)(pts)
    vg = jax.jit(jax.value_and_grad(loss))
    jitted = vg(pts)
    np.testing.assert_allclose(float(jitted[0]), float(eager[0]), rtol=1e-13)
    assert _rel(jitted[1], eager[1]) < 1e-12

    lr = 1e-6 / float(jnp.max(jnp.abs(eager[1])))

    def step(p, _):
        v, g = jax.value_and_grad(loss)(p)
        return p - lr * g, v

    p_scan, v_scan = jax.jit(lambda p: jax.lax.scan(step, p, None, length=3))(pts)
    p_loop, v_loop = pts, []
    for _ in range(3):
        v, g = vg(p_loop)
        v_loop.append(float(v))
        p_loop = p_loop - lr * g
    np.testing.assert_allclose(np.asarray(v_scan), v_loop, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(p_scan), np.asarray(p_loop), atol=1e-14)
    assert not np.array_equal(np.asarray(p_scan), np.asarray(pts))


# ---------------------------------------------------------------------------
# jaccpot's audit: trace probe, dependency sweep, ragged side effect
# ---------------------------------------------------------------------------


def _payload_loss(payload, ndev, n):
    coef = jnp.asarray(np.random.default_rng(9).normal(size=(n, _K)))
    if payload == "xi":
        return lambda p, w: _xi_loss_of(_xi_dist(p, w, ndev).counts)
    return lambda p, w: _cdf_loss_of(_knn_dist(p, ndev, cdf=True).cdf) + _dist_loss(
        _knn_dist(p, ndev).distances, coef
    )


@pytest.mark.parametrize("payload", ["xi", "knn"])
def test_trace_probe_no_topology_loop_carries_a_gradient(payload):
    """``make_jaxpr(grad)`` traces; no topology loop is reached by a live float."""
    ndev = _MULTI
    pts = jnp.asarray(_faces_and_corners(256, seed=2))
    w = jnp.ones((256,))
    loss = _payload_loss(payload, ndev, 256)
    jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(pts, w)

    jaxpr = jax.make_jaxpr(loss)(pts, w)
    whiles, live = live_while_loops(jaxpr, [True, True])
    assert whiles == [], whiles
    for prim in ("while", "sort", "top_k", "argsort"):
        assert prim not in live, (prim, live)
    # the values do reach the statistic: through both exchanges and the gathers
    assert live.get("gather", 0) > 0 and live.get("all_gather", 0) > 0

    # control: without the stop_gradient seam the walks and tree loops are live
    ctrl, _ = live_while_loops(jaxpr, [True, True], stop_at_stop_gradient=False)
    where = " ".join(c.where for c in ctrl)
    assert "_dual_tree_walk_mutual" in where and "compute_tree_geometry" in where


def _corner_and_face_picks(pts, n_pick=4):
    """Rows closest to a corner and to a face: their partners sit on other domains."""
    p = np.asarray(pts)
    to_face = np.minimum(p, _L - p)
    corner = np.argsort(np.max(to_face, axis=1))[: n_pick // 2]
    face = np.argsort(np.min(to_face, axis=1))[: n_pick // 2]
    return np.unique(np.concatenate([corner, face]))


@pytest.mark.parametrize("payload", ["xi", "knn"])
def test_dependency_sweep_across_faces_and_devices(payload):
    """Perturb one particle: exactly the right gradients change, on every device.

    The perturbed rows hug a corner or a face, so most of their partners are in
    another Morton domain through the periodic image. The change in the whole
    gradient must match brute force (the cross-device cotangents came back), and
    rows that brute force leaves untouched must stay untouched (nothing spurious
    reaches them).
    """
    ndev = _MULTI
    if ndev < 2:
        pytest.skip("needs >= 2 devices")
    n = 400
    pts = _faces_and_corners(n, seed=6)
    p0 = jnp.asarray(pts)
    w = jnp.ones((n,))
    coef = jnp.asarray(np.random.default_rng(9).normal(size=(n, _K)))
    delta = 1e-4

    if payload == "xi":
        dist = jax.jit(jax.grad(lambda p: _xi_loss_of(_xi_dist(p, w, ndev).counts)))

        def brute(x):
            return jax.grad(lambda p: _xi_loss_of(_xi_brute(p, w)))(jnp.asarray(x))

    else:
        dist = jax.jit(
            jax.grad(lambda p: _dist_loss(_knn_dist(p, ndev).distances, coef))
        )

        def brute(x):
            idx, ok = _brute_nbrs(x, x, self_exclude=True)
            return jax.grad(
                lambda p: _dist_loss(
                    jnp.where(ok, _knn_dist_brute(p, p, idx, ok), jnp.inf), coef
                )
            )(jnp.asarray(x))

    g0, b0 = np.asarray(dist(p0)), np.asarray(brute(pts))
    scale = float(np.max(np.abs(b0)))
    picks = _corner_and_face_picks(pts)
    crossed = 0
    for j in picks:
        x = pts.copy()
        x[j] = np.mod(x[j] + delta * np.array([1.0, -0.7, 0.4]), _L)
        dg = np.asarray(dist(jnp.asarray(x))) - g0
        db = np.asarray(brute(x)) - b0
        changed = np.any(db != 0.0, axis=1)
        assert changed[j] and changed.sum() > 1
        assert np.max(np.abs(dg - db)) < 1e-10 * scale
        assert np.max(np.abs(dg[~changed])) < 1e-12 * scale
        # some of the changed rows are partners through the periodic image
        raw = np.linalg.norm(pts[changed] - pts[j], axis=1)
        crossed += int(np.sum(raw > 0.5 * _L))
    assert crossed > 0, "no perturbation reached a partner across a face"


@pytest.mark.parametrize("payload", ["xi", "knn"])
def test_forward_survives_a_gradient(payload):
    """A forward evaluated after a gradient is bit-identical to one before.

    jaccpot's regression guard: on jax 0.9.0.x the native ragged all-to-all's
    reverse pass left every later exchange returning its fill value, so the
    forward silently lost the whole cross-domain halo. On CPU the exchange is the
    all_gather fallback, which this pins; on GPU it is what ``"auto"`` resolves to.
    """
    ndev = _MULTI
    pts = jnp.asarray(_clustered(400, seed=12))
    w = jnp.ones((400,))
    loss = _payload_loss(payload, ndev, 400)

    def forward():
        if payload == "xi":
            return np.asarray(_xi_dist(pts, w, ndev).counts)
        return np.asarray(_knn_dist(pts, ndev).distances)

    before = forward()
    jax.block_until_ready(jax.jit(jax.grad(loss))(pts, w))
    after = forward()
    assert np.array_equal(before, after, equal_nan=True)


@pytest.mark.skipif(
    os.environ.get("YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX", "0") != "1",
    reason=(
        "opt-in, GPU only: differentiates through the native ragged all-to-all and "
        "checks the forward afterwards (jaccpot's JAX_RAGGED_GRAD_FIXED_VERSION "
        "measurement). Set YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1 on >= 2 GPUs."
    ),
)
@pytest.mark.parametrize("payload", ["xi", "knn"])
def test_native_ragged_exchange_survives_a_gradient(payload):
    if jax.default_backend() != "gpu" or device_count() < 2:
        pytest.skip("the native ragged all-to-all needs >= 2 GPUs")
    ndev = min(device_count(), 8)
    pts = jnp.asarray(_clustered(400, seed=12))
    w = jnp.ones((400,))
    coef = jnp.asarray(np.random.default_rng(9).normal(size=(400, _K)))

    def run(exchange):
        if payload == "xi":
            f = lambda p: _xi_loss_of(_xi_dist(p, w, ndev, exchange=exchange).counts)
            fwd = lambda: np.asarray(_xi_dist(pts, w, ndev, exchange=exchange).counts)
        else:
            f = lambda p: _dist_loss(
                _knn_dist(p, ndev, exchange=exchange).distances, coef
            )
            fwd = lambda: np.asarray(_knn_dist(pts, ndev, exchange=exchange).distances)
        before = fwd()
        g = np.asarray(jax.jit(jax.grad(f))(pts))
        return before, g, fwd()

    before, g_native, after = run("native")
    assert np.array_equal(before, after, equal_nan=True), (
        "the native ragged all-to-all changed the forward after a gradient: keep "
        "exchange='buf' on this jax"
    )
    _, g_buf, _ = run("buf")
    assert _rel(g_native, g_buf) < 1e-12


# ---------------------------------------------------------------------------
# the per-device body in a caller's shard_map; configuration
# ---------------------------------------------------------------------------


def test_gradient_through_the_shard_body_in_a_callers_shard_map():
    from jax import shard_map
    from jax.sharding import PartitionSpec as P

    ndev = _MULTI
    mesh = _mesh(ndev)
    pts = _clustered(400, seed=13)
    lo, hi = soft_window_cutoffs(_EDGES, _SHARP, _EPS)
    per = -(-len(pts) // ndev)
    pos = np.zeros((ndev * per, 3))
    pos[: len(pts)] = pts
    counts = np.array([min(per, len(pts) - d * per) for d in range(ndev)])
    cfg = _cfg(len(pts), ndev)

    def body(x, c):
        return pair_counts_shard(
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
        ).counts

    run = jax.jit(
        shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
            out_specs=P(),
            check_vma=False,
        )
    )
    g = jax.grad(lambda x: _xi_loss_of(run(x, jnp.asarray(counts))))(jnp.asarray(pos))
    gb = jax.grad(lambda p: _xi_loss_of(_xi_brute(p, jnp.ones(len(pts)))))(
        jnp.asarray(pts)
    )
    assert _rel(np.asarray(g)[: len(pts)], gb) < _TOL
    assert not np.any(np.asarray(g)[len(pts) :])


def test_unknown_exchange_is_refused():
    with pytest.raises(ValueError, match="exchange"):
        DistributedStatsConfig(
            capacity=16, halo_capacity=16, exchange="ragged"
        ).resolved()
