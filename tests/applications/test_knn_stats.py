"""Tests for the differentiable kNN-CDF (``yggdrax.applications.knn_stats``).

* the tree search finds the exact neighbour distances (``scipy`` reference);
* the soft CDF converges to the hard CDF as the sharpness grows;
* the "fixed indices, recomputed distances" gradient equals autodiff through a
  dense ``top_k`` and central finite differences (double precision, no ties);
* ``jit``, ``grad`` and ``vmap`` over catalogues compose;
* in a periodic box all of the above hold against a brute-force minimum-image
  reference, with particles on faces and corners;
* on one device it agrees with :func:`yggdrax.distributed.stats.distributed_knn`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

from yggdrax.applications.knn_stats import (
    cdf_distance,
    hard_knn_cdf,
    knn_cdf,
    knn_cdf_loss,
    knn_distances,
    knn_indices,
    knn_log_distances,
    peaked_cdf,
)
from yggdrax.applications.knn_stats.baselines import (
    brute_force_knn_cdf,
    brute_force_knn_log_distances,
    scipy_knn_distances,
)

KS = (1, 2, 5)
R_GRID = jnp.geomspace(0.03, 0.6, 18)


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _points(n: int, seed: int) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)


def _weighted(cdf: jax.Array) -> jax.Array:
    """A scalar that depends on every (k, r) entry with a distinct weight."""

    w = jnp.arange(1, cdf.size + 1, dtype=cdf.dtype).reshape(cdf.shape)
    return jnp.sum(w * cdf)


# ---------------------------------------------------------------------------
# Neighbour search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["tree", "dense", "tiled"])
def test_distances_match_scipy_random_queries(backend):
    data, queries = _points(400, 0), _points(250, 1)
    idx = knn_indices(data, queries, max(KS), backend=backend, leaf_size=8)
    r = np.asarray(knn_distances(data, queries, idx, KS))
    ref = scipy_knn_distances(np.asarray(data), np.asarray(queries), KS)
    np.testing.assert_allclose(r, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", ["tree", "dense"])
def test_distances_match_scipy_self_queries(backend):
    data = _points(350, 2)
    idx = knn_indices(data, data, max(KS), exclude_self=True, backend=backend)
    assert not np.any(np.asarray(idx) == np.arange(350)[:, None])
    r = np.asarray(knn_distances(data, data, idx, KS))
    ref = scipy_knn_distances(np.asarray(data), None, KS)
    np.testing.assert_allclose(r, ref, rtol=1e-12, atol=1e-14)


def test_tree_cdf_matches_dense_reference():
    data, queries = _points(500, 3), _points(300, 4)
    cdf = knn_cdf(data, queries, KS, R_GRID, sharpness=25.0)
    ref = brute_force_knn_cdf(data, queries, KS, R_GRID, sharpness=25.0)
    np.testing.assert_allclose(np.asarray(cdf), np.asarray(ref), atol=1e-14)


def test_cdf_shape_range_and_monotonicity():
    data, queries = _points(300, 5), _points(200, 6)
    cdf, peaked = knn_cdf(data, queries, KS, R_GRID, sharpness=20.0, return_peaked=True)
    assert cdf.shape == (len(KS), R_GRID.shape[0])
    c = np.asarray(cdf)
    assert np.all((c >= 0.0) & (c <= 1.0))
    assert np.all(np.diff(c, axis=1) >= 0.0)  # non-decreasing in r
    assert np.all(np.diff(c, axis=0) <= 1e-15)  # CDF_k >= CDF_{k+1}
    np.testing.assert_allclose(np.asarray(peaked), np.minimum(c, 1.0 - c))
    np.testing.assert_allclose(np.asarray(peaked_cdf(cdf)), np.asarray(peaked))


# ---------------------------------------------------------------------------
# Soft -> hard
# ---------------------------------------------------------------------------


def test_soft_cdf_converges_to_hard_cdf():
    data, queries = _points(400, 7), _points(600, 8)
    hard = np.asarray(hard_knn_cdf(data, queries, KS, R_GRID))
    # The hard CDF against scipy's distances, independently of our search.
    r_ref = scipy_knn_distances(np.asarray(data), np.asarray(queries), KS)
    hard_ref = np.mean(r_ref[:, :, None] < np.asarray(R_GRID)[None, None, :], axis=0)
    np.testing.assert_allclose(hard, hard_ref)

    sharpnesses = (5.0, 50.0, 500.0, 5e3, 5e5)
    errs = []
    for s in sharpnesses:
        soft = np.asarray(knn_cdf(data, queries, KS, R_GRID, sharpness=s))
        errs.append(np.max(np.abs(soft - hard)))
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    # Each query is off by sigmoid(-s * gap), its gap to the grid in log r, so
    # the error is bounded by the queries' own gaps -- tight, not a guess.
    gaps = np.abs(np.log(np.asarray(R_GRID))[None, None, :] - np.log(r_ref)[..., None])
    bound = np.max(np.mean(expit(-sharpnesses[-1] * gaps), axis=0))
    assert errs[-1] <= bound * (1 + 1e-6), (errs, bound)
    assert errs[-1] < 1e-4, errs


def test_soft_cdf_bias_is_second_order_in_width():
    """With many queries the CDF is smooth and the bias falls as ``1/s^2``."""

    data, queries = _points(300, 9), _points(20000, 10)
    hard = np.asarray(hard_knn_cdf(data, queries, (1,), R_GRID))
    b1 = np.max(
        np.abs(np.asarray(knn_cdf(data, queries, (1,), R_GRID, sharpness=10.0)) - hard)
    )
    b2 = np.max(
        np.abs(np.asarray(knn_cdf(data, queries, (1,), R_GRID, sharpness=20.0)) - hard)
    )
    # Doubling s should cut the bias by ~4; allow for sampling noise.
    assert 2.5 < b1 / b2 < 6.0, (b1, b2)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("self_queries", [False, True])
def test_gradient_matches_dense_topk_autodiff(self_queries):
    data = _points(300, 11)
    queries = None if self_queries else _points(200, 12)

    def tree_scalar(d):
        return _weighted(knn_cdf(d, queries, KS, R_GRID, sharpness=15.0))

    def dense_scalar(d):
        return _weighted(brute_force_knn_cdf(d, queries, KS, R_GRID, sharpness=15.0))

    g_tree = np.asarray(jax.grad(tree_scalar)(data))
    g_dense = np.asarray(jax.grad(dense_scalar)(data))
    assert np.max(np.abs(g_dense)) > 1.0  # the check is not vacuous
    np.testing.assert_allclose(g_tree, g_dense, rtol=1e-10, atol=1e-10)


def test_gradient_wrt_queries_matches_dense_topk_autodiff():
    data, queries = _points(300, 13), _points(150, 14)

    def tree_scalar(q):
        return _weighted(knn_cdf(data, q, KS, R_GRID, sharpness=15.0))

    def dense_scalar(q):
        return _weighted(brute_force_knn_cdf(data, q, KS, R_GRID, sharpness=15.0))

    np.testing.assert_allclose(
        np.asarray(jax.grad(tree_scalar)(queries)),
        np.asarray(jax.grad(dense_scalar)(queries)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_log_distance_gradient_matches_dense_topk():
    """Per-query ``log r_k`` Jacobian-vector product, before any CDF smoothing."""

    data, queries = _points(200, 15), _points(80, 16)
    idx = knn_indices(data, queries, max(KS))
    tangent = jax.random.normal(jax.random.PRNGKey(17), data.shape, dtype=data.dtype)

    def tree_logr(d):
        return jnp.log(knn_distances(d, queries, idx, KS))

    def dense_logr(d):
        return brute_force_knn_log_distances(d, queries, KS)

    _, jvp_tree = jax.jvp(tree_logr, (data,), (tangent,))
    _, jvp_dense = jax.jvp(dense_logr, (data,), (tangent,))
    np.testing.assert_allclose(np.asarray(jvp_tree), np.asarray(jvp_dense), atol=1e-11)


@pytest.mark.parametrize("self_queries", [False, True])
def test_gradient_matches_central_finite_differences(self_queries):
    data = _points(200, 18)
    queries = None if self_queries else _points(150, 19)

    @jax.jit
    def scalar(flat):
        d = flat.reshape(data.shape)
        return _weighted(knn_cdf(d, queries, KS, R_GRID, sharpness=10.0))

    x = data.reshape(-1)
    grad = np.asarray(jax.grad(scalar)(x))
    # Coordinates with the largest gradients: those whose kNN role is real, so
    # the comparison is not dominated by zeros.
    picks = np.argsort(-np.abs(grad))[:12]
    h = 1e-6
    fd = np.array(
        [
            (float(scalar(x.at[i].add(h))) - float(scalar(x.at[i].add(-h)))) / (2 * h)
            for i in picks
        ]
    )
    rel = np.max(np.abs(grad[picks] - fd) / np.maximum(np.abs(fd), 1e-8))
    assert rel < 1e-5, rel


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_jit_matches_eager():
    data, queries = _points(250, 20), _points(120, 21)
    eager = knn_cdf(data, queries, KS, R_GRID, sharpness=12.0)
    jitted = jax.jit(lambda d, q, r, s: knn_cdf(d, q, KS, r, sharpness=s))(
        data, queries, R_GRID, 12.0
    )
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-15)


def test_vmap_over_catalogues_and_grad():
    cats = jax.random.uniform(jax.random.PRNGKey(22), (3, 200, 3), dtype=jnp.float64)
    queries = _points(100, 23)

    def per_cat(d):
        return knn_cdf(d, queries, KS, R_GRID, sharpness=12.0)

    batched = jax.jit(jax.vmap(per_cat))(cats)
    looped = jnp.stack([per_cat(c) for c in cats])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), atol=1e-15)

    total = lambda cs: jnp.sum(jax.vmap(lambda d: _weighted(per_cat(d)))(cs))
    g_batched = jax.jit(jax.grad(total))(cats)
    g_looped = jnp.stack([jax.grad(lambda d: _weighted(per_cat(d)))(c) for c in cats])
    np.testing.assert_allclose(np.asarray(g_batched), np.asarray(g_looped), atol=1e-12)


def test_vmap_over_query_sets():
    data = _points(200, 24)
    qsets = jax.random.uniform(jax.random.PRNGKey(25), (2, 90, 3), dtype=jnp.float64)
    batched = jax.vmap(lambda q: knn_cdf(data, q, KS, R_GRID, sharpness=12.0))(qsets)
    for b, q in zip(batched, qsets):
        np.testing.assert_allclose(
            np.asarray(b),
            np.asarray(knn_cdf(data, q, KS, R_GRID, sharpness=12.0)),
            atol=1e-15,
        )


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


def test_loss_is_zero_at_target_and_descends():
    queries = _points(300, 26)
    target_data, data = _points(250, 27), _points(250, 28)
    target = knn_cdf(target_data, queries, KS, R_GRID, sharpness=15.0)

    loss = lambda d: knn_cdf_loss(d, queries, target, KS, R_GRID, sharpness=15.0)
    assert float(loss(target_data)) == 0.0
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(target_data)), 0.0)

    value, grad = jax.jit(jax.value_and_grad(loss))(data)
    assert float(value) > 0.0
    step = 1e-4 / float(jnp.max(jnp.abs(grad)))
    assert float(loss(data - step * grad)) < float(value)


def test_cdf_distance_weights_and_peaked():
    a = jnp.array([[0.1, 0.4, 0.9], [0.0, 0.2, 0.7]])
    b = jnp.array([[0.2, 0.4, 0.8], [0.0, 0.3, 0.7]])
    np.testing.assert_allclose(float(cdf_distance(a, b)), np.mean((a - b) ** 2))
    w = jnp.array([[1.0], [0.0]])  # first k only
    np.testing.assert_allclose(
        float(cdf_distance(a, b, weights=w)), np.mean((a[0] - b[0]) ** 2)
    )
    pa, pb = np.minimum(a, 1 - a), np.minimum(b, 1 - b)
    np.testing.assert_allclose(
        float(cdf_distance(a, b, peaked=True)), np.mean((pa - pb) ** 2)
    )


# ---------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ks", [(), (0, 1)])
def test_bad_neighbour_orders_raise(ks):
    data = _points(50, 30)
    with pytest.raises(ValueError):
        knn_cdf(data, None, ks, R_GRID, sharpness=10.0)


# ---------------------------------------------------------------------------
# Periodic box
# ---------------------------------------------------------------------------

_L = 1.0
P_GRID = jnp.geomspace(0.02, 0.4, 16)  # below L/2


def _faces_and_corners(
    n: int, seed: int, box=(_L, _L, _L), *, exact: bool = True
) -> jax.Array:
    """Uniform points; ~20 % within 1e-3 L of a face (a few exactly on one),
    ~5 % within 1e-3 L of a corner, one exactly on the corner.

    ``exact=False`` drops the exactly placed points, for a query set that must
    not coincide with a data point (``r = 0`` is a degenerate input: the
    gradient of ``r`` is undefined there).
    """

    rng = np.random.default_rng(seed)
    lengths = np.asarray(box, dtype=np.float64)
    pts = rng.uniform(0.0, 1.0, (n, 3)) * lengths
    m = n // 5
    axis = rng.integers(0, 3, m)
    side = rng.integers(0, 2, m)
    offs = rng.uniform(0.0, 1e-3, m) * lengths[axis]
    pts[np.arange(m), axis] = np.where(side == 1, lengths[axis] - offs, offs)
    if exact:
        pts[:3, 0] = 0.0  # exactly on the x = 0 face
    c = n // 20
    corners = rng.integers(0, 2, (c, 3)) * lengths
    corners = corners + rng.uniform(-1e-3, 1e-3, (c, 3)) * lengths
    pts[m : m + c] = np.mod(corners, lengths)
    if exact:
        pts[m] = 0.0  # exactly on the corner
    return jnp.asarray(pts)


@pytest.mark.parametrize("backend", ["tree", "dense", "tiled"])
@pytest.mark.parametrize("self_queries", [False, True])
def test_periodic_distances_match_brute_force_and_scipy(backend, self_queries):
    data = _faces_and_corners(400, 31)
    queries = None if self_queries else _faces_and_corners(300, 32)
    q = data if queries is None else queries
    idx = knn_indices(
        data, q, max(KS), exclude_self=self_queries, backend=backend, box_size=_L
    )
    logr = np.asarray(knn_log_distances(data, q, idx, KS, box_size=_L))
    brute = np.asarray(brute_force_knn_log_distances(data, queries, KS, box_size=_L))
    np.testing.assert_allclose(logr, brute, rtol=1e-12, atol=1e-13)
    ref = scipy_knn_distances(np.asarray(data), queries, KS, box_size=_L)
    np.testing.assert_allclose(np.exp(logr), ref, rtol=1e-12, atol=1e-14)


def test_periodic_anisotropic_box_matches_scipy():
    box = (1.0, 1.5, 2.0)
    data, queries = _faces_and_corners(300, 33, box), _faces_and_corners(200, 34, box)
    idx = knn_indices(data, queries, max(KS), box_size=box)
    r = np.asarray(knn_distances(data, queries, idx, KS, box_size=box))
    ref = scipy_knn_distances(np.asarray(data), np.asarray(queries), KS, box_size=box)
    np.testing.assert_allclose(r, ref, rtol=1e-12, atol=1e-14)


def test_periodic_neighbour_across_a_face_and_a_corner():
    data = jnp.array(
        [[0.001, 0.5, 0.5], [0.5, 0.5, 0.5], [0.002, 0.002, 0.002], [0.3, 0.7, 0.2]]
    )
    queries = jnp.array([[0.999, 0.5, 0.5], [0.999, 0.999, 0.999]])
    r = np.asarray(
        knn_distances(
            data, queries, knn_indices(data, queries, 1, box_size=_L), 1, box_size=_L
        )
    )[:, 0]
    np.testing.assert_allclose(r, [0.002, np.sqrt(3) * 0.003], rtol=1e-10)
    # Without the box the same queries are far from both.
    r_open = np.asarray(knn_distances(data, queries, knn_indices(data, queries, 1), 1))[
        :, 0
    ]
    assert np.all(r_open > 0.4)


def test_periodic_positions_may_lie_outside_the_box():
    """Images of the same catalogue give the same CDF and the same gradient."""

    data, queries = _faces_and_corners(300, 35), _faces_and_corners(200, 36)
    rng = np.random.default_rng(37)
    shifts = rng.integers(-2, 3, data.shape) * _L
    moved = data + shifts
    f = lambda d: _weighted(
        knn_cdf(d, queries, KS, P_GRID, sharpness=15.0, box_size=_L)
    )
    np.testing.assert_allclose(float(f(moved)), float(f(data)), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jax.grad(f)(moved)), np.asarray(jax.grad(f)(data)), atol=1e-9
    )


def test_periodic_cdf_is_translation_invariant():
    data, queries = _faces_and_corners(300, 38), _faces_and_corners(250, 39)
    shift = jnp.array([0.37, -0.81, 0.5])
    base = knn_cdf(data, queries, KS, P_GRID, sharpness=15.0, box_size=_L)
    moved = knn_cdf(
        data + shift, queries + shift, KS, P_GRID, sharpness=15.0, box_size=_L
    )
    np.testing.assert_allclose(np.asarray(moved), np.asarray(base), atol=1e-12)
    # and the open-boundary CDF is not, so the check has teeth
    open_moved = knn_cdf(
        jnp.mod(data + shift, _L),
        jnp.mod(queries + shift, _L),
        KS,
        P_GRID,
        sharpness=15.0,
    )
    open_base = knn_cdf(data, queries, KS, P_GRID, sharpness=15.0)
    assert float(jnp.max(jnp.abs(open_moved - open_base))) > 1e-3


@pytest.mark.parametrize("self_queries", [False, True])
def test_periodic_gradient_matches_brute_force_and_finite_differences(self_queries):
    data = _faces_and_corners(250, 40)
    queries = None if self_queries else _faces_and_corners(200, 41)

    @jax.jit
    def tree_scalar(d):
        return _weighted(knn_cdf(d, queries, KS, P_GRID, sharpness=10.0, box_size=_L))

    def dense_scalar(d):
        return _weighted(
            brute_force_knn_cdf(d, queries, KS, P_GRID, sharpness=10.0, box_size=_L)
        )

    g = np.asarray(jax.grad(tree_scalar)(data))
    g_dense = np.asarray(jax.grad(dense_scalar)(data))
    assert np.max(np.abs(g_dense)) > 1.0
    np.testing.assert_allclose(g, g_dense, rtol=1e-10, atol=1e-10)

    # Finite differences on the largest gradients and on particles at a face or
    # a corner (rows 0-2 sit on x = 0, row 50 on the corner).
    flat = g.reshape(-1)
    picks = list(np.argsort(-np.abs(flat))[:8]) + [0, 3, 6, 150, 151, 152]
    h = 1e-6
    x = data.reshape(-1)
    f = lambda v: tree_scalar(v.reshape(data.shape))
    fd = np.array(
        [
            (float(f(x.at[i].add(h))) - float(f(x.at[i].add(-h)))) / (2 * h)
            for i in picks
        ]
    )
    err = np.abs(flat[picks] - fd)
    assert np.all(err < 1e-5 * np.maximum(np.abs(fd), 1.0)), (flat[picks], fd)


def test_periodic_gradient_wrt_queries_matches_brute_force():
    data, queries = _faces_and_corners(250, 42), _faces_and_corners(150, 43)
    f = lambda q: _weighted(knn_cdf(data, q, KS, P_GRID, sharpness=10.0, box_size=_L))
    fb = lambda q: _weighted(
        brute_force_knn_cdf(data, q, KS, P_GRID, sharpness=10.0, box_size=_L)
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(f)(queries)),
        np.asarray(jax.grad(fb)(queries)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_periodic_jit_vmap_and_loss():
    cats = jnp.stack([_faces_and_corners(200, s) for s in (44, 45)])
    queries = _faces_and_corners(120, 46)
    per_cat = lambda d: knn_cdf(d, queries, KS, P_GRID, sharpness=12.0, box_size=_L)
    batched = jax.jit(jax.vmap(per_cat))(cats)
    for b, c in zip(batched, cats):
        np.testing.assert_allclose(np.asarray(b), np.asarray(per_cat(c)), atol=1e-15)

    target = per_cat(cats[0])
    loss = lambda d: knn_cdf_loss(
        d, queries, target, KS, P_GRID, sharpness=12.0, box_size=_L
    )
    assert float(loss(cats[0])) == 0.0
    value, grad = jax.jit(jax.value_and_grad(loss))(cats[1])
    step = 1e-4 / float(jnp.max(jnp.abs(grad)))
    assert float(loss(cats[1] - step * grad)) < float(value)


def test_periodic_grid_past_half_box_is_refused():
    data = _faces_and_corners(60, 47)
    with pytest.raises(ValueError, match="min\\(box_size\\)/2"):
        knn_cdf(
            data, None, KS, jnp.geomspace(0.05, 0.5, 4), sharpness=10.0, box_size=_L
        )
    with pytest.raises(ValueError):
        hard_knn_cdf(
            data, None, KS, jnp.geomspace(0.05, 0.7, 4), box_size=(1.0, 2.0, 2.0)
        )


# ---------------------------------------------------------------------------
# Agreement with the multi-device driver (C1) on one device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("self_queries", [False, True])
def test_agrees_with_distributed_knn_on_one_device(self_queries):
    """Same data, k, grid and sharpness: same CDF and same gradients.

    ``r_max`` is set above every ``r_k`` (checked), so the distributed search's
    cap never truncates and the two estimators are the same function.
    """

    from yggdrax.distributed import make_mesh
    from yggdrax.distributed.stats import DistributedStatsConfig, distributed_knn

    k, r_max, sharp = 4, 0.3, 20.0
    ks = tuple(range(1, k + 1))
    data = _faces_and_corners(400, 48)
    # Queries avoid the exactly placed data points: at r = 0 the gradient of r
    # is undefined, and the two codes regularise it differently (B1 floors d^2,
    # C1 takes sqrt(0) and returns NaN).
    queries = None if self_queries else _faces_and_corners(300, 49, exact=False)
    n = data.shape[0] + (0 if queries is None else queries.shape[0])
    cfg = DistributedStatsConfig.for_problem(n, halo_fraction=8.0)
    mesh = make_mesh(1)

    r_ref = scipy_knn_distances(np.asarray(data), queries, ks, box_size=_L)
    assert r_ref.max() < r_max

    def c1(d, q):
        res = distributed_knn(
            d,
            k=k,
            r_max=r_max,
            box_size=_L,
            config=cfg,
            queries=q,
            r_grid=P_GRID,
            sharpness=sharp,
            mesh=mesh,
        )
        assert res.cdf is not None
        return res.cdf

    def b1(d, q):
        cdf = knn_cdf(d, q, ks, P_GRID, sharpness=sharp, box_size=_L)
        assert isinstance(cdf, jax.Array)
        return cdf

    np.testing.assert_allclose(
        np.asarray(b1(data, queries)), np.asarray(c1(data, queries)), atol=1e-12
    )
    if queries is None:
        g_b1 = jax.grad(lambda d: _weighted(b1(d, None)))(data)
        g_c1 = jax.grad(lambda d: _weighted(c1(d, None)))(data)
        np.testing.assert_allclose(np.asarray(g_b1), np.asarray(g_c1), atol=1e-9)
    else:
        gb = jax.grad(lambda d, q: _weighted(b1(d, q)), argnums=(0, 1))(data, queries)
        gc = jax.grad(lambda d, q: _weighted(c1(d, q)), argnums=(0, 1))(data, queries)
        for a, b in zip(gb, gc):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)
