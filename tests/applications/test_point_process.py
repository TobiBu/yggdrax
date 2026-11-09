"""The point-process forward model composed with the clustering statistics (B3).

* the model's ``xi`` matches the analytic fixed-count Thomas ``xi`` (and not the
  Poisson one), and a non-Gaussian profile matches its Monte Carlo prediction;
* ``d(statistic)/d(theta)`` through the whole chain (positions, wrap, partition,
  tree rebuild, walk, soft statistic) equals central finite differences of the
  same chain, and forward- and reverse-mode agree;
* the Fisher matrix is symmetric and positive definite, and the Hartlap factor
  is right;
* a Levenberg–Marquardt fit recovers the parameters exactly on its own data and within
  the Fisher errors on an independent realisation.

Everything runs under ``jit``; ensembles run under ``vmap``. Sizes are tiny
(N = 512); the module takes 2-3 min on CPU, most of it compiling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.point_process import (
    child_offsets,
    concat_statistics,
    draw_ensemble,
    draw_thomas,
    ensemble_jacobians,
    ensemble_statistics,
    fisher_forecast,
    fit_parameters,
    hartlap_factor,
    make_knn_cdf_statistic,
    make_xi_statistic,
    profile_norm,
    profile_xi_fixed_counts_mc,
    query_grid,
    sample_covariance,
    soft_shell_volumes,
    statistic_jacobian,
    thomas_positions,
    thomas_xi_fixed_counts,
    thomas_xi_poisson,
    window_average,
)
from yggdrax.applications.point_process.inference import central_difference_jacobian
from yggdrax.distributed import make_mesh
from yggdrax.distributed.stats import soft_window_cutoffs


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


L = 1.0
P, M = 64, 8
N = P * M
SIGMA = 0.03
THETA = (SIGMA, 1.0)
XI_S, EPS = 20.0, 1e-6
EDGES = np.geomspace(0.2 * SIGMA, 5.0 * SIGMA, 7)
R_CUT = soft_window_cutoffs(EDGES, XI_S, EPS)


class _Setup:
    """Statistics built once per module (their programs are cached)."""

    def __init__(self):
        mesh = make_mesh(1)
        self.xi = make_xi_statistic(
            EDGES, box_size=L, n_points=N, sharpness=XI_S, eps=EPS, mesh=mesh
        )
        self.knn = make_knn_cdf_statistic(
            (1, 2),
            np.geomspace(0.06, 0.16, 3),
            box_size=L,
            n_points=N,
            queries=query_grid(8, L),
            sharpness=20.0,
            mesh=mesh,
        )
        self.both = concat_statistics(self.xi, self.knn)


@pytest.fixture(scope="module")
def setup():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield _Setup()
    finally:
        jax.config.update("jax_enable_x64", old)


@pytest.fixture(scope="module")
def ensemble(setup):
    """Covariance ensemble (vmapped under jit) and an ensemble of Jacobians."""
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        theta = jnp.asarray(THETA)
        n_cov = 32
        draws = draw_ensemble(jax.random.PRNGKey(11), n_cov, P, M)
        vals, over = jax.jit(lambda t, d: ensemble_statistics(setup.both, t, d, L))(
            theta, draws
        )
        assert not bool(jnp.any(over))
        draws_j = draw_ensemble(jax.random.PRNGKey(12), 4, P, M)
        _, jacs, over_j = jax.jit(lambda t, d: ensemble_jacobians(setup.both, t, d, L))(
            theta, draws_j
        )
        assert not bool(jnp.any(over_j))
        yield np.asarray(vals), np.asarray(jacs), n_cov
    finally:
        jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_positions_are_wrapped_and_beta_one_is_gaussian():
    d = draw_thomas(jax.random.PRNGKey(0), P, M)
    x = jax.jit(lambda t: thomas_positions(t, d, L))(jnp.asarray(THETA))
    assert x.shape == (N, 3)
    assert bool(jnp.all((x >= 0.0) & (x < L)))
    off = child_offsets(jnp.asarray(THETA), d.normals)
    np.testing.assert_allclose(off, SIGMA * d.normals, rtol=1e-14, atol=1e-16)
    assert abs(float(profile_norm(1.0)) - 1.0) < 1e-14


@pytest.mark.parametrize("beta", [0.7, 1.0, 1.5])
def test_profile_shape_keeps_the_rms_size(beta):
    z = jax.random.normal(jax.random.PRNGKey(3), (1 << 20, 3), dtype=jnp.float64)
    off = child_offsets(jnp.asarray([SIGMA, beta]), z)
    ms = float(jnp.mean(jnp.sum(off * off, axis=-1))) / (3.0 * SIGMA**2)
    assert abs(ms - 1.0) < 1e-2


def test_soft_shell_volumes_tile_the_binned_shell():
    vb = soft_shell_volumes(EDGES, 200.0, soft_window_cutoffs(EDGES, 200.0, 1e-10))
    shell = 4.0 / 3.0 * np.pi * (EDGES[-1] ** 3 - EDGES[0] ** 3)
    assert abs(vb.sum() / shell - 1.0) < 1e-2


def test_xi_matches_the_fixed_count_thomas_process(setup, ensemble):
    vals, _, n_cov = ensemble
    xi = vals[:, : setup.xi.size]
    mean = xi.mean(0)
    prec = hartlap_factor(n_cov, xi.shape[1]) * np.linalg.inv(
        np.asarray(sample_covariance(jnp.asarray(xi))) / n_cov
    )
    fixed = window_average(
        lambda r: thomas_xi_fixed_counts(r, SIGMA, P, M, L**3), EDGES, XI_S, R_CUT
    )
    poisson = window_average(
        lambda r: thomas_xi_poisson(r, SIGMA, P / L**3), EDGES, XI_S, R_CUT
    )
    chi2_fixed = float((mean - fixed) @ prec @ (mean - fixed))
    chi2_poisson = float((mean - poisson) @ prec @ (mean - poisson))
    dof = xi.shape[1]
    # chi2 with 6 dof: 99.9% quantile 22.5. The Poisson amplitude is 1/(1 - 1/m)
    # = 14 % high at m = 8 and must be rejected.
    assert chi2_fixed < 22.5, (chi2_fixed, dof)
    assert chi2_poisson > 4.0 * 22.5, chi2_poisson


def test_non_gaussian_profile_matches_monte_carlo(setup):
    theta = jnp.asarray([SIGMA, 1.5])
    draws = draw_ensemble(jax.random.PRNGKey(21), 16, P, M)
    # batches of 4 under lax.map: the bounded-memory path of the ensembles
    vals, over = jax.jit(
        lambda t, d: ensemble_statistics(setup.xi, t, d, L, batch_size=4)
    )(theta, draws)
    assert not bool(jnp.any(over))
    vals = np.asarray(vals)
    mc = profile_xi_fixed_counts_mc(
        jax.random.PRNGKey(5), theta, EDGES, XI_S, R_CUT, P, M, L**3
    )
    pull = (vals.mean(0) - mc) / (vals.std(0, ddof=1) / 4.0)
    assert np.all(np.abs(pull) < 4.0), pull


# ---------------------------------------------------------------------------
# gradients through the full chain
# ---------------------------------------------------------------------------


def test_jacobian_matches_finite_differences_of_the_full_chain(setup):
    theta = jnp.asarray([SIGMA, 1.2])
    d = draw_thomas(jax.random.PRNGKey(7), P, M)
    v, jac, over = jax.jit(lambda t: statistic_jacobian(setup.both, t, d, L))(theta)
    assert not bool(over)
    jac = np.asarray(jac)
    assert np.all(np.abs(jac).max(axis=0) > 0)

    fd_fn = jax.jit(lambda h: central_difference_jacobian(setup.both, theta, d, L, h))

    def err(rel, rows):
        fd, o = fd_fn(rel * theta)
        assert not bool(o)
        a = jac[rows]
        return np.abs(np.asarray(fd)[rows] - a).max(axis=0) / np.abs(a).max(axis=0)

    xi_rows = slice(0, setup.xi.size)
    knn_rows = slice(setup.xi.size, setup.both.size)
    # at a small step the FD of the whole chain (tree rebuilt each time) is AD
    assert np.all(err(1e-5, xi_rows) < 1e-6)
    assert np.all(err(1e-5, knn_rows) < 1e-5)
    # and the xi error falls as h^2 (truncation, not a wrong derivative)
    e3, e4 = err(1e-3, xi_rows), err(1e-4, xi_rows)
    assert np.all(e3 / e4 > 30.0), (e3, e4)


def test_reverse_mode_gradient_equals_forward_jacobian(setup):
    """``jax.grad`` of a scalar loss (reverse mode) against ``w @ jacfwd``."""
    d = draw_thomas(jax.random.PRNGKey(9), P, M)
    w = jax.random.normal(jax.random.PRNGKey(10), (setup.both.size,))

    def loss(t):
        v, _ = setup.both.fn(thomas_positions(t, d, L))
        return jnp.sum(w * v * v)

    theta = jnp.asarray([SIGMA, 1.1])
    g = jax.jit(jax.grad(loss))(theta)
    v, jac, _ = jax.jit(lambda t: statistic_jacobian(setup.both, t, d, L))(theta)
    np.testing.assert_allclose(g, (2.0 * w * v) @ jac, rtol=1e-9)


# ---------------------------------------------------------------------------
# Fisher forecast
# ---------------------------------------------------------------------------


def test_hartlap_factor():
    assert hartlap_factor(100, 10) == pytest.approx(88.0 / 99.0)
    with pytest.raises(ValueError):
        hartlap_factor(12, 10)


def test_fisher_matrix_is_symmetric_positive_definite(setup, ensemble):
    vals, jacs, n_cov = ensemble
    j = jacs.mean(0)
    cov = np.asarray(sample_covariance(jnp.asarray(vals)))
    p_xi = setup.xi.size
    out = {}
    for name, s in (
        ("xi", slice(0, p_xi)),
        ("knn", slice(p_xi, setup.both.size)),
        ("both", slice(0, setup.both.size)),
    ):
        fc = fisher_forecast(j[s], cov[s, s], n_cov)
        f = fc.fisher
        np.testing.assert_allclose(f, f.T, rtol=0, atol=0)
        assert np.all(np.linalg.eigvalsh(f) > 0)
        assert np.all(fc.errors > 0) and np.all(np.isfinite(fc.errors))
        assert np.all(fc.errors >= fc.conditional_errors * (1 - 1e-12))
        assert abs(fc.correlation[0, 1]) < 1.0
        assert fc.hartlap == pytest.approx(
            (n_cov - (s.stop - s.start) - 2) / (n_cov - 1)
        )
        out[name] = fc
    # more data never loses information: J^T C^-1 J of the joint vector dominates
    # each sub-vector's (before the p-dependent Hartlap factors)
    raw = {k: v.fisher / v.hartlap for k, v in out.items()}
    for sub in ("xi", "knn"):
        assert np.all(
            np.linalg.eigvalsh(raw["both"] - raw[sub])
            > -1e-8 * np.abs(raw["both"]).max()
        )


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_recovery_on_own_and_independent_realisations(setup, ensemble):
    vals, jacs, n_cov = ensemble
    stat = setup.both
    cov = np.asarray(sample_covariance(jnp.asarray(vals)))
    fc = fisher_forecast(jacs.mean(0), cov, n_cov)
    prec = jnp.asarray(fc.hartlap * np.linalg.inv(cov))
    theta = jnp.asarray(THETA)
    start = jnp.asarray([1.25 * SIGMA, 0.8])
    lower, upper = jnp.asarray([0.2 * SIGMA, 0.3]), jnp.asarray([5 * SIGMA, 3.0])
    d_a = draw_ensemble(jax.random.PRNGKey(31), 1, P, M)
    d_b = draw_thomas(jax.random.PRNGKey(32), P, M)

    fit = jax.jit(
        lambda target, d: fit_parameters(
            stat, target, d, start, prec, L, n_steps=10, lower=lower, upper=upper
        )
    )
    sv = jax.jit(lambda d: stat.fn(thomas_positions(theta, d, L)))

    # its own statistic: an exact fixed point at the truth (not a test of
    # inference, a test that the optimiser and the Jacobian are right)
    own, _ = sv(jax.tree.map(lambda a: a[0], d_a))
    res = fit(own, d_a)
    assert not bool(res.overflow)
    np.testing.assert_allclose(res.theta, theta, rtol=1e-6)
    assert float(res.chi2_history[-1]) < 1e-8 * float(res.chi2_history[0])

    # an independent realisation: chi^2 never increases, the fit ends at least as
    # close to the target as the truth is (so it reached the minimum's basin, not
    # a step limit), and lands within the Fisher error (target and model are both
    # noisy: expected scatter sqrt(2) sigma_F)
    target, _ = sv(d_b)
    res = fit(target, d_a)
    assert not bool(res.overflow)
    chi2 = np.asarray(res.chi2_history)
    assert np.all(np.diff(chi2) <= 0.0)
    r_true = np.asarray(target) - np.asarray(own)
    chi2_truth = float(r_true @ np.asarray(prec) @ r_true)
    assert chi2[-1] <= chi2_truth, (chi2[-1], chi2_truth)
    pull = (np.asarray(res.theta) - np.asarray(theta)) / (np.sqrt(2.0) * fc.errors)
    assert np.all(np.abs(pull) < 4.0), pull
