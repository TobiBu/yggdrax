"""Jacobians, Fisher forecasts and fits through the point-process forward model.

The chain is ``theta -> positions (fixed draws) -> statistic -> loss``. Every
link is JAX: the tree, walk and halo are rebuilt from the positions inside each
call (topology carries no cotangent), and the statistic's values are smooth in
the positions. So ``jax.jacfwd`` gives ``dS/dtheta`` per realisation, ``vmap``
(or a batched ``lax.map``) runs an ensemble of realisations, and a fit is a
jitted ``lax.scan`` of Levenberg–Marquardt steps.

* :func:`statistic_jacobian`, :func:`central_difference_jacobian`: the autodiff
  Jacobian of one realisation and the finite-difference check of the same chain.
* :func:`ensemble_statistics`, :func:`ensemble_jacobians`: over a batch of draws.
* :func:`fisher_forecast`: ``F = h J^T C^-1 J`` with the Hartlap factor ``h``.
* :func:`fit_parameters`: Levenberg–Marquardt fit of one set of draws (one
  realisation, or the mean of several) to a target data vector.

Because the draws are fixed, ``d/dtheta E[S] = E[dS/dtheta]``: the mean of the
per-realisation Jacobians is an unbiased estimate of the Jacobian of the mean,
which is what the Fisher matrix needs.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .model import ThomasDraws, thomas_positions
from .statistics import Statistic

__all__ = [
    "FisherForecast",
    "FitResult",
    "central_difference_jacobian",
    "ensemble_jacobians",
    "ensemble_statistics",
    "fisher_forecast",
    "fit_parameters",
    "forward",
    "hartlap_factor",
    "sample_covariance",
    "statistic_jacobian",
]


def forward(
    stat: Statistic, box_size: float
) -> Callable[[Array, ThomasDraws], tuple[Array, Array]]:
    """The full chain ``(theta, draws) -> (statistic, overflow)``.

    Parameters
    ----------
    stat:
        The statistic.
    box_size:
        Periodic box length.

    Returns
    -------
    callable
        ``f(theta, draws)``; traceable, differentiable in ``theta``.
    """

    def f(theta: Array, draws: ThomasDraws) -> tuple[Array, Array]:
        return stat.fn(thomas_positions(theta, draws, box_size))

    return f


def statistic_jacobian(
    stat: Statistic, theta: Array, draws: ThomasDraws, box_size: float
) -> tuple[Array, Array, Array]:
    """Statistic and its forward-mode Jacobian for one realisation.

    Parameters
    ----------
    stat:
        The statistic.
    theta:
        ``(n_params,)`` parameters.
    draws:
        One realisation's draws.
    box_size:
        Periodic box length.

    Returns
    -------
    tuple of Array
        ``(values (p,), jacobian (p, n_params), overflow)``.
    """
    f = forward(stat, box_size)

    def g(t: Array) -> tuple[Array, tuple[Array, Array]]:
        v, o = f(t, draws)
        return v, (v, o)

    jac, (values, overflow) = jax.jacfwd(g, has_aux=True)(jnp.asarray(theta))
    return values, jac, overflow


def central_difference_jacobian(
    stat: Statistic,
    theta: Array,
    draws: ThomasDraws,
    box_size: float,
    steps: Array,
) -> tuple[Array, Array]:
    """Central finite differences of the full chain, tree rebuild included.

    Parameters
    ----------
    stat:
        The statistic.
    theta:
        ``(n_params,)`` parameters.
    draws:
        One realisation's draws (the same at every evaluation).
    box_size:
        Periodic box length.
    steps:
        ``(n_params,)`` absolute steps ``h_i``.

    Returns
    -------
    tuple of Array
        ``(jacobian (p, n_params), overflow)``; column ``i`` is
        ``(S(theta + h_i e_i) - S(theta - h_i e_i)) / 2 h_i``.
    """
    f = forward(stat, box_size)
    theta = jnp.asarray(theta)
    steps = jnp.asarray(steps, theta.dtype)
    eye = jnp.eye(theta.shape[0], dtype=theta.dtype)

    def column(i: Array) -> tuple[Array, Array]:
        h = steps[i]
        vp, op = f(theta + h * eye[i], draws)
        vm, om = f(theta - h * eye[i], draws)
        return (vp - vm) / (2.0 * h), op | om

    cols, over = jax.lax.map(column, jnp.arange(theta.shape[0]))
    return cols.T, jnp.any(over)


def _ensemble_map(fn: Callable, draws: ThomasDraws, batch_size: Optional[int]):
    n = int(draws.parents.shape[0])
    if batch_size is None or batch_size >= n:
        return jax.vmap(fn)(draws)
    return jax.lax.map(fn, draws, batch_size=batch_size)


def ensemble_statistics(
    stat: Statistic,
    theta: Array,
    draws: ThomasDraws,
    box_size: float,
    batch_size: Optional[int] = None,
) -> tuple[Array, Array]:
    """The statistic of every realisation of a batch of draws.

    Parameters
    ----------
    stat:
        The statistic.
    theta:
        ``(n_params,)`` parameters, shared.
    draws:
        Batched draws (leading axis ``n``, :func:`~.model.draw_ensemble`).
    box_size:
        Periodic box length.
    batch_size:
        Realisations per ``vmap`` batch; ``None`` vmaps all at once. Smaller
        batches bound the memory (``lax.map(..., batch_size=...)``).

    Returns
    -------
    tuple of Array
        ``(values (n, p), overflow (n,))``.
    """
    f = forward(stat, box_size)
    theta = jnp.asarray(theta)
    return _ensemble_map(lambda d: f(theta, d), draws, batch_size)


def ensemble_jacobians(
    stat: Statistic,
    theta: Array,
    draws: ThomasDraws,
    box_size: float,
    batch_size: Optional[int] = None,
) -> tuple[Array, Array, Array]:
    """Statistic and autodiff Jacobian of every realisation of a batch.

    Parameters
    ----------
    stat:
        The statistic.
    theta:
        ``(n_params,)`` parameters, shared.
    draws:
        Batched draws (leading axis ``n``).
    box_size:
        Periodic box length.
    batch_size:
        As :func:`ensemble_statistics`.

    Returns
    -------
    tuple of Array
        ``(values (n, p), jacobians (n, p, n_params), overflow (n,))``.
    """
    theta = jnp.asarray(theta)
    return _ensemble_map(
        lambda d: statistic_jacobian(stat, theta, d, box_size), draws, batch_size
    )


def sample_covariance(values: Array) -> Array:
    """Unbiased sample covariance of an ensemble of data vectors.

    Parameters
    ----------
    values:
        ``(n, p)`` data vectors.

    Returns
    -------
    Array
        ``(p, p)`` covariance with the ``1/(n - 1)`` normalisation.
    """
    x = values - jnp.mean(values, axis=0)
    return x.T @ x / (values.shape[0] - 1)


def hartlap_factor(n_realisations: int, n_data: int) -> float:
    """Hartlap, Simon & Schneider (2007) debiasing factor of an inverted sample covariance.

    Parameters
    ----------
    n_realisations:
        Realisations ``n`` behind the covariance.
    n_data:
        Data-vector length ``p``.

    Returns
    -------
    float
        ``(n - p - 2) / (n - 1)``; ``h C^-1`` is unbiased for Gaussian data.

    Raises
    ------
    ValueError
        If ``n <= p + 2`` (the inverse has no finite mean).
    """
    if n_realisations <= n_data + 2:
        raise ValueError(
            f"need more than p + 2 = {n_data + 2} realisations, got {n_realisations}"
        )
    return (n_realisations - n_data - 2) / (n_realisations - 1)


class FisherForecast(NamedTuple):
    """A Fisher forecast and its ingredients.

    Attributes
    ----------
    fisher:
        ``(n_params, n_params)`` Fisher matrix ``h J^T C^-1 J``.
    param_covariance:
        ``F^-1``, the forecast parameter covariance.
    errors:
        ``sqrt(diag F^-1)``, marginalised 1-sigma errors.
    conditional_errors:
        ``1 / sqrt(diag F)``, the error on each parameter with the other fixed.
    correlation:
        Parameter correlation matrix.
    hartlap:
        The factor ``h`` that was applied.
    """

    fisher: np.ndarray
    param_covariance: np.ndarray
    errors: np.ndarray
    conditional_errors: np.ndarray
    correlation: np.ndarray
    hartlap: float


def fisher_forecast(
    jacobian: Array | np.ndarray,
    data_covariance: Array | np.ndarray,
    n_realisations: int,
) -> FisherForecast:
    """Gaussian-likelihood Fisher matrix from a Jacobian and a sample covariance.

    Parameters
    ----------
    jacobian:
        ``(p, n_params)`` ``d<S>/dtheta`` (e.g. the ensemble mean of the
        autodiff Jacobians).
    data_covariance:
        ``(p, p)`` sample covariance of ``S`` over ``n_realisations``.
    n_realisations:
        Realisations behind the covariance (for the Hartlap factor).

    Returns
    -------
    FisherForecast
        The forecast. The parameter-independent covariance approximation is
        used (no ``dC/dtheta`` term).
    """
    j = np.asarray(jacobian, dtype=np.float64)
    c = np.asarray(data_covariance, dtype=np.float64)
    h = hartlap_factor(n_realisations, c.shape[0])
    f = h * j.T @ np.linalg.solve(c, j)
    f = 0.5 * (f + f.T)
    cov = np.linalg.inv(f)
    err = np.sqrt(np.diag(cov))
    return FisherForecast(
        fisher=f,
        param_covariance=cov,
        errors=err,
        conditional_errors=1.0 / np.sqrt(np.diag(f)),
        correlation=cov / np.outer(err, err),
        hartlap=h,
    )


class FitResult(NamedTuple):
    """Trajectory of a Levenberg–Marquardt fit.

    Attributes
    ----------
    theta:
        ``(n_params,)`` final (best) parameters.
    theta_history:
        ``(n_steps + 1, n_params)`` accepted point after each step, starting
        point first.
    chi2_history:
        ``(n_steps + 1,)`` ``chi^2`` at each point of ``theta_history``
        (non-increasing).
    damping_history:
        ``(n_steps + 1,)`` Levenberg–Marquardt ``lambda`` after each step.
    overflow:
        True if any evaluation overflowed a capacity.
    """

    theta: Array
    theta_history: Array
    chi2_history: Array
    damping_history: Array
    overflow: Array


def fit_parameters(
    stat: Statistic,
    target: Array,
    draws: ThomasDraws,
    theta0: Array,
    precision: Array,
    box_size: float,
    *,
    n_steps: int = 10,
    damping: float = 1e-2,
    damping_up: float = 4.0,
    damping_down: float = 3.0,
    max_rel_step: float = 0.3,
    lower: Optional[Array] = None,
    upper: Optional[Array] = None,
    batch_size: Optional[int] = None,
) -> FitResult:
    """Fit the parameters of a set of draws to a target data vector.

    Minimises ``chi^2(theta) = r^T W r`` with ``r = target - mean_i S(theta,
    draws_i)`` by Levenberg–Marquardt: the step solves
    ``(J^T W J + lambda diag(J^T W J)) delta = J^T W r`` with ``J`` the autodiff
    Jacobian, i.e. the gradient of ``chi^2`` preconditioned by its damped
    Gauss–Newton Hessian. A step that lowers ``chi^2`` is accepted and
    ``lambda`` shrinks; otherwise it is rejected and ``lambda`` grows. Each step
    costs one forward-mode Jacobian evaluation (at the proposed point, reused
    when it is accepted). The loop is a ``lax.scan``, so the whole fit traces
    and jits as one program, with the trees rebuilt at every evaluation.

    Parameters
    ----------
    stat:
        The statistic.
    target:
        ``(p,)`` data vector to fit (e.g. of an independent realisation).
    draws:
        Batched draws of the model, leading axis ``m >= 1``; the model is the
        mean statistic over them.
    theta0:
        ``(n_params,)`` starting point.
    precision:
        ``(p, p)`` weight ``W`` (an inverse covariance).
    box_size:
        Periodic box length.
    n_steps:
        Levenberg–Marquardt steps (accepted or rejected).
    damping:
        Initial ``lambda``.
    damping_up:
        Factor on ``lambda`` after a rejected step.
    damping_down:
        Divisor of ``lambda`` after an accepted step.
    max_rel_step:
        Each step is clipped to ``max_rel_step * |theta|`` per parameter.
    lower:
        Optional ``(n_params,)`` lower bounds (a step is clipped to them).
    upper:
        Optional ``(n_params,)`` upper bounds.
    batch_size:
        Realisations per ``vmap`` batch of the model mean.

    Returns
    -------
    FitResult
        The trajectory.
    """
    theta0 = jnp.asarray(theta0)
    target = jnp.asarray(target)
    w = jnp.asarray(precision)
    lo = (
        jnp.full_like(theta0, -jnp.inf)
        if lower is None
        else jnp.asarray(lower, theta0.dtype)
    )
    hi = (
        jnp.full_like(theta0, jnp.inf)
        if upper is None
        else jnp.asarray(upper, theta0.dtype)
    )

    def model(theta: Array) -> tuple[Array, Array, Array]:
        v, j, o = _ensemble_map(
            lambda d: statistic_jacobian(stat, theta, d, box_size), draws, batch_size
        )
        return jnp.mean(v, axis=0), jnp.mean(j, axis=0), jnp.any(o)

    def chi2_of(v: Array) -> Array:
        r = target - v
        return r @ w @ r

    v0, j0, o0 = model(theta0)
    lam0 = jnp.asarray(damping, theta0.dtype)

    def step(carry, _):
        theta, v, j, chi2, lam, over = carry
        r = target - v
        jtw = j.T @ w
        a = jtw @ j
        delta = jnp.linalg.solve(a + lam * jnp.diag(jnp.diag(a)), jtw @ r)
        lim = max_rel_step * jnp.abs(theta)
        trial = jnp.clip(theta + jnp.clip(delta, -lim, lim), lo, hi)
        v_t, j_t, o_t = model(trial)
        chi2_t = chi2_of(v_t)
        ok = chi2_t < chi2

        def pick(new, old):
            return jnp.where(ok, new, old)

        carry = (
            pick(trial, theta),
            pick(v_t, v),
            pick(j_t, j),
            pick(chi2_t, chi2),
            jnp.where(ok, lam / damping_down, lam * damping_up),
            over | o_t,
        )
        return carry, (carry[0], carry[3], carry[4])

    init = (theta0, v0, j0, chi2_of(v0), lam0, o0)
    (theta_f, _, _, _, _, over), (hist, chi2, lams) = jax.lax.scan(
        step, init, None, length=n_steps
    )
    return FitResult(
        theta=theta_f,
        theta_history=jnp.concatenate([theta0[None], hist]),
        chi2_history=jnp.concatenate([init[3][None], chi2]),
        damping_history=jnp.concatenate([lam0[None], lams]),
        overflow=over,
    )
