"""Analytic two-point functions of the Thomas process and soft-window averages.

The Thomas process with a Poisson number of parents (density ``n_p``) and a
Poisson number of children per parent, each child offset by ``N(0, sigma^2 I)``,
has

.. code-block:: text

    xi(r) = p_Delta(r) / n_p,       p_Delta(r) = exp(-r^2 / 4 sigma^2) / (4 pi sigma^2)^(3/2)

where ``p_Delta`` is the density of the separation of two children of one
parent (the difference of two offsets, ``N(0, 2 sigma^2 I)``).

**Fixed counts.** :mod:`.model` fixes the number of parents ``P`` and of children
per parent ``m`` (``N = P m``). Of the ``N(N-1)/2`` pairs, ``P m(m-1)/2`` are
pairs of siblings, distributed with ``p_Delta``; every other pair joins two
independent uniform parents' children, so its separation is uniform in the box.
The natural estimator's expectation is therefore exactly

.. code-block:: text

    xi(r) = (m - 1)/(N - 1) * (V p_Delta(r) - 1)

against ``V p_Delta(r) / P`` for the Poisson process: a factor
``(1 - 1/m) N/(N-1)`` on the amplitude (``m = 16``: -6 %) and a constant
``-(m-1)/(N-1)`` (the integral constraint of a fixed total). Both matter at the
sizes used here and are applied in the tests. Periodic images of a sibling pair
are neglected; they enter at ``exp(-(L/2)^2 / 4 sigma^2)``.

For a profile shape ``beta != 1`` (:func:`~.model.child_offsets`) ``p_Delta``
has no closed form; :func:`profile_xi_fixed_counts_mc` estimates its window
averages by Monte Carlo over sibling pairs, which is exact in expectation.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax.distributed.stats import soft_window

from .model import child_offsets

__all__ = [
    "analytic_rr",
    "profile_xi_fixed_counts_mc",
    "soft_shell_volumes",
    "thomas_separation_pdf",
    "thomas_xi_fixed_counts",
    "thomas_xi_poisson",
    "window_average",
]


def thomas_separation_pdf(r: Array, sigma: float) -> Array:
    """Density of the separation vector of two siblings, ``N(0, 2 sigma^2 I)``.

    Parameters
    ----------
    r:
        Separations.
    sigma:
        Offset width per axis.

    Returns
    -------
    Array
        ``exp(-r^2 / 4 sigma^2) / (4 pi sigma^2)^(3/2)`` (a 3-D density).
    """
    r = jnp.asarray(r)
    return (
        jnp.exp(-(r * r) / (4.0 * sigma * sigma))
        / (4.0 * math.pi * sigma * sigma) ** 1.5
    )


def thomas_xi_poisson(r: Array, sigma: float, parent_density: float) -> Array:
    """``xi(r)`` of the Thomas process with Poisson counts.

    Parameters
    ----------
    r:
        Separations.
    sigma:
        Offset width per axis.
    parent_density:
        Mean parents per unit volume, ``n_p``.

    Returns
    -------
    Array
        ``exp(-r^2/4 sigma^2) / (n_p (4 pi sigma^2)^(3/2))``.
    """
    return thomas_separation_pdf(r, sigma) / parent_density


def thomas_xi_fixed_counts(
    r: Array, sigma: float, n_parents: int, n_children: int, volume: float
) -> Array:
    """``xi(r)`` of the Thomas process with fixed counts (the natural estimator's mean).

    Parameters
    ----------
    r:
        Separations.
    sigma:
        Offset width per axis.
    n_parents:
        Parents ``P``.
    n_children:
        Children per parent ``m``.
    volume:
        Box volume ``V``.

    Returns
    -------
    Array
        ``(m - 1)/(N - 1) (V p_Delta(r) - 1)`` with ``N = P m``.
    """
    n = n_parents * n_children
    frac = (n_children - 1) / (n - 1)
    return frac * (volume * thomas_separation_pdf(r, sigma) - 1.0)


def _quadrature(r_cut: tuple[float, float], log: bool, n_quad: int):
    r_lo, r_hi = r_cut
    if log:
        u = np.linspace(math.log(r_lo), math.log(r_hi), n_quad)
        r = np.exp(u)
        jac = r  # dr = r du
        x = u
    else:
        r = np.linspace(r_lo, r_hi, n_quad)
        jac = np.ones_like(r)
        x = r
    # composite Simpson weights on the uniform grid (n_quad odd)
    h = (x[-1] - x[0]) / (n_quad - 1)
    w = np.ones(n_quad)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return r, w * h / 3.0 * jac


def soft_shell_volumes(
    edges: np.ndarray,
    sharpness: float,
    r_cut: tuple[float, float],
    log: bool = True,
    n_quad: int = 40001,
) -> np.ndarray:
    """``V_b = integral over [r_lo, r_hi) of W_b(r) 4 pi r^2 dr`` (soft shell volumes).

    Parameters
    ----------
    edges:
        ``(nbins + 1,)`` bin edges.
    sharpness:
        Soft-window sharpness (:func:`~yggdrax.distributed.stats.soft_window`).
    r_cut:
        The statistic's hard cut ``(r_lo, r_hi)``.
    log:
        Windows in ``ln r``.
    n_quad:
        Simpson nodes (odd); the window is smooth on ``1/sharpness``.

    Returns
    -------
    numpy.ndarray
        ``(nbins,)`` volumes.
    """
    r, w = _quadrature(r_cut, log, n_quad | 1)
    win = np.asarray(soft_window(jnp.asarray(r), jnp.asarray(edges), sharpness, log))
    return np.einsum("q,qb->b", w * 4.0 * math.pi * r * r, win)


def analytic_rr(
    edges: np.ndarray,
    sharpness: float,
    r_cut: tuple[float, float],
    n_points: int,
    volume: float,
    log: bool = True,
) -> np.ndarray:
    """Expected soft pair counts of ``n_points`` uniform points in a periodic box.

    Parameters
    ----------
    edges:
        ``(nbins + 1,)`` bin edges.
    sharpness:
        Window sharpness.
    r_cut:
        Hard cut ``(r_lo, r_hi)``, below ``L/2``.
    n_points:
        Points ``N``.
    volume:
        Box volume ``V``.
    log:
        Windows in ``ln r``.

    Returns
    -------
    numpy.ndarray
        ``RR_b = N(N-1)/2 * V_b / V`` (exact for a periodic box: the
        minimum-image separation of two uniform points is uniform in the ball of
        radius ``L/2``).
    """
    vb = soft_shell_volumes(edges, sharpness, r_cut, log)
    return 0.5 * n_points * (n_points - 1) * vb / volume


def window_average(
    xi_fn: Callable[[np.ndarray], np.ndarray],
    edges: np.ndarray,
    sharpness: float,
    r_cut: tuple[float, float],
    log: bool = True,
    n_quad: int = 40001,
) -> np.ndarray:
    """``integral W_b xi dV / integral W_b dV``: what the soft natural estimator measures.

    Parameters
    ----------
    xi_fn:
        ``xi(r)`` on an array of separations.
    edges:
        ``(nbins + 1,)`` bin edges.
    sharpness:
        Window sharpness.
    r_cut:
        Hard cut ``(r_lo, r_hi)``.
    log:
        Windows in ``ln r``.
    n_quad:
        Simpson nodes.

    Returns
    -------
    numpy.ndarray
        ``(nbins,)`` window-averaged ``xi``.
    """
    r, w = _quadrature(r_cut, log, n_quad | 1)
    win = np.asarray(soft_window(jnp.asarray(r), jnp.asarray(edges), sharpness, log))
    dv = w * 4.0 * math.pi * r * r
    num = np.einsum("q,q,qb->b", dv, np.asarray(xi_fn(r)), win)
    return num / np.einsum("q,qb->b", dv, win)


def profile_xi_fixed_counts_mc(
    key: Array,
    theta: Array,
    edges: np.ndarray,
    sharpness: float,
    r_cut: tuple[float, float],
    n_parents: int,
    n_children: int,
    volume: float,
    n_samples: int = 1 << 20,
    log: bool = True,
) -> np.ndarray:
    """Window-averaged fixed-count ``xi`` for any profile shape, by Monte Carlo.

    Draws ``n_samples`` sibling pairs, two independent offsets
    (:func:`~.model.child_offsets`) each, and averages the windows of their
    separation: ``E[W_b(|o_1 - o_2|) 1_cut] = integral W_b p_Delta dV``.

    Parameters
    ----------
    key:
        PRNG key.
    theta:
        ``(sigma, beta)``.
    edges:
        ``(nbins + 1,)`` bin edges.
    sharpness:
        Window sharpness.
    r_cut:
        Hard cut ``(r_lo, r_hi)``.
    n_parents:
        Parents ``P``.
    n_children:
        Children per parent ``m``.
    volume:
        Box volume ``V``.
    n_samples:
        Sibling pairs drawn.
    log:
        Windows in ``ln r``.

    Returns
    -------
    numpy.ndarray
        ``(nbins,)`` window averages of ``(m-1)/(N-1) (V p_Delta - 1)``; the
        Monte Carlo error is ``O(n_samples^-1/2)`` relative per bin.
    """
    theta = jnp.asarray(theta, jnp.float64)
    z = jax.random.normal(key, (2, n_samples, 3), dtype=jnp.float64)
    d = child_offsets(theta, z[0]) - child_offsets(theta, z[1])
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    inside = (r >= r_cut[0]) & (r < r_cut[1])
    win = soft_window(jnp.maximum(r, 1e-300), jnp.asarray(edges), sharpness, log)
    mean_w = np.asarray(jnp.mean(jnp.where(inside[:, None], win, 0.0), axis=0))
    vb = soft_shell_volumes(edges, sharpness, r_cut, log)
    n = n_parents * n_children
    frac = (n_children - 1) / (n - 1)
    return frac * (volume * mean_w / vb - 1.0)
