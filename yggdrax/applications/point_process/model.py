"""A reparameterised Thomas (Neyman–Scott) point process in a periodic box.

The forward model of the point-process demonstration. ``n_parents`` parents are
uniform in ``[0, L)^3``. Each has exactly ``n_children`` children, offset from it
by an isotropic random vector, and the children are wrapped into the box.

Reparameterisation
------------------
The positions are a smooth function of the parameters at fixed random draws:

.. code-block:: text

    parents  p_a   ~ U[0, L)^3                (fixed)
    normals  z_ai  ~ N(0, I_3)                (fixed)
    x_ai = wrap( p_a + sigma * c(beta) * |z_ai|^(beta - 1) * z_ai )

Every random number is drawn once, from a key; the parameters enter only through
the offset map. So ``jax.grad`` of anything computed from ``x`` with respect to
``(sigma, beta)`` is the pathwise (reparameterisation) derivative, and
``E[grad S] = grad E[S]`` for any statistic ``S``.

The discrete parts are conditioned on: the number of parents and the number of
children per parent are fixed per realisation, not Poisson. This changes the
two-point function by a known factor (:mod:`.analytic`).

Parameters
----------
``sigma``
    The RMS size of a cluster per axis: ``E|offset|^2 = 3 sigma^2`` for every
    ``beta``.
``beta``
    The profile shape at fixed RMS size. The offset radius is
    ``r = sigma c(beta) rho^beta`` with ``rho = |z| ~ chi_3``, a monotone map of
    the Gaussian radius, i.e. a smooth inverse-CDF reparameterisation of the
    radial distribution. The cluster's density profile is

    .. code-block:: text

        n(r) ∝ r^(3/beta - 3) exp(-(r / (c sigma))^(2/beta) / 2)

    ``beta = 1`` is the Gaussian (the Thomas process). ``beta > 1`` gives a
    central cusp of slope ``3 - 3/beta`` (``beta = 1.5`` is the NFW inner slope
    ``r^-1``) and a stretched-exponential tail; ``beta < 1`` a cored, sharper-
    edged cluster. It plays the role of the halo concentration in the halo
    model: at fixed size it moves pairs between small and intermediate
    separations, so it changes the shape of ``xi(r)`` rather than its
    amplitude, and it moves the local density that kNN-CDFs measure.

``c(beta) = sqrt(3 / E[rho^(2 beta)])`` with
``E[rho^(2 beta)] = 2^beta Gamma(3/2 + beta) / Gamma(3/2)`` keeps the RMS size at
``sigma``, so the two parameters are as orthogonal as a one-line definition
makes them.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array

from yggdrax.periodic import wrap_positions

__all__ = [
    "PARAM_NAMES",
    "ThomasDraws",
    "child_offsets",
    "draw_ensemble",
    "draw_thomas",
    "profile_norm",
    "thomas_positions",
]

#: Order of the entries of the parameter vector ``theta``.
PARAM_NAMES: tuple[str, str] = ("sigma", "beta")


class ThomasDraws(NamedTuple):
    """The fixed random numbers of one realisation.

    Attributes
    ----------
    parents:
        ``(n_parents, 3)`` parent positions in units of the box, ``U[0, 1)``.
    normals:
        ``(n_parents, n_children, 3)`` standard-normal offsets.
    """

    parents: Array
    normals: Array


def draw_thomas(
    key: Array, n_parents: int, n_children: int, dtype: jnp.dtype = jnp.float64
) -> ThomasDraws:
    """Draw the random numbers of one realisation.

    Parameters
    ----------
    key:
        PRNG key; the realisation's seed.
    n_parents:
        Parents, fixed.
    n_children:
        Children per parent, fixed.
    dtype:
        Floating dtype of the draws.

    Returns
    -------
    ThomasDraws
        Uniform parents (in box units) and standard-normal offsets.
    """
    k_p, k_z = jax.random.split(key)
    parents = jax.random.uniform(k_p, (n_parents, 3), dtype=dtype)
    normals = jax.random.normal(k_z, (n_parents, n_children, 3), dtype=dtype)
    return ThomasDraws(parents=parents, normals=normals)


def draw_ensemble(
    key: Array,
    n_realisations: int,
    n_parents: int,
    n_children: int,
    dtype: jnp.dtype = jnp.float64,
) -> ThomasDraws:
    """Draw ``n_realisations`` independent realisations.

    Parameters
    ----------
    key:
        PRNG key of the ensemble; realisation ``i`` uses ``split(key, n)[i]``.
    n_realisations:
        Ensemble size.
    n_parents:
        Parents per realisation.
    n_children:
        Children per parent.
    dtype:
        Floating dtype.

    Returns
    -------
    ThomasDraws
        Batched draws, with a leading axis of length ``n_realisations``.
    """
    keys = jax.random.split(key, n_realisations)
    return jax.vmap(lambda k: draw_thomas(k, n_parents, n_children, dtype))(keys)


def profile_norm(beta: Array | float) -> Array:
    """``c(beta) = sqrt(3 / E[rho^(2 beta)])`` for ``rho ~ chi_3``.

    Parameters
    ----------
    beta:
        Profile shape.

    Returns
    -------
    Array
        The normalisation that keeps ``E|offset|^2 = 3 sigma^2``; ``c(1) = 1``.
    """
    beta = jnp.asarray(beta)
    log_moment = beta * jnp.log(2.0) + gammaln(1.5 + beta) - gammaln(1.5)
    return jnp.sqrt(3.0 * jnp.exp(-log_moment))


def child_offsets(theta: Array, normals: Array) -> Array:
    """Offsets of the children from their parents.

    Parameters
    ----------
    theta:
        ``(2,)`` parameters ``(sigma, beta)``.
    normals:
        ``(..., 3)`` standard-normal draws.

    Returns
    -------
    Array
        ``sigma c(beta) |z|^(beta - 1) z``, same shape as ``normals``. At
        ``beta = 1`` exactly ``sigma z``.
    """
    sigma, beta = theta[0], theta[1]
    rho2 = jnp.sum(normals * normals, axis=-1, keepdims=True)
    # |z|^(beta-1) = exp((beta-1)/2 ln|z|^2); the floor only guards z = 0 exactly
    scale = jnp.exp(0.5 * (beta - 1.0) * jnp.log(jnp.maximum(rho2, 1e-300)))
    return sigma * profile_norm(beta) * scale * normals


def thomas_positions(theta: Array, draws: ThomasDraws, box_size: float) -> Array:
    """Positions of one realisation at parameters ``theta``.

    Parameters
    ----------
    theta:
        ``(2,)`` parameters ``(sigma, beta)``; ``sigma`` in box-length units of
        ``box_size`` (the same units as the positions).
    draws:
        One realisation's fixed draws (:func:`draw_thomas`).
    box_size:
        Periodic box length ``L``.

    Returns
    -------
    Array
        ``(n_parents * n_children, 3)`` positions in ``[0, L)^3``, children of
        one parent contiguous. Differentiable in ``theta`` (the wrap is the
        identity's derivative away from the faces, a set of measure zero).
    """
    theta = jnp.asarray(theta, dtype=draws.normals.dtype)
    parents = box_size * draws.parents
    x = parents[:, None, :] + child_offsets(theta, draws.normals)
    return wrap_positions(x.reshape(-1, 3), box_size)
