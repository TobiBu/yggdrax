"""Soft (smooth) radial bin-membership windows for differentiable pair counts.

A classical pair-counting correlation-function estimator sorts each pair into a
hard radial bin: the per-bin count is a sum of indicator functions
``1[r_k <= r < r_{k+1}]``. That indicator has zero gradient almost everywhere
and a non-differentiable jump at the edge, so the resulting histogram cannot be
differentiated with respect to particle positions or bin edges.

We replace the hard indicator with a smooth window built from a difference of
logistic sigmoids,

    w_k(r) = sigmoid(s * (u - a_k)) - sigmoid(s * (u - b_k)),

where ``[a_k, b_k)`` are the (optionally log-transformed) edges of bin ``k``,
``u`` is ``r`` (or ``log r`` in log-space), and ``s`` is a sharpness (inverse
transition width). ``w_k`` is smooth in ``r``, the edges, and ``s``; as
``s -> inf`` it converges pointwise to the hard indicator. Because adjacent
bins share an edge (``b_k == a_{k+1}``), the shared sigmoid terms cancel and the
windows form an approximate partition of unity across the binned range.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def make_log_edges(
    r_min: float, r_max: float, num_bins: int
) -> Float[Array, " nbins+1"]:
    """Return ``num_bins + 1`` logarithmically spaced radial bin edges.

    Args:
        r_min: Smallest bin edge (must be > 0).
        r_max: Largest bin edge.
        num_bins: Number of bins.

    Returns:
        Array of ``num_bins + 1`` edges, geometrically spaced.
    """
    return jnp.geomspace(r_min, r_max, num_bins + 1)


def make_linear_edges(
    r_min: float, r_max: float, num_bins: int
) -> Float[Array, " nbins+1"]:
    """Return ``num_bins + 1`` linearly spaced radial bin edges.

    Args:
        r_min: Smallest bin edge.
        r_max: Largest bin edge.
        num_bins: Number of bins.

    Returns:
        Array of ``num_bins + 1`` edges, linearly spaced.
    """
    return jnp.linspace(r_min, r_max, num_bins + 1)


def _transform(values: Array, log: bool) -> Array:
    """Map separations/edges to the coordinate the window operates on."""
    if log:
        return jnp.log(jnp.clip(values, min=1e-30))
    return values


def soft_bin_weights(
    separations: Float[Array, " ..."],
    edges: Float[Array, " nbins+1"],
    sharpness: float,
    *,
    log: bool = True,
) -> Float[Array, " ... nbins"]:
    """Soft bin-membership weights for each separation and bin.

    Args:
        separations: Pair separations, arbitrary leading shape ``...``.
        edges: Monotonically increasing bin edges, shape ``(nbins + 1,)``.
        sharpness: Inverse transition width ``s`` of the logistic window.
            Larger is sharper; the hard-bin limit is ``s -> inf``. In log-space
            (``log=True``) this acts on ``log r``, so it is a dimensionless
            steepness in dex.
        log: If True, bin in ``log(separation)`` (constant fractional width);
            otherwise bin linearly.

    Returns:
        Weights with shape ``separations.shape + (nbins,)``; entry ``[..., k]``
        is the soft membership of that separation in bin ``k``.
    """
    u = _transform(separations, log)[..., None]  # (..., 1)
    edge_u = _transform(edges, log)  # (nbins+1,)
    gate = jax.nn.sigmoid(sharpness * (u - edge_u))  # (..., nbins+1)
    return gate[..., :-1] - gate[..., 1:]  # (..., nbins)


def hard_bin_weights(
    separations: Float[Array, " ..."],
    edges: Float[Array, " nbins+1"],
    *,
    log: bool = True,
) -> Float[Array, " ... nbins"]:
    """Hard (non-differentiable) bin membership, for reference and tests.

    Bin ``k`` covers ``[edges[k], edges[k+1])``. Used to check that
    :func:`soft_bin_weights` converges to this as ``sharpness -> inf``.

    Args:
        separations: Pair separations, arbitrary leading shape ``...``.
        edges: Monotonically increasing bin edges, shape ``(nbins + 1,)``.
        log: If True, compare in ``log(separation)`` (matches
            :func:`soft_bin_weights`); otherwise compare linearly. The result
            is identical either way since the transform is monotonic, but the
            flag is accepted for a symmetric call signature.

    Returns:
        0/1 weights with shape ``separations.shape + (nbins,)``.
    """
    u = _transform(separations, log)[..., None]
    edge_u = _transform(edges, log)
    lower = edge_u[:-1]
    upper = edge_u[1:]
    return ((u >= lower) & (u < upper)).astype(separations.dtype)
