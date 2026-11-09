"""Brute-force references for the differentiable kNN-CDF.

* :func:`brute_force_knn_log_distances` -- dense ``(q, n)`` distance matrix and
  ``lax.top_k`` on the live positions. ``top_k`` is differentiable in the values
  it selects, so autodiff through it is the independent reference for the
  tree path's "fixed indices, recomputed distances" gradient. ``O(q n)`` memory.
* :func:`brute_force_knn_cdf` -- the soft CDF on top of it.
* :func:`scipy_knn_distances` -- host-side ``scipy.spatial.KDTree`` distances,
  for checking the neighbour search itself. ``scipy`` is imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax.applications.knn_stats.estimator import (
    _normalise_ks,
    soft_cdf_from_log_distances,
)
from yggdrax.periodic import BoxSize, minimum_image, normalize_box_size


def brute_force_knn_log_distances(
    data: Array,
    queries: Optional[Array],
    ks: Sequence[int] | int,
    *,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """``log r_k`` from a dense distance matrix and ``top_k``, fully autodiffed.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``, or ``None`` for the data themselves
        with the diagonal (each point's own distance) excluded.
    ks
        Neighbour orders (1-based).
    box_size
        Periodic box: every query-data difference is mapped to its minimum
        image, so this is the brute-force minimum-image reference. ``None``
        for open boundaries.

    Returns
    -------
    Array
        ``log r_k(q)``, shape ``(q, len(ks))``.
    """

    ks_tuple = _normalise_ks(ks)
    exclude_self = queries is None
    queries_arr = data if queries is None else queries
    box = normalize_box_size(box_size, int(data.shape[-1]))
    delta = minimum_image(queries_arr[:, None, :] - data[None, :, :], box)
    d2 = jnp.sum(delta * delta, axis=-1)
    if exclude_self:
        n = d2.shape[0]
        d2 = d2.at[jnp.arange(n), jnp.arange(n)].set(jnp.inf)
    neg_top, _ = jax.lax.top_k(-d2, max(ks_tuple))
    columns = jnp.asarray([k - 1 for k in ks_tuple], dtype=jnp.int32)
    d2_k = jnp.maximum(-jnp.take(neg_top, columns, axis=1), jnp.finfo(d2.dtype).tiny)
    return 0.5 * jnp.log(d2_k)


def brute_force_knn_cdf(
    data: Array,
    queries: Optional[Array],
    ks: Sequence[int] | int,
    r_grid: Array,
    *,
    sharpness: float | Array,
    box_size: Optional[BoxSize] = None,
) -> Array:
    """Soft kNN-CDF from the dense ``top_k`` reference.

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``, or ``None`` for the data themselves.
    ks
        Neighbour orders (1-based).
    r_grid
        Radii, shape ``(g,)``, all ``> 0``.
    sharpness
        Inverse step width in ``log r``.
    box_size
        Periodic box (minimum image), or ``None``.

    Returns
    -------
    Array
        ``CDF_k(r)``, shape ``(len(ks), g)``.
    """

    log_r_k = brute_force_knn_log_distances(data, queries, ks, box_size=box_size)
    return soft_cdf_from_log_distances(log_r_k, jnp.asarray(r_grid), sharpness)


def scipy_knn_distances(
    data: np.ndarray,
    queries: Optional[np.ndarray],
    ks: Sequence[int] | int,
    *,
    box_size: Optional[BoxSize] = None,
) -> np.ndarray:
    """``r_k`` from ``scipy.spatial.KDTree`` (host, exact).

    Parameters
    ----------
    data
        Data points, shape ``(n, dim)``.
    queries
        Query points, shape ``(q, dim)``, or ``None`` for the data themselves
        (each point's own zero distance skipped).
    ks
        Neighbour orders (1-based).
    box_size
        Periodic box, passed to ``KDTree(boxsize=...)``; the points are
        wrapped into it first. ``None`` for open boundaries.

    Returns
    -------
    numpy.ndarray
        ``r_k(q)``, shape ``(q, len(ks))``.
    """

    from scipy.spatial import KDTree

    ks_tuple = _normalise_ks(ks)
    box = normalize_box_size(box_size, int(np.shape(data)[-1]))
    data_np = np.asarray(data, dtype=np.float64)
    query_np = data_np if queries is None else np.asarray(queries, dtype=np.float64)
    if box is not None:
        lengths = np.asarray(box)
        data_np = (
            np.mod(data_np, lengths) % lengths
        )  # mod can round up to L; again -> 0
        query_np = np.mod(query_np, lengths) % lengths
        tree = KDTree(data_np, boxsize=lengths)
    else:
        tree = KDTree(data_np)
    # A data point's nearest "neighbour" is itself at distance 0: skip it.
    shift = 1 if queries is None else 0
    dist, _ = tree.query(query_np, k=max(ks_tuple) + shift)
    dist = np.asarray(dist).reshape(query_np.shape[0], -1)
    return dist[:, [k - 1 + shift for k in ks_tuple]]


__all__ = [
    "brute_force_knn_cdf",
    "brute_force_knn_log_distances",
    "scipy_knn_distances",
]
