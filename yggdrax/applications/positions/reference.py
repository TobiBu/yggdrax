"""Dense ``O(N^2)`` references of the matching statistics, for validation.

:func:`dense_statistics` computes what :meth:`PositionMatcher.statistics
<yggdrax.applications.positions.matching.PositionMatcher.statistics>` computes,
without any tree: every minimum-image pair for ``DD``, and every particle
sorted by distance for the ``k``-th neighbours. ``jax.grad`` through it is the
exact gradient of the estimator, the reference for the tree's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from yggdrax.applications.corrfunc.binning import soft_bin_weights
from yggdrax.applications.knn_stats.estimator import soft_cdf_from_log_distances
from yggdrax.periodic import minimum_image

from .matching import MatchStatistics, PositionMatcher

__all__ = ["dense_statistics"]


def _dense_log_r(
    data: Array, queries: Array, ks: tuple[int, ...], box: float, self_query: bool
) -> Array:
    delta = minimum_image(queries[:, None, :] - data[None, :, :], box)
    d2 = jnp.sum(delta * delta, axis=-1)
    if self_query:
        d2 = jnp.where(jnp.eye(d2.shape[0], dtype=bool), jnp.inf, d2)
    d2 = jnp.maximum(d2, jnp.finfo(d2.dtype).tiny)
    log_r = 0.5 * jnp.log(jnp.sort(d2, axis=1)[:, : max(ks)])
    return log_r[:, jnp.asarray([k - 1 for k in ks], dtype=jnp.int32)]


def dense_statistics(matcher: PositionMatcher, positions: Array) -> MatchStatistics:
    """Brute-force ξ counts and kNN-CDFs with ``matcher``'s spec.

    Parameters
    ----------
    matcher:
        Supplies the spec, ``RR`` and the random queries.
    positions:
        ``(N, 3)`` catalogue; differentiable. ``N`` up to a few thousand.

    Returns
    -------
    MatchStatistics
        ``overflow`` is always false.
    """
    spec = matcher.spec
    dtype = positions.dtype
    n = positions.shape[0]
    delta = minimum_image(positions[:, None, :] - positions[None, :, :], spec.box_size)
    d2 = jnp.sum(delta * delta, axis=-1)
    iu = jnp.triu_indices(n, k=1)
    r = jnp.sqrt(d2[iu])
    w = soft_bin_weights(r, jnp.asarray(spec.xi_edges, dtype), spec.xi_sharpness)
    counts = jnp.sum(w, axis=0)
    xi = counts / jnp.asarray(matcher.rr, dtype) - 1.0
    r_grid = jnp.asarray(spec.knn_r_grid, dtype)
    cdfs = []
    if spec.use_randoms:
        assert matcher.randoms is not None
        q = jax.lax.stop_gradient(matcher.randoms.astype(dtype))
        lr = _dense_log_r(positions, q, spec.knn_ks, spec.box_size, False)
        cdfs.append(soft_cdf_from_log_distances(lr, r_grid, spec.knn_sharpness))
    if spec.use_data:
        lr = _dense_log_r(positions, positions, spec.knn_ks, spec.box_size, True)
        cdfs.append(soft_cdf_from_log_distances(lr, r_grid, spec.knn_sharpness))
    return MatchStatistics(counts, xi, jnp.stack(cdfs), jnp.asarray(False))
