"""Pluggable clustering statistics for the point-process forward model.

A statistic is a :class:`Statistic`: a function ``positions -> (values,
overflow)`` plus a label per value. Everything downstream (Jacobians, Fisher
forecasts, fits) sees only that function, so another estimator drops in by
writing one more builder. Two are provided (periodic box):

* :func:`make_xi_statistic`: ``xi`` from the natural estimator ``DD/RR - 1``
  with soft binned ``DD`` and the analytic soft ``RR`` of the periodic box
  (:func:`natural_estimator_rr`). ``DD`` comes from
  :func:`~yggdrax.distributed.stats.distributed_pair_counts` (any number of
  devices, 1 included; ``backend="distributed"``) or from the single-device
  traced estimator with the bin-aware window rule
  (:func:`~yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`,
  ``backend="traced"``), which also takes the smooth acceptance rule and
  ``membership=``.
* :func:`make_knn_cdf_statistic`: soft kNN-CDFs of volume-filling query points
  (:func:`~yggdrax.distributed.stats.distributed_knn`), the fraction of queries
  whose ``k``-th nearest data point lies within ``r``.

Swapping in later estimators (the ``knn_stats`` application, say) means
wrapping them in a :class:`Statistic` with the same contract; nothing else
changes.

The ``overflow`` flag is the pipeline's "a static capacity was too small"
signal. It must be read: a truncated halo or walk loses pairs silently.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax.applications.corrfunc.traced import (
    TracedPairCountConfig,
    traced_soft_pair_counts,
)
from yggdrax.distributed.stats import (
    DistributedStatsConfig,
    distributed_knn,
    distributed_pair_counts,
    soft_window_cutoffs,
)

from .analytic import analytic_rr

__all__ = [
    "Statistic",
    "concat_statistics",
    "make_knn_cdf_statistic",
    "make_xi_statistic",
    "natural_estimator_rr",
    "query_grid",
]


class Statistic(NamedTuple):
    """A data vector computed from positions.

    Attributes
    ----------
    fn:
        ``positions (N, 3) -> (values (p,), overflow ())``. Traceable and
        differentiable in the positions; ``overflow`` is a bool scalar.
    labels:
        ``p`` names, one per value.
    """

    fn: Callable[[Array], tuple[Array, Array]]
    labels: tuple[str, ...]

    @property
    def size(self) -> int:
        """Length ``p`` of the data vector."""
        return len(self.labels)


def _default_config(n_rows: int, mesh: Optional[jax.sharding.Mesh], **kw):
    ndev = int(mesh.size) if mesh is not None else jax.device_count()
    halo = 0.05 if ndev == 1 else 2.0
    kw.setdefault("halo_fraction", halo)
    return DistributedStatsConfig.for_problem(max(1, -(-n_rows // ndev)), **kw)


def natural_estimator_rr(
    edges: Sequence[float] | np.ndarray,
    *,
    box_size: float,
    n_points: int,
    sharpness: float,
    eps: float = 1e-6,
    log: bool = True,
) -> np.ndarray:
    """Analytic soft ``RR`` of the natural estimator in a periodic cube.

    ``RR_b = N(N-1)/2 * V_b / L^3`` with ``V_b`` the soft shell volume of bin
    ``b`` over the hard cut ``(r_lo, r_hi)`` =
    :func:`~yggdrax.distributed.stats.soft_window_cutoffs` ``(edges, sharpness,
    eps)`` (:func:`~.analytic.analytic_rr`). Exact for uniform points while the
    cut stays below ``L/2``, where the minimum-image separation of two uniform
    points is uniform in the ball.

    The distributed ``DD`` drops pairs outside the cut exactly; the traced one
    drops only pairs whose window is at most ``window_eps`` and sums the rest.
    The two differ by the window mass beyond the cut, at most ``eps`` per pair
    and in practice a relative ``O(eps / sharpness)`` per bin, so one ``RR``
    serves both.

    Parameters
    ----------
    edges:
        ``(nbins + 1,)`` increasing bin edges.
    box_size:
        Periodic box length ``L``.
    n_points:
        Points per catalogue.
    sharpness:
        Soft-window sharpness (in ``ln r`` when ``log``).
    eps:
        Window value at the hard cut.
    log:
        Windows in ``ln r``.

    Returns
    -------
    numpy.ndarray
        ``(nbins,)`` expected soft pair counts.

    Raises
    ------
    ValueError
        If the hard cut reaches ``L/2``.
    """
    e = np.asarray(edges, dtype=np.float64)
    r_cut = soft_window_cutoffs(e, sharpness, eps, log)
    if r_cut[1] >= 0.5 * box_size:
        raise ValueError(
            f"the hard cut r_hi = {r_cut[1]:.4g} reaches L/2; raise sharpness or eps"
        )
    return analytic_rr(e, sharpness, r_cut, n_points, box_size**3, log)


def make_xi_statistic(
    edges: Sequence[float] | np.ndarray,
    *,
    box_size: float,
    n_points: int,
    sharpness: float = 20.0,
    eps: float = 1e-6,
    config: Optional[DistributedStatsConfig] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    backend: Literal["distributed", "traced"] = "distributed",
    traced_config: Optional[TracedPairCountConfig] = None,
) -> Statistic:
    """``xi`` in soft log bins from the natural estimator with analytic ``RR``.

    Parameters
    ----------
    edges:
        ``(nbins + 1,)`` increasing bin edges.
    box_size:
        Periodic box length ``L``.
    n_points:
        Points per catalogue (``RR`` depends on it).
    sharpness:
        Soft-window sharpness in ``ln r``.
    eps:
        Window value at the hard cut (:func:`~yggdrax.distributed.stats.soft_window_cutoffs`);
        the cut's upper end must stay below ``L/2``.
    config:
        Capacities of ``backend="distributed"``; default
        :meth:`~yggdrax.distributed.stats.DistributedStatsConfig.for_problem`.
    mesh:
        Device mesh of ``backend="distributed"``; default all devices.
    backend:
        ``"distributed"`` (default): ``DD`` from
        :func:`~yggdrax.distributed.stats.distributed_pair_counts`.
        ``"traced"``: ``DD`` from
        :func:`~yggdrax.applications.corrfunc.traced.traced_soft_pair_counts`
        on one device with ``box_size``, whose acceptance rule (window or
        geometric, hard or smooth) is ``traced_config``'s.
    traced_config:
        Required by ``backend="traced"``: plan it with
        :func:`~yggdrax.applications.corrfunc.traced.plan_traced_pair_counts`
        ``(..., edges=edges, sharpness=sharpness, box_size=box_size)`` on
        catalogues that bracket the ones the statistic will see, ideally with
        ``window_eps=eps``.

    Returns
    -------
    Statistic
        ``xi_b = DD_b / RR_b - 1`` per bin.

    Raises
    ------
    ValueError
        If the hard cut reaches ``L/2``, the backend is unknown, or
        ``backend="traced"`` has no ``traced_config``.
    """
    e = np.asarray(edges, dtype=np.float64)
    rr = natural_estimator_rr(
        e, box_size=box_size, n_points=n_points, sharpness=sharpness, eps=eps
    )
    if backend == "distributed":
        cfg = config if config is not None else _default_config(n_points, mesh)

        def fn(positions: Array) -> tuple[Array, Array]:
            res = distributed_pair_counts(
                positions,
                e,
                box_size=box_size,
                config=cfg,
                sharpness=sharpness,
                eps=eps,
                mesh=mesh,
            )
            return res.counts / jnp.asarray(rr, res.counts.dtype) - 1.0, res.overflow

    elif backend == "traced":
        if traced_config is None:
            raise ValueError(
                'backend="traced" needs traced_config (plan_traced_pair_counts)'
            )
        tcfg = traced_config

        def fn(positions: Array) -> tuple[Array, Array]:
            res = traced_soft_pair_counts(
                positions,
                jnp.asarray(e, positions.dtype),
                tcfg,
                sharpness=sharpness,
                box_size=box_size,
            )
            return res.counts / jnp.asarray(rr, res.counts.dtype) - 1.0, res.overflow

    else:
        raise ValueError(f"unknown backend {backend!r}")

    centres = np.sqrt(e[:-1] * e[1:])
    return Statistic(fn=fn, labels=tuple(f"xi(r={c:.4g})" for c in centres))


def query_grid(n_side: int, box_size: float, dtype: Any = jnp.float64) -> Array:
    """A regular grid of ``n_side^3`` query points at the cell centres.

    Parameters
    ----------
    n_side:
        Points per axis.
    box_size:
        Box length ``L``.
    dtype:
        Floating dtype.

    Returns
    -------
    Array
        ``(n_side^3, 3)`` points; the volume-filling quadrature of a kNN-CDF,
        with no shot noise of its own.
    """
    g = (jnp.arange(n_side, dtype=dtype) + 0.5) * (box_size / n_side)
    xx, yy, zz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def make_knn_cdf_statistic(
    ks: Sequence[int],
    r_grid: Sequence[float] | np.ndarray,
    *,
    box_size: float,
    n_points: int,
    queries: Array,
    sharpness: float = 15.0,
    r_max: Optional[float] = None,
    leak: float = 1e-6,
    config: Optional[DistributedStatsConfig] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> Statistic:
    """Soft kNN-CDFs of fixed query points: ``P(r_k <= r)`` for each ``k`` and ``r``.

    The step in ``ln r`` is ``sigmoid(s (ln r - ln r_k))``. A query with fewer
    than ``k`` data points within ``r_max`` has ``r_k = inf`` and contributes 0;
    the jump to that from ``r_k = r_max`` is a discontinuity of size
    ``sigmoid(s ln(r / r_max))`` in the statistic, so ``r_max`` defaults to
    ``max(r_grid) * leak^(-1/s)``, where it is below ``leak`` everywhere.

    Parameters
    ----------
    ks:
        Neighbour ranks, e.g. ``(1, 2, 4)``.
    r_grid:
        Radii at which each CDF is read.
    box_size:
        Periodic box length ``L``.
    n_points:
        Data points per catalogue (sizes the capacities).
    queries:
        ``(Q, 3)`` fixed query points (:func:`query_grid`), not differentiated.
    sharpness:
        Soft-step sharpness ``s`` in ``ln r``.
    r_max:
        Search cap, below ``L/2``; see above for the default.
    leak:
        Largest allowed discontinuity at ``r_max`` (sets the default).
    config:
        Capacities; default
        :meth:`~yggdrax.distributed.stats.DistributedStatsConfig.for_problem`
        for data plus queries.
    mesh:
        Device mesh; default all devices.

    Returns
    -------
    Statistic
        ``len(ks) * len(r_grid)`` values, ``k``-major.

    Raises
    ------
    ValueError
        If ``r_max`` reaches ``L/2``.
    """
    ks = tuple(int(k) for k in ks)
    rg = np.asarray(r_grid, dtype=np.float64)
    if r_max is None:
        r_max = float(rg.max() * math.exp(math.log(1.0 / leak) / sharpness))
    if r_max >= 0.5 * box_size:
        raise ValueError(
            f"r_max = {r_max:.4g} reaches L/2; raise sharpness, lower the grid or "
            "pass r_max (and accept a larger leak)"
        )
    k_max = max(ks)
    rows = np.asarray(ks) - 1
    q = jax.lax.stop_gradient(jnp.asarray(queries))
    cfg = (
        config
        if config is not None
        else _default_config(n_points + int(q.shape[0]), mesh)
    )

    def fn(positions: Array) -> tuple[Array, Array]:
        res = distributed_knn(
            positions,
            k=k_max,
            r_max=r_max,
            box_size=box_size,
            config=cfg,
            queries=q.astype(positions.dtype),
            r_grid=rg,
            sharpness=sharpness,
            mesh=mesh,
        )
        assert res.cdf is not None
        return res.cdf[rows].reshape(-1), res.overflow

    labels = tuple(f"cdf_k{k}(r={r:.4g})" for k in ks for r in rg)
    return Statistic(fn=fn, labels=labels)


def concat_statistics(*stats: Statistic) -> Statistic:
    """One statistic that concatenates several data vectors.

    Parameters
    ----------
    *stats:
        Statistics on the same positions.

    Returns
    -------
    Statistic
        Values concatenated in order; overflow is the OR.
    """

    def fn(positions: Array) -> tuple[Array, Array]:
        outs = [s.fn(positions) for s in stats]
        values = jnp.concatenate([o[0] for o in outs])
        overflow = jnp.any(jnp.stack([jnp.asarray(o[1]) for o in outs]))
        return values, overflow

    labels: tuple[str, ...] = ()
    for s in stats:
        labels = labels + s.labels
    return Statistic(fn=fn, labels=labels)
