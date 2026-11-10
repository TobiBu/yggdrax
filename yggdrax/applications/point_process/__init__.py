"""A differentiable point-process forward model composed with yggdrax statistics.

``parameters -> particle positions -> xi / kNN-CDF -> loss``, all in JAX, with
``jax.grad`` (or ``jacfwd``) with respect to the model parameters through the tree
rebuild. Built for the parameter-inference demonstration of the yggdrax paper:
gradient checks against finite differences of the full chain, Fisher forecasts
from an ensemble of realisations, and recovery of the parameters of one
realisation from the statistics of an independent one.

Modules:

* ``model``      -- a reparameterised Thomas (Neyman–Scott) process in a periodic
  box: cluster size ``sigma`` and radial-profile shape ``beta``.
* ``analytic``   -- the Thomas ``xi(r)`` (Poisson and fixed counts), analytic
  soft ``RR`` of a periodic box, soft-window averages.
* ``statistics`` -- pluggable data vectors: ``xi`` (natural estimator, on
  :mod:`yggdrax.distributed.stats` or the traced single-device estimator) and
  soft kNN-CDFs on :mod:`yggdrax.distributed.stats`.
* ``inference``  -- Jacobians, finite differences, ensembles, Fisher forecasts,
  Levenberg–Marquardt fits.

The bench script ``bench/point_process/forecast.py`` produces the paper's JSON
and figures; ``docs/point_process.md`` is the design note.
"""

from .analytic import (
    analytic_rr,
    profile_xi_fixed_counts_mc,
    soft_shell_volumes,
    thomas_separation_pdf,
    thomas_xi_fixed_counts,
    thomas_xi_poisson,
    window_average,
)
from .inference import (
    FisherForecast,
    FitResult,
    central_difference_jacobian,
    ensemble_jacobians,
    ensemble_statistics,
    fisher_forecast,
    fit_parameters,
    forward,
    hartlap_factor,
    sample_covariance,
    statistic_jacobian,
)
from .model import (
    PARAM_NAMES,
    ThomasDraws,
    child_offsets,
    draw_ensemble,
    draw_thomas,
    profile_norm,
    thomas_positions,
)
from .statistics import (
    Statistic,
    concat_statistics,
    make_knn_cdf_statistic,
    make_xi_statistic,
    natural_estimator_rr,
    query_grid,
)

__all__ = [
    "PARAM_NAMES",
    "FisherForecast",
    "FitResult",
    "Statistic",
    "ThomasDraws",
    "analytic_rr",
    "central_difference_jacobian",
    "child_offsets",
    "concat_statistics",
    "draw_ensemble",
    "draw_thomas",
    "ensemble_jacobians",
    "ensemble_statistics",
    "fisher_forecast",
    "fit_parameters",
    "forward",
    "hartlap_factor",
    "make_knn_cdf_statistic",
    "make_xi_statistic",
    "natural_estimator_rr",
    "profile_norm",
    "profile_xi_fixed_counts_mc",
    "query_grid",
    "sample_covariance",
    "soft_shell_volumes",
    "statistic_jacobian",
    "thomas_positions",
    "thomas_separation_pdf",
    "thomas_xi_fixed_counts",
    "thomas_xi_poisson",
    "window_average",
]
