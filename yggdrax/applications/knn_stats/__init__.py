"""Differentiable k-nearest-neighbour CDFs on the yggdrax KD-tree.

The kNN-CDFs of Banerjee & Abel (2021, MNRAS 500, 5479), made differentiable
with respect to every particle coordinate: neighbour identities come from the
traced KD-tree search and are held constant, distances are recomputed from the
live positions, and the indicator ``1[r_k < r]`` becomes a logistic step in
``log r``. ``box_size`` gives minimum-image kNN-CDFs in a periodic box. See
:mod:`yggdrax.applications.knn_stats.estimator` for the definition, the
bias/sharpness trade-off, and how this single-device API relates to the
multi-device :func:`yggdrax.distributed.stats.distributed_knn`.

Modules:

* ``estimator`` -- :func:`knn_cdf`, :func:`knn_cdf_loss` and their parts.
* ``baselines`` -- dense ``top_k`` and ``scipy`` references for validation.
"""

from yggdrax.applications.knn_stats.estimator import (
    cdf_distance,
    hard_knn_cdf,
    knn_cdf,
    knn_cdf_loss,
    knn_distances,
    knn_indices,
    knn_log_distances,
    peaked_cdf,
)

__all__ = [
    "cdf_distance",
    "hard_knn_cdf",
    "knn_cdf",
    "knn_cdf_loss",
    "knn_distances",
    "knn_indices",
    "knn_log_distances",
    "peaked_cdf",
]
