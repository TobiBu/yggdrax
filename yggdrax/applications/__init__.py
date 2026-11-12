"""Applications built on top of the yggdrax tree engine.

Each sub-package is a self-contained, differentiable solver that uses the core
near/far partition for something other than gravity. They are shipped with the
package -- a user can run tree-accelerated SVGD or the differentiable pair-count
estimator without any paper material -- but they are *not* part of the core
tree/traversal public API and are imported explicitly, never re-exported from
:mod:`yggdrax`.

Sub-packages:

* :mod:`yggdrax.applications.corrfunc` -- differentiable two-point correlation
  functions via soft-binned dual-tree pair counting.
* :mod:`yggdrax.applications.knn_stats` -- differentiable k-nearest-neighbour
  CDFs (Banerjee & Abel 2021) on the traced KD-tree search.
* :mod:`yggdrax.applications.svgd` -- tree-accelerated Stein variational
  gradient descent using the kernel-aware pair-policy hook.
* :mod:`yggdrax.applications.point_process` -- a differentiable Thomas-process
  forward model composed with the distributed xi / kNN-CDF statistics:
  parameter gradients, Fisher forecasts and recovery on independent
  realisations.
* :mod:`yggdrax.applications.positions` -- move every particle of a periodic
  catalogue until its traced ξ and kNN-CDFs match a target's, with gradients
  through both tree estimators.

Two modules fit parameters with ``optax``, which is not a core runtime
dependency: :mod:`yggdrax.applications.svgd.bandwidth_learning` and the
optimisation loop of :mod:`yggdrax.applications.positions` (imported when it
runs); install ``yggdrax[applications]`` to use them.
"""
