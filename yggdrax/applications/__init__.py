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
* :mod:`yggdrax.applications.svgd` -- tree-accelerated Stein variational
  gradient descent using the kernel-aware pair-policy hook.

Two modules (:mod:`yggdrax.applications.svgd.bandwidth_learning` and
:mod:`yggdrax.applications.corrfunc.inference_demo`) fit parameters with
``optax``, which is not a core runtime dependency; install ``yggdrax[applications]``
to use them.
"""
