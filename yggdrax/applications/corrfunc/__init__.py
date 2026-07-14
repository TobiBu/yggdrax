"""Differentiable two-point correlation functions on the yggdrax tree engine.

Modules (see the paper plan, Phase 2):

* ``binning``       -- soft/smooth bin-membership windows replacing hard bins.
* ``estimator``     -- dual-tree pair counting accumulating per-bin soft counts.
* ``baselines``     -- thin Corrfunc / TreeCorr wrappers for validation.
* ``inference_demo`` -- gradient-based parameter recovery using the estimator.
"""
