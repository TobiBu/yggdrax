"""Differentiable two-point correlation functions on the yggdrax tree engine.

Modules:

* ``binning``       -- soft/smooth bin-membership windows replacing hard bins.
* ``estimator``     -- dual-tree pair counting accumulating per-bin soft counts
                       (host-built topology, differentiated at fixed topology).
* ``traced``        -- the same estimator as one traced function, tree rebuilt
                       inside (``jit``/``grad``/``scan``/``vmap``-safe).
* ``baselines``     -- thin Corrfunc / TreeCorr wrappers for validation.
* ``inference_demo`` -- gradient-based parameter recovery using the estimator.
"""
