"""Differentiable two-point correlation functions on the yggdrax tree engine.

Modules:

* ``binning``       -- soft/smooth bin-membership windows replacing hard bins.
* ``estimator``     -- dual-tree pair counting accumulating per-bin soft counts
                       (host-built topology, differentiated at fixed topology).
* ``traced``        -- the same estimator as one traced function, tree rebuilt
                       inside (``jit``/``grad``/``scan``/``vmap``-safe), in an
                       open or periodic box.
* ``baselines``     -- brute-force references and thin Corrfunc / TreeCorr
                       wrappers for validation.

The natural estimator ``xi = DD / RR - 1`` with the analytic ``RR`` of a
periodic box is
:func:`yggdrax.applications.point_process.make_xi_statistic` (``backend=
"traced"`` runs it on :func:`traced.traced_soft_pair_counts`); parameter
inference on top of it is :mod:`yggdrax.applications.point_process`.
"""
