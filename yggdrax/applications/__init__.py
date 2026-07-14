"""Paper-specific application code built on top of the yggdrax tree engine.

Everything under :mod:`yggdrax.applications` is application/case-study code for
the differentiable-tree paper (branch ``paper/differentiable-applications``).
It deliberately lives outside the maintained core tree/traversal public API so
that paper-specific experiments do not leak into the library surface.

Sub-packages:

* :mod:`yggdrax.applications.corrfunc` -- differentiable two-point correlation
  functions via soft-binned dual-tree pair counting.
* :mod:`yggdrax.applications.svgd` -- tree-accelerated Stein variational
  gradient descent using the KD-tree pair-policy hook.
"""
