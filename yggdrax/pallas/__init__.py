"""yggdrax's Pallas kernels and the compatibility layer they share.

* :mod:`yggdrax.pallas.mutual_walk` -- the mutual dual-tree walk with one Pallas
  launch per wavefront round. Its public names are exported from
  :mod:`yggdrax.interactions` (``mutual_walk_pallas``, ``PallasWalkResult``,
  ``pallas_mutual_walk_supported``), and ``dual_tree_walk_mutual(...,
  backend="pallas" | "auto")`` runs it behind the flat walk's interface.
* :mod:`yggdrax.pallas._compat` -- the ``pallas_call`` API differences across JAX
  versions and the sm_80 hardware gate, for every Pallas kernel in yggdrax
  (including :mod:`yggdrax.applications.svgd`).
"""
