"""Tree-accelerated, differentiable Stein Variational Gradient Descent (SVGD).

Modules (see the paper plan, Phase 3):

* ``kernel``            -- RBF kernel, median heuristic, and Stein pair terms.
* ``targets``           -- toy target distributions with analytic scores.
* ``sampler``           -- tree-accelerated Stein update: near field exact,
  far field a monopole (centre-of-mass) expansion, on the radix/octree
  backend (d <= 3).
* ``exact``             -- reference O(N^2) SVGD for validation (small N).
* ``bandwidth_learning`` -- backprop a validation loss through SVGD steps to
  learn the kernel bandwidth instead of using the median heuristic.
"""
