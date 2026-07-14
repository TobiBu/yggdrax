"""Tree-accelerated, differentiable Stein Variational Gradient Descent (SVGD).

Modules (see the paper plan, Phase 3):

* ``kernel``            -- RBF kernel and its gradient terms.
* ``sampler``           -- dual-tree SVGD update via the KD-tree pair policy.
* ``exact``             -- reference O(N^2) SVGD for validation (small N).
* ``bandwidth_learning`` -- backprop a validation loss through SVGD steps to
  learn the kernel bandwidth instead of using the median heuristic.
"""
