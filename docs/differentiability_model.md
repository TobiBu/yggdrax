# yggdrax differentiability model

*Phase 0 deliverable for the differentiable-applications paper. This is the
precise statement of what is and is not differentiable in yggdrax; paper
section 2 reads from this document.*

## Summary

yggdrax is differentiable **conditioned on a fixed tree topology and a fixed
near/far interaction partition**. There are no custom derivative rules
anywhere in the package (`grep` for `custom_jvp`/`custom_vjp`/`stop_gradient`
returns nothing): differentiability is *structural*. Gradients flow through
ordinary floating-point `jnp` operations and stop at integer, sorting, and
comparison operations. The practical rule is **differentiate through the
values, not the combinatorics**.

## What is differentiable

Given a topology, the following are smooth functions of the continuous inputs
(particle positions, kernel bandwidths, bin edges/widths) and carry exact
reverse-mode gradients:

- **Per-node geometry.** `compute_tree_geometry` (`yggdrax/geometry.py`,
  impl `yggdrax/_geometry_impl.py:159-215`) produces node `center`,
  `half_extent`, `max_extent`, and `radius` from the gathered sorted
  positions via min / max / mean reductions. These are (sub)differentiable
  w.r.t. positions.
- **Pairwise geometry and kernel evaluations.** Squared distances between
  nodes or particles (`dist_sq`, exposed to pair policies at
  `yggdrax/_interactions_impl.py:773`) and any kernel evaluated on them.
- **Soft / smooth binned or counted quantities.** A *hard* pair count is an
  integer and non-differentiable; a count accumulated with a smooth
  bin-membership weight (e.g. a sigmoid window on separation) is
  differentiable w.r.t. both positions and bin parameters. This is the basis
  of the differentiable correlation-function estimator (paper section 5).

This is validated numerically: `tests/unit/test_gradient_correctness.py`
compares `jax.grad` of a smooth observable built on node centres against
central finite differences, swept over the MAC parameter `theta`, on all
three backends (radix, octree, KD-tree). Away from accept/reject boundaries
the maximum relative error is `~1e-6` in float64.

## What is not differentiable

These outputs are discrete or piecewise-constant and correctly carry **no**
gradient (JAX returns zero or an integer type):

- **Morton ordering.** `morton_encode_impl` quantises coordinates with
  `jnp.rint(...).astype(uint64)` and bit-spreads them (`yggdrax/morton.py:72`);
  `sort_by_morton` uses `jnp.argsort` (`yggdrax/morton.py:156`). Both break
  gradients w.r.t. the ordering.
- **Tree topology.** `parent`, `left_child`, `right_child`, `node_ranges`,
  and the particle permutation are integer-valued and piecewise-constant in
  the positions.
- **MAC accept/reject.** The multipole acceptance criterion `_compute_mac_ok`
  returns booleans from `<=` comparisons
  (`yggdrax/_interactions_impl.py:338-395`); the resulting per-pair actions
  (`_ACTION_ACCEPT` / `_ACTION_NEAR` / `_ACTION_REFINE`) and `interaction_tags`
  are integers. The near/far partition is therefore piecewise-constant.

## The measure-zero boundary

The one subtlety is exactly *at* a MAC accept/reject boundary: an
infinitesimal change in a position can flip a node pair between the far
(multipole) and near (direct) branches, producing a jump in any quantity that
is computed differently on the two branches. This is a measure-zero set of
configurations, and both branches are individually smooth, so:

- almost everywhere the topology and partition are locally constant and the
  autodiff gradient is exact (the smooth-branch gradient);
- an optimizer that crosses a boundary sees a finite step, not an unbounded
  one, because the two branches agree to the approximation order set by
  `theta`;
- the practical prescription is to keep `theta` in a regime where the
  multipole approximation error is small, so the branch discontinuity is
  bounded by that same error. The frequency of boundary crossings during a
  representative optimization is quantified empirically in the MAC-accuracy
  experiment (paper section 3, `bench/differentiability/mac_accuracy.py`).

## Backend parity

All three backends satisfy the same FMM-core topology contract
(`yggdrax/tree.py:135-142`) and, for radix and KD-tree, run the *identical*
dual-tree walk (`yggdrax/_interactions_impl.py:3880`); the octree has a
separate but contract-equivalent traversal seam. The KD-tree backend
therefore exposes the same `pair_policy`, `return_result`, and
`interaction_tags` surface as radix/octree, verified by
`tests/unit/test_kd_tree_parity.py` and
`tests/unit/test_interactions_backend_parity.py`. Because the topologies
differ (a KD median split is not a Morton radix split), per-node counts are
not expected to match across backends -- only the structural contract,
determinism, and continuous geometry are backend invariants.

## Consequence for the applications

Both case studies rebuild the tree inside the traced/differentiated function
each step and accumulate a continuous per-pair quantity *downstream* of the
`pair_policy` (which itself only selects and tags pairs, and does not thread a
float accumulator). The gradient of the downstream reduction is well defined
and finite even though the tree is only piecewise-constant, as demonstrated by
`test_gradient_correctness.py::test_gradient_through_rebuilt_tree_is_finite`.
