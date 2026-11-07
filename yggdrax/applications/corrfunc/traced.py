"""Soft pair counts as ONE traced function, with the tree rebuilt inside it.

:mod:`~yggdrax.applications.corrfunc.estimator` builds its near/far partition on
the host (:func:`~yggdrax.applications.corrfunc.estimator.build_pair_topology`)
and differentiates only the accumulation, so a caller has to rebuild the tree
*outside* ``jax.grad`` and cannot put the estimator in a ``lax.scan`` or a
``vmap``. This module removes that split. :func:`traced_soft_pair_counts` is a
pure, static-shape JAX function of the positions:

    positions -> build_static_cells_tree -> compute_tree_geometry
              -> mutual_walk_inputs -> dual_tree_walk_mutual
              -> soft counts

so ``jax.jit(jax.grad(loss))`` rebuilds the tree on every evaluation, and the
same function runs inside ``lax.scan`` (a rebuild per step) and under ``vmap``
(a rebuild per catalogue).

Estimator. With the soft window ``w_k`` of
:func:`~yggdrax.applications.corrfunc.binning.soft_bin_weights`, the counts are
``DD_k = sum_{i<j} w_k(r_ij)``, split by the walk's canonical lists (every
unordered node pair at most once, ``a < b``):

* **within a leaf**: every pair ``i < j`` of the leaf, exactly;
* **near leaf pair** ``(a, b)``: every particle pair across the two leaves,
  exactly;
* **far node pair** ``(A, B)``: ``count_A * count_B * w_k(|com_A - com_B|)``,
  the monopole at the centres of mass.

These three sets tile the unordered particle pairs, so at a ``theta`` small
enough that the walk accepts nothing as far the result is the brute-force sum
(up to summation order).

Differentiability. The topology -- Morton codes, leaf ranges, the pair lists --
is built from ``stop_gradient(positions)`` and carries no gradient. The counts
are then a smooth function of the positions *given* that topology: gradients
flow through the exact near-field separations and through the centres of mass
of the far pairs, the fixed-topology gradient of ``docs/differentiability_model.md``.
The topology is piecewise constant in the positions, so this is the gradient of
the estimator almost everywhere.

Accuracy. "Small theta" is not "no far pairs": a single-particle leaf has
radius 0, so the non-strict Dehnen test accepts a pair of them at any theta --
harmlessly, since their monopole is the exact pair term. At a finite theta the
far error is the monopole's: an accepted pair's separations span about
``d (1 +- theta)``, which with a sharp window can straddle a bin edge. Cell
leaves are small (often a few particles), so the mutual walk accepts many
pairs inside the binned range, and at equal theta this path is markedly less
accurate than the host path over count-bucket leaves (n = 1500 uniform, leaf
16, 10 log bins, sharpness 100: max per-bin error 2e-3 at theta 0.2, 4e-2 at
0.4). Choose theta well below the log bin width, or larger leaves.

Static shapes. Every capacity -- leaf count, wavefront queue, far and near list
lengths -- lives in a :class:`TracedPairCountConfig`, sized outside ``jit`` by
:func:`plan_traced_pair_counts` from an untimed probe (the pattern of
``bench/differentiability/scaling.py::_fast_path``). A traced call that needs
more than its config allows does not fail: it returns a result whose overflow
flags say so, and its counts are then incomplete. Callers must check
:attr:`TracedPairCounts.overflow` -- under ``scan`` or ``vmap`` an ``any`` over
the stacked flags.

Superseded host-side helpers (kept for now, not deleted). For gradients that
include a rebuild, this module replaces:

* ``estimator.build_pair_topology`` -- the host-side near/far list assembly
  (numpy CSR expansion of ``build_interactions_and_neighbors``), which cannot
  be traced and so cannot run inside ``grad``/``scan``/``vmap``;
* ``estimator.soft_pair_counts_from_topology`` together with ``PairTopology`` --
  its role (accumulation at a fixed topology) is played by the accumulation
  half of :func:`traced_soft_pair_counts`, which reads the walk's flat
  canonical lists directly, so no halving of symmetric lists is needed;
* ``estimator.soft_pair_counts`` -- the eager build-then-accumulate wrapper;
* the per-step host rebuild in ``inference_demo.recover_parameter`` -- see
  ``inference_demo.recover_parameter_traced``.

``estimator._accumulate_block_pairs`` (the chunked, checkpointed near-field
reduction) is shared, not superseded. The old path still supports the
``octree`` and ``kdtree`` backends, which this module does not.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax import MACType
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import dual_tree_walk_mutual, mutual_walk_inputs

from .binning import soft_bin_weights
from .estimator import _DEFAULT_NEAR_CHUNK, _accumulate_block_pairs

__all__ = [
    "TracedPairCountConfig",
    "TracedPairCounts",
    "plan_traced_pair_counts",
    "traced_soft_pair_counts",
]


@dataclasses.dataclass(frozen=True)
class TracedPairCountConfig:
    """Static parameters and capacities of :func:`traced_soft_pair_counts`.

    Frozen and hashable, so it can be closed over or passed as a static
    argument of ``jax.jit``. Build one with :func:`plan_traced_pair_counts`
    unless you know the capacities.

    Attributes:
        leaf_size: Maximum particles per cell leaf; also the static width of
            the padded near-field leaf blocks.
        leaf_capacity: Static number of leaves the tree is padded to.
        max_pair_queue: Wavefront capacity of the mutual walk, in node pairs.
        far_cap: Capacity of the canonical far node-pair list.
        near_cap: Capacity of the canonical near leaf-pair list.
        theta: Opening angle of the acceptance criterion.
        mac_type: Geometric acceptance criterion (``"bh"``, ``"dehnen"`` or
            ``"engblom"``).
        near_chunk: Leaf pairs per chunk of the near-field reduction; bounds the
            live ``(chunk, leaf_size, leaf_size, nbins)`` intermediate. Does not
            change the result beyond summation order.
    """

    leaf_size: int
    leaf_capacity: int
    max_pair_queue: int
    far_cap: int
    near_cap: int
    theta: float = 0.5
    mac_type: MACType = "dehnen"
    near_chunk: int = _DEFAULT_NEAR_CHUNK


class TracedPairCounts(NamedTuple):
    """Output of :func:`traced_soft_pair_counts`.

    ``counts`` is the differentiable result; every other field is a
    non-differentiable diagnostic. ``overflow`` is the OR of the four specific
    flags: when it is set the counts miss pairs and must not be used.
    ``num_leaves``, ``far_count``, ``near_count`` and ``peak_wavefront`` are
    what the call *needed* (counted before truncation), which is what a caller
    re-plans from after an overflow.
    """

    counts: Array  # (nbins,)
    overflow: Array  # bool
    leaf_overflow: Array  # bool: more cell leaves needed than leaf_capacity
    queue_overflow: Array  # bool: a walk round needed > max_pair_queue
    far_overflow: Array  # bool: far_count > far_cap
    near_overflow: Array  # bool: near_count > near_cap
    num_leaves: Array  # int: live cell leaves needed
    far_count: Array  # int: canonical far pairs needed
    near_count: Array  # int: canonical near pairs needed
    peak_wavefront: Array  # int: largest wavefront any round tried to push


class _Topology(NamedTuple):
    """Integer (non-differentiable) structure of one traced build + walk."""

    order: Array  # (n,) Morton slot -> particle index
    node_ranges: Array  # (num_nodes, 2) inclusive particle-slot ranges
    num_internal: int
    walk: object  # MutualWalkResult
    leaf_overflow: Array
    num_leaves: Array


def _resolve_bounds(positions: Array, bounds) -> tuple[Array, Array]:
    """Morton domain: the given bounds or the padded box of ``positions``."""
    if bounds is None:
        return infer_bounds(positions)
    lo, hi = bounds
    return jax.lax.stop_gradient(jnp.asarray(lo)), jax.lax.stop_gradient(
        jnp.asarray(hi)
    )


def _build_topology(
    positions: Array, config: TracedPairCountConfig, bounds
) -> _Topology:
    """Rebuild the tree and run the mutual walk, all traceable, no gradient.

    The evaluator and the planner both call this, so a plan sized by the probe
    is a plan for exactly the walk the evaluator runs.
    """
    pos = jax.lax.stop_gradient(positions)
    box = _resolve_bounds(pos, bounds)
    masses = jnp.ones(pos.shape[0], dtype=pos.dtype)
    tree, pos_sorted, _, _, leaf_overflow, num_leaves = build_static_cells_tree(
        pos,
        masses,
        box,
        leaf_size=config.leaf_size,
        leaf_capacity=config.leaf_capacity,
        return_reordered=True,
        return_overflow=True,
        return_num_leaves=True,
    )
    # max_leaf_size: leaf_size is traced under jit; without the static bound
    # the geometry's leaf gather falls back to (num_leaves, n, 3).
    geometry = compute_tree_geometry(tree, pos_sorted, max_leaf_size=config.leaf_size)
    w = mutual_walk_inputs(
        tree, geometry, mac_type=config.mac_type, mask_empty_nodes=True
    )
    walk = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        config.theta,
        w.root,
        max_pair_queue=config.max_pair_queue,
        far_cap=config.far_cap,
        near_cap=config.near_cap,
        mac_type=config.mac_type,
        node_active=w.node_active,
    )
    return _Topology(
        order=tree.particle_indices,
        node_ranges=tree.node_ranges,
        num_internal=int(tree.left_child.shape[0]),
        walk=walk,
        leaf_overflow=leaf_overflow,
        num_leaves=num_leaves,
    )


def traced_soft_pair_counts(
    positions: Array,
    edges: Array,
    config: TracedPairCountConfig,
    *,
    sharpness: float | Array = 100.0,
    log: bool = True,
    bounds: tuple[Array, Array] | None = None,
) -> TracedPairCounts:
    """Soft pair counts ``sum_{i<j} w_k(r_ij)``, rebuilding the tree inside.

    A pure, static-shape function of ``positions``: safe under ``jax.jit``,
    ``jax.grad`` (the rebuild happens inside the differentiated function),
    ``lax.scan`` and ``vmap``. See the module docstring for the estimator and
    its gradient.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``. Differentiable.
        edges: Radial bin edges, shape ``(nbins + 1,)``. Differentiable.
        config: Static capacities and walk parameters, usually from
            :func:`plan_traced_pair_counts`.
        sharpness: Soft-window sharpness (may be traced).
        log: Bin in ``log`` separation.
        bounds: Morton domain ``(min_corner, max_corner)``. ``None`` infers the
            padded box of ``positions`` on every call (no gradient flows into
            it). Fixed bounds keep the leaf structure comparable across calls;
            every particle must lie inside them.

    Returns:
        A :class:`TracedPairCounts`. ``counts`` has shape ``(nbins,)``; read
        ``overflow`` before trusting it.
    """
    topo = _build_topology(positions, config, bounds)
    walk = topo.walk
    dtype = positions.dtype
    n = int(positions.shape[0])
    nbins = int(edges.shape[0]) - 1
    leaf = int(config.leaf_size)
    ni = topo.num_internal
    ranges = topo.node_ranges

    # Differentiable: gather the live positions through the integer order.
    pos_sorted = positions[topo.order]

    # --- padded leaf blocks (leaf_capacity, leaf_size, 3) + validity ---
    # Leaves are nodes [ni, ni + leaf_capacity); an EMPTY padding leaf has the
    # range (n, n - 1), i.e. length 0. A cell leaf never holds more than
    # leaf_size particles (the partition splits a full deepest cell), so the
    # block width is exact.
    leaf_start = ranges[ni:, 0]
    leaf_len = jnp.clip(ranges[ni:, 1] - leaf_start + 1, 0, leaf)
    ramp = jnp.arange(leaf, dtype=leaf_start.dtype)
    slots = jnp.clip(leaf_start[:, None] + ramp[None, :], 0, n - 1)
    leaf_mask = (ramp[None, :] < leaf_len[:, None]).astype(dtype)
    blocks = pos_sorted[slots]
    leaf_rows = jnp.arange(int(config.leaf_capacity), dtype=leaf_start.dtype)

    # within-leaf pairs i < j (strict upper triangle), each once
    triu = jnp.triu(jnp.ones((leaf, leaf), dtype=dtype), k=1)
    within = _accumulate_block_pairs(
        blocks,
        leaf_mask,
        leaf_rows,
        leaf_rows,
        edges,
        sharpness,
        log,
        nbins,
        triu,
        int(config.near_chunk),
    )

    # near leaf pairs: canonical (a < b), each once, -1 padded
    near_live = walk.near_a >= 0
    near_t = jnp.where(near_live, walk.near_a - ni, 0)
    near_s = jnp.where(near_live, walk.near_b - ni, 0)
    cross = _accumulate_block_pairs(
        blocks,
        leaf_mask,
        near_t,
        near_s,
        edges,
        sharpness,
        log,
        nbins,
        None,
        int(config.near_chunk),
        row_weight=near_live.astype(dtype),
    )

    # far node pairs: monopole at the centres of mass, each once
    far_live = walk.far_a >= 0
    fa = jnp.where(far_live, walk.far_a, 0)
    fb = jnp.where(far_live, walk.far_b, 0)
    prefix = jnp.concatenate(
        [jnp.zeros((1, 3), dtype), jnp.cumsum(pos_sorted, axis=0)], axis=0
    )

    def com_and_count(node: Array) -> tuple[Array, Array]:
        start, end = ranges[node, 0], ranges[node, 1]
        count = jnp.maximum(end - start + 1, 0)
        total = prefix[end + 1] - prefix[start]
        safe = jnp.maximum(count, 1).astype(dtype)
        return total / safe[:, None], count.astype(dtype)

    com_a, cnt_a = com_and_count(fa)
    com_b, cnt_b = com_and_count(fb)
    d_far = jnp.sqrt(jnp.sum((com_a - com_b) ** 2, axis=-1) + 1e-30)
    w_far = soft_bin_weights(d_far, edges, sharpness, log=log)  # (far_cap, nbins)
    pair_weight = jnp.where(far_live, cnt_a * cnt_b, 0.0)
    far = jnp.sum(pair_weight[:, None] * w_far, axis=0)

    leaf_overflow = jnp.asarray(topo.leaf_overflow, bool)
    overflow = (
        leaf_overflow | walk.queue_overflow | walk.far_overflow | walk.near_overflow
    )
    return TracedPairCounts(
        counts=within + cross + far,
        overflow=overflow,
        leaf_overflow=leaf_overflow,
        queue_overflow=walk.queue_overflow,
        far_overflow=walk.far_overflow,
        near_overflow=walk.near_overflow,
        num_leaves=topo.num_leaves,
        far_count=walk.far_count,
        near_count=walk.near_count,
        peak_wavefront=walk.peak_wavefront,
    )


def _pow2(x: int) -> int:
    """Smallest power of two >= ``x`` (and >= 2)."""
    return 1 << max(int(x) - 1, 1).bit_length()


def plan_traced_pair_counts(
    positions: Array | Sequence[Array],
    *,
    theta: float = 0.5,
    leaf_size: int = 16,
    mac_type: MACType = "dehnen",
    bounds: tuple[Array, Array] | None = None,
    near_chunk: int = _DEFAULT_NEAR_CHUNK,
    leaf_slack: float = 1.25,
    list_slack: float = 1.5,
    max_doublings: int = 16,
) -> TracedPairCountConfig:
    """Size the static capacities of :func:`traced_soft_pair_counts` by probing.

    Runs the traced build + walk eagerly (outside the caller's ``jit``) on each
    probe catalogue with growing capacities until nothing overflows, then sizes
    each capacity from what the probes *needed* times a slack, rounded up to a
    power of two for the walk's lists (fewer distinct compiled shapes). With a
    batch, the capacities cover the largest need over the batch.

    Probe with every catalogue the traced function will see, or with ones that
    bracket them (e.g. the start and the target of an optimisation). A
    configuration is only a guess for unseen positions: the traced call
    reports, never hides, an overflow.

    Args:
        positions: One catalogue ``(n, 3)``, a batch ``(B, n, 3)``, or a
            sequence of catalogues.
        theta: Opening angle.
        leaf_size: Maximum particles per cell leaf.
        mac_type: Geometric acceptance criterion.
        bounds: Fixed Morton domain, as for :func:`traced_soft_pair_counts`.
            Pass the same value to both.
        near_chunk: Near-field chunk size stored in the config.
        leaf_slack: Multiplier on the probed live-leaf count.
        list_slack: Multiplier on the probed wavefront peak and list lengths.
        max_doublings: Give up after this many capacity increases.

    Returns:
        A :class:`TracedPairCountConfig`.

    Raises:
        RuntimeError: If the walk capacities do not converge.
    """
    if isinstance(positions, (list, tuple)):
        catalogues = [jnp.asarray(p) for p in positions]
    else:
        arr = jnp.asarray(positions)
        catalogues = [arr] if arr.ndim == 2 else [arr[b] for b in range(arr.shape[0])]

    leaf = int(leaf_size)
    need_leaves = need_queue = need_far = need_near = 1
    for pos in catalogues:
        n = int(pos.shape[0])
        box = _resolve_bounds(pos, bounds)
        # --- live leaves: the count is taken before the cut, so one probe at
        # any capacity gives the exact need. ---
        cap0 = max(64, 4 * (-(-n // leaf)))
        out = build_static_cells_tree(
            pos,
            jnp.ones(n, pos.dtype),
            box,
            leaf_size=leaf,
            leaf_capacity=cap0,
            return_num_leaves=True,
        )
        live = int(out[-1])
        leaf_capacity = int(leaf_slack * live) + 8

        # --- walk: grow the caps until the probe fits, then record its need ---
        q, far_cap, near_cap = 1 << 12, 1 << 14, 1 << 14
        for _ in range(max_doublings):
            cfg = TracedPairCountConfig(
                leaf_size=leaf,
                leaf_capacity=leaf_capacity,
                max_pair_queue=q,
                far_cap=far_cap,
                near_cap=near_cap,
                theta=float(theta),
                mac_type=mac_type,
                near_chunk=int(near_chunk),
            )
            walk = jax.jit(lambda p, c=cfg: _build_topology(p, c, bounds).walk)(pos)
            qo, fo, no = (
                bool(walk.queue_overflow),
                bool(walk.far_overflow),
                bool(walk.near_overflow),
            )
            if not (qo or fo or no):
                break
            q *= 4 if qo else 1
            far_cap *= 4 if fo else 1
            near_cap *= 4 if no else 1
        else:
            raise RuntimeError(
                f"mutual walk capacities did not converge after {max_doublings} "
                f"increases (n={n}, theta={theta}, leaf_size={leaf})"
            )
        need_leaves = max(need_leaves, live)
        need_queue = max(need_queue, int(walk.peak_wavefront))
        need_far = max(need_far, int(walk.far_count))
        need_near = max(need_near, int(walk.near_count))

    return TracedPairCountConfig(
        leaf_size=leaf,
        leaf_capacity=int(leaf_slack * need_leaves) + 8,
        max_pair_queue=_pow2(int(np.ceil(list_slack * need_queue)) + 1),
        far_cap=_pow2(int(np.ceil(list_slack * need_far)) + 1),
        near_cap=_pow2(int(np.ceil(list_slack * need_near)) + 1),
        theta=float(theta),
        mac_type=mac_type,
        near_chunk=int(near_chunk),
    )
