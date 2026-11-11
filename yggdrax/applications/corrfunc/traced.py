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

These three sets tile the unordered particle pairs.

Near-field culling (``sub_block``, on by default from the planner). A cell
leaf is usually larger than the windows' reach and holds fewer particles than
its block width, so most slots of a dense leaf-pair block are padding or out of
reach (N = 1e5 uniform, 10 log bins to 0.04 L, leaf 32: 1.5e8 slots, 5.9e7 real
pairs, about 2e6 within reach). Each leaf's Morton-ordered slots are cut into
runs of ``sub_block``; a near leaf pair is expanded into its sub-block pairs,
those whose bounding boxes are farther apart (minimum image) than the
separation where every window is below ``1e-15`` are dropped, and the rest are
summed as dense ``sub_block x sub_block`` blocks (3.2e7 slots at 8). The counts
change by at most ``1e-15`` per dropped particle pair; forward 2.3x and
value+grad 2.8x faster on CPU there (4-5x on a Thomas catalogue).

Acceptance. Which node pairs are far is decided by
``TracedPairCountConfig.acceptance``:

* ``"window"`` (default). Every particle separation of nodes ``a`` and ``b``
  lies in ``[d - s, d + s]``, with ``d`` the distance of the box centres and
  ``s`` the sum of the bounding-sphere radii. A pair is

  - **pruned** (``dual_tree_walk_mutual(pair_discard=...)``: not emitted, not
    refined) when every bin's window is at most ``window_eps`` over that
    range -- beyond the last edge, or below the first if it is above 0;
  - **accepted** as far when every bin's window varies by at most
    ``window_eps`` over it -- deep inside one bin, so the monopole
    ``n_a n_b w_k(d_com)`` misses each particle pair by at most ``window_eps``;
  - otherwise refined, down to leaf pairs summed exactly.

  The error is then set by ``window_eps``, not by ``theta`` (unused) or the
  leaf size. With a sharp window almost nothing inside ``[r_min, r_max]`` is
  flat enough to accept, so the tree mostly prunes and the rest is exact
  pairwise work. Measured (n = 1500 uniform, 10 log bins over [0.01, 0.5],
  sharpness 100, eps 1e-6): max per-bin relative error 2e-10 at leaf 16,
  4e-13 at leaf 32.
* ``"geometric"``. The opening-angle MAC of ``mac_type`` at ``theta``. Its far
  error is the monopole's over a separation range of about ``d (1 +- theta)``,
  which straddles bin edges for a sharp window; cell leaves are small, so many
  such pairs are accepted (same setup, leaf 16: 2e-3 at theta 0.2, 4e-2 at
  0.4, 1e-1 at 0.6). Note that a single-particle leaf has radius 0, so this
  rule accepts a pair of them at any theta (exactly, as a point pair).

Smooth acceptance (``smooth_band = beta > 1``). Both rules above are hard:
each node-pair decision is piecewise constant in the positions, so the counts
jump where a pair switches between far, refined and pruned -- by up to the
rule's error on that pair. The smooth rule blends instead. Each visited pair
``P`` gets a far fraction ``sigma_P`` and a keep fraction ``kappa_P`` in
``[0, 1]``, C2 ramps (in log) of the window variation / window maximum over the
pair's separation range (window rule; band ``[eps / beta, eps]``) or of
``(r_a + r_b) / d`` (geometric; band ``[theta / beta, theta]``), and contributes

    C(P) = kappa_P * (sigma_P * far_P + (1 - sigma_P) * sum_children C(Q)),

with a leaf pair's ``sum_children`` its exact sum. Only band pairs
(``0 < sigma < 1`` or ``kappa < 1``) are both emitted far and refined, via the
walk's ``pair_band`` hook, which also records each emitted pair's nearest band
ancestor. The weight of every emitted term is then a product along that
ancestry, recomputed here from LIVE node geometry (an exact range min / max of
the positions), so ``grad`` includes the ``d sigma / dx`` terms. Two more
choices make the blend continuous at fixed leaf membership: the walk splits the
node with more particles (not the larger radius, whose order flips with the
positions and would switch the decomposition under a band pair), and pruning
is blended like acceptance. What remains discontinuous is a change of leaf
membership (a particle crossing a cell face), whose jump is bounded by the far
error of the pairs involved. See ``docs/differentiating_a_tree_code.md``.

Periodic box (``box_size=L``). Every separation is the minimum-image one: the
near field through :func:`yggdrax.periodic.minimum_image`, the far monopole at
the minimum image of the centre-of-mass displacement, and every acceptance hook
on the minimum-image centre distance ``d``, which the hooks recompute from the
node centres. The walk itself runs without ``box_size``, so its periodic guard
(far only if ``d + r_a + r_b < L/2``) is not applied. A radial count does not
need it: through any image, a particle pair's minimum-image separation lies in
``[d - s, d + s]`` (``s = r_a + r_b``), and so does the minimum image of the COM
displacement. That interval is all the window rule's bound uses, and it is the
geometric rule's monopole interval too. The guard would instead send every
node pair beyond ``L/2`` (about half of all pairs in a cube) to exact leaf
pairs. Positions are wrapped into the box and the cell tree is built on it.
With ``membership`` each live position is taken at its image nearest its
membership position, so a face crossing does not restructure the tree. Keep
the windows' reach (``r_max`` plus the ``window_eps`` tail) below ``L/2`` for
the result to be the usual periodic statistic; the analytic ``RR`` of
:func:`yggdrax.applications.point_process.natural_estimator_rr` assumes it.

Differentiability. The topology -- Morton codes, leaf ranges, the pair lists --
is built from ``stop_gradient(positions)`` and carries no gradient. The counts
are then a smooth function of the positions *given* that topology: gradients
flow through the exact near-field separations and through the centres of mass
of the far pairs: the fixed-topology gradient. The topology is piecewise
constant in the positions, so this is the gradient of the estimator almost
everywhere; across a switch the estimator jumps (see "Smooth acceptance" and
``docs/differentiating_a_tree_code.md`` for how accurate it is and where it
breaks).

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
* the per-step host rebuild of the former ``inference_demo`` module, whose
  parameter recovery is now :mod:`yggdrax.applications.point_process`.

``estimator._accumulate_block_pairs`` (the chunked, checkpointed near-field
reduction) is shared, not superseded. The old path still supports the
``octree`` and ``kdtree`` backends, which this module does not.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax import MACType
from yggdrax._interactions_impl import _compute_mac_ok
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import (
    MutualWalkResult,
    dual_tree_walk_mutual,
    mutual_walk_inputs,
)
from yggdrax.periodic import (
    BoxSize,
    minimum_image,
    normalize_box_size,
    wrap_positions,
)

from .binning import soft_bin_weights
from .estimator import _DEFAULT_NEAR_CHUNK, _accumulate_block_pairs

# Default ``TracedPairCountConfig.window_eps``: an absolute bound per particle
# pair and bin. The per-bin RELATIVE error it gives is far smaller in practice
# (2e-10 at n = 1500 uniform, 10 log bins over [0.01, 0.5], sharpness 100,
# leaf 16); tests/applications/test_corrfunc_traced.py holds it below 1e-6
# across leaf sizes, clustering and window shapes.
_DEFAULT_WINDOW_EPS = 1e-6

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
        theta: Opening angle; used only by ``acceptance="geometric"``.
        mac_type: Geometric criterion (``"bh"``, ``"dehnen"`` or
            ``"engblom"``) of ``acceptance="geometric"``.
        near_chunk: Leaf pairs per chunk of the near-field reduction; bounds the
            live ``(chunk, leaf_size, leaf_size, nbins)`` intermediate. Does not
            change the result beyond summation order.
        acceptance: ``"window"`` (default) accepts a node pair only where every
            bin's soft window varies by at most ``window_eps`` over the pair's
            whole separation range, and prunes a pair whose windows are all at
            most ``window_eps`` there (see the module docstring); the error is
            then set by ``window_eps``, independent of ``theta`` and the leaf
            size. ``"geometric"`` is the plain opening-angle MAC.
        window_eps: Absolute tolerance on the window, per particle pair and bin,
            of ``acceptance="window"``.
        smooth_band: Ratio ``beta >= 1`` of the smooth (blended) acceptance
            band; ``1.0`` (default) is the hard rule. With ``beta > 1`` a node
            pair inside the band contributes ``sigma * far + (1 - sigma) *
            refined`` (and, for the window rule's pruning, ``kappa`` times
            itself), with ``sigma``, ``kappa`` smooth in the positions, which
            makes the counts continuous across acceptance switches (see the
            module docstring and ``docs/differentiating_a_tree_code.md``). The
            band is ``[eps / beta, eps]`` on the window variation (window
            rule) or ``[theta / beta, theta]`` on ``(r_a + r_b) / d``
            (geometric), so the error bound of the hard rule at the outer edge
            still holds.
        band_cap: Capacity of the walk's band list (``smooth_band > 1`` only).
        sub_block: Sub-block width of the culled near field; ``0`` (default)
            evaluates every near leaf pair as one dense ``leaf_size`` x
            ``leaf_size`` block. With ``sub_block = B`` (a divisor of
            ``leaf_size``) each leaf's Morton-ordered slots are cut into blocks
            of ``B``; a near leaf pair is expanded into its sub-block pairs, the
            pairs whose bounding boxes are farther apart (minimum image) than
            the separation where every window is below ``1e-15`` are dropped,
            and the rest are evaluated as dense ``B x B`` blocks. Same counts up
            to that ``1e-15`` per dropped particle pair; far less work when the
            leaves are larger than the windows' reach. See
            :func:`plan_traced_pair_counts`, which turns it on.
        sub_pair_cap: Capacity of the culled sub-block pair list
            (``sub_block > 0`` only).
    """

    leaf_size: int
    leaf_capacity: int
    max_pair_queue: int
    far_cap: int
    near_cap: int
    theta: float = 0.5
    mac_type: MACType = "dehnen"
    near_chunk: int = _DEFAULT_NEAR_CHUNK
    acceptance: Literal["window", "geometric"] = "window"
    window_eps: float = _DEFAULT_WINDOW_EPS
    smooth_band: float = 1.0
    band_cap: int = 0
    sub_block: int = 0
    sub_pair_cap: int = 0

    @property
    def smooth(self) -> bool:
        """Whether the blended (continuous) acceptance rule is on."""
        return float(self.smooth_band) > 1.0


class TracedPairCounts(NamedTuple):
    """Output of :func:`traced_soft_pair_counts`.

    ``counts`` is the differentiable result; every other field is a
    non-differentiable diagnostic. ``overflow`` is the OR of the four specific
    flags (plus ``band_overflow`` and ``sub_overflow``): when it is set the
    counts miss pairs and must not be used.
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
    band_overflow: Array  # bool: band_count > band_cap (smooth rule only)
    band_count: Array  # int: band pairs needed (0 for the hard rule)
    sub_overflow: Array  # bool: sub_count > sub_pair_cap (sub_block > 0 only)
    sub_count: Array  # int: culled sub-block pairs needed (0 without sub_block)


class _Topology(NamedTuple):
    """Integer (non-differentiable) structure of one traced build + walk."""

    order: Array  # (n,) Morton slot -> particle index
    node_ranges: Array  # (num_nodes, 2) inclusive particle-slot ranges
    num_internal: int
    walk: MutualWalkResult
    leaf_overflow: Array
    num_leaves: Array


def _window_min_max(edges_u: Array, sharpness: Array, u_lo: Array, u_hi: Array):
    """``(w_min, w_max)`` per bin of the soft windows over ``[u_lo, u_hi]``."""
    lower, upper = edges_u[:-1], edges_u[1:]

    def window(u):
        u = u[:, None]
        return jax.nn.sigmoid(sharpness * (u - lower)) - jax.nn.sigmoid(
            sharpness * (u - upper)
        )

    mid = 0.5 * (lower + upper)
    peak = jnp.clip(mid[None, :], u_lo[:, None], u_hi[:, None])
    w_peak = jax.nn.sigmoid(sharpness * (peak - lower)) - jax.nn.sigmoid(
        sharpness * (peak - upper)
    )
    w_lo, w_hi = window(u_lo), window(u_hi)
    return jnp.minimum(w_lo, w_hi), jnp.maximum(jnp.maximum(w_lo, w_hi), w_peak)


def _separation_interval_u(dist_sq, radius_a, radius_b, log: bool):
    """Bounds of every particle separation of a node pair, in window coordinates."""
    d = jnp.sqrt(dist_sq)
    s = radius_a + radius_b
    lo = d - s
    hi = d + s
    if log:
        tiny = jnp.finfo(d.dtype).tiny
        # lo <= 0: the nodes may touch, so the range reaches r -> 0 (u -> -inf)
        return jnp.log(jnp.maximum(lo, tiny)), jnp.log(jnp.maximum(hi, tiny))
    return lo, hi


def _make_window_predicates(log: bool):
    """``(accept, discard)`` callables for ``dual_tree_walk_mutual``.

    Both take ``data = (edges_u, sharpness, eps)``. ``accept``: every bin's
    window varies by at most ``eps`` over the pair's separation range, so the
    monopole term ``n_a n_b w_k(d_com)`` is within ``eps`` per particle pair of
    the exact sum. ``discard``: every window is at most ``eps`` there, so the
    pair contributes at most ``eps`` per particle pair and is pruned.
    """

    def accept(data, a, b, dist_sq, radius_a, radius_b):
        del a, b
        edges_u, sharpness, eps = data
        u_lo, u_hi = _separation_interval_u(dist_sq, radius_a, radius_b, log)
        w_min, w_max = _window_min_max(edges_u, sharpness, u_lo, u_hi)
        return jnp.all(w_max - w_min <= eps, axis=-1)

    def discard(data, a, b, dist_sq, radius_a, radius_b):
        del a, b
        edges_u, sharpness, eps = data
        u_lo, u_hi = _separation_interval_u(dist_sq, radius_a, radius_b, log)
        _, w_max = _window_min_max(edges_u, sharpness, u_lo, u_hi)
        return jnp.all(w_max <= eps, axis=-1)

    return accept, discard


# Module-level, so the walk's static arguments hash the same on every call.
_WINDOW_PREDICATES = {
    True: _make_window_predicates(True),
    False: _make_window_predicates(False),
}


# --------------------------------------------------------------------------
# smooth (blended) acceptance
# --------------------------------------------------------------------------


def _smootherstep(t: Array) -> Array:
    """C2 ramp ``6t^5 - 15t^4 + 10t^3`` on ``[0, 1]`` (``t`` already clipped)."""
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _ramp(x: Array, lo: Array, log_beta: Array) -> Array:
    """0 for ``x <= lo``, 1 for ``x >= lo * beta``, C2 in ``log x`` between.

    ``x`` is floored at ``lo`` before the log, so ``x = 0`` gives a finite value
    and a zero (not NaN) gradient.
    """
    t = jnp.log(jnp.maximum(x, lo) / lo) / log_beta
    return _smootherstep(jnp.clip(t, 0.0, 1.0))


def _safe_sqrt(x: Array) -> Array:
    """``sqrt(x)`` with a zero gradient at ``x = 0``; same values as sqrt."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _blend_fractions(kind: str, log: bool, params, dist_sq, radius_a, radius_b):
    """``(sigma, kappa)`` of node pairs: far fraction and keep fraction.

    ``kind="window"``, ``params = (edges_u, sharpness, eps, log_beta)``: with
    ``v`` the largest per-bin window variation and ``m`` the largest window over
    the pair's separation range ``[d - s, d + s]``, ``sigma`` ramps from 1 at
    ``v <= eps / beta`` to 0 at ``v >= eps``, and ``kappa`` from 0 at
    ``m <= eps / beta`` to 1 at ``m >= eps``.

    ``kind="geometric"``, ``params = (theta, log_beta)``: ``sigma`` ramps from 1
    at ``(r_a + r_b) / d <= theta / beta`` to 0 at ``>= theta``; ``kappa = 1``.

    Smooth in the node geometry, so -- evaluated on differentiable centres and
    radii -- in the positions; the walk evaluates the same function on
    ``stop_gradient`` geometry to decide what to emit.
    """
    if kind == "window":
        edges_u, sharpness, eps, log_beta = params
        d = _safe_sqrt(dist_sq)
        s = radius_a + radius_b
        lo, hi = d - s, d + s
        if log:
            tiny = jnp.finfo(d.dtype).tiny
            lo, hi = jnp.log(jnp.maximum(lo, tiny)), jnp.log(jnp.maximum(hi, tiny))
        w_min, w_max = _window_min_max(edges_u, sharpness, lo, hi)
        v = jnp.max(w_max - w_min, axis=-1)
        m = jnp.max(w_max, axis=-1)
        eps_lo = eps / jnp.exp(log_beta)
        sigma = 1.0 - _ramp(v, eps_lo, log_beta)
        kappa = _ramp(m, eps_lo, log_beta)
        return sigma, kappa
    theta, log_beta = params
    s = radius_a + radius_b
    # ratio^2 = (s / d)^2. A self pair (d = 0) is never far: sigma = 0, and the
    # double ``where`` keeps its gradient zero instead of NaN.
    apart = dist_sq > 0
    ratio_sq = (s * s) / jnp.where(apart, dist_sq, 1.0)
    theta_lo_sq = (theta / jnp.exp(log_beta)) ** 2
    sigma = 1.0 - _ramp(ratio_sq, theta_lo_sq, 2.0 * log_beta)
    sigma = jnp.where(apart, sigma, 0.0)
    return sigma, jnp.ones_like(sigma)


def _make_smooth_predicates(kind: str, log: bool):
    """``(accept, discard, band)`` hooks of the smooth rule for the walk.

    ``data = (params, node_radius)``. The walk's own ``radii`` argument is the
    position-independent split key (the node count), so the hooks read the
    bounding radii from ``data`` by node index.
    """

    def fractions(data, a, b, dist_sq):
        params, node_radius = data
        return _blend_fractions(
            kind, log, params, dist_sq, node_radius[a], node_radius[b]
        )

    def accept(data, a, b, dist_sq, radius_a, radius_b):
        del radius_a, radius_b
        sigma, kappa = fractions(data, a, b, dist_sq)
        return (sigma > 0) & (kappa > 0)

    def discard(data, a, b, dist_sq, radius_a, radius_b):
        del radius_a, radius_b
        _, kappa = fractions(data, a, b, dist_sq)
        return kappa <= 0

    def band(data, a, b, dist_sq, radius_a, radius_b):
        del radius_a, radius_b
        sigma, kappa = fractions(data, a, b, dist_sq)
        return ((sigma > 0) & (sigma < 1)) | (kappa < 1)

    return accept, (discard if kind == "window" else None), band


_SMOOTH_PREDICATES = {
    (kind, log): _make_smooth_predicates(kind, log)
    for kind in ("window", "geometric")
    for log in (True, False)
}


# --------------------------------------------------------------------------
# periodic box
# --------------------------------------------------------------------------


def _make_geometric_accept(mac_type: MACType):
    """The walk's own opening-angle test as a ``pair_accept`` hook.

    ``data`` is ``theta``. Used only in a periodic box, where the hook has to
    see the minimum-image distance (:func:`_min_image_hook`); the non-periodic
    geometric rule keeps the walk's built-in test.
    """

    def accept(data, a, b, dist_sq, radius_a, radius_b):
        theta = data
        return _compute_mac_ok(
            mac_type=mac_type,
            theta_sq=theta * theta,
            dist_sq=dist_sq,
            extent_target=radius_a,
            extent_source=radius_b,
            valid_pairs=jnp.ones(dist_sq.shape, dtype=bool),
            different_nodes=a != b,
        )

    return accept


_GEOMETRIC_ACCEPT = {m: _make_geometric_accept(m) for m in ("bh", "dehnen", "engblom")}


@functools.lru_cache(maxsize=None)
def _min_image_hook(hook, box: tuple[float, ...]):
    """``hook`` evaluated on the minimum-image centre distance in ``box``.

    The wrapped hook takes ``data = (inner_data, centers)``, recomputes the
    squared centre distance through :func:`yggdrax.periodic.minimum_image` and
    calls ``hook(inner_data, a, b, dist_sq_mi, radius_a, radius_b)``. Cached,
    so the same ``(hook, box)`` gives the same callable and the walk's static
    arguments hash equal across calls.

    The walk is run *without* its own ``box_size`` on purpose: its periodic
    guard refuses far acceptance unless ``d + r_a + r_b < L/2``, which a
    radial pair count does not need (see the module docstring, "Periodic
    box") and which would force every node pair beyond ``L/2`` -- about half
    of all pairs in a cube -- down to exact leaf pairs under the geometric
    rule.
    """

    def wrapped(data, a, b, dist_sq, radius_a, radius_b):
        del dist_sq
        inner, centers = data
        delta = minimum_image(centers[b] - centers[a], box)
        return hook(inner, a, b, jnp.sum(delta * delta, axis=-1), radius_a, radius_b)

    return wrapped


def _periodic_positions(
    positions: Array, membership: Array | None, box: tuple[float, ...]
) -> tuple[Array, Array | None]:
    """``(positions, membership)`` as a periodic evaluation uses them.

    Without ``membership`` the positions are wrapped into ``[0, L)``; the wrap
    has a unit Jacobian. With it, ``membership`` is wrapped (no gradient) and
    each live position is moved to its image nearest its membership position,
    so a leaf's particles stay together when one crosses a face during a
    trajectory (a wrap would throw it to the far side and blow up its leaf's
    bounding sphere). Live positions may then lie slightly outside the box;
    every distance below is minimum-image, so that is harmless.
    """
    if membership is None:
        return wrap_positions(positions, box), None
    home = wrap_positions(jax.lax.stop_gradient(membership), box)
    return home + minimum_image(positions - home, box), home


def _resolve_box(
    box_size: BoxSize | None, positions: Array, bounds
) -> tuple[float, ...] | None:
    """Normalised periodic box, rejecting ``bounds`` given alongside it."""
    box = normalize_box_size(box_size, int(jnp.shape(positions)[-1]))
    if box is not None and bounds is not None:
        raise ValueError("pass either bounds or box_size, not both")
    return box


def _smooth_params(config: "TracedPairCountConfig", dtype, edges, sharpness, log):
    """The ``params`` of :func:`_blend_fractions` for ``config``, no gradient."""
    log_beta = jnp.log(jnp.asarray(config.smooth_band, dtype=dtype))
    if config.acceptance == "geometric":
        return (jnp.asarray(config.theta, dtype=dtype), log_beta)
    if edges is None or sharpness is None:
        raise ValueError('acceptance="window" needs edges and sharpness')
    edges_ng = jax.lax.stop_gradient(jnp.asarray(edges, dtype=dtype))
    return (
        jnp.log(edges_ng) if log else edges_ng,
        jax.lax.stop_gradient(jnp.asarray(sharpness, dtype=dtype)),
        jnp.asarray(config.window_eps, dtype=dtype),
        log_beta,
    )


def _range_min_max_table(values: Array) -> tuple[Array, Array]:
    """Sparse tables of running min / max over power-of-two windows.

    ``tables[k][i]`` covers slots ``[i, i + 2^k)`` (clamped to the end), so any
    inclusive range is the min / max of two overlapping table entries. Built
    with ``minimum`` / ``maximum`` only, so the result is exactly the range's
    min / max and differentiable (to the extremal element).
    """
    n = int(values.shape[0])
    levels = max(1, int(np.ceil(np.log2(max(n, 1)))) + 1)
    mins, maxs = [values], [values]
    for k in range(1, levels):
        step = 1 << (k - 1)
        idx = jnp.minimum(jnp.arange(n) + step, n - 1)
        mins.append(jnp.minimum(mins[-1], mins[-1][idx]))
        maxs.append(jnp.maximum(maxs[-1], maxs[-1][idx]))
    return jnp.stack(mins), jnp.stack(maxs)


def _node_geometry(tables, ranges: Array, nodes: Array) -> tuple[Array, Array]:
    """Bounding-box centre and sphere radius of ``nodes``, differentiable.

    The same formulas as :func:`yggdrax.geometry.compute_tree_geometry`
    (min / max of the node's particles, ``half = max(half, 1e-6)``,
    ``radius = |half|``), evaluated on live positions, so the values equal the
    walk's ``stop_gradient`` geometry while gradients reach the extremal
    particles.
    """
    mins, maxs = tables
    levels = int(mins.shape[0])
    start = ranges[nodes, 0]
    end = jnp.maximum(ranges[nodes, 1], start)
    length = end - start + 1
    k = sum((length >= (1 << j)).astype(start.dtype) for j in range(1, levels))
    right = end - (jnp.asarray(1, start.dtype) << k) + 1
    lo = jnp.minimum(mins[k, start], mins[k, right])
    hi = jnp.maximum(maxs[k, start], maxs[k, right])
    center = 0.5 * (lo + hi)
    half = jnp.maximum(0.5 * (hi - lo), jnp.asarray(1e-6, dtype=lo.dtype))
    return center, jnp.linalg.norm(half, axis=-1)


def _pair_fractions(kind, log, params, tables, ranges, a, b, box=None):
    """Differentiable ``(sigma, kappa)`` of node pairs ``(a, b)``."""
    ca, ra = _node_geometry(tables, ranges, a)
    cb, rb = _node_geometry(tables, ranges, b)
    delta = minimum_image(cb - ca, box)
    return _blend_fractions(kind, log, params, jnp.sum(delta * delta, axis=-1), ra, rb)


def _ancestry_products(factor: Array, parent: Array) -> Array:
    """``prod`` of ``factor`` over each band slot and all its band ancestors.

    Pointer doubling over the ``parent`` links (``-1`` at a root): after ``j``
    rounds each slot holds the product over its next ``2^j`` ancestors, so
    ``ceil(log2(cap)) + 1`` rounds cover any chain. Differentiable in
    ``factor``.
    """
    cap = int(factor.shape[0])
    prod, anc = factor, parent
    for _ in range(max(1, int(np.ceil(np.log2(max(cap, 2)))) + 1)):
        has = anc >= 0
        safe = jnp.where(has, anc, 0)
        prod = prod * jnp.where(has, prod[safe], 1.0)
        anc = jnp.where(has, anc[safe], -1)
    return prod


def _resolve_bounds(positions: Array, bounds) -> tuple[Array, Array]:
    """Morton domain: the given bounds or the padded box of ``positions``."""
    if bounds is None:
        return infer_bounds(positions)
    lo, hi = bounds
    return jax.lax.stop_gradient(jnp.asarray(lo)), jax.lax.stop_gradient(
        jnp.asarray(hi)
    )


def _build_topology(
    positions: Array,
    config: TracedPairCountConfig,
    bounds,
    edges: Array | None = None,
    sharpness: float | Array | None = None,
    log: bool = True,
    membership: Array | None = None,
    box: tuple[float, ...] | None = None,
) -> _Topology:
    """Rebuild the tree and run the mutual walk, all traceable, no gradient.

    The evaluator and the planner both call this, so a plan sized by the probe
    is a plan for exactly the walk the evaluator runs. ``edges``, ``sharpness``
    and ``log`` are needed (and only read) by ``acceptance="window"``.
    ``membership`` (default ``positions``) are the positions the tree's leaf
    partition is built from; the node geometry and the walk always use
    ``positions``. With a periodic ``box`` (a normalised tuple) both arrive
    already through :func:`_periodic_positions`; the tree is built on the box
    and every acceptance hook sees minimum-image centre distances.
    """
    pos = jax.lax.stop_gradient(positions)
    build_pos = pos if membership is None else jax.lax.stop_gradient(membership)
    masses = jnp.ones(pos.shape[0], dtype=pos.dtype)
    tree, pos_sorted, _, _, leaf_overflow, num_leaves = build_static_cells_tree(
        build_pos,
        masses,
        bounds=None if box is not None else _resolve_bounds(build_pos, bounds),
        box_size=box,
        leaf_size=config.leaf_size,
        leaf_capacity=config.leaf_capacity,
        return_reordered=True,
        return_overflow=True,
        return_num_leaves=True,
    )
    if membership is not None:
        pos_sorted = pos[tree.particle_indices]
    # max_leaf_size: leaf_size is traced under jit; without the static bound
    # the geometry's leaf gather falls back to (num_leaves, n, 3).
    geometry = compute_tree_geometry(tree, pos_sorted, max_leaf_size=config.leaf_size)
    w = mutual_walk_inputs(
        tree, geometry, mac_type=config.mac_type, mask_empty_nodes=True
    )
    # Every branch picks the walk's centres, radii and hooks; one call below.
    accept = discard = band = None
    data = None
    band_cap = 0
    centers = jnp.asarray(geometry.center)
    walk_mac = None
    if config.smooth:
        if config.acceptance not in ("window", "geometric"):
            raise ValueError(f"unknown acceptance {config.acceptance!r}")
        accept, discard, band = _SMOOTH_PREDICATES[(config.acceptance, bool(log))]
        params = _smooth_params(config, pos.dtype, edges, sharpness, log)
        node_radius = jnp.asarray(geometry.radius, dtype=pos.dtype)
        # The walk's ``radii`` only pick which node of a refined pair to split
        # (the larger). Bounding radii move with the positions, so their order
        # flips and the decomposition below a band pair would switch between
        # two splits: a jump. The node COUNT is piecewise constant at fixed
        # leaf membership, so the split order is too. The hooks read the true
        # radii from ``data``.
        radii = jnp.maximum(
            tree.node_ranges[:, 1] - tree.node_ranges[:, 0] + 1, 0
        ).astype(pos.dtype)
        data = (params, node_radius)
        band_cap = max(1, int(config.band_cap))
    elif config.acceptance == "geometric":
        centers = w.centers
        radii = w.radii
        if box is None:
            walk_mac = config.mac_type
        else:
            accept = _GEOMETRIC_ACCEPT[config.mac_type]
            data = jnp.asarray(config.theta, dtype=pos.dtype)
    elif config.acceptance == "window":
        if edges is None or sharpness is None:
            raise ValueError('acceptance="window" needs edges and sharpness')
        accept, discard = _WINDOW_PREDICATES[bool(log)]
        edges_ng = jax.lax.stop_gradient(jnp.asarray(edges, dtype=pos.dtype))
        edges_u = jnp.log(edges_ng) if log else edges_ng
        data = (
            edges_u,
            jax.lax.stop_gradient(jnp.asarray(sharpness, dtype=pos.dtype)),
            jnp.asarray(config.window_eps, dtype=pos.dtype),
        )
        # Bounding spheres about the box centres, NOT the MAC's (possibly
        # scaled, depth-padded) extents: the rule needs every particle
        # separation of a pair inside [d - s, d + s].
        radii = jnp.asarray(geometry.radius, dtype=pos.dtype)
    else:
        raise ValueError(f"unknown acceptance {config.acceptance!r}")
    if box is not None:
        accept, discard, band = (
            None if hook is None else _min_image_hook(hook, box)
            for hook in (accept, discard, band)
        )
        data = (data, centers)
    walk = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        centers,
        radii,
        config.theta,
        w.root,
        max_pair_queue=config.max_pair_queue,
        far_cap=config.far_cap,
        near_cap=config.near_cap,
        mac_type=walk_mac,
        node_active=w.node_active,
        pair_accept=accept,
        pair_discard=discard,
        pair_band=band,
        band_cap=band_cap,
        pair_accept_data=data,
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
    membership: Array | None = None,
    box_size: BoxSize | None = None,
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
        membership: Optional positions ``(n, 3)`` that fix the tree's LEAF
            PARTITION (no gradient flows into them); the node geometry, the
            acceptance decisions and every separation still use
            ``positions``, so the estimator stays correct to the same error
            bound. ``None`` (default) builds the partition from ``positions``.
            Pass the start of an HMC trajectory or of a line search: with the
            smooth rule the counts are then continuous in ``positions`` along
            the whole of it, because a particle crossing a cell face no longer
            restructures the tree. Must lie inside ``bounds`` when given.
        box_size: Optional periodic box ``[0, L)^3``: a float (cube) or one
            length per dimension. Static (a Python value). Every separation is
            then the minimum-image one, the tree is built on the box (the
            positions are wrapped; any coordinates are accepted) and
            ``bounds`` must be ``None``. With ``membership``, each live
            position is taken at its image nearest its membership position, so
            a particle crossing a face does not restructure the tree either.
            Keep the largest separation the windows reach below ``L / 2`` (see
            the module docstring, "Periodic box"). ``None`` (default) is the
            open-boundary estimator, unchanged.

    Returns:
        A :class:`TracedPairCounts`. ``counts`` has shape ``(nbins,)``; read
        ``overflow`` before trusting it.
    """
    box = _resolve_box(box_size, positions, bounds)
    if box is not None:
        positions, membership = _periodic_positions(positions, membership, box)
    topo = _build_topology(
        positions,
        config,
        bounds,
        edges,
        sharpness,
        log,
        membership=membership,
        box=box,
    )
    return _accumulate(positions, edges, config, topo, sharpness, log, box=box)


# Window value below which a particle pair may be dropped by the sub-block
# culling of the near field: ln(1 / 1e-15) / sharpness beyond the outermost
# edges every window is at most 1e-15.
_SUBBLOCK_TAIL_LOG = float(np.log(1e15))


def _window_reach(edges: Array, sharpness, log: bool, dtype) -> Array:
    """Separation beyond which every soft window is below ``1e-15``. No grad."""
    e_last = jax.lax.stop_gradient(jnp.asarray(edges, dtype=dtype)[-1])
    du = _SUBBLOCK_TAIL_LOG / jax.lax.stop_gradient(jnp.asarray(sharpness, dtype))
    return e_last * jnp.exp(du) if log else e_last + du


def _leaf_blocks(
    pos_sorted: Array, ranges: Array, num_internal: int, config: TracedPairCountConfig
) -> tuple[Array, Array]:
    """Padded leaf blocks ``(leaf_capacity, leaf_size, 3)`` and their 0/1 mask.

    Leaves are nodes ``[ni, ni + leaf_capacity)``; an EMPTY padding leaf has
    the range ``(n, n - 1)``, i.e. length 0. A cell leaf never holds more than
    ``leaf_size`` particles (the partition splits a full deepest cell), so the
    block width is exact.
    """
    n = int(pos_sorted.shape[0])
    leaf = int(config.leaf_size)
    leaf_start = ranges[num_internal:, 0]
    leaf_len = jnp.clip(ranges[num_internal:, 1] - leaf_start + 1, 0, leaf)
    ramp = jnp.arange(leaf, dtype=leaf_start.dtype)
    slots = jnp.clip(leaf_start[:, None] + ramp[None, :], 0, n - 1)
    leaf_mask = (ramp[None, :] < leaf_len[:, None]).astype(pos_sorted.dtype)
    return pos_sorted[slots], leaf_mask


def _culled_subblock_pairs(
    blocks: Array,
    leaf_mask: Array,
    near_t: Array,
    near_s: Array,
    near_live: Array,
    reach: Array,
    sub_block: int,
    cap: int,
    box: tuple[float, ...] | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sub-block pairs of the near leaf pairs that can hold an in-reach pair.

    Each leaf block is cut into ``k = leaf_size / sub_block`` runs of
    consecutive (Morton-ordered, hence spatially compact) slots. For every near
    leaf pair and each of its ``k^2`` sub-block pairs, the minimum-image
    distance of the two bounding boxes is a lower bound on every particle
    separation between them (per axis ``|mi(c_b - c_a)| - (h_a + h_b)``,
    floored at 0). Pairs whose bound reaches ``reach`` -- where every window is
    below ``1e-15`` -- and pairs with an empty side are dropped; the rest are
    compacted, in order, into ``cap`` slots. Topology only: no gradient.

    Returns:
        ``(rows_a, rows_b, pair, valid, count)``: sub-block row indices into
        ``blocks.reshape(-1, sub_block, 3)``, the near-list slot each came from,
        the live mask of the ``cap`` slots, and the number needed.
    """
    b = int(sub_block)
    lc, leaf, dim = blocks.shape
    k = leaf // b
    pos = jax.lax.stop_gradient(blocks).reshape(lc, k, b, dim)
    occ = (leaf_mask.reshape(lc, k, b) > 0)[..., None]
    big = jnp.asarray(jnp.finfo(pos.dtype).max, pos.dtype)
    lo = jnp.min(jnp.where(occ, pos, big), axis=2)
    hi = jnp.max(jnp.where(occ, pos, -big), axis=2)
    filled = jnp.any(occ[..., 0], axis=-1)  # (lc, k)
    center = jnp.where(filled[..., None], 0.5 * (lo + hi), 0.0)
    half = jnp.where(filled[..., None], 0.5 * (hi - lo), 0.0)

    ca, ha = center[near_t][:, :, None, :], half[near_t][:, :, None, :]
    cb, hb = center[near_s][:, None, :, :], half[near_s][:, None, :, :]
    gap = jnp.maximum(jnp.abs(minimum_image(cb - ca, box)) - (ha + hb), 0.0)
    close = jnp.sum(gap * gap, axis=-1) < reach * reach
    keep = (
        near_live[:, None, None]
        & filled[near_t][:, :, None]
        & filled[near_s][:, None, :]
        & close
    ).reshape(-1)
    idx_dtype = near_t.dtype
    order = jnp.cumsum(keep.astype(idx_dtype)) - 1
    count = jnp.sum(keep.astype(idx_dtype))
    target = jnp.where(keep & (order < cap), order, cap)
    flat = jnp.arange(keep.shape[0], dtype=idx_dtype)
    pair = flat // (k * k)
    u = (flat // k) % k
    v = flat % k
    rows_a = near_t[pair] * k + u
    rows_b = near_s[pair] * k + v
    out = jnp.zeros((cap,), idx_dtype)
    rows_a = out.at[target].set(rows_a, mode="drop")
    rows_b = out.at[target].set(rows_b, mode="drop")
    pair = out.at[target].set(pair, mode="drop")
    valid = jnp.arange(cap, dtype=idx_dtype) < jnp.minimum(count, cap)
    return rows_a, rows_b, pair, valid, count


def _accumulate(
    positions: Array,
    edges: Array,
    config: TracedPairCountConfig,
    topo: _Topology,
    sharpness: float | Array,
    log: bool,
    box: tuple[float, ...] | None = None,
) -> TracedPairCounts:
    """The differentiable half: soft counts of ``positions`` over ``topo``.

    Split from the build so a caller can hold a topology fixed (the benches in
    ``bench/differentiability`` evaluate a step's end point on the start
    point's topology to separate the smooth change from the switch). With a
    periodic ``box``, ``positions`` are the ones :func:`_periodic_positions`
    returned for the build, and every separation is minimum-image.
    """
    walk = topo.walk
    dtype = positions.dtype
    nbins = int(edges.shape[0]) - 1
    leaf = int(config.leaf_size)
    ni = topo.num_internal
    ranges = topo.node_ranges

    # Differentiable: gather the live positions through the integer order.
    pos_sorted = positions[topo.order]

    # padded leaf blocks (leaf_capacity, leaf_size, 3) + validity
    blocks, leaf_mask = _leaf_blocks(pos_sorted, ranges, ni, config)
    leaf_rows = jnp.arange(int(config.leaf_capacity), dtype=ranges.dtype)

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
        box_size=box,
    )

    # near leaf pairs: canonical (a < b), each once, -1 padded
    near_live = walk.near_a >= 0
    near_t = jnp.where(near_live, walk.near_a - ni, 0)
    near_s = jnp.where(near_live, walk.near_b - ni, 0)
    # far node pairs (live slots): needed by the smooth weights below
    far_live = walk.far_a >= 0
    fa = jnp.where(far_live, walk.far_a, 0)
    fb = jnp.where(far_live, walk.far_b, 0)
    if config.smooth:
        near_factor, far_factor = _smooth_weights(
            walk, config, edges, sharpness, log, pos_sorted, ranges, fa, fb, box
        )
        near_weight = jnp.where(near_live, near_factor, 0.0)
    else:
        near_weight = near_live.astype(dtype)
        far_factor = None
    sub = int(config.sub_block)
    if sub > 0:
        if leaf % sub:
            raise ValueError(f"sub_block {sub} must divide leaf_size {leaf}")
        cap = max(1, int(config.sub_pair_cap))
        rows_a, rows_b, pair, valid, sub_count = _culled_subblock_pairs(
            blocks,
            leaf_mask,
            near_t,
            near_s,
            near_live,
            _window_reach(edges, sharpness, log, dtype),
            sub,
            cap,
            box,
        )
        sub_overflow = sub_count > cap
        cross = _accumulate_block_pairs(
            blocks.reshape(-1, sub, blocks.shape[-1]),
            leaf_mask.reshape(-1, sub),
            rows_a,
            rows_b,
            edges,
            sharpness,
            log,
            nbins,
            None,
            # same live (chunk, block, block, nbins) intermediate as the dense path
            int(config.near_chunk) * (leaf // sub) ** 2,
            row_weight=jnp.where(valid, near_weight[pair], 0.0),
            box_size=box,
        )
    else:
        sub_overflow = jnp.asarray(False)
        sub_count = jnp.zeros((), dtype=walk.near_count.dtype)
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
            row_weight=near_weight,
            box_size=box,
        )

    # far node pairs: monopole at the centres of mass, each once
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
    # Periodic: the minimum image of the COM displacement. It may differ from
    # the image some particle pairs of the node pair use, but every one of
    # those separations lies in the same [d - s, d + s] as this one, which is
    # all the window rule's bound needs (module docstring, "Periodic box").
    delta_far = minimum_image(com_a - com_b, box)
    d_far = jnp.sqrt(jnp.sum(delta_far * delta_far, axis=-1) + 1e-30)
    w_far = soft_bin_weights(d_far, edges, sharpness, log=log)  # (far_cap, nbins)
    pair_weight = jnp.where(far_live, cnt_a * cnt_b, 0.0)
    if far_factor is not None:
        pair_weight = pair_weight * far_factor
    far = jnp.sum(pair_weight[:, None] * w_far, axis=0)

    leaf_overflow = jnp.asarray(topo.leaf_overflow, bool)
    overflow = (
        leaf_overflow
        | walk.queue_overflow
        | walk.far_overflow
        | walk.near_overflow
        | sub_overflow
    )
    if walk.band_overflow is not None and walk.band_count is not None:
        band_overflow = jnp.asarray(walk.band_overflow, bool)
        band_count = walk.band_count
        overflow = overflow | band_overflow
    else:
        band_overflow = jnp.asarray(False)
        band_count = jnp.zeros((), dtype=walk.far_count.dtype)
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
        band_overflow=band_overflow,
        band_count=band_count,
        sub_overflow=sub_overflow,
        sub_count=sub_count,
    )


def _smooth_weights(
    walk: MutualWalkResult,
    config: TracedPairCountConfig,
    edges: Array,
    sharpness: float | Array,
    log: bool,
    pos_sorted: Array,
    ranges: Array,
    fa: Array,
    fb: Array,
    box: tuple[float, ...] | None = None,
) -> tuple[Array, Array]:
    """Blend weights of the smooth rule: ``(near_factor, far_factor)``.

    Recomputed on live node geometry, so the gradient includes the
    ``d(sigma)/dx`` terms: the counts are then continuous across acceptance
    switches and their gradient is the gradient of the blended estimator, not of
    a frozen-weight surrogate. ``near_factor[i]`` is the product of the band
    factors ``kappa (1 - sigma)`` over the ancestry of near pair ``i`` (itself
    included); ``far_factor[i]`` is ``sigma kappa`` of far pair ``(fa, fb)[i]``
    times the product over its strict ancestors.
    """
    if (
        walk.band_a is None
        or walk.band_b is None
        or walk.band_parent is None
        or walk.band_accepted is None
        or walk.band_count is None
        or walk.far_parent is None
        or walk.near_parent is None
    ):
        raise ValueError("the smooth rule needs a walk run with pair_band")
    dtype = pos_sorted.dtype
    kind = config.acceptance
    params = _smooth_params(config, dtype, edges, sharpness, log)
    tables = _range_min_max_table(pos_sorted)
    cap = int(walk.band_a.shape[0])
    band_live = jnp.arange(cap) < jnp.minimum(walk.band_count, cap)
    ba = jnp.where(band_live, walk.band_a, 0)
    bb = jnp.where(band_live, walk.band_b, 0)
    b_sigma, b_kappa = _pair_fractions(kind, log, params, tables, ranges, ba, bb, box)
    b_factor = b_kappa * (1.0 - b_sigma * walk.band_accepted.astype(dtype))
    band_prod = _ancestry_products(
        jnp.where(band_live, b_factor, 1.0),
        jnp.where(band_live, walk.band_parent, -1),
    )

    def hang(parent: Array) -> Array:
        has = parent >= 0
        return jnp.where(has, band_prod[jnp.where(has, parent, 0)], 1.0)

    f_sigma, f_kappa = _pair_fractions(kind, log, params, tables, ranges, fa, fb, box)
    return hang(walk.near_parent), f_sigma * f_kappa * hang(walk.far_parent)


def _pow2(x: int) -> int:
    """Smallest power of two >= ``x`` (and >= 2)."""
    return 1 << max(int(x) - 1, 1).bit_length()


def plan_traced_pair_counts(
    positions: Array | Sequence[Array],
    *,
    edges: Array | None = None,
    sharpness: float | Array | None = None,
    log: bool = True,
    acceptance: Literal["window", "geometric"] = "window",
    window_eps: float = _DEFAULT_WINDOW_EPS,
    smooth_band: float = 1.0,
    theta: float = 0.5,
    leaf_size: int = 16,
    mac_type: MACType = "dehnen",
    bounds: tuple[Array, Array] | None = None,
    box_size: BoxSize | None = None,
    near_chunk: int = _DEFAULT_NEAR_CHUNK,
    sub_block: int | None = None,
    leaf_slack: float = 1.25,
    list_slack: float = 1.5,
    min_list_cap: int = 256,
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
        edges: Bin edges the traced call will use; required by
            ``acceptance="window"``, whose walk depends on them.
        sharpness: Window sharpness the traced call will use; required by
            ``acceptance="window"``.
        log: Bin in ``log`` separation, as for the traced call.
        acceptance: ``"window"`` or ``"geometric"``; see
            :class:`TracedPairCountConfig`.
        window_eps: Window tolerance of ``acceptance="window"``.
        smooth_band: Blend-band ratio of the smooth rule; ``1.0`` is the hard
            rule. See :class:`TracedPairCountConfig`.
        theta: Opening angle of ``acceptance="geometric"``.
        leaf_size: Maximum particles per cell leaf.
        mac_type: Geometric acceptance criterion.
        bounds: Fixed Morton domain, as for :func:`traced_soft_pair_counts`.
            Pass the same value to both.
        box_size: Periodic box, as for :func:`traced_soft_pair_counts`. Pass
            the same value to both.
        near_chunk: Near-field chunk size stored in the config.
        sub_block: Sub-block width of the culled near field (see
            :class:`TracedPairCountConfig`). ``None`` (default) picks 8 when
            ``leaf_size`` is a multiple of 8 and at least 16 and ``edges`` and
            ``sharpness`` are given (the culling reads the windows' reach), else
            0; ``0`` keeps the dense leaf-pair near field.
        leaf_slack: Multiplier on the probed live-leaf count.
        list_slack: Multiplier on the probed wavefront peak and list lengths.
        min_list_cap: Floor on the queue and list capacities. A list the
            probes barely use (the window rule's far list, often a handful of
            pairs) would otherwise get a cap a slightly different catalogue
            overflows; a floor costs almost nothing.
        max_doublings: Give up after this many capacity increases.

    Returns:
        A :class:`TracedPairCountConfig`.

    Raises:
        RuntimeError: If the walk capacities do not converge.
        ValueError: If ``acceptance="window"`` is planned without ``edges``
            and ``sharpness``.
    """
    if acceptance == "window" and (edges is None or sharpness is None):
        raise ValueError('acceptance="window" is planned from edges and sharpness')
    if isinstance(positions, (list, tuple)):
        catalogues = [jnp.asarray(p) for p in positions]
    else:
        arr = jnp.asarray(positions)
        catalogues = [arr] if arr.ndim == 2 else [arr[b] for b in range(arr.shape[0])]

    leaf = int(leaf_size)
    if sub_block is None:
        sub_block = 8 if (leaf % 8 == 0 and leaf >= 16) else 0
        if edges is None or sharpness is None:
            sub_block = 0
    sub_block = int(sub_block)
    if sub_block and (leaf % sub_block or edges is None or sharpness is None):
        raise ValueError(
            "sub_block must divide leaf_size and needs edges and sharpness, got "
            f"sub_block={sub_block}, leaf_size={leaf}"
        )
    if float(smooth_band) < 1.0:
        raise ValueError(f"smooth_band must be >= 1, got {smooth_band}")
    smooth = float(smooth_band) > 1.0
    need_leaves = need_queue = need_far = need_near = need_band = need_sub = 1
    for pos in catalogues:
        n = int(pos.shape[0])
        periodic = _resolve_box(box_size, pos, bounds)
        if periodic is not None:
            pos, _ = _periodic_positions(pos, None, periodic)
        # --- live leaves: the count is taken before the cut, so one probe at
        # any capacity gives the exact need. ---
        cap0 = max(64, 4 * (-(-n // leaf)))
        out = build_static_cells_tree(
            pos,
            jnp.ones(n, pos.dtype),
            bounds=None if periodic is not None else _resolve_bounds(pos, bounds),
            box_size=periodic,
            leaf_size=leaf,
            leaf_capacity=cap0,
            return_num_leaves=True,
        )
        live = int(out[-1])
        leaf_capacity = int(leaf_slack * live) + 8

        # --- walk: grow the caps until the probe fits, then record its need ---
        q, far_cap, near_cap = 1 << 12, 1 << 14, 1 << 14
        band_cap = (1 << 14) if smooth else 0
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
                acceptance=acceptance,
                window_eps=float(window_eps),
                smooth_band=float(smooth_band),
                band_cap=band_cap,
            )
            # one compiled probe per capacity set: the walk and, when sub-blocks
            # are on, the culled sub-block count of the same topology
            walk, sub_count = jax.jit(
                lambda p, c=cfg: _probe(
                    p, c, bounds, edges, sharpness, log, periodic, sub_block
                )
            )(pos)
            qo, fo, no = (
                bool(walk.queue_overflow),
                bool(walk.far_overflow),
                bool(walk.near_overflow),
            )
            bo = smooth and bool(walk.band_overflow)
            if not (qo or fo or no or bo):
                break
            q *= 4 if qo else 1
            far_cap *= 4 if fo else 1
            near_cap *= 4 if no else 1
            band_cap *= 4 if bo else 1
        else:
            raise RuntimeError(
                f"mutual walk capacities did not converge after {max_doublings} "
                f"increases (n={n}, theta={theta}, leaf_size={leaf})"
            )
        need_leaves = max(need_leaves, live)
        need_queue = max(need_queue, int(walk.peak_wavefront))
        need_far = max(need_far, int(walk.far_count))
        need_near = max(need_near, int(walk.near_count))
        if smooth:
            need_band = max(need_band, int(walk.band_count))
        need_sub = max(need_sub, int(sub_count))

    return TracedPairCountConfig(
        leaf_size=leaf,
        leaf_capacity=int(leaf_slack * need_leaves) + 8,
        max_pair_queue=_pow2(
            max(min_list_cap, int(np.ceil(list_slack * need_queue)) + 1)
        ),
        far_cap=_pow2(max(min_list_cap, int(np.ceil(list_slack * need_far)) + 1)),
        near_cap=_pow2(max(min_list_cap, int(np.ceil(list_slack * need_near)) + 1)),
        theta=float(theta),
        mac_type=mac_type,
        near_chunk=int(near_chunk),
        acceptance=acceptance,
        window_eps=float(window_eps),
        smooth_band=float(smooth_band),
        band_cap=(
            _pow2(max(min_list_cap, int(np.ceil(list_slack * need_band)) + 1))
            if smooth
            else 0
        ),
        sub_block=sub_block,
        sub_pair_cap=(
            _pow2(max(min_list_cap, int(np.ceil(list_slack * need_sub)) + 1))
            if sub_block
            else 0
        ),
    )


def _probe(pos, config, bounds, edges, sharpness, log, box, sub_block):
    """Planner probe: the walk of ``pos`` and its culled sub-block pair count."""
    topo = _build_topology(pos, config, bounds, edges, sharpness, log, box=box)
    if not sub_block:
        return topo.walk, jnp.zeros((), jnp.int32)
    ni = topo.num_internal
    blocks, mask = _leaf_blocks(pos[topo.order], topo.node_ranges, ni, config)
    live = topo.walk.near_a >= 0
    count = _culled_subblock_pairs(
        blocks,
        mask,
        jnp.where(live, topo.walk.near_a - ni, 0),
        jnp.where(live, topo.walk.near_b - ni, 0),
        live,
        _window_reach(edges, sharpness, log, pos.dtype),
        sub_block,
        1,
        box,
    )[4]
    return topo.walk, count
