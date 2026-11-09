"""Distributed clustering statistics in a periodic box (C1).

Two payloads, one pipeline: soft binned pair counts for the two-point correlation
function, and k-nearest-neighbour distances for kNN-CDFs, on particles sharded
across the devices of a 1-D mesh. ``docs/distributed_statistics.md`` is the design
note; in brief, per device inside one ``shard_map``:

1. wrap and SFC-partition the particles into contiguous Morton domains;
2. build a cell tree over the local particles and publish an occupancy cut of it,
   each cell as a ball whose radius is inflated by the cell's REACH;
3. walk the local tree against every other device's balls (minimum image, prune
   disjoint spheres, never accept far) to decide which leaves each device needs;
4. ship those leaves' particles (once per destination) with one ragged
   all-to-all: the HALO;
5. build one tree over ``[local ; halo]`` and walk it with the payload's
   acceptance, pruning node pairs that hold no (query, data) particle pair;
6. ξ: accumulate with ownership weights ``(o_i + o_j) / 2`` and ``psum``;
   kNN: a streamed, segmented top-k over each query leaf's candidate leaves.

The result does not depend on the device count (to summation order) when no far
pair is accepted, which is the default :func:`exact_acceptance`.

Topology (codes, sorts, trees, walks, send lists) is computed from
``stop_gradient`` positions; every value the statistics read is the live input
routed through integer gathers and the two (linear) exchanges. ``jax.grad`` with
respect to positions (and weights) therefore runs through both drivers at fixed
topology: the reverse pass of each exchange returns the halo's cotangents to the
owners, and the result equals the single-device gradient (C2; see the design
note's *Gradients* section).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map as _shard_map  # jax >= 0.8 (the package's floor)
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from .._tree_impl import build_static_cells_tree
from ..dtypes import INDEX_DTYPE, as_index, require_x64
from ..geometry import compute_tree_geometry
from ..interactions import dual_tree_walk_mutual
from ..periodic import BoxSize, minimum_image, normalize_box_size, wrap_positions
from .comm import ragged_all_to_all_exchange
from .export import build_send_buffers, export_walk
from .partition import sfc_partition
from .sharding import AXIS_NAME, make_mesh

__all__ = [
    "DistributedKNN",
    "DistributedPairCounts",
    "DistributedStatsConfig",
    "PairAcceptance",
    "distributed_knn",
    "distributed_pair_counts",
    "exact_acceptance",
    "geometric_acceptance",
    "knn_shard",
    "pair_counts_shard",
    "soft_window",
    "soft_window_cutoffs",
]

# Particle flag columns carried through the partition and the halo exchange.
_QUERY = 0  # a query / owned particle: its pairs (ξ) or neighbours (kNN) are wanted
_DATA = 1  # a data particle: it can be somebody's pair partner or neighbour


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedStatsConfig:
    """Static capacities of the distributed statistics pipeline.

    Every buffer in the pipeline has a static length, and every one of them can
    overflow; each has a flag on the result. ``None`` fields are derived by
    :meth:`resolved`. :meth:`for_problem` gives generous defaults.

    Attributes
    ----------
    capacity:
        Particle rows per device after the partition.
    halo_capacity:
        Halo particles a device can receive.
    leaf_size:
        Particles per cell-tree leaf (both trees).
    leaf_capacity:
        Leaves of the local tree. Default ``capacity`` (always enough: a live
        leaf holds at least one particle).
    halo_leaf_capacity:
        Leaves of the ``[local ; halo]`` tree. Default
        ``capacity + halo_capacity``.
    summary_max_leaves:
        Leaves per published summary cell (:func:`occupancy_cut`). Smaller
        cells import a thinner halo and cost a larger ``all_gather``.
    summary_capacity:
        Cells per device. Default ``leaf_capacity``.
    export_queue, export_near_cap:
        Wavefront and (cell, leaf) list of the export walk.
    send_leaf_capacity:
        Distinct (destination, leaf) rows a device sends. Default
        ``export_near_cap``.
    send_capacity:
        Particles a device sends, summed over destinations. Default
        ``2 * halo_capacity``.
    max_pair_queue, far_cap, near_cap:
        The walk over ``[local ; halo]``.
    first_pass_reach_factor:
        kNN only: in the local first pass a node looks this many of its own
        bounding radii beyond its sphere. Larger finds more first-pass bounds
        (a thinner halo) at more first-pass work; it never affects correctness.
    num_samples:
        Partition samples per device (:func:`sfc_partition`).
    chunk:
        Leaf pairs (or query leaves) evaluated per step of the near-field scan.
    exchange:
        Implementation of the partition and halo exchanges: ``"auto"``,
        ``"buf"`` or ``"native"``
        (:func:`~yggdrax.distributed.comm.resolve_ragged_method`). ``"auto"`` is
        safe under ``jax.grad`` too: it picks the native ragged all-to-all only
        on GPU/TPU with ``jax >=`` :data:`~yggdrax.distributed.comm.RAGGED_NATIVE_FIXED_JAX`,
        the release that also fixed the native op's reverse-pass side effect.
    """

    capacity: int
    halo_capacity: int
    leaf_size: int = 16
    leaf_capacity: Optional[int] = None
    halo_leaf_capacity: Optional[int] = None
    summary_max_leaves: int = 4
    summary_capacity: Optional[int] = None
    export_queue: int = 1 << 13
    export_near_cap: int = 1 << 14
    send_leaf_capacity: Optional[int] = None
    send_capacity: Optional[int] = None
    max_pair_queue: int = 1 << 14
    far_cap: int = 1 << 12
    near_cap: int = 1 << 16
    first_pass_reach_factor: float = 2.0
    num_samples: int = 64
    chunk: int = 256
    exchange: str = "auto"

    def resolved(self) -> "DistributedStatsConfig":
        """Fill every ``None`` capacity from the others.

        Returns
        -------
        DistributedStatsConfig
            A copy with no ``None`` field.
        """
        if self.exchange not in ("auto", "buf", "native"):
            raise ValueError(
                f"exchange must be 'auto', 'buf' or 'native'; got {self.exchange!r}"
            )
        leaf_cap = self.leaf_capacity or self.capacity
        return replace(
            self,
            leaf_capacity=leaf_cap,
            halo_leaf_capacity=self.halo_leaf_capacity
            or (self.capacity + self.halo_capacity),
            summary_capacity=self.summary_capacity or leaf_cap,
            send_leaf_capacity=self.send_leaf_capacity or self.export_near_cap,
            send_capacity=self.send_capacity or 2 * self.halo_capacity,
        )

    @classmethod
    def for_problem(
        cls,
        n_per_device: int,
        *,
        halo_fraction: float = 1.0,
        slack: float = 1.5,
        leaf_size: int = 16,
        **overrides: Any,
    ) -> "DistributedStatsConfig":
        """Generous capacities for about ``n_per_device`` particles per device.

        Parameters
        ----------
        n_per_device:
            Mean particles per device (all particles / devices).
        halo_fraction:
            Expected halo size as a fraction of ``n_per_device``. The halo is a
            shell of thickness ``reach`` around each domain, so this grows with
            ``reach / domain size``.
        slack:
            Multiplier on the partition capacity for load imbalance.
        leaf_size:
            Cell-tree leaf size.
        **overrides:
            Any field, set explicitly.

        Returns
        -------
        DistributedStatsConfig
            The resolved configuration.
        """
        cap = max(leaf_size, int(math.ceil(slack * n_per_device)))
        halo = max(leaf_size, int(math.ceil(halo_fraction * slack * n_per_device)))
        n_comb = cap + halo
        pair_cap = 1 << max(10, int(math.ceil(math.log2(max(n_comb, 2) * 8))))
        fields: dict[str, Any] = dict(
            capacity=cap,
            halo_capacity=halo,
            leaf_size=leaf_size,
            export_queue=pair_cap,
            export_near_cap=pair_cap,
            max_pair_queue=pair_cap,
            near_cap=2 * pair_cap,
            far_cap=pair_cap,
        )
        fields.update(overrides)
        return cls(**fields).resolved()


class PairAcceptance(NamedTuple):
    """A pluggable node-pair rule for the walk over ``[local ; halo]``.

    Both callables have :func:`~yggdrax.interactions.dual_tree_walk_mutual`'s
    ``pair_accept`` signature, ``f(data, a, b, dist_sq, radius_a, radius_b)`` ->
    bool mask, and must be symmetric in ``a`` and ``b``. ``dist_sq`` is the
    minimum-image distance of the node centres; the radii BOUND every particle's
    distance from its node centre (``geometry.radius``).

    Attributes
    ----------
    accept:
        Far (monopole) pairs; ``None`` accepts none.
    discard:
        Pairs to prune; ``None`` prunes none beyond the driver's own range and
        ownership prunes, which are always applied.
    data:
        Pytree handed to both as ``data``; traced.
    """

    accept: Optional[Callable[..., Array]] = None
    discard: Optional[Callable[..., Array]] = None
    data: Any = None


def _geometric_accept(data, a, b, dist_sq, radius_a, radius_b):
    """``(r_a + r_b) < theta d`` on bounding radii."""
    del a, b
    theta = data
    s = radius_a + radius_b
    return s * s < theta * theta * dist_sq


def exact_acceptance() -> PairAcceptance:
    """No far pairs: every pair inside the cut is evaluated particle by particle.

    Returns
    -------
    PairAcceptance
        The default rule; results agree across device counts to round-off.
    """
    return PairAcceptance()


def geometric_acceptance(theta: float) -> PairAcceptance:
    """Far pairs by the opening angle on bounding radii, ``r_a + r_b < theta d``.

    The monopole error depends on the tree, hence on the partition: results
    agree across device counts only to that error.

    Parameters
    ----------
    theta:
        Opening angle.

    Returns
    -------
    PairAcceptance
        The rule, with ``theta`` as its (traced) data.
    """
    return PairAcceptance(accept=_geometric_accept, data=jnp.asarray(theta))


def _never(data, a, b, dist_sq, radius_a, radius_b):
    """Accept nothing."""
    del data, a, b, radius_a, radius_b
    return jnp.zeros(jnp.shape(dist_sq), dtype=bool)


def _disjoint(data, a, b, dist_sq, radius_a, radius_b):
    """Prune when the two balls do not intersect (the export walk's rule)."""
    del data, a, b
    s = radius_a + radius_b
    return dist_sq > s * s


@functools.lru_cache(maxsize=None)
def _composed(
    accept: Optional[Callable[..., Array]], discard: Optional[Callable[..., Array]]
) -> tuple[Callable[..., Array], Callable[..., Array]]:
    """The walk's ``(pair_accept, pair_discard)`` around a payload's rule.

    Cached, so the same rule gives the same (hashable, static) callables on every
    call and the walk does not recompile.
    """

    def composed_accept(data, a, b, dist_sq, radius_a, radius_b):
        user, _cq, _cd, _lo, _reach = data
        if accept is None:
            return _never(None, a, b, dist_sq, radius_a, radius_b)
        return accept(user, a, b, dist_sq, radius_a, radius_b)

    def composed_discard(data, a, b, dist_sq, radius_a, radius_b):
        user, cq, cd, lo, reach = data
        d = jnp.sqrt(jnp.maximum(dist_sq, 0.0))
        s = radius_a + radius_b
        # every separation of the pair lies in [d - s, d + s]. The pair is needed
        # only if one side's queries can reach the other side's data: a query
        # node's reach bounds how far its queries look (r_hi for pair counts).
        gap = d - s
        a_needs_b = (cq[a] > 0) & (cd[b] > 0) & (gap < reach[a])
        b_needs_a = (cq[b] > 0) & (cd[a] > 0) & (gap < reach[b])
        out = ~(a_needs_b | b_needs_a) | (d + s < lo)
        if discard is not None:
            out = out | jnp.asarray(
                discard(user, a, b, dist_sq, radius_a, radius_b), dtype=bool
            )
        return out

    return composed_accept, composed_discard


# ---------------------------------------------------------------------------
# soft windows
# ---------------------------------------------------------------------------


def soft_window(r: Array, edges: Array, sharpness: Any, log: bool = True) -> Array:
    """Soft bin membership ``sigmoid(s (u - a_b)) - sigmoid(s (u - b_b))``.

    The same window as
    :func:`yggdrax.applications.corrfunc.binning.soft_bin_weights`, restated here
    so that the core package does not import the applications.

    Parameters
    ----------
    r:
        Separations, any shape ``(...)``; must be positive when ``log``.
    edges:
        ``(nbins + 1,)`` increasing edges.
    sharpness:
        Inverse transition width, in ``u`` units.
    log:
        Bin in ``u = ln r`` (else ``u = r``).

    Returns
    -------
    Array
        ``(..., nbins)`` weights.
    """
    edges = jnp.asarray(edges, dtype=jnp.asarray(r).dtype)
    if log:
        u = jnp.log(jnp.maximum(r, jnp.finfo(jnp.asarray(r).dtype).tiny))
        eu = jnp.log(edges)
    else:
        u, eu = r, edges
    gate = jax.nn.sigmoid(sharpness * (u[..., None] - eu))
    return gate[..., :-1] - gate[..., 1:]


def soft_window_cutoffs(
    edges: Any, sharpness: float, eps: float = 1e-8, log: bool = True
) -> tuple[float, float]:
    """The hard cut ``[r_lo, r_hi)`` outside which every window is below ``eps``.

    For ``u`` beyond the last edge ``b`` the last window is at most
    ``sigmoid(-s (u - b)) <= exp(-s (u - b))``, so ``u_hi = b + ln(1/eps) / s``;
    the lower cut mirrors it below the first edge.

    Parameters
    ----------
    edges:
        Concrete bin edges.
    sharpness:
        Concrete window sharpness.
    eps:
        Largest window value dropped by the cut.
    log:
        Windows in ``ln r``.

    Returns
    -------
    tuple of float
        ``(r_lo, r_hi)``.
    """
    e = np.asarray(edges, dtype=np.float64)
    du = math.log(1.0 / float(eps)) / float(sharpness)
    if log:
        return float(e[0] * math.exp(-du)), float(e[-1] * math.exp(du))
    return float(max(e[0] - du, 0.0)), float(e[-1] + du)


# ---------------------------------------------------------------------------
# shared stages
# ---------------------------------------------------------------------------


def _morton_box(box: Optional[tuple[float, ...]], bounds, dtype):
    """The fixed Morton box: the periodic box (as the cell tree has it) or bounds."""
    if box is not None:
        hi = jnp.asarray(box, dtype=dtype) * (1.0 + 2.0**-20)
        return jnp.zeros((3,), dtype=dtype), hi
    if bounds is None:
        raise ValueError("a non-periodic run needs explicit bounds")
    lo, hi = bounds
    return jnp.asarray(lo, dtype=dtype), jnp.asarray(hi, dtype=dtype)


class _Tree(NamedTuple):
    """One cell tree's integer structure, plus its (frozen) geometry."""

    order: Array  # sorted slot -> input row
    ranges: Array  # (nodes, 2) inclusive sorted-slot ranges
    left: Array  # (nodes,) -1 at leaves
    right: Array
    root: Array
    active: Array  # (nodes,) non-empty
    center: Array  # (nodes, 3)
    radius: Array  # (nodes,) bounding radius about ``center``
    parent: Array
    num_internal: int
    overflow: Array


def _cell_tree(pos, num_valid, box, bounds, leaf_size: int, leaf_capacity: int):
    """Capacity-padded cell tree over the leading ``num_valid`` rows (no gradient)."""
    pos = jax.lax.stop_gradient(pos)
    masses = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    kw: dict[str, Any] = dict(
        leaf_size=int(leaf_size),
        leaf_capacity=int(leaf_capacity),
        return_reordered=True,
        return_overflow=True,
        num_valid=num_valid,
    )
    if box is not None:
        out = build_static_cells_tree(pos, masses, box_size=box, **kw)
    else:
        out = build_static_cells_tree(pos, masses, bounds, **kw)
    tree, pos_sorted, _m, _inv, overflow = out
    geom = compute_tree_geometry(tree, pos_sorted, max_leaf_size=int(leaf_size))
    ni = int(tree.left_child.shape[0])
    nl = int(tree.parent.shape[0]) - ni
    idx = INDEX_DTYPE
    pad = jnp.full((nl,), -1, idx)
    ranges = jnp.asarray(tree.node_ranges, idx)
    return _Tree(
        order=jnp.asarray(tree.particle_indices, idx),
        ranges=ranges,
        left=jnp.concatenate([jnp.asarray(tree.left_child, idx), pad]),
        right=jnp.concatenate([jnp.asarray(tree.right_child, idx), pad]),
        root=jnp.argmin(jnp.asarray(tree.parent)).astype(idx),
        active=ranges[:, 1] >= ranges[:, 0],
        center=jnp.asarray(geom.center),
        radius=jnp.asarray(geom.radius, pos.dtype),
        parent=jnp.asarray(tree.parent, idx),
        num_internal=ni,
        overflow=jnp.asarray(overflow, bool),
    )


def _node_sums(t: _Tree, values_sorted: Array) -> Array:
    """Per-node sums of a per-sorted-slot quantity (``(n,)`` or ``(n, k)``)."""
    n = values_sorted.shape[0]
    zero = jnp.zeros((1,) + values_sorted.shape[1:], values_sorted.dtype)
    prefix = jnp.concatenate([zero, jnp.cumsum(values_sorted, axis=0)])
    start = jnp.clip(t.ranges[:, 0], 0, n)
    stop = jnp.clip(t.ranges[:, 1] + 1, 0, n)
    stop = jnp.maximum(stop, start)
    return prefix[stop] - prefix[start]


def _leaf_slots(t: _Tree, n: int, leaf_size: int) -> tuple[Array, Array]:
    """``(num_leaves, leaf_size)`` sorted slots of each leaf and their validity."""
    lr = t.ranges[t.num_internal :]
    start = lr[:, 0]
    length = jnp.clip(lr[:, 1] - start + 1, 0, leaf_size)
    ramp = jnp.arange(leaf_size, dtype=start.dtype)
    slots = jnp.clip(start[:, None] + ramp[None, :], 0, max(n - 1, 0))
    return slots, ramp[None, :] < length[:, None]


def _node_max(t: _Tree, values_sorted: Array, n: int, leaf_size: int) -> Array:
    """Per-node max of a non-negative per-sorted-slot value (0 on empty nodes).

    Leaves reduce their slots; internal nodes then take the max of their
    descendants by repeated scatter-max to the parent until nothing changes (at
    most the tree height). Topology data only: no gradient is wanted here.
    """
    values_sorted = jax.lax.stop_gradient(values_sorted)
    slots, mask = _leaf_slots(t, n, leaf_size)
    leaf_vals = jnp.max(jnp.where(mask, values_sorted[slots], 0.0), axis=1)
    nodes = int(t.ranges.shape[0])
    vals = jnp.zeros((nodes,), values_sorted.dtype).at[t.num_internal :].set(leaf_vals)
    parent = jnp.where(t.parent >= 0, t.parent, as_index(nodes))

    def step(state):
        v, _ = state
        nv = v.at[parent].max(v, mode="drop")
        return nv, jnp.any(nv != v)

    vals, _ = jax.lax.while_loop(lambda st: st[1], step, (vals, jnp.asarray(True)))
    return vals


def _walk(t: _Tree, box, acceptance: PairAcceptance, cq, cd, r_lo, reach, cfg):
    """The mutual walk over one cell tree with the composed rule.

    ``reach`` is a scalar or a ``(nodes,)`` per-node query reach.
    """
    accept, discard = _composed(acceptance.accept, acceptance.discard)
    dtype = t.center.dtype
    nodes = int(t.ranges.shape[0])
    data = (
        acceptance.data,
        cq,
        cd,
        jnp.asarray(r_lo, dtype),
        jnp.broadcast_to(jnp.asarray(reach, dtype), (nodes,)),
    )
    return dual_tree_walk_mutual(
        t.left,
        t.right,
        t.center,
        t.radius,
        0.0,
        t.root,
        max_pair_queue=int(cfg.max_pair_queue),
        far_cap=int(cfg.far_cap),
        near_cap=int(cfg.near_cap),
        node_active=t.active,
        box_size=box,
        periodic_radii=t.radius,
        pair_accept=accept,
        pair_discard=discard,
        pair_accept_data=data,
    )


def _partition(pos, weights, flags, gid, count, ndev, box, bounds, cfg):
    """Wrap and sample-sort into Morton domains; clamp the count to the capacity."""
    if box is not None:
        pos = wrap_positions(pos, box)
    lo, hi = _morton_box(box, bounds, pos.dtype)
    p_pos, p_w, _codes, p_count, payload = sfc_partition(
        pos,
        weights,
        ndev,
        output_capacity=int(cfg.capacity),
        bounds=(jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)),
        num_samples=int(cfg.num_samples),
        payload={"flags": flags, "gid": gid},
        payload_fill={"flags": 0, "gid": -1},
        count=count,
        exchange_method=cfg.exchange,
    )
    assert payload is not None  # a payload was passed, so one comes back
    overflow = p_count > as_index(int(cfg.capacity))
    p_count = jnp.minimum(p_count, as_index(int(cfg.capacity)))
    live = jnp.arange(int(cfg.capacity), dtype=INDEX_DTYPE) < p_count
    p_flags = jnp.where(live[:, None], payload["flags"], 0)
    p_w = jnp.where(live, p_w, 0.0)
    return p_pos, p_w, p_flags, payload["gid"], p_count, overflow


class _Halo(NamedTuple):
    pos: Array
    weights: Array
    flags: Array
    gid: Array
    num: Array  # live halo rows (not ``count``: that shadows ``tuple.count``)
    diag: dict


def _import_halo(t, pos, weights, flags, gid, count, cell_reach, ndev, box, cfg):
    """Summary, export walk, send buffers and the halo exchange of one device.

    ``cell_reach`` is ``(summary_capacity,)`` per cell, or a scalar for all.
    """
    from .summary import occupancy_cut  # local: keeps the import graph shallow

    cap = int(cfg.capacity)
    S = int(cfg.summary_capacity)
    me = jax.lax.axis_index(AXIS_NAME)
    cut = occupancy_cut(
        t.parent,
        t.ranges,
        t.num_internal,
        max_leaves=int(cfg.summary_max_leaves),
        capacity=S,
        num_valid=count,
    )
    cells = cut.cells
    live_cell = cells >= 0
    safe = jnp.where(live_cell, cells, 0)
    reach = cell_reach(cut) if callable(cell_reach) else cell_reach
    c_center = t.center[safe]
    c_radius = jnp.where(live_cell, t.radius[safe] + reach, 0.0)
    all_center = jax.lax.all_gather(c_center, AXIS_NAME, tiled=False)
    all_radius = jax.lax.all_gather(c_radius, AXIS_NAME, tiled=False)
    all_live = jax.lax.all_gather(live_cell, AXIS_NAME, tiled=False)

    walk_fn = functools.partial(
        dual_tree_walk_mutual,
        box_size=box,
        pair_accept=_never,
        pair_discard=_disjoint,
    )
    ex = export_walk(
        t.left,
        t.right,
        t.center,
        t.radius,
        t.root,
        all_center,
        all_radius,
        all_live,
        0.0,
        me,
        max_pair_queue=int(cfg.export_queue),
        far_cap=8,
        near_cap=int(cfg.export_near_cap),
        node_active=t.active,
        walk_fn=walk_fn,
    )
    num_nodes = int(t.ranges.shape[0])
    sb = build_send_buffers(
        ex.near_cell,
        ex.near_node,
        ex.near_count,
        ndev=ndev,
        max_cells=S,
        num_nodes=num_nodes,
        node_capacity=int(cfg.send_leaf_capacity),
        csr_capacity=int(cfg.export_near_cap),
    )

    # expand each shipped leaf to its particles, keeping the destination grouping
    R = int(cfg.send_leaf_capacity)
    rows = sb.node_rows
    live_row = rows >= 0
    rr = t.ranges[jnp.where(live_row, rows, 0)]
    plen = jnp.where(live_row, jnp.maximum(rr[:, 1] - rr[:, 0] + 1, 0), 0)
    cum = jnp.concatenate([jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(plen)])
    row_off = jnp.concatenate(
        [jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(sb.node_sizes.astype(INDEX_DTYPE))]
    )
    row_off = jnp.clip(row_off, 0, R)
    send_sizes = cum[row_off[1:]] - cum[row_off[:-1]]
    total = cum[-1]
    C = int(cfg.send_capacity)
    q = jnp.arange(C, dtype=INDEX_DTYPE)
    r_of = jnp.clip(jnp.searchsorted(cum[1:], q, side="right"), 0, R - 1)
    within = q - cum[r_of]
    q_live = q < total
    slot = jnp.clip(rr[r_of, 0] + within, 0, cap - 1)
    src = t.order[slot]
    s_pos = jnp.where(q_live[:, None], pos[src], 0.0)
    s_w = jnp.where(q_live, weights[src], 0.0)
    s_flags = jnp.where(q_live[:, None], flags[src], 0).astype(pos.dtype)
    s_gid = jnp.where(q_live, gid[src], -1)

    H = int(cfg.halo_capacity)
    block = jnp.concatenate([s_pos, s_w[:, None], s_flags], axis=1)
    # the halo import: linear in the values, so its reverse pass (the transpose
    # of the exchange) returns every halo cotangent to the row it was sent from
    recv, recv_sizes, _ = ragged_all_to_all_exchange(
        block, send_sizes, output_capacity=H, method=cfg.exchange
    )
    recv_gid, _, _ = ragged_all_to_all_exchange(
        s_gid[:, None],
        send_sizes,
        output_capacity=H,
        fill_value=-1,
        method=cfg.exchange,
    )
    n_halo = jnp.sum(recv_sizes).astype(INDEX_DTYPE)
    halo_overflow = n_halo > as_index(H)
    n_halo = jnp.minimum(n_halo, as_index(H))
    live = jnp.arange(H, dtype=INDEX_DTYPE) < n_halo
    h_flags = jnp.where(live[:, None], recv[:, 4:6], 0.0).astype(flags.dtype)
    # a halo particle is never this device's query: its owner counts it
    h_flags = h_flags.at[:, _QUERY].set(0)
    diag = dict(
        summary_overflow=jnp.asarray(cut.overflow, bool),
        export_overflow=ex.near_overflow | ex.queue_overflow | ex.far_overflow,
        send_overflow=sb.node_overflow | (total > as_index(C)),
        halo_overflow=halo_overflow,
        halo_count=n_halo,
        export_count=ex.near_count,
        send_count=total,
    )
    return _Halo(
        pos=jnp.where(live[:, None], recv[:, :3], 0.0),
        weights=jnp.where(live, recv[:, 3], 0.0),
        flags=h_flags,
        gid=jnp.where(live, recv_gid[:, 0], -1),
        num=n_halo,
        diag=diag,
    )


def _combine(pos, weights, flags, count, halo: _Halo):
    """``[local ; halo]`` with the live rows compacted to the front."""
    cap = pos.shape[0]
    H = halo.pos.shape[0]
    live = jnp.concatenate(
        [
            jnp.arange(cap, dtype=INDEX_DTYPE) < count,
            jnp.arange(H, dtype=INDEX_DTYPE) < halo.num,
        ]
    )
    perm = jnp.argsort(~live, stable=True).astype(INDEX_DTYPE)
    c_pos = jnp.concatenate([pos, halo.pos])[perm]
    c_w = jnp.concatenate([weights, halo.weights])[perm]
    c_flags = jnp.concatenate([flags, halo.flags])[perm]
    return c_pos, c_w, c_flags, perm, count + halo.num


def _check_reach(box, reach: float, what: str) -> None:
    if box is not None and not float(reach) < 0.5 * min(box):
        raise ValueError(
            f"{what} = {reach} must be below min(box_size)/2 = {0.5 * min(box)}: "
            "beyond it the minimum image is ambiguous"
        )


def _any(x: Array) -> Array:
    """True on every device if ``x`` is true on any."""
    return jax.lax.psum(jnp.asarray(x, jnp.int32), AXIS_NAME) > 0


def _scan_chunks(fn, n_items: int, chunk: int, acc0, *arrays):
    """``acc + sum fn(chunk of arrays)`` over ``n_items`` rows, ``chunk`` at a time.

    ``fn`` is rematerialised (``jax.checkpoint``) in the reverse pass: a chunk's
    residuals are its integer inputs, not its ``(chunk, leaf, leaf, nbins)``
    intermediates, so a gradient stores O(rows) rather than O(pairs x bins). The
    forward is unchanged.
    """
    fn = jax.checkpoint(fn, prevent_cse=False)
    chunk = max(1, min(int(chunk), max(n_items, 1)))
    n_chunks = -(-n_items // chunk)
    pad = n_chunks * chunk - n_items

    def prep(a):
        a = jnp.asarray(a)
        if pad:
            a = jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)])
        return a.reshape((n_chunks, chunk) + a.shape[1:])

    xs = tuple(prep(a) for a in arrays)

    def body(acc, x):
        return acc + fn(*x), None

    out, _ = jax.lax.scan(body, acc0, xs)
    return out


# ---------------------------------------------------------------------------
# payload 1: soft binned pair counts
# ---------------------------------------------------------------------------


class DistributedPairCounts(NamedTuple):
    """Soft pair counts and the pipeline's diagnostics.

    Attributes
    ----------
    counts:
        ``(nbins,)`` ``sum_{i<j} w_i w_j W_b(r_ij)`` over pairs with
        ``r_lo <= r_ij < r_hi`` (minimum image). Replicated.
    overflow:
        True if any capacity overflowed on any device. **Must be read.**
    flags:
        Each overflow flag by stage, replicated.
    stats:
        Per-device counters ``(ndev,)``: particles, halo, walk lists.
    """

    counts: Array
    overflow: Array
    flags: dict
    stats: dict


def _block_pair_sums(
    xa, wa, oa, ma, xb, wb, ob, mb, extra, edges, sharpness, log, lo, hi, box
):
    """``sum_{i in a, j in b} c_ij w_i w_j W(r_ij) [lo <= r < hi]`` per bin.

    Blocks are ``(C, L, ...)``; ``extra`` is a ``(C, L, L)`` (or broadcastable)
    pair mask. ``c_ij = (o_i + o_j) / 2``.
    """
    delta = minimum_image(xb[:, None, :, :] - xa[:, :, None, :], box)
    r2 = jnp.sum(delta * delta, axis=-1)
    valid = ma[:, :, None] & mb[:, None, :] & extra
    # masked pairs (self, padding, duplicates) never reach the sqrt at r = 0
    r = jnp.sqrt(jnp.where(valid, r2, 1.0))
    inside = valid & (r >= lo) & (r < hi)
    c = 0.5 * (oa[:, :, None] + ob[:, None, :])
    pw = jnp.where(inside, wa[:, :, None] * wb[:, None, :] * c, 0.0)
    win = soft_window(r, edges, sharpness, log)
    return jnp.einsum("cij,cijb->b", pw, win)


def pair_counts_shard(
    positions: Array,
    weights: Array,
    count: Array,
    edges: Array,
    *,
    ndev: int,
    config: DistributedStatsConfig,
    box_size: Optional[BoxSize],
    r_lo: float,
    r_hi: float,
    sharpness: Any = 100.0,
    log: bool = True,
    acceptance: Optional[PairAcceptance] = None,
    bounds: Optional[tuple[Array, Array]] = None,
) -> DistributedPairCounts:
    """One device's body of :func:`distributed_pair_counts`, for a ``shard_map``.

    Call inside ``shard_map`` over :data:`AXIS_NAME`. ``positions``/``weights``
    are this device's input rows (any rows: the partition redistributes them),
    the first ``count`` live.

    Parameters
    ----------
    positions, weights, count:
        This device's input shard, ``(rows, 3)`` / ``(rows,)`` / scalar.
    edges:
        ``(nbins + 1,)`` bin edges.
    ndev:
        Mesh size. Static.
    config:
        Capacities. Static.
    box_size:
        Periodic box, or ``None`` with ``bounds``. Static.
    r_lo, r_hi:
        The hard cut (:func:`soft_window_cutoffs`); ``r_hi`` is the halo reach.
        Static.
    sharpness, log:
        The soft window.
    acceptance:
        Node-pair rule; default :func:`exact_acceptance`.
    bounds:
        Morton box of a non-periodic run.

    Returns
    -------
    DistributedPairCounts
        Replicated counts and diagnostics; ``stats`` entries are per device
        (shape ``(1,)`` here, ``(ndev,)`` once gathered by the ``shard_map``).
    """
    require_x64("distributed statistics (Morton codes, global ids)")
    cfg = config.resolved()
    box = normalize_box_size(box_size, 3)
    _check_reach(box, r_hi, "r_hi")
    acc = acceptance if acceptance is not None else exact_acceptance()
    gid_in = jnp.arange(positions.shape[0], dtype=jnp.int64)  # replaced by caller ids
    flags_in = jnp.ones((positions.shape[0], 2), jnp.int32)
    return _pair_counts_body(
        positions,
        weights,
        flags_in,
        gid_in,
        count,
        edges,
        ndev=ndev,
        cfg=cfg,
        box=box,
        bounds=bounds,
        r_lo=float(r_lo),
        r_hi=float(r_hi),
        sharpness=sharpness,
        log=log,
        acc=acc,
    )


def _pair_counts_body(
    positions,
    weights,
    flags,
    gid,
    count,
    edges,
    *,
    ndev,
    cfg,
    box,
    bounds,
    r_lo,
    r_hi,
    sharpness,
    log,
    acc,
):
    """Partition, halo, walk, accumulate, ``psum``."""
    dtype = positions.dtype
    p_pos, p_w, p_flags, _p_gid, p_count, part_over = _partition(
        positions, weights, flags, gid, count, ndev, box, bounds, cfg
    )
    lo_b, hi_b = _morton_box(box, bounds, dtype)
    t_loc = _cell_tree(
        p_pos, p_count, box, (lo_b, hi_b), cfg.leaf_size, cfg.leaf_capacity
    )
    halo = _import_halo(
        t_loc,
        p_pos,
        p_w,
        p_flags,
        _p_gid,
        p_count,
        jnp.asarray(r_hi, dtype),
        ndev,
        box,
        cfg,
    )
    c_pos, c_w, c_flags, _perm, n_live = _combine(p_pos, p_w, p_flags, p_count, halo)
    t = _cell_tree(
        c_pos, n_live, box, (lo_b, hi_b), cfg.leaf_size, cfg.halo_leaf_capacity
    )
    n = c_pos.shape[0]
    # live values in the frozen order (integer gathers only)
    xs = c_pos[t.order]
    live_s = jnp.arange(n, dtype=INDEX_DTYPE) < n_live
    ws = jnp.where(live_s, c_w[t.order], 0.0)
    fq = jnp.where(live_s, c_flags[t.order, _QUERY], 0)
    fd = jnp.where(live_s, c_flags[t.order, _DATA], 0)
    cq = _node_sums(t, fq.astype(INDEX_DTYPE))
    cd = _node_sums(t, fd.astype(INDEX_DTYPE))
    walk = _walk(t, box, acc, cq, cd, r_lo, r_hi, cfg)

    ls = int(cfg.leaf_size)
    ni = t.num_internal
    slots, smask = _leaf_slots(t, n, ls)
    smask = smask & live_s[slots]
    os_ = fq.astype(dtype)
    nb = int(jnp.shape(edges)[0]) - 1
    edges = jnp.asarray(edges, dtype)
    zero = jnp.zeros((nb,), dtype)

    def blocks(leaf):
        sl = slots[leaf]
        return xs[sl], ws[sl], os_[sl], smask[leaf]

    def pairs(ta, tb, ok, extra):
        xa, wa, oa, ma = blocks(ta)
        xb, wb, ob, mb = blocks(tb)
        ma = ma & ok[:, None]
        return _block_pair_sums(
            xa,
            wa,
            oa,
            ma,
            xb,
            wb,
            ob,
            mb,
            extra,
            edges,
            sharpness,
            log,
            r_lo,
            r_hi,
            box,
        )

    num_leaves = slots.shape[0]
    triu = jnp.triu(jnp.ones((ls, ls), bool), k=1)[None]
    leaf_ids = jnp.arange(num_leaves, dtype=INDEX_DTYPE)
    leaf_ok = t.active[ni:] & (cq[ni:] > 0)
    within = _scan_chunks(
        lambda a, ok: pairs(a, a, ok, triu),
        num_leaves,
        cfg.chunk,
        zero,
        leaf_ids,
        leaf_ok,
    )
    near_ok = walk.near_a >= 0
    near = _scan_chunks(
        lambda a, b, ok: pairs(a, b, ok, True),
        int(walk.near_a.shape[0]),
        cfg.chunk,
        zero,
        jnp.where(near_ok, walk.near_a - ni, 0),
        jnp.where(near_ok, walk.near_b - ni, 0),
        near_ok,
    )

    # far pairs: closed-form monopole of the ownership weights at the centroids
    acc_dtype = jnp.float64
    cnt = _node_sums(t, live_s.astype(INDEX_DTYPE)).astype(acc_dtype)
    w_tot = _node_sums(t, ws.astype(acc_dtype))
    w_own = _node_sums(t, (ws * os_).astype(acc_dtype))
    centroid = _node_sums(t, jnp.where(live_s[:, None], xs, 0.0).astype(acc_dtype))
    centroid = (centroid / jnp.maximum(cnt, 1.0)[:, None]).astype(dtype)
    far_ok = walk.far_a >= 0
    fa = jnp.where(far_ok, walk.far_a, 0)
    fb = jnp.where(far_ok, walk.far_b, 0)
    d_far = minimum_image(centroid[fb] - centroid[fa], box)
    r_far = jnp.sqrt(jnp.where(far_ok, jnp.sum(d_far * d_far, axis=-1), 1.0))
    pw_far = 0.5 * (w_own[fa] * w_tot[fb] + w_tot[fa] * w_own[fb])
    pw_far = jnp.where(far_ok & (r_far >= r_lo) & (r_far < r_hi), pw_far, 0.0)
    far = jnp.einsum(
        "p,pb->b", pw_far.astype(dtype), soft_window(r_far, edges, sharpness, log)
    )

    local = within + near + far
    counts = jax.lax.psum(local, AXIS_NAME)
    walk_over = walk.near_overflow | walk.far_overflow | walk.queue_overflow
    flags_out = dict(
        partition_overflow=_any(part_over),
        leaf_overflow=_any(t_loc.overflow | t.overflow),
        summary_overflow=_any(halo.diag["summary_overflow"]),
        export_overflow=_any(halo.diag["export_overflow"]),
        send_overflow=_any(halo.diag["send_overflow"]),
        halo_overflow=_any(halo.diag["halo_overflow"]),
        walk_overflow=_any(walk_over),
    )
    overflow = functools.reduce(lambda a, b: a | b, flags_out.values())
    stats = dict(
        particles=p_count[None],
        halo=halo.diag["halo_count"][None],
        export_pairs=halo.diag["export_count"][None],
        sent=halo.diag["send_count"][None],
        near_pairs=walk.near_count[None],
        far_pairs=walk.far_count[None],
        peak_wavefront=walk.peak_wavefront[None],
    )
    return DistributedPairCounts(
        counts=counts, overflow=overflow, flags=flags_out, stats=stats
    )


def _split_for_mesh(positions, ndev, *extra):
    """Split ``(N, ...)`` arrays into ``ndev`` equal padded input shards.

    Traceable: the pad is a ``jnp.concatenate``, so positions (and any ``extra``
    array) may be tracers under ``jit``/``grad``, and the cotangent of a padded
    shard is the plain slice back to the input rows. Only the per-device counts
    are host values (they depend on shapes alone).
    """
    positions = jnp.asarray(positions)
    n = int(positions.shape[0])
    if n < 1:
        raise ValueError("need at least one particle")
    per = -(-n // ndev)
    counts = np.array(
        [max(0, min(per, n - d * per)) for d in range(ndev)], dtype=np.int64
    )
    total = ndev * per

    def pad(a, fill):
        a = jnp.asarray(a)
        tail = jnp.full((total - n,) + a.shape[1:], fill, dtype=a.dtype)
        return jnp.concatenate([a, tail])

    return counts, pad(positions, 0.0), tuple(pad(a, f) for a, f in extra)


def _resolve_mesh(mesh, ndev):
    if mesh is not None:
        return mesh
    return make_mesh(ndev)


@functools.lru_cache(maxsize=64)
def _pair_count_program(
    mesh, cfg, box, bounds_key, r_lo, r_hi, log, accept, discard, nbins, has_bounds
):
    ndev = int(mesh.size)
    acc_proto = (accept, discard)

    def body(pos, w, gid, count, edges, sharpness, acc_data, lo_b, hi_b):
        acc = PairAcceptance(acc_proto[0], acc_proto[1], acc_data)
        flags = jnp.ones((pos.shape[0], 2), jnp.int32)
        res = _pair_counts_body(
            pos,
            w,
            flags,
            gid,
            count[0],
            edges,
            ndev=ndev,
            cfg=cfg,
            box=box,
            bounds=(lo_b, hi_b) if has_bounds else None,
            r_lo=r_lo,
            r_hi=r_hi,
            sharpness=sharpness,
            log=log,
            acc=acc,
        )
        return res.counts, res.overflow, res.flags, res.stats

    rep = P()
    sh = P(AXIS_NAME)
    fn = _shard_map(
        body,
        mesh=mesh,
        in_specs=(sh, sh, sh, sh, rep, rep, rep, rep, rep),
        out_specs=(rep, rep, rep, sh),
        check_vma=False,
    )
    return jax.jit(fn)


def distributed_pair_counts(
    positions: Any,
    edges: Any,
    *,
    box_size: Optional[BoxSize],
    config: DistributedStatsConfig,
    weights: Any = None,
    sharpness: float = 100.0,
    log: bool = True,
    eps: float = 1e-8,
    r_cut: Optional[tuple[float, float]] = None,
    acceptance: Optional[PairAcceptance] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    bounds: Optional[tuple[Any, Any]] = None,
) -> DistributedPairCounts:
    """Soft binned pair counts of ``positions`` sharded over a device mesh.

    ``C_b = sum_{i<j} w_i w_j W_b(r_ij)`` over unordered pairs inside the cut
    ``[r_lo, r_hi)``, with ``r_ij`` the minimum-image distance and ``W_b``
    :func:`soft_window`. See ``docs/distributed_statistics.md``.

    Parameters
    ----------
    positions:
        ``(N, 3)`` particle positions, host or device. Wrapped into the box.
    edges:
        ``(nbins + 1,)`` increasing bin edges (concrete).
    box_size:
        Periodic box; ``None`` for a non-periodic run with ``bounds``.
    config:
        Capacities (:meth:`DistributedStatsConfig.for_problem`).
    weights:
        ``(N,)`` particle weights, default 1.
    sharpness, log:
        The soft window.
    eps:
        Window value below which pairs are cut (sets the default ``r_cut``).
    r_cut:
        Explicit ``(r_lo, r_hi)``; default :func:`soft_window_cutoffs`.
    acceptance:
        Node-pair rule, default :func:`exact_acceptance`.
    mesh:
        1-D mesh over :data:`AXIS_NAME`; default all devices.
    bounds:
        Morton box ``(lo, hi)`` of a non-periodic run.

    Returns
    -------
    DistributedPairCounts
        Counts, the overflow flag (read it), per-stage flags and per-device
        counters.

    Notes
    -----
    Differentiable with respect to ``positions`` and ``weights`` (and traceable:
    it may sit inside ``jit``, ``grad`` or ``lax.scan``). The gradient is that of
    the fixed-topology statistic: the tree, walk and halo are rebuilt from the
    current positions on every call, but carry no cotangent. With the default
    :func:`exact_acceptance` it equals the single-device gradient to round-off.
    ``edges``, ``sharpness`` and ``eps`` must be concrete (they set the cut).
    """
    require_x64("distributed statistics (Morton codes, global ids)")
    mesh = _resolve_mesh(mesh, None)
    ndev = int(mesh.size)
    cfg = config.resolved()
    box = normalize_box_size(box_size, 3)
    e = np.asarray(edges, dtype=np.float64)
    r_lo, r_hi = (
        r_cut if r_cut is not None else soft_window_cutoffs(e, sharpness, eps, log)
    )
    _check_reach(box, r_hi, "r_hi")
    acc = acceptance if acceptance is not None else exact_acceptance()
    pos_in = jnp.asarray(positions)
    n = int(pos_in.shape[0])
    dtype = pos_in.dtype
    w = jnp.ones((n,), dtype) if weights is None else jnp.asarray(weights).astype(dtype)
    counts, pos_f, (w_f, gid_f) = _split_for_mesh(
        pos_in, ndev, (w, 0.0), (np.arange(n, dtype=np.int64), -1)
    )
    has_bounds = bounds is not None
    lo_b, hi_b = (
        (np.asarray(bounds[0]), np.asarray(bounds[1]))
        if has_bounds and bounds is not None
        else (np.zeros(3), np.ones(3))
    )
    prog = _pair_count_program(
        mesh,
        cfg,
        box,
        None,
        float(r_lo),
        float(r_hi),
        bool(log),
        acc.accept,
        acc.discard,
        int(e.shape[0]) - 1,
        has_bounds,
    )
    out = prog(
        pos_f,
        w_f,
        jnp.asarray(gid_f, jnp.int64),
        jnp.asarray(counts),
        jnp.asarray(e, dtype),
        jnp.asarray(sharpness, dtype),
        acc.data,
        jnp.asarray(lo_b, dtype),
        jnp.asarray(hi_b, dtype),
    )
    c, overflow, flags, stats = out
    return DistributedPairCounts(counts=c, overflow=overflow, flags=flags, stats=stats)


# ---------------------------------------------------------------------------
# payload 2: kNN distances
# ---------------------------------------------------------------------------


class DistributedKNN(NamedTuple):
    """kNN distances of every query, and the pipeline's diagnostics.

    Attributes
    ----------
    distances:
        ``(rows, k)`` sorted minimum-image distances to the ``k`` nearest data
        points (self excluded), ``inf`` past the last neighbour found within
        ``r_max``. Rows are in the PARTITIONED order; read them with ``gid``.
    gid:
        ``(rows,)`` the input row of each output row; ``-1`` on padding and on
        non-query rows.
    cdf:
        ``(k, n_r)`` fraction of queries with ``r_k <= r`` on the radius grid
        (soft when ``sharpness`` was given), or ``None`` without a grid.
        Replicated.
    overflow, flags, stats:
        As :class:`DistributedPairCounts`. ``flags`` also reports
        ``local_pass_overflow``, which is not an error (a truncated first pass
        only thickens the halo) and is not part of ``overflow``.
    """

    distances: Array
    gid: Array
    cdf: Optional[Array]
    overflow: Array
    flags: dict
    stats: dict


def _merge_topk(va, ia, vb, ib, k: int):
    """The ``k`` smallest of two sorted candidate sets (values and indices)."""
    v = jnp.concatenate([va, vb], axis=-1)
    i = jnp.concatenate([ia, ib], axis=-1)
    neg, pos = jax.lax.top_k(-v, k)
    return -neg, jnp.take_along_axis(i, pos, axis=-1)


def _knn_on_tree(t: _Tree, xs, fq, fd, live_s, n, k, r_max, box, cfg, node_reach=None):
    """kNN of every query slot of one tree among its data slots within ``r_max``.

    The walk prunes node pairs with no (query, data) particle pair and pairs
    beyond the query side's reach. Every surviving leaf pair, in both
    orientations, plus each leaf with itself, is an EDGE (query leaf, source
    leaf). Edges are sorted by query leaf and streamed in chunks of
    ``cfg.chunk``: each edge's ``(leaf, leaf)`` block gives a per-query top-k, a
    segmented associative scan merges the edges of one query leaf inside the
    chunk, and the segment's last edge merges into the running best. Work is
    proportional to the edges; nothing is padded to the busiest leaf.

    ``node_reach`` (per node, optional; default ``r_max``) bounds how far a
    node's queries look. Exact when every node's reach covers each of its
    queries' true ``r_k`` (or ``r_max``); any smaller reach still returns upper
    bounds, which is all the first pass needs.

    Returns
    -------
    tuple
        ``(nbr, d2, walk_overflow, walk)``: ``(n, k)`` neighbour sorted slots
        (``-1`` where none) and squared distances (``inf``) per sorted slot.
    """
    dtype = xs.dtype
    ls = int(cfg.leaf_size)
    ni = t.num_internal
    cq = _node_sums(t, fq.astype(INDEX_DTYPE))
    cd = _node_sums(t, fd.astype(INDEX_DTYPE))
    if node_reach is None:
        node_reach = r_max
    walk = _walk(t, box, exact_acceptance(), cq, cd, 0.0, node_reach, cfg)
    slots, smask = _leaf_slots(t, n, ls)
    smask = smask & live_s[slots]
    nl = slots.shape[0]

    near_ok = walk.near_a >= 0
    na = jnp.where(near_ok, walk.near_a - ni, 0)
    nb_ = jnp.where(near_ok, walk.near_b - ni, 0)
    leaves = jnp.arange(nl, dtype=INDEX_DTYPE)
    tgt = jnp.concatenate([na, nb_, leaves])
    src = jnp.concatenate([nb_, na, leaves])
    ok = jnp.concatenate([near_ok, near_ok, t.active[ni:]])
    ok = ok & (cq[ni + tgt] > 0) & (cd[ni + src] > 0)
    key = jnp.where(ok, tgt, as_index(nl))
    order = jnp.argsort(key, stable=True)
    e_tgt = key[order]
    e_src = src[order]
    n_edges = jnp.sum(ok.astype(INDEX_DTYPE))
    E = int(e_tgt.shape[0])
    C = max(1, min(int(cfg.chunk), E))
    pad = (-E) % C
    e_tgt = jnp.concatenate([e_tgt, jnp.full((pad,), nl, e_tgt.dtype)])
    e_src = jnp.concatenate([e_src, jnp.zeros((pad,), e_src.dtype)])
    # one past the last real edge; the sentinel ``nl`` never matches a leaf
    e_next = jnp.concatenate([e_tgt[1:], jnp.full((1,), nl, e_tgt.dtype)])

    r2max = jnp.asarray(r_max, dtype) ** 2
    inf = jnp.asarray(jnp.inf, dtype)
    kk = min(int(k), ls)
    best_v = jnp.full((n + 1, k), inf, dtype)
    best_i = jnp.full((n + 1, k), -1, INDEX_DTYPE)
    q_ok_all = smask & (fq[slots] > 0)
    d_ok_all = smask & (fd[slots] > 0)

    def chunk_body(state):
        c, bv, bi = state
        start = c * C
        ct = jax.lax.dynamic_slice(e_tgt, (start,), (C,))
        cs = jax.lax.dynamic_slice(e_src, (start,), (C,))
        cn = jax.lax.dynamic_slice(e_next, (start,), (C,))
        live_e = ct < nl
        tl = jnp.where(live_e, ct, 0)
        q_sl, s_sl = slots[tl], slots[cs]
        q_ok = q_ok_all[tl] & live_e[:, None]
        s_ok = d_ok_all[cs]
        delta = minimum_image(xs[s_sl][:, None, :, :] - xs[q_sl][:, :, None, :], box)
        r2 = jnp.sum(delta * delta, axis=-1)  # (C, ls, ls)
        good = (
            q_ok[:, :, None]
            & s_ok[:, None, :]
            & (s_sl[:, None, :] != q_sl[:, :, None])
            & (r2 <= r2max)
        )
        r2 = jnp.where(good, r2, inf)
        neg, pos = jax.lax.top_k(-r2, kk)
        v = -neg
        i = jnp.where(
            jnp.isfinite(v),
            jnp.take_along_axis(
                jnp.broadcast_to(s_sl[:, None, :], r2.shape), pos, axis=-1
            ),
            -1,
        )
        if kk < k:
            v = jnp.concatenate([v, jnp.full(v.shape[:-1] + (k - kk,), inf, dtype)], -1)
            i = jnp.concatenate(
                [i, jnp.full(i.shape[:-1] + (k - kk,), -1, i.dtype)], -1
            )
        # segmented merge of the edges that share a query leaf (contiguous)
        first = jnp.concatenate([jnp.ones((1,), bool), ct[1:] != ct[:-1]])

        def op(x, y):
            fx, vx, ix = x
            fy, vy, iy = y
            mv, mi = _merge_topk(vx, ix, vy, iy, k)
            fy3 = fy[:, None, None]
            return fx | fy, jnp.where(fy3, vy, mv), jnp.where(fy3, iy, mi)

        _, sv, si = jax.lax.associative_scan(op, (first, v, i))
        last = live_e & ((ct != cn) | (jnp.arange(C) == C - 1))
        rows = jnp.where(last[:, None] & smask[tl], q_sl, as_index(n))
        mv, mi = _merge_topk(bv[rows], bi[rows], sv, si, k)
        bv = bv.at[rows].set(mv, mode="drop")
        bi = bi.at[rows].set(mi, mode="drop")
        # the dropped row ``n`` is the scratch row: keep it clean
        bv = bv.at[n].set(inf)
        bi = bi.at[n].set(-1)
        return c + 1, bv, bi

    n_chunks = (n_edges + as_index(C) - 1) // as_index(C)
    _, best_v, best_i = jax.lax.while_loop(
        lambda st: st[0] < n_chunks, chunk_body, (as_index(0), best_v, best_i)
    )
    walk_over = walk.near_overflow | walk.far_overflow | walk.queue_overflow
    return best_i[:n], best_v[:n], walk_over, walk


def _cell_max(t: _Tree, cut, values_sorted, n):
    """Per summary cell, the max of a per-sorted-slot value over the cell's slots."""
    cells = cut.cells
    live = cells >= 0
    big = as_index(n + 1)
    starts = jnp.where(live, t.ranges[jnp.where(live, cells, 0), 0], big)
    order = jnp.argsort(starts)
    s_sorted = starts[order]
    slot = jnp.arange(n, dtype=INDEX_DTYPE)
    pos_ = jnp.clip(jnp.searchsorted(s_sorted, slot, side="right") - 1, 0, None)
    cell_of = order[pos_]
    ends = jnp.where(live, t.ranges[jnp.where(live, cells, 0), 1], -1)
    inside = (slot >= s_sorted[pos_]) & (slot <= ends[cell_of])
    vals = jnp.where(inside, values_sorted, 0.0)
    seg = jnp.where(inside, cell_of, as_index(cells.shape[0]))
    m = jax.ops.segment_max(vals, seg, num_segments=int(cells.shape[0]) + 1)
    return jnp.where(live, jnp.maximum(m[:-1], 0.0), 0.0)


def _knn_body(
    positions,
    flags,
    gid,
    count,
    r_grid,
    sharpness,
    *,
    k,
    r_max,
    adaptive_reach,
    ndev,
    cfg,
    box,
    bounds,
):
    """Partition, local pass, halo, combined pass, CDF reduction."""
    dtype = positions.dtype
    weights = jnp.ones((positions.shape[0],), dtype)
    p_pos, _p_w, p_flags, p_gid, p_count, part_over = _partition(
        positions, weights, flags, gid, count, ndev, box, bounds, cfg
    )
    lo_b, hi_b = _morton_box(box, bounds, dtype)
    cap = int(cfg.capacity)
    t_loc = _cell_tree(
        p_pos, p_count, box, (lo_b, hi_b), cfg.leaf_size, cfg.leaf_capacity
    )
    live_loc = jnp.arange(cap, dtype=INDEX_DTYPE) < p_count
    xs_loc = jax.lax.stop_gradient(p_pos)[t_loc.order]
    live_ls = live_loc[t_loc.order] & (jnp.arange(cap, dtype=INDEX_DTYPE) < p_count)
    fq_loc = jnp.where(live_ls, p_flags[t_loc.order, _QUERY], 0)
    fd_loc = jnp.where(live_ls, p_flags[t_loc.order, _DATA], 0)

    loc_over = jnp.asarray(False)
    bound = jnp.zeros((cap,), dtype)
    if adaptive_reach:
        # pass 1: local kNN -> an upper bound on each query's true r_k
        # A cheap, density-adaptive candidate set: each node looks a few of its
        # own radii beyond its sphere. Whatever it finds is an upper bound.
        r1 = jnp.minimum(float(cfg.first_pass_reach_factor) * t_loc.radius, r_max)
        _nbr1, d2_loc, w_over1, _ = _knn_on_tree(
            t_loc, xs_loc, fq_loc, fd_loc, live_ls, cap, k, r_max, box, cfg, r1
        )
        # benign: missing candidates only make a bound larger, never wrong
        loc_over = w_over1
        bound = jnp.sqrt(jnp.minimum(d2_loc[:, k - 1], r_max * r_max))
        bound = jnp.where(fq_loc > 0, bound, 0.0)

        def reach(cut):
            return _cell_max(t_loc, cut, bound, cap)

        cell_reach: Any = reach
    else:
        cell_reach = jnp.asarray(r_max, dtype)

    halo = _import_halo(
        t_loc,
        p_pos,
        jnp.ones((cap,), dtype),
        p_flags,
        p_gid,
        p_count,
        cell_reach,
        ndev,
        box,
        cfg,
    )
    c_pos, _c_w, c_flags, perm, n_live = _combine(
        p_pos, jnp.ones((cap,), dtype), p_flags, p_count, halo
    )
    t = _cell_tree(
        c_pos, n_live, box, (lo_b, hi_b), cfg.leaf_size, cfg.halo_leaf_capacity
    )
    n = c_pos.shape[0]
    xs = c_pos[t.order]
    live_s = jnp.arange(n, dtype=INDEX_DTYPE) < n_live
    fq = jnp.where(live_s, c_flags[t.order, _QUERY], 0)
    fd = jnp.where(live_s, c_flags[t.order, _DATA], 0)
    # topology from frozen positions; distances recomputed from live ones would be
    # the C2 step (the top-k indices are discrete, the distances are not)
    if adaptive_reach:
        # each local query's first-pass bound, carried to its combined slot
        inv = (
            jnp.zeros((n,), INDEX_DTYPE).at[perm].set(jnp.arange(n, dtype=INDEX_DTYPE))
        )
        bound_comp = (
            jnp.zeros((n,), dtype)
            .at[inv[t_loc.order]]
            .set(jnp.where(live_ls, bound, 0.0))
        )
        reach_s = jnp.where(fq > 0, bound_comp[t.order], 0.0)
        # the walk prunes on ``gap < reach`` (strict): pad so that a neighbour
        # exactly at a query's bound is kept
        node_reach: Any = (
            _node_max(t, reach_s, n, int(cfg.leaf_size)) * (1.0 + 1e-12)
            + jnp.finfo(dtype).tiny
        )
    else:
        node_reach = None
    nbr, _d2, w_over, walk = _knn_on_tree(
        t, jax.lax.stop_gradient(xs), fq, fd, live_s, n, k, r_max, box, cfg, node_reach
    )
    # values from the live positions at the frozen neighbour indices
    has = nbr >= 0
    delta = minimum_image(xs[jnp.where(has, nbr, 0)] - xs[:, None, :], box)
    r2 = jnp.where(has, jnp.sum(delta * delta, axis=-1), 1.0)
    d_sorted = jnp.where(has, jnp.sqrt(r2), jnp.inf)
    # back to partitioned local rows: sorted slot -> combined row -> local row
    comb_row = perm[t.order]
    local_rows = jnp.where(live_s & (comb_row < cap), comb_row, as_index(cap))
    dist = (
        jnp.full((cap + 1, k), jnp.inf, dtype).at[local_rows].set(d_sorted, mode="drop")
    )[:cap]
    is_q = live_loc & (p_flags[:, _QUERY] > 0)
    dist = jnp.where(is_q[:, None], dist, jnp.inf)
    out_gid = jnp.where(is_q, p_gid, -1)

    cdf = None
    if r_grid is not None:
        rg = jnp.asarray(r_grid, dtype)
        safe_d = jnp.where(jnp.isfinite(dist), dist, 2.0 * jnp.max(rg) + 1.0)
        if sharpness is None:
            step = (safe_d[:, :, None] <= rg[None, None, :]).astype(dtype)
        else:
            step = jax.nn.sigmoid(
                sharpness * (jnp.log(rg)[None, None, :] - jnp.log(safe_d)[:, :, None])
            )
        num = jnp.sum(jnp.where(is_q[:, None, None], step, 0.0), axis=0)
        num = jax.lax.psum(num, AXIS_NAME)
        nq = jax.lax.psum(jnp.sum(is_q.astype(dtype)), AXIS_NAME)
        cdf = num / jnp.maximum(nq, 1.0)

    flags_out = dict(
        partition_overflow=_any(part_over),
        leaf_overflow=_any(t_loc.overflow | t.overflow),
        summary_overflow=_any(halo.diag["summary_overflow"]),
        export_overflow=_any(halo.diag["export_overflow"]),
        send_overflow=_any(halo.diag["send_overflow"]),
        halo_overflow=_any(halo.diag["halo_overflow"]),
        walk_overflow=_any(w_over),
    )
    overflow = functools.reduce(lambda a, b: a | b, flags_out.values())
    # reported, not fatal: a truncated first pass only thickens the halo
    flags_out["local_pass_overflow"] = _any(loc_over)
    stats = dict(
        particles=p_count[None],
        halo=halo.diag["halo_count"][None],
        export_pairs=halo.diag["export_count"][None],
        sent=halo.diag["send_count"][None],
        near_pairs=walk.near_count[None],
        peak_wavefront=walk.peak_wavefront[None],
    )
    return dist, out_gid, cdf, overflow, flags_out, stats


def knn_shard(
    positions: Array,
    count: Array,
    *,
    k: int,
    r_max: float,
    ndev: int,
    config: DistributedStatsConfig,
    box_size: Optional[BoxSize],
    is_query: Optional[Array] = None,
    is_data: Optional[Array] = None,
    gid: Optional[Array] = None,
    r_grid: Optional[Array] = None,
    sharpness: Optional[Any] = None,
    adaptive_reach: bool = True,
    bounds: Optional[tuple[Array, Array]] = None,
) -> DistributedKNN:
    """One device's body of :func:`distributed_knn`, for a ``shard_map``.

    Parameters
    ----------
    positions, count:
        This device's input rows and live count.
    k, r_max:
        Neighbours wanted and the search radius cap. Static.
    ndev, config, box_size, bounds:
        As :func:`pair_counts_shard`.
    is_query, is_data:
        ``(rows,)`` bool masks; default every row is both (data as queries).
    gid:
        ``(rows,)`` int64 global ids; default the local row index.
    r_grid, sharpness:
        Optional CDF grid and soft-step sharpness (``None``: hard step).
    adaptive_reach:
        Run the local first pass that thins the halo.

    Returns
    -------
    DistributedKNN
        Per-row distances in the partitioned order and the replicated CDF.
    """
    require_x64("distributed statistics (Morton codes, global ids)")
    cfg = config.resolved()
    box = normalize_box_size(box_size, 3)
    _check_reach(box, r_max, "r_max")
    rows = positions.shape[0]
    q = jnp.ones((rows,), bool) if is_query is None else jnp.asarray(is_query, bool)
    d = jnp.ones((rows,), bool) if is_data is None else jnp.asarray(is_data, bool)
    flags = jnp.stack([q, d], axis=1).astype(jnp.int32)
    g = jnp.arange(rows, dtype=jnp.int64) if gid is None else jnp.asarray(gid)
    dist, out_gid, cdf, overflow, fl, st = _knn_body(
        positions,
        flags,
        g,
        count,
        r_grid,
        sharpness,
        k=int(k),
        r_max=float(r_max),
        adaptive_reach=bool(adaptive_reach),
        ndev=ndev,
        cfg=cfg,
        box=box,
        bounds=bounds,
    )
    return DistributedKNN(dist, out_gid, cdf, overflow, fl, st)


@functools.lru_cache(maxsize=64)
def _knn_program(mesh, cfg, box, k, r_max, adaptive_reach, has_grid, soft, has_bounds):
    ndev = int(mesh.size)

    def body(pos, flags, gid, count, r_grid, sharpness, lo_b, hi_b):
        dist, out_gid, cdf, overflow, fl, st = _knn_body(
            pos,
            flags,
            gid,
            count[0],
            r_grid if has_grid else None,
            sharpness if soft else None,
            k=k,
            r_max=r_max,
            adaptive_reach=adaptive_reach,
            ndev=ndev,
            cfg=cfg,
            box=box,
            bounds=(lo_b, hi_b) if has_bounds else None,
        )
        if cdf is None:
            cdf = jnp.zeros((k, 1), pos.dtype)
        return dist, out_gid, cdf, overflow, fl, st

    rep = P()
    sh = P(AXIS_NAME)
    fn = _shard_map(
        body,
        mesh=mesh,
        in_specs=(sh, sh, sh, sh, rep, rep, rep, rep),
        out_specs=(sh, sh, rep, rep, rep, sh),
        check_vma=False,
    )
    return jax.jit(fn)


def distributed_knn(
    positions: Any,
    *,
    k: int,
    r_max: float,
    box_size: Optional[BoxSize],
    config: DistributedStatsConfig,
    queries: Any = None,
    r_grid: Any = None,
    sharpness: Optional[float] = None,
    adaptive_reach: bool = True,
    mesh: Optional[jax.sharding.Mesh] = None,
    bounds: Optional[tuple[Any, Any]] = None,
) -> DistributedKNN:
    """kNN distances (and optionally kNN-CDFs) with the data sharded over a mesh.

    Every query gets the minimum-image distances to its ``k`` nearest data
    points, exact within ``r_max`` and ``inf`` past the last neighbour found
    there. See ``docs/distributed_statistics.md`` for the two-pass halo.

    Parameters
    ----------
    positions:
        ``(N, 3)`` data points.
    k:
        Neighbours per query.
    r_max:
        Search radius cap, below ``min(box_size)/2``.
    box_size:
        Periodic box, or ``None`` with ``bounds``.
    config:
        Capacities.
    queries:
        ``(Q, 3)`` query points (e.g. randoms). ``None``: the data are the
        queries, self excluded.
    r_grid:
        Optional radius grid for the CDF reduction.
    sharpness:
        Soft step in ``ln r`` for the CDF; ``None`` is the hard step.
    adaptive_reach:
        Thin the halo with a local first pass.
    mesh:
        1-D mesh over :data:`AXIS_NAME`; default all devices.
    bounds:
        Morton box of a non-periodic run.

    Returns
    -------
    DistributedKNN
        ``distances`` reordered to the INPUT order of the queries (``(Q, k)``,
        or ``(N, k)`` without ``queries``), ``gid`` the matching input ids,
        the CDF and the diagnostics.

    Notes
    -----
    Differentiable with respect to ``positions`` and ``queries`` (and
    traceable). The neighbour indices are topology, recomputed from the current
    positions on every call but carrying no cotangent; the distances are
    recomputed from the live positions at those indices, so a gradient reaches
    both the query and each of its neighbours, on whichever device owns it. An
    ``inf`` entry (no neighbour within ``r_max``) has no gradient: mask it
    before using it in a loss. The hard CDF is piecewise constant (zero
    gradient); pass ``sharpness`` for a differentiable one.
    """
    require_x64("distributed statistics (Morton codes, global ids)")
    mesh = _resolve_mesh(mesh, None)
    ndev = int(mesh.size)
    cfg = config.resolved()
    box = normalize_box_size(box_size, 3)
    _check_reach(box, r_max, "r_max")
    data = jnp.asarray(positions)
    nd = int(data.shape[0])
    if queries is None:
        pts = data
        q = np.ones((nd,), bool)
        dflag = np.ones((nd,), bool)
        gid = np.arange(nd, dtype=np.int64)
        n_out = nd
    else:
        qq = jnp.asarray(queries).astype(data.dtype)
        nq = int(qq.shape[0])
        pts = jnp.concatenate([data, qq])
        q = np.concatenate([np.zeros(nd, bool), np.ones(nq, bool)])
        dflag = ~q
        gid = np.concatenate([np.full(nd, -1, np.int64), np.arange(nq, dtype=np.int64)])
        n_out = nq
    flags = np.stack([q, dflag], axis=1).astype(np.int32)
    counts, pos_f, (flags_f, gid_f) = _split_for_mesh(pts, ndev, (flags, 0), (gid, -1))
    has_grid = r_grid is not None
    soft = sharpness is not None
    has_bounds = bounds is not None
    dtype = data.dtype
    lo_b, hi_b = (
        (np.asarray(bounds[0]), np.asarray(bounds[1]))
        if has_bounds and bounds is not None
        else (np.zeros(3), np.ones(3))
    )
    prog = _knn_program(
        mesh,
        cfg,
        box,
        int(k),
        float(r_max),
        bool(adaptive_reach),
        has_grid,
        soft,
        has_bounds,
    )
    rg = jnp.asarray(r_grid if has_grid else np.ones(1), dtype)
    dist, out_gid, cdf, overflow, fl, st = prog(
        pos_f,
        jnp.asarray(flags_f),
        jnp.asarray(gid_f, jnp.int64),
        jnp.asarray(counts),
        rg,
        jnp.asarray(sharpness if soft else 1.0, dtype),
        jnp.asarray(lo_b, dtype),
        jnp.asarray(hi_b, dtype),
    )
    # back to the input order on the device (a scatter, so it differentiates);
    # padding and non-query rows aim one past the end and are dropped. The ids
    # stay int64: past 2^31 particles an INDEX_DTYPE (int32) cast would wrap.
    rows = jnp.where(out_gid >= 0, out_gid, n_out).astype(jnp.int64)
    result = (
        jnp.full((n_out, int(k)), jnp.inf, dist.dtype).at[rows].set(dist, mode="drop")
    )
    return DistributedKNN(
        distances=result,
        gid=jnp.arange(n_out),
        cdf=cdf if has_grid else None,
        overflow=overflow,
        flags=fl,
        stats=st,
    )
