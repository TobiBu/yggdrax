"""Differentiable, tree-accelerated two-point pair-count estimator.

The estimator computes a soft-binned pair-count histogram
``DD_k = sum_{i<j} w_k(r_ij)`` (see :mod:`~yggdrax.applications.corrfunc.binning`)
using the yggdrax dual-tree traversal to avoid the :math:`O(N^2)` sum:

* **Near field** (leaf--leaf pairs the traversal did not accept as far):
  evaluated *exactly*, per particle pair, so the near contribution is exact.
* **Far field** (well-separated node pairs accepted by the MAC): each far
  node pair ``(A, B)`` contributes ``count_A * count_B * w_k(d_AB)`` where
  ``d_AB`` is the distance between the node centres of mass -- a monopole
  approximation whose error is controlled by the opening angle ``theta``.

Design (matching the differentiability model, paper section 2): the tree
topology and near/far partition are built once as an integer
:class:`PairTopology` (non-differentiable), and the soft counts are a smooth,
reverse-mode-differentiable function of the particle positions *given* that
fixed topology. Gradients therefore flow through the continuous separations
and centres of mass, exactly as characterised in section 2.

Backend note: use ``backend="radix"`` (default) or ``"octree"``. These give
provably exact, disjoint, complete pair coverage. The KD-tree stores a pivot
particle at every internal node that is absent from the leaf near-list, so it
does *not* cover all pairs and must not be used for this estimator.

Accounting: both the far node-pair list and the near leaf-neighbour list are
fully symmetric (every ``(A, B)`` also appears as ``(B, A)``), so those
contributions are halved; within-leaf pairs are counted once directly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from yggdrax import (
    DualTreeTraversalConfig,
    MACType,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)
from yggdrax.applications.corrfunc.binning import soft_bin_weights


class PairTopology(NamedTuple):
    """Integer index structure describing the near/far partition of pairs.

    All fields are integer/static (no gradient). The differentiable soft
    counts are computed from particle positions *given* this topology by
    :func:`soft_pair_counts_from_topology`.
    """

    order: Array  # (n,) sorted-slot -> original particle id
    leaf_slots: Array  # (L, max_leaf) padded sorted-slot indices per leaf
    leaf_mask: Array  # (L, max_leaf) 1.0 valid / 0.0 pad
    near_target_row: Array  # (P,) target-leaf row index into leaf_slots
    near_source_row: Array  # (P,) source-leaf row index into leaf_slots
    far_src_start: Array  # (F,) inclusive start slot of far source node
    far_src_end: Array  # (F,) inclusive end slot of far source node
    far_tgt_start: Array  # (F,) inclusive start slot of far target node
    far_tgt_end: Array  # (F,) inclusive end slot of far target node
    num_particles: int


def build_pair_topology(
    positions: Array,
    *,
    theta: float = 0.5,
    leaf_size: int = 16,
    backend: str = "radix",
    mac_type: MACType = "dehnen",
    traversal_config: DualTreeTraversalConfig | None = None,
) -> PairTopology:
    """Build the near/far pair partition for ``positions`` (non-differentiable).

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        theta: Opening angle for the multipole acceptance criterion.
        leaf_size: Target leaf occupancy for the tree build.
        backend: ``"radix"``, ``"octree"``, or ``"kdtree"`` (all give exact
            coverage). ``"kdtree"`` uses the leaf-only bucket KD-tree, which
            stores every point in a leaf and so tiles all pairs; the heap
            KD-tree used by the KNN kernels does not and is not used here.
        mac_type: Acceptance criterion, e.g. ``"dehnen"``.
        traversal_config: Optional explicit capacities; auto-sized if None.

    Returns:
        A :class:`PairTopology` describing leaf blocks, near leaf pairs, and
        far node pairs.
    """
    masses = jnp.ones(positions.shape[0], dtype=positions.dtype)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=backend,
        build_mode="adaptive",
        leaf_size=leaf_size,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type=mac_type,
        traversal_config=traversal_config,
    )

    node_ranges = np.asarray(tree.node_ranges)  # inclusive [start, end]
    order = np.asarray(tree.particle_indices)
    n = int(tree.num_particles)

    # --- leaves: the near-list rows enumerate every leaf and tile [0, n) ---
    leaf_ids = np.asarray(neighbors.leaf_indices)
    leaf_start = node_ranges[leaf_ids, 0]
    leaf_end = node_ranges[leaf_ids, 1]
    leaf_len = leaf_end - leaf_start + 1
    max_leaf = int(leaf_len.max())
    num_leaves = leaf_ids.shape[0]

    ramp = np.arange(max_leaf)[None, :]
    leaf_slots = leaf_start[:, None] + ramp  # (L, max_leaf)
    leaf_mask = (ramp < leaf_len[:, None]).astype(np.float32)
    leaf_slots = np.clip(leaf_slots, 0, n - 1)  # padded entries masked out

    # Map leaf node id -> row index (for the near-neighbour CSR).
    node_to_row = np.full(int(node_ranges.shape[0]), -1, dtype=np.int64)
    node_to_row[leaf_ids] = np.arange(num_leaves)

    # --- near: expand the leaf-neighbour CSR to (target_row, source_row) ---
    # Vectorized: row r owns neighbours[offsets[r]:offsets[r+1]]; repeat the row
    # index by its count and map each neighbour node id to its leaf row.
    n_off = np.asarray(neighbors.offsets)
    n_nb = np.asarray(neighbors.neighbors)
    row_counts = (n_off[1:] - n_off[:-1]).astype(np.int64)
    near_target_row = np.repeat(np.arange(num_leaves, dtype=np.int64), row_counts)
    near_source_row = node_to_row[n_nb[: int(n_off[-1])]].astype(np.int64)

    # --- far: node pairs (inclusive slot spans) ---
    far_src = np.asarray(interactions.sources)
    far_tgt = np.asarray(interactions.targets)

    return PairTopology(
        order=jnp.asarray(order),
        leaf_slots=jnp.asarray(leaf_slots),
        leaf_mask=jnp.asarray(leaf_mask),
        near_target_row=jnp.asarray(near_target_row),
        near_source_row=jnp.asarray(near_source_row),
        far_src_start=jnp.asarray(node_ranges[far_src, 0]),
        far_src_end=jnp.asarray(node_ranges[far_src, 1]),
        far_tgt_start=jnp.asarray(node_ranges[far_tgt, 0]),
        far_tgt_end=jnp.asarray(node_ranges[far_tgt, 1]),
        num_particles=n,
    )


def _leaf_blocks(pos_sorted: Array, topo: PairTopology) -> Array:
    """Gather padded per-leaf particle blocks, shape ``(L, max_leaf, 3)``."""
    return pos_sorted[topo.leaf_slots]


def _pairwise_dist(a: Array, b: Array) -> Array:
    """Euclidean distances between blocks ``a`` and ``b`` over last-but-one axis.

    Args:
        a: Array ``(..., m, 3)``.
        b: Array ``(..., p, 3)``.

    Returns:
        Distances ``(..., m, p)``.
    """
    d = a[..., :, None, :] - b[..., None, :, :]
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)


def soft_pair_counts_from_topology(
    positions: Array,
    topo: PairTopology,
    edges: Array,
    sharpness: float,
    *,
    log: bool = True,
) -> Array:
    """Differentiable soft-binned pair counts given a fixed pair topology.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``. Differentiable input.
        topo: Pair topology from :func:`build_pair_topology`.
        edges: Radial bin edges, shape ``(nbins + 1,)``.
        sharpness: Soft-window sharpness.
        log: Bin in ``log`` separation.

    Returns:
        Per-bin soft pair counts, shape ``(nbins,)``, differentiable in
        ``positions``.
    """
    pos_sorted = positions[topo.order]
    nbins = int(edges.shape[0]) - 1

    # --- near field: exact per-pair soft counts ---
    blocks = _leaf_blocks(pos_sorted, topo)  # (L, max_leaf, 3)
    mask = topo.leaf_mask  # (L, max_leaf)

    # within-leaf pairs (strict upper triangle), counted once.
    r_within = _pairwise_dist(blocks, blocks)  # (L, max_leaf, max_leaf)
    w_within = soft_bin_weights(r_within, edges, sharpness, log=log)
    pair_mask = mask[:, :, None] * mask[:, None, :]  # (L, ml, ml)
    ml = blocks.shape[1]
    triu = jnp.triu(jnp.ones((ml, ml), dtype=pos_sorted.dtype), k=1)
    within = jnp.sum(w_within * (pair_mask * triu)[..., None], axis=(0, 1, 2))

    # cross-leaf near pairs; the neighbour list is symmetric, so halve.
    tgt_blocks = blocks[topo.near_target_row]  # (P, ml, 3)
    src_blocks = blocks[topo.near_source_row]
    tgt_mask = mask[topo.near_target_row]
    src_mask = mask[topo.near_source_row]
    r_cross = _pairwise_dist(tgt_blocks, src_blocks)  # (P, ml, ml)
    w_cross = soft_bin_weights(r_cross, edges, sharpness, log=log)
    cross_mask = tgt_mask[:, :, None] * src_mask[:, None, :]
    cross = 0.5 * jnp.sum(w_cross * cross_mask[..., None], axis=(0, 1, 2))

    # --- far field: monopole (centre-of-mass) approximation; halve ---
    if topo.far_src_start.shape[0] > 0:
        # Per-node centre of mass and count via prefix sums over sorted pos.
        prefix = jnp.concatenate(
            [jnp.zeros((1, 3), pos_sorted.dtype), jnp.cumsum(pos_sorted, axis=0)]
        )
        cnt_prefix = jnp.arange(topo.num_particles + 1, dtype=pos_sorted.dtype)

        def com_and_count(start, end):
            total = prefix[end + 1] - prefix[start]
            count = cnt_prefix[end + 1] - cnt_prefix[start]
            return total / count[..., None], count

        com_src, cnt_src = com_and_count(topo.far_src_start, topo.far_src_end)
        com_tgt, cnt_tgt = com_and_count(topo.far_tgt_start, topo.far_tgt_end)
        d_far = jnp.sqrt(jnp.sum((com_src - com_tgt) ** 2, axis=-1) + 1e-30)
        w_far = soft_bin_weights(d_far, edges, sharpness, log=log)  # (F, nbins)
        far = 0.5 * jnp.sum((cnt_src * cnt_tgt)[:, None] * w_far, axis=0)
    else:
        far = jnp.zeros(nbins, dtype=pos_sorted.dtype)

    return within + cross + far


def soft_pair_counts(
    positions: Array,
    edges: Array,
    *,
    theta: float = 0.5,
    sharpness: float = 100.0,
    leaf_size: int = 16,
    backend: str = "radix",
    mac_type: MACType = "dehnen",
    traversal_config: DualTreeTraversalConfig | None = None,
    log: bool = True,
) -> Array:
    """Tree-accelerated differentiable soft pair counts (build + accumulate).

    Convenience wrapper that builds the pair topology and evaluates the soft
    counts. For gradients w.r.t. positions at a fixed topology (the intended
    use), build the topology once with :func:`build_pair_topology` and
    differentiate :func:`soft_pair_counts_from_topology`.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        edges: Radial bin edges, shape ``(nbins + 1,)``.
        theta: Opening angle for the MAC.
        sharpness: Soft-window sharpness.
        leaf_size: Target leaf occupancy.
        backend: ``"radix"`` or ``"octree"``.
        mac_type: Acceptance criterion.
        traversal_config: Optional explicit traversal capacities.
        log: Bin in ``log`` separation.

    Returns:
        Per-bin soft pair counts, shape ``(nbins,)``.
    """
    topo = build_pair_topology(
        positions,
        theta=theta,
        leaf_size=leaf_size,
        backend=backend,
        mac_type=mac_type,
        traversal_config=traversal_config,
    )
    return soft_pair_counts_from_topology(positions, topo, edges, sharpness, log=log)
