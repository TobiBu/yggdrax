"""Tree-accelerated, differentiable SVGD via a far-field monopole expansion.

The empirical Stein update sums over all particle pairs, :math:`O(N^2)`. We
split the sum with the yggdrax dual-tree partition (the same near/far machinery
used for gravity and correlation functions):

* **Near field** -- pairs the traversal did not accept as far are summed
  *exactly*, per particle pair, over leaf blocks.
* **Far field** -- each well-separated source node ``B`` is summarised by its
  particle count, centre of mass, and summed score, and this monopole is
  evaluated directly at every target particle (an M2P/treecode step):

  .. math::

      \\sum_{j \\in B} \\big[k(x_j,x_i)\\,s_j + k(x_j,x_i)(x_i-x_j)/h^2\\big]
      \\approx k(c_B, x_i)\\,\\big[S_B + n_B (x_i - c_B)/h^2\\big],

  with :math:`c_B` the centre of mass, :math:`n_B=|B|`, :math:`S_B=\\sum_j s_j`.
  The approximation is controlled by the opening angle: it is exact as
  :math:`\\theta \\to 0` (nothing accepted far) and degrades gracefully as
  ``theta`` opens.

The tree partition is a discrete, non-differentiable topology built once per
step; the update is a smooth function of positions, scores, and bandwidth
``h`` given that partition, so gradients flow for bandwidth learning.

Backend: the default ``"leaf_kdtree"`` is a leaf-only bucket KD-tree that
stores every particle in a leaf and so tiles all pairs exactly, in *arbitrary
dimension* -- SVGD targets need not be 3-D. The 3-D radix and octree backends
are also available (targets in :math:`d<3` are padded to 3-D for the tree build
only). All kernel/geometry is evaluated in the true dimension either way.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)
from yggdrax.applications.svgd.kernel import stein_pair_terms
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.kdtree import build_leaf_kdtree

# Tree geometry is a pure device computation whose output shape is fixed by the
# tree structure (``num_nodes``), which for a given ``(n, leaf_size)`` does not
# depend on the particle positions. Jitting it therefore compiles once and reuses
# across per-step rebuilds, collapsing the eager op-dispatch overhead that
# otherwise dominates the build (~490 ms -> ~0.5 ms per rebuild). ``max_leaf_size``
# is a static staging-buffer cap.
_jit_compute_tree_geometry = jax.jit(
    compute_tree_geometry, static_argnames=("max_leaf_size",)
)


# Traversal action codes, mirroring ``yggdrax._interactions_impl``.
_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2

#: Tag on an accepted far pair that is summarised by its monopole.
FAR_TAG_MONOPOLE = 0
#: Tag on an accepted far pair whose kernel contribution is negligible. Such a
#: pair is accepted purely to stop the walk descending into it; the partition
#: then drops it, so it costs nothing in the update.
FAR_TAG_IGNORE = 1


def _kernel_cutoff_pair_policy(
    policy_state,
    *,
    valid_pairs,
    mac_ok,
    different_nodes,
    target_leaf,
    source_leaf,
    same_node,
    target_nodes,
    source_nodes,
    center_target,
    center_source,
    dist_sq,
    extent_target,
    extent_source,
):
    """Traversal policy that drops node pairs the RBF kernel cannot reach.

    The Stein kernel ``exp(-r^2 / 2h^2)`` has compact effective support, so a
    node pair whose *closest possible* particle separation
    ``d - r_A - r_B`` already exceeds the cutoff contributes nothing -- at any
    opening angle. Gravity has no such pairs, which is why the built-in MAC has
    no notion of them: it summarises every well-separated pair as a monopole and
    evaluates it at every target particle, and for this kernel the great majority
    of those evaluations return zero to floating-point precision.

    Three outcomes per pair:

    * **drop** -- beyond the cutoff: accept (so the walk stops descending) and
      tag :data:`FAR_TAG_IGNORE` so the partition never expands it;
    * **monopole** -- the MAC accepts and the pair is within the cutoff: accept
      and tag :data:`FAR_TAG_MONOPOLE`, the usual far field;
    * otherwise the built-in near/refine decision stands.

    The decision is symmetric under swapping target and source, so the two
    directed evaluations the traversal performs always agree.

    Args:
        policy_state: The cutoff radius ``r_cut`` (a scalar), in the same length
            units as the particle positions.
        valid_pairs: Mask of live pairs in the wavefront.
        mac_ok: Built-in multipole acceptance decision per pair.
        different_nodes: Mask of pairs whose two nodes differ.
        target_leaf: Whether the target node is a leaf.
        source_leaf: Whether the source node is a leaf.
        same_node: Unused.
        target_nodes: Unused.
        source_nodes: Unused.
        center_target: Unused.
        center_source: Unused.
        dist_sq: Squared distance between the two node centres.
        extent_target: Conservative radius of the target node.
        extent_source: Conservative radius of the source node.

    Returns:
        ``(actions, tags)`` as integer arrays over the pair batch.
    """
    del same_node, target_nodes, source_nodes, center_target, center_source

    r_cut = jnp.asarray(policy_state, dtype=dist_sq.dtype)
    gap = jnp.sqrt(dist_sq) - extent_target - extent_source
    negligible = valid_pairs & different_nodes & (gap > r_cut)

    near = valid_pairs & (~mac_ok) & target_leaf & source_leaf & different_nodes
    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=INDEX_DTYPE)
    actions = jnp.where(mac_ok, as_index(_ACTION_ACCEPT), actions)
    actions = jnp.where(near, as_index(_ACTION_NEAR), actions)
    actions = jnp.where(negligible, as_index(_ACTION_ACCEPT), actions)
    tags = jnp.where(negligible, as_index(FAR_TAG_IGNORE), as_index(FAR_TAG_MONOPOLE))
    return actions, tags


class SvgdTraversal(NamedTuple):
    """Device-side output of the SVGD dual-tree walk, before host assembly.

    Everything here is still a device array: the tree's particle order and node
    ranges, the near-neighbour CSR of the leaf walk, and the accepted far node
    pairs. :func:`assemble_svgd_topology` is what pulls it to the host.
    """

    order: Array  # (n,) sorted-slot -> original particle id
    node_ranges: Array  # (num_nodes, 2) inclusive [start, end] slot range
    leaf_ids: Array  # (L,) node id of each leaf, in row order
    near_offsets: Array  # (L + 1,) CSR offsets into ``near_neighbors``
    near_neighbors: Array  # (>= near_offsets[-1],) source *node* ids
    far_src: Array  # (F,) source node of each accepted far pair
    far_tgt: Array  # (F,) target node of each accepted far pair
    far_tags: Array  # (F,) FAR_TAG_MONOPOLE / FAR_TAG_IGNORE per far pair
    num_particles: int


class SvgdTopology(NamedTuple):
    """Integer partition for one tree-accelerated Stein update (non-diff)."""

    order: Array  # (n,) sorted-slot -> original particle id
    leaf_slots: Array  # (L, max_leaf) padded sorted-slot indices per leaf
    leaf_mask: Array  # (L, max_leaf) 1.0 valid / 0.0 pad
    near_target_row: Array  # (P,) target-leaf row (directional, complete)
    near_source_row: Array  # (P,) source-leaf row
    far_tgt_slot: Array  # (M,) sorted-slot of each far target particle
    far_src_start: Array  # (M,) inclusive start slot of the far source node
    far_src_end: Array  # (M,) inclusive end slot of the far source node
    num_far_pairs: int  # far node pairs kept (M is their expansion, >= this)
    num_particles: int


def _pad_to_3d(points: Array) -> Array:
    d = points.shape[1]
    if d == 3:
        return points
    if d > 3:
        raise ValueError(
            "the far-field SVGD sampler uses the 3-D radix/octree backends; "
            f"got dimension {d}. Use d <= 3."
        )
    pad = jnp.zeros((points.shape[0], 3 - d), dtype=points.dtype)
    return jnp.concatenate([points, pad], axis=1)


def build_svgd_traversal(
    particles: Float[Array, "n d"],
    *,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    traversal_config: DualTreeTraversalConfig | None = None,
    kernel_cutoff: float | None = None,
) -> SvgdTraversal:
    """Run the dual-tree walk for the Stein update and return its device output.

    This is the whole device half of :func:`build_svgd_topology`: tree build,
    geometry, traversal. Nothing is pulled to the host, so a caller can time it
    on its own (``jax.block_until_ready`` on the returned tuple), reuse one walk
    for several partitions, or replace the host assembly with its own.

    Args:
        particles: Particle positions, shape ``(n, d)``. Arbitrary ``d`` with
            the default ``leaf_kdtree`` backend; ``d <= 3`` for radix/octree.
        theta: Opening angle for the multipole acceptance criterion.
        leaf_size: Target leaf occupancy for the tree build.
        backend: ``"leaf_kdtree"`` (default, dimension-general, exact coverage),
            ``"radix"``, or ``"octree"`` (both 3-D only).
        traversal_config: Optional explicit traversal capacities.
        kernel_cutoff: Optional kernel cutoff radius ``r_cut``. When given, node
            pairs whose closest possible separation exceeds it are dropped by
            :func:`_kernel_cutoff_pair_policy` instead of being refined or
            summarised; see :func:`svgd_phi` for the ``c * h`` convention.
            ``None`` (default) reproduces the plain MAC traversal exactly.

    Returns:
        An :class:`SvgdTraversal`.

    Raises:
        ValueError: If a 3-D-only backend is requested for ``d != 3``, the
            backend name is unknown, or ``kernel_cutoff`` is not positive.
    """
    if kernel_cutoff is not None and not kernel_cutoff > 0.0:
        raise ValueError(f"kernel_cutoff must be positive; got {kernel_cutoff!r}")
    if backend == "leaf_kdtree":
        tree = build_leaf_kdtree(particles, leaf_size=leaf_size)
        pos_sorted = particles[tree.particle_indices]
    elif backend in ("radix", "octree"):
        if particles.shape[1] > 3:
            raise ValueError(
                f"backend={backend!r} is 3-D only; use 'leaf_kdtree' for "
                f"dimension {particles.shape[1]}."
            )
        pts3d = _pad_to_3d(particles)
        masses = jnp.ones(pts3d.shape[0], dtype=pts3d.dtype)
        tree = Tree.from_particles(
            pts3d,
            masses,
            tree_type=backend,
            build_mode="adaptive",
            leaf_size=leaf_size,
            return_reordered=True,
        )
        pos_sorted = tree.positions_sorted
    else:
        raise ValueError(
            f"unknown backend {backend!r}; use 'leaf_kdtree', 'radix', or " "'octree'."
        )
    geometry = _jit_compute_tree_geometry(tree, pos_sorted, max_leaf_size=leaf_size)
    if kernel_cutoff is None:
        interactions, neighbors = build_interactions_and_neighbors(
            tree,
            geometry,
            theta=theta,
            mac_type="dehnen",
            traversal_config=traversal_config,
        )
        far_src = interactions.sources
        far_tgt = interactions.targets
        far_tags = jnp.full(far_src.shape, FAR_TAG_MONOPOLE, dtype=INDEX_DTYPE)
    else:
        # The tags only exist when a policy is installed, and they are carried on
        # the compact far-pair payload rather than on the sparse per-node list.
        interactions, neighbors, far_pairs = build_interactions_and_neighbors(
            tree,
            geometry,
            theta=theta,
            mac_type="dehnen",
            traversal_config=traversal_config,
            pair_policy=_kernel_cutoff_pair_policy,
            policy_state=float(kernel_cutoff),
            return_compact_far_pairs=True,
        )
        del interactions
        far_src = far_pairs.sources
        far_tgt = far_pairs.targets
        far_tags = far_pairs.tags
    return SvgdTraversal(
        order=tree.particle_indices,
        node_ranges=tree.node_ranges,
        leaf_ids=neighbors.leaf_indices,
        near_offsets=neighbors.offsets,
        near_neighbors=neighbors.neighbors,
        far_src=far_src,
        far_tgt=far_tgt,
        far_tags=far_tags,
        num_particles=int(tree.num_particles),
    )


def assemble_svgd_topology(walk: SvgdTraversal) -> SvgdTopology:
    """Turn a dual-tree walk into the near/far partition of the Stein update.

    This is the host half of :func:`build_svgd_topology`: it pulls the walk's
    integer arrays to the host and expands them into per-leaf slot blocks, the
    directional near-pair rows, and the far target/source slot ranges.

    Args:
        walk: Device output of :func:`build_svgd_traversal`.

    Returns:
        An :class:`SvgdTopology`.
    """
    node_ranges = np.asarray(walk.node_ranges)  # inclusive [start, end]
    order = np.asarray(walk.order)
    n = int(walk.num_particles)

    # Leaves tile [0, n); build padded per-leaf slot blocks.
    leaf_ids = np.asarray(walk.leaf_ids)
    leaf_start = node_ranges[leaf_ids, 0]
    leaf_end = node_ranges[leaf_ids, 1]
    leaf_len = leaf_end - leaf_start + 1
    max_leaf = int(leaf_len.max())
    num_leaves = leaf_ids.shape[0]
    ramp = np.arange(max_leaf)[None, :]
    leaf_slots = np.clip(leaf_start[:, None] + ramp, 0, n - 1)
    leaf_mask = (ramp < leaf_len[:, None]).astype(np.float32)

    node_to_row = np.full(int(node_ranges.shape[0]), -1, dtype=np.int64)
    node_to_row[leaf_ids] = np.arange(num_leaves)

    # Directional near leaf pairs (complete; NOT halved -- each target receives
    # from each source, and the symmetric entry handles the reverse).
    n_off = np.asarray(walk.near_offsets)
    n_nb = np.asarray(walk.near_neighbors)
    row_counts = (n_off[1:] - n_off[:-1]).astype(np.int64)
    near_target_row = np.repeat(np.arange(num_leaves, dtype=np.int64), row_counts)
    near_source_row = node_to_row[n_nb[: int(n_off[-1])]].astype(np.int64)

    # Far field: expand each far pair's TARGET node to its particles; each such
    # particle receives the monopole of the paired SOURCE node.
    far_src = np.asarray(walk.far_src)
    far_tgt = np.asarray(walk.far_tgt)
    far_tags = np.asarray(walk.far_tags)
    if far_tags.shape[0] == far_src.shape[0]:
        keep = far_tags != FAR_TAG_IGNORE
        if not keep.all():
            far_src = far_src[keep]
            far_tgt = far_tgt[keep]
    tgt_start = node_ranges[far_tgt, 0]
    tgt_end = node_ranges[far_tgt, 1]
    tgt_len = (tgt_end - tgt_start + 1).astype(np.int64)
    far_tgt_slot = (
        np.concatenate([np.arange(s, e + 1) for s, e in zip(tgt_start, tgt_end)])
        if far_src.shape[0] > 0
        else np.zeros(0, dtype=np.int64)
    )
    far_src_start = np.repeat(node_ranges[far_src, 0], tgt_len)
    far_src_end = np.repeat(node_ranges[far_src, 1], tgt_len)

    return SvgdTopology(
        order=jnp.asarray(order),
        leaf_slots=jnp.asarray(leaf_slots),
        leaf_mask=jnp.asarray(leaf_mask),
        near_target_row=jnp.asarray(near_target_row),
        near_source_row=jnp.asarray(near_source_row),
        far_tgt_slot=jnp.asarray(far_tgt_slot.astype(np.int64)),
        far_src_start=jnp.asarray(far_src_start.astype(np.int64)),
        far_src_end=jnp.asarray(far_src_end.astype(np.int64)),
        num_far_pairs=int(far_src.shape[0]),
        num_particles=n,
    )


def build_svgd_topology(
    particles: Float[Array, "n d"],
    *,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    traversal_config: DualTreeTraversalConfig | None = None,
    kernel_cutoff: float | None = None,
) -> SvgdTopology:
    """Build the near/far Stein-update partition for ``particles``.

    Args:
        particles: Particle positions, shape ``(n, d)``. Arbitrary ``d`` with
            the default ``leaf_kdtree`` backend; ``d <= 3`` for radix/octree.
        theta: Opening angle for the multipole acceptance criterion.
        leaf_size: Target leaf occupancy for the tree build.
        backend: ``"leaf_kdtree"`` (default, dimension-general, exact coverage),
            ``"radix"``, or ``"octree"`` (both 3-D only).
        traversal_config: Optional explicit traversal capacities.
        kernel_cutoff: Optional kernel cutoff radius; see
            :func:`build_svgd_traversal`.

    Returns:
        An :class:`SvgdTopology`.
    """
    return assemble_svgd_topology(
        build_svgd_traversal(
            particles,
            theta=theta,
            leaf_size=leaf_size,
            backend=backend,
            traversal_config=traversal_config,
            kernel_cutoff=kernel_cutoff,
        )
    )


def svgd_phi_from_topology(
    particles: Float[Array, "n d"],
    scores: Float[Array, "n d"],
    h: float | Float[Array, ""],
    topo: SvgdTopology,
) -> Float[Array, "n d"]:
    """Tree-accelerated Stein update given a fixed partition (differentiable).

    Args:
        particles: Particle positions, shape ``(n, d)``.
        scores: Target score at each particle, shape ``(n, d)``.
        h: Kernel bandwidth.
        topo: Partition from :func:`build_svgd_topology`.

    Returns:
        Update directions, shape ``(n, d)``.
    """
    n, d = particles.shape
    pos = particles[topo.order]  # sorted order
    sco = scores[topo.order]
    phi = jnp.zeros((n, d), dtype=particles.dtype)

    # --- near field: exact per-pair Stein terms ---
    blocks_x = pos[topo.leaf_slots]  # (L, ml, d)
    blocks_s = sco[topo.leaf_slots]
    mask = topo.leaf_mask  # (L, ml)

    # within-leaf: target axis 1, source axis 2.
    terms = stein_pair_terms(
        blocks_x[:, :, None, :], blocks_x[:, None, :, :], blocks_s[:, None, :, :], h
    )  # (L, ml, ml, d)
    within = jnp.sum(terms * mask[:, None, :, None], axis=2)  # (L, ml, d)
    phi = phi.at[topo.leaf_slots].add(within * mask[..., None])

    # cross-leaf near pairs (directional).
    tgt_x = blocks_x[topo.near_target_row]  # (P, ml, d)
    src_x = blocks_x[topo.near_source_row]
    src_s = blocks_s[topo.near_source_row]
    src_m = mask[topo.near_source_row]
    tgt_m = mask[topo.near_target_row]
    cterms = stein_pair_terms(
        tgt_x[:, :, None, :], src_x[:, None, :, :], src_s[:, None, :, :], h
    )  # (P, ml, ml, d)
    cross = jnp.sum(cterms * src_m[:, None, :, None], axis=2)  # (P, ml, d)
    tgt_slots = topo.leaf_slots[topo.near_target_row]  # (P, ml)
    phi = phi.at[tgt_slots].add(cross * tgt_m[..., None])

    # --- far field: monopole (M2P) ---
    if topo.far_tgt_slot.shape[0] > 0:
        zero_x = jnp.zeros((1, d), pos.dtype)
        pos_prefix = jnp.concatenate([zero_x, jnp.cumsum(pos, axis=0)])
        sco_prefix = jnp.concatenate([zero_x, jnp.cumsum(sco, axis=0)])
        cnt_prefix = jnp.arange(n + 1, dtype=pos.dtype)
        s, e = topo.far_src_start, topo.far_src_end
        count = (cnt_prefix[e + 1] - cnt_prefix[s])[:, None]  # (M, 1)
        sum_x = pos_prefix[e + 1] - pos_prefix[s]  # (M, d)
        sum_s = sco_prefix[e + 1] - sco_prefix[s]  # (M, d)
        com = sum_x / count
        x_i = pos[topo.far_tgt_slot]  # (M, d)
        d2 = jnp.sum((x_i - com) ** 2, axis=-1, keepdims=True)
        kB = jnp.exp(-d2 / (2.0 * h**2))
        contrib = kB * (sum_s + count * (x_i - com) / (h**2))  # (M, d)
        phi = phi.at[topo.far_tgt_slot].add(contrib)

    phi = phi / n
    # phi is in sorted order; scatter back to original particle order.
    return jnp.zeros_like(phi).at[topo.order].set(phi)


# Fused, compiled accumulation. Given a (fixed) partition the Stein update is a
# pure array computation; jitting it collapses the eager per-op dispatch into one
# kernel (~1.5x faster per step even when the partition shapes vary a little).
_jit_svgd_phi_from_topology = jax.jit(svgd_phi_from_topology)


def _cutoff_radius(
    cutoff_bandwidths: float | None, h: float | Float[Array, ""]
) -> float | None:
    """Return the absolute cutoff radius for ``c`` bandwidths, or ``None``.

    Args:
        cutoff_bandwidths: Cutoff ``c`` in bandwidths, or ``None``.
        h: Kernel bandwidth.

    Returns:
        ``c * h`` as a Python float, or ``None`` when no cutoff is requested.
    """
    if cutoff_bandwidths is None:
        return None
    return float(cutoff_bandwidths) * float(h)


def svgd_phi(
    particles: Float[Array, "n d"],
    scores: Float[Array, "n d"],
    h: float | Float[Array, ""],
    *,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    traversal_config: DualTreeTraversalConfig | None = None,
    cutoff_bandwidths: float | None = None,
) -> Float[Array, "n d"]:
    """Tree-accelerated Stein update (build partition + accumulate).

    Args:
        particles: Particle positions, shape ``(n, d)``.
        scores: Target score at each particle, shape ``(n, d)``.
        h: Kernel bandwidth.
        theta: Opening angle.
        leaf_size: Target leaf occupancy.
        backend: ``"radix"`` or ``"octree"``.
        traversal_config: Optional explicit traversal capacities.
        cutoff_bandwidths: Kernel cutoff ``c``, in bandwidths: node pairs whose
            closest possible separation exceeds ``c * h`` are dropped, at a
            relative cost bounded by ``exp(-c^2 / 2)`` (1.5e-8 at ``c = 6``).
            ``None`` (default) keeps the monopole-everything far field.

    Returns:
        Update directions, shape ``(n, d)``.
    """
    topo = build_svgd_topology(
        particles,
        theta=theta,
        leaf_size=leaf_size,
        backend=backend,
        traversal_config=traversal_config,
        kernel_cutoff=_cutoff_radius(cutoff_bandwidths, h),
    )
    return _jit_svgd_phi_from_topology(particles, scores, h, topo)


def tree_svgd_step(
    particles: Float[Array, "n d"],
    score_fn: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
    h: float | Float[Array, ""],
    step_size: float,
    *,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    traversal_config: DualTreeTraversalConfig | None = None,
    cutoff_bandwidths: float | None = None,
) -> Float[Array, "n d"]:
    """One tree-accelerated SVGD step.

    Args:
        particles: Current particles, shape ``(n, d)``.
        score_fn: Target score function, ``(n, d) -> (n, d)``.
        h: Kernel bandwidth.
        step_size: Update step size.
        theta: Opening angle.
        leaf_size: Target leaf occupancy.
        backend: ``"radix"`` or ``"octree"``.
        traversal_config: Optional explicit traversal capacities.
        cutoff_bandwidths: Kernel cutoff in bandwidths; see :func:`svgd_phi`.

    Returns:
        Updated particles, shape ``(n, d)``.
    """
    scores = score_fn(particles)
    phi = svgd_phi(
        particles,
        scores,
        h,
        theta=theta,
        leaf_size=leaf_size,
        backend=backend,
        traversal_config=traversal_config,
        cutoff_bandwidths=cutoff_bandwidths,
    )
    return particles + step_size * phi


def run_tree_svgd(
    particles: Float[Array, "n d"],
    score_fn: Callable[[Float[Array, "n d"]], Float[Array, "n d"]],
    h: float | Float[Array, ""],
    step_size: float,
    num_steps: int,
    *,
    theta: float = 0.4,
    leaf_size: int = 32,
    backend: str = "leaf_kdtree",
    traversal_config: DualTreeTraversalConfig | None = None,
    cutoff_bandwidths: float | None = None,
) -> Float[Array, "n d"]:
    """Run tree-accelerated SVGD for ``num_steps`` steps.

    Args:
        particles: Initial particles, shape ``(n, d)``.
        score_fn: Target score function, ``(n, d) -> (n, d)``.
        h: Kernel bandwidth (fixed across steps).
        step_size: Update step size.
        num_steps: Number of SVGD steps.
        theta: Opening angle.
        leaf_size: Target leaf occupancy.
        backend: ``"radix"`` or ``"octree"``.
        traversal_config: Optional explicit traversal capacities.
        cutoff_bandwidths: Kernel cutoff in bandwidths; see :func:`svgd_phi`.

    Returns:
        Final particles, shape ``(n, d)``.
    """
    p = particles
    for _ in range(num_steps):
        p = tree_svgd_step(
            p,
            score_fn,
            h,
            step_size,
            theta=theta,
            leaf_size=leaf_size,
            backend=backend,
            traversal_config=traversal_config,
            cutoff_bandwidths=cutoff_bandwidths,
        )
    return p
