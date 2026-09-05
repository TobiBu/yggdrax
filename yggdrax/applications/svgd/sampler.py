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
from yggdrax.applications.svgd.pallas_farfield import farfield_stein
from yggdrax.applications.svgd.pallas_nearfield import (
    nearfield_stein,
    prefer_fused_nearfield,
)
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
    near_target_row: Array  # (Q,) one row of each UNORDERED near leaf pair
    near_source_row: Array  # (Q,) the other row; row_a < row_b throughout
    near_dir_target: Array  # (2Q,) DIRECTED pairs, ascending in target row
    near_dir_source: Array  # (2Q,) the source row of each directed pair
    near_dir_offsets: Array  # (L + 1,) CSR offsets into the LIVE directed pairs
    near_live: Array  # (Q,) 1.0 real pair / 0.0 capacity padding
    near_dir_live: Array  # (2Q,) the same for the directed list
    far_leaf_offsets: Array  # (L + 1,) CSR offsets into the LIVE far entries
    far_leaf_source: Array  # (P,) index into the far source-node table
    far_leaf_live: Array  # (P,) 1.0 real entry / 0.0 capacity padding
    far_node_start: Array  # (F,) inclusive start slot of each far source node
    far_node_end: Array  # (F,) inclusive end slot of each far source node
    far_node_offsets: Array  # (F + 1,) CSR offsets, the entry list by SOURCE
    far_entry_perm: Array  # (P,) leaf-order entry index, in source order
    num_far_pairs: int  # accepted far node pairs kept
    num_far_contribs: int  # M, the per-particle expansion, never materialised
    num_near_leaf_pairs: int  # DIRECTED near pairs, i.e. 2 * len(near_target_row)
    num_particles: int


def _round_up_capacity(count: int, policy: str | int) -> int:
    """Return the padded capacity for ``count`` under ``policy``.

    Args:
        count: The number of real entries.
        policy: ``"exact"`` for no padding, ``"bucket"`` for the next eighth of
            an octave, ``"pow2"`` for the next power of two, or an explicit
            integer capacity.

    Returns:
        The capacity to allocate, never less than ``count``.

    Raises:
        ValueError: If an explicit capacity is smaller than ``count``, or the
            policy name is unknown.
    """
    if isinstance(policy, int) and not isinstance(policy, bool):
        if policy < count:
            raise ValueError(
                f"capacity {policy} is smaller than the {count} entries the "
                "partition actually has; the update would silently drop pairs"
            )
        return int(policy)
    if policy == "exact":
        return int(count)
    if count <= 1:
        return 1
    octave = 1 << (int(count) - 1).bit_length()
    if policy == "pow2":
        return octave
    if policy == "bucket":
        # An eighth of an octave: at most 12.5 % padding, and stable against
        # fluctuations up to that size. The counts move by ~1 % between rebuilds
        # (41075..41401 at N = 1e4), so this is far more headroom than needed,
        # where "pow2" happens to cost 1.6x on the same data.
        granule = max(1, octave >> 3)
        return int(-(-int(count) // granule) * granule)
    raise ValueError(
        f"capacity must be 'exact', 'bucket', 'pow2' or an int; got {policy!r}"
    )


def _pad_int(values: np.ndarray, capacity: int, fill: int) -> np.ndarray:
    """Pad a 1-D integer array up to ``capacity`` with ``fill``."""
    if values.shape[0] >= capacity:
        return values
    tail = np.full((capacity - values.shape[0],), fill, dtype=values.dtype)
    return np.concatenate([values, tail])


def _live_mask(count: int, capacity: int) -> np.ndarray:
    """Return a float32 0/1 mask marking the first ``count`` of ``capacity``."""
    live = np.zeros((capacity,), dtype=np.float32)
    live[:count] = 1.0
    return live


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


def _assemble_far_on_host(
    node_ranges: np.ndarray,
    walk: SvgdTraversal,
    leaf_start: np.ndarray,
    num_leaves: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Numpy counterpart of :func:`_assemble_far_on_device`, for small partitions.

    Semantically identical, down to keying the monopole table by node id, so a
    test can build both and compare. It exists because the device path pays a
    fixed dispatch cost and a synchronisation whatever the size, which below a
    few thousand far pairs is more than the numpy work it replaces.

    Args:
        node_ranges: Inclusive ``[start, end]`` slot range of every node.
        walk: Device output of :func:`build_svgd_traversal`.
        leaf_start: First sorted slot of each leaf, in leaf-row order.
        num_leaves: Number of leaves ``L``.

    Returns:
        ``(leaf_offsets, leaf_source, node_offsets, entry_perm, node_start,
        node_end, num_far_entries, num_far_contribs)``.
    """
    far_src = np.asarray(walk.far_src)
    far_tgt = np.asarray(walk.far_tgt)
    far_tags = np.asarray(walk.far_tags)
    if far_tags.shape[0] == far_src.shape[0]:
        keep = far_tags != FAR_TAG_IGNORE
        if not keep.all():
            far_src, far_tgt = far_src[keep], far_tgt[keep]

    num_nodes = int(node_ranges.shape[0])
    tgt_start = node_ranges[far_tgt, 0]
    tgt_end = node_ranges[far_tgt, 1]
    num_far_contribs = int((tgt_end - tgt_start + 1).sum())

    by_start = np.argsort(leaf_start, kind="stable")
    sorted_start = leaf_start[by_start]
    first = np.searchsorted(sorted_start, tgt_start, side="right") - 1
    last = np.searchsorted(sorted_start, tgt_end, side="right") - 1

    pair_order = np.argsort(far_src, kind="stable")
    counts = (last - first + 1).astype(np.int64)[pair_order]
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return (
            np.zeros((num_leaves + 1,), dtype=np.int64),
            empty,
            np.zeros((num_nodes + 1,), dtype=np.int64),
            empty,
            node_ranges[:, 0],
            node_ranges[:, 1],
            0,
            num_far_contribs,
        )

    seg_start = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(seg_start, counts)
    entry_leaf = by_start[
        np.repeat(first.astype(np.int64)[pair_order], counts) + within
    ]
    entry_source = np.repeat(far_src[pair_order].astype(np.int64), counts)
    node_offsets = np.concatenate(
        [[0], np.cumsum(np.bincount(entry_source, minlength=num_nodes))]
    ).astype(np.int64)

    by_leaf = np.argsort(entry_leaf, kind="stable")
    leaf_offsets = np.concatenate(
        [[0], np.cumsum(np.bincount(entry_leaf, minlength=num_leaves))]
    ).astype(np.int64)
    entry_perm = np.empty(total, dtype=np.int64)
    entry_perm[by_leaf] = np.arange(total, dtype=np.int64)
    return (
        leaf_offsets,
        entry_source[by_leaf],
        node_offsets,
        entry_perm,
        node_ranges[:, 0],
        node_ranges[:, 1],
        total,
        num_far_contribs,
    )


def _assemble_far_on_device(
    walk: SvgdTraversal, leaf_start: np.ndarray, num_leaves: int
) -> tuple[Array, Array, Array, Array, Array, int, int]:
    """Build the leaf-major far CSR and its transpose without leaving the device.

    The host version of this was 271 ms of a 633 ms build at N = 1e5: two
    orderings of 2.8 million entries, two ``searchsorted`` over 1.1 million
    pairs, and several ``np.repeat`` of the same size. Every input is already a
    device array when :func:`assemble_svgd_topology` receives it, so none of
    that needed to come to the host in the first place.

    Two departures from the host path, both of which simplify it:

    * **The monopole table is keyed by node id**, not by a compacted index over
      the distinct far source nodes. That removes the dense remap, and it makes
      the table's shape ``num_nodes`` -- fixed by ``(n, leaf_size)`` -- rather
      than a data-dependent ``F``, so it does not churn between rebuilds. The
      table is roughly twice as long and every row is a handful of scalars.
    * **Pairs the policy dropped are given a leaf count of zero** rather than
      being filtered out, because a boolean filter has a data-dependent length
      and would need its own host synchronisation. They then expand to nothing.

    Exactly one scalar crosses to the host: the total entry count, which is a
    static shape everything downstream needs.

    Args:
        walk: Device output of :func:`build_svgd_traversal`.
        leaf_start: First sorted slot of each leaf, in leaf-row order.
        num_leaves: Number of leaves ``L``.

    Returns:
        ``(leaf_offsets, leaf_source, node_offsets, entry_perm, node_ranges,
        num_far_entries, num_far_contribs)``.
    """
    ranges = walk.node_ranges
    far_src, far_tgt = walk.far_src, walk.far_tgt
    num_nodes = int(ranges.shape[0])

    keep = jnp.ones(far_src.shape, dtype=bool)
    if walk.far_tags.shape[0] == far_src.shape[0]:
        keep = walk.far_tags != FAR_TAG_IGNORE

    tgt_start = ranges[far_tgt, 0]
    tgt_end = ranges[far_tgt, 1]
    span = jnp.where(keep, tgt_end - tgt_start + 1, 0)
    num_far_contribs = int(span.sum())

    # Leaf rows are not assumed to be slot-ordered, so the node -> leaf-range
    # lookup goes through a sort of the leaf starts rather than the row index.
    starts = jnp.asarray(leaf_start)
    by_start = jnp.argsort(starts)
    sorted_start = starts[by_start]
    first = jnp.searchsorted(sorted_start, tgt_start, side="right") - 1
    last = jnp.searchsorted(sorted_start, tgt_end, side="right") - 1
    counts = jnp.where(keep, last - first + 1, 0)

    # Expand in SOURCE order, so the transpose CSR the reverse pass needs comes
    # out grouped for free. Ordering 1.1 million pairs and then expanding is far
    # cheaper than expanding first and ordering 2.8 million entries, whose key
    # is no longer nearly sorted -- 20 ms against 180 ms when this was on the
    # host, and the same asymmetry holds here.
    pair_order = jnp.argsort(far_src)
    counts_src = counts[pair_order]
    total = int(counts_src.sum())
    if total == 0:
        empty = as_index(jnp.zeros((0,), dtype=INDEX_DTYPE))
        zeros = as_index(jnp.zeros((num_leaves + 1,), dtype=INDEX_DTYPE))
        node_zeros = as_index(jnp.zeros((num_nodes + 1,), dtype=INDEX_DTYPE))
        return zeros, empty, node_zeros, empty, ranges, 0, num_far_contribs

    seg_start = jnp.cumsum(counts_src) - counts_src
    within = jnp.arange(total, dtype=counts_src.dtype) - jnp.repeat(
        seg_start, counts_src, total_repeat_length=total
    )
    entry_leaf = by_start[
        jnp.repeat(first[pair_order], counts_src, total_repeat_length=total) + within
    ]
    entry_source = jnp.repeat(
        far_src[pair_order], counts_src, total_repeat_length=total
    )
    node_offsets = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(jnp.bincount(entry_source, length=num_nodes)).astype(
                INDEX_DTYPE
            ),
        ]
    )

    # Then the forward order: by target leaf. Only the grouping matters, since
    # the entries within a leaf are summed, so the sort need not be stable.
    by_leaf = jnp.argsort(entry_leaf)
    leaf_offsets = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(jnp.bincount(entry_leaf, length=num_leaves)).astype(INDEX_DTYPE),
        ]
    )
    # The reverse needs the map the other way -- leaf-order index to its position
    # in source order -- which is the inverse of ``by_leaf``, built by scatter.
    entry_perm = (
        jnp.zeros((total,), dtype=INDEX_DTYPE)
        .at[by_leaf]
        .set(jnp.arange(total, dtype=INDEX_DTYPE))
    )
    return (
        as_index(leaf_offsets),
        as_index(entry_source[by_leaf]),
        as_index(node_offsets),
        as_index(entry_perm),
        ranges,
        total,
        num_far_contribs,
    )


def assemble_svgd_topology(
    walk: SvgdTraversal, capacity: str | int = "exact"
) -> SvgdTopology:
    """Turn a dual-tree walk into the near/far partition of the Stein update.

    The counterpart to :func:`build_svgd_traversal`. The leaf blocks and the
    near-pair rows are assembled on the host (they are ``O(L)`` and ``O(P)``,
    both small); the far-entry expansion, which is by far the largest array in
    the partition, is assembled on device.

    Args:
        walk: Device output of :func:`build_svgd_traversal`.
        capacity: Padding policy for the two data-dependent lengths, the near
            pair count and the far entry count *M*. ``"exact"`` (default) pads
            nothing. ``"pow2"`` rounds both up to a power of two, which makes
            the partition's **shapes** stable across rebuilds so the jitted
            update compiles once instead of once per step; ``"bucket"`` does the
            same at an eighth of an octave, which is stable on the observed ~1 %
            rebuild-to-rebuild drift and wastes far less. An integer pins the
            capacity explicitly.

    Returns:
        An :class:`SvgdTopology`.

    Note:
        Shape stability is worth more than it sounds for a per-step-rebuild
        sampler. Six rebuilds of a perturbed N = 1e4 cloud produce six distinct
        ``(near pairs, M)`` signatures -- 41075/428752, 41141/412000,
        41168/420016, 41179/419152, 41401/406768, 41255/407584 -- so every step
        retraces and recompiles the update. Padding costs a little arithmetic on
        masked-out entries and buys back all of that.
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
    num_directed = int(near_target_row.shape[0])

    # Halve the list. The walk emits (A -> B) and (B -> A) for every near leaf
    # pair, and the kernel value is shared between them: the contribution to i
    # from j is k*s_j + k*(x_i - x_j)/h^2 and the contribution to j from i is
    # k*s_i - k*(x_i - x_j)/h^2, one exp for both. Keeping one entry per
    # unordered pair halves the exp count and the tensor reverse mode has to
    # deal with.
    upper = near_target_row < near_source_row
    kept = int(upper.sum())
    if 2 * kept != num_directed:
        raise ValueError(
            "the near leaf-pair list is not symmetric: "
            f"{num_directed} directed entries, {kept} with row_a < row_b. The "
            "Stein partition is only complete when every near pair appears in "
            "both directions."
        )
    # Keep the directed list too: it is what the segment-sum accumulation needs,
    # and it is already exactly what that wants -- np.repeat emits it ascending
    # in target row, so there is nothing to sort. (Concatenating the halved list
    # with its mirror would double it; it is *already* both directions.)
    dir_target = near_target_row
    dir_source = near_source_row
    near_target_row = near_target_row[upper]
    near_source_row = near_source_row[upper]

    num_pairs = int(near_target_row.shape[0])
    pair_cap = _round_up_capacity(num_pairs, capacity)
    near_live = _live_mask(num_pairs, pair_cap)
    # Padding pairs a leaf with itself and weights it zero, so it contributes
    # nothing through either accumulation.
    near_target_row = _pad_int(near_target_row, pair_cap, 0)
    near_source_row = _pad_int(near_source_row, pair_cap, 0)

    # CSR offsets over the directed list. np.repeat emits it ascending in target
    # row, so the offsets are just the running counts -- and because they stop at
    # ``num_directed`` they are also what tells a consumer where the capacity
    # padding starts, without depending on the padding's contents. Their shape is
    # (L + 1) whatever the pair count, so a kernel driven by them does not
    # retrace when the count moves.
    dir_offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int64)

    dir_cap = _round_up_capacity(num_directed, capacity)
    near_dir_live = _live_mask(num_directed, dir_cap)
    # The directed list must stay non-decreasing in target row for the segmented
    # reduction, so it pads with the *last* leaf, not the first.
    dir_target = _pad_int(dir_target, dir_cap, num_leaves - 1)
    dir_source = _pad_int(dir_source, dir_cap, num_leaves - 1)

    # Far field, leaf-major. Every input is already a device array, and at
    # N = 1e5 the host version of this was 271 ms of a 633 ms build, so it goes
    # to the device once there is enough of it to pay the dispatch.
    far_pairs_kept = (
        int(np.asarray(walk.far_tags != FAR_TAG_IGNORE).sum())
        if (walk.far_tags.shape[0] == walk.far_src.shape[0])
        else int(walk.far_src.shape[0])
    )

    if far_pairs_kept >= _DEVICE_FAR_ASSEMBLY_MIN_PAIRS:
        (
            far_leaf_offsets,
            far_leaf_source,
            far_node_offsets,
            far_entry_perm,
            far_ranges,
            num_far_entries,
            num_far_contribs,
        ) = _assemble_far_on_device(walk, leaf_start, num_leaves)
        far_node_start = as_index(far_ranges[:, 0])
        far_node_end = as_index(far_ranges[:, 1])
    else:
        (
            far_leaf_offsets,
            far_leaf_source,
            far_node_offsets,
            far_entry_perm,
            far_node_start,
            far_node_end,
            num_far_entries,
            num_far_contribs,
        ) = _assemble_far_on_host(node_ranges, walk, leaf_start, num_leaves)

    far_cap = _round_up_capacity(num_far_entries, capacity)
    far_leaf_live = _live_mask(num_far_entries, far_cap)
    # Padding entries point at source node 0; the CSR bounds keep the kernel
    # away from them and ``far_leaf_live`` masks them for the pure-JAX path.
    if far_cap != num_far_entries:
        tail = jnp.zeros((far_cap - num_far_entries,), dtype=far_leaf_source.dtype)
        far_leaf_source = jnp.concatenate([jnp.asarray(far_leaf_source), tail])
        far_entry_perm = jnp.concatenate(
            [jnp.asarray(far_entry_perm), jnp.zeros_like(tail)]
        )

    return SvgdTopology(
        order=jnp.asarray(order),
        leaf_slots=jnp.asarray(leaf_slots),
        leaf_mask=jnp.asarray(leaf_mask),
        near_target_row=jnp.asarray(near_target_row),
        near_source_row=jnp.asarray(near_source_row),
        near_dir_target=jnp.asarray(dir_target),
        near_dir_source=jnp.asarray(dir_source),
        near_dir_offsets=as_index(jnp.asarray(dir_offsets)),
        near_live=jnp.asarray(near_live),
        near_dir_live=jnp.asarray(near_dir_live),
        far_leaf_offsets=as_index(jnp.asarray(far_leaf_offsets)),
        far_leaf_source=as_index(jnp.asarray(far_leaf_source)),
        far_leaf_live=jnp.asarray(far_leaf_live),
        far_node_start=as_index(jnp.asarray(far_node_start)),
        far_node_end=as_index(jnp.asarray(far_node_end)),
        far_node_offsets=as_index(jnp.asarray(far_node_offsets)),
        far_entry_perm=as_index(jnp.asarray(far_entry_perm)),
        num_far_pairs=far_pairs_kept,
        num_far_contribs=num_far_contribs,
        num_near_leaf_pairs=num_directed,
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
    capacity: str | int = "exact",
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
        capacity: Shape-padding policy; see :func:`assemble_svgd_topology`.
            ``"pow2"`` is what a per-step-rebuild sampler wants.

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
        ),
        capacity=capacity,
    )


#: Target byte size of one chunk's ``(chunk, ml, ml, d)`` near-field tensor.
#: Reverse mode rematerialises the chunk rather than storing it, so this caps
#: peak device memory of the near field independently of the pair count.
#:
#: 256 MiB, measured, not guessed. At N = 1e4 (41778 unordered pairs, ml = 32,
#: float64) on an A100:
#:
#: ===========  =======  ==========  =====
#: chunk bytes  chunks   fwd (ms)    fwd+grad (ms)
#: ===========  =======  ==========  =====
#:      64 MiB       16       10.12  17.46
#:     256 MiB        4        2.19   9.97
#:     979 MiB        1        2.29   8.62
#: ===========  =======  ==========  =====
#:
#: 64 MiB gives the best forward-to-gradient *ratio* (1.72) purely by making the
#: forward 4.6x slower, which is not a speed-up of anything. 256 MiB gives the
#: best forward and near-best total while still bounding memory; one chunk is
#: marginally faster and bounds nothing (5 GiB by N = 3e4).
_NEAR_CHUNK_BYTES = 256 << 20

#: Target byte size of one chunk's ``(chunk, ml, d)`` far-field tensor, matching
#: :data:`_NEAR_CHUNK_BYTES`'s role for the near field.
_FAR_CHUNK_BYTES = 256 << 20

#: Accepted far pairs above which the whole far assembly runs on the device.
#: Below it the per-call dispatch and the one scalar synchronisation cost more
#: than the numpy work they replace, exactly as the old expansion threshold did.
_DEVICE_FAR_ASSEMBLY_MIN_PAIRS = 1 << 14

#: Far entries above which the leaf-order sort runs on the device. numpy's
#: stable sort of 2.8 million small-range integer keys costs 239 ms (217 ms as
#: int32, 59 ms unstable); ``jnp.argsort`` plus both transfers costs 11 ms. Below
#: a million or so the dispatch and the round trip cost more than they save.
_DEVICE_FAR_SORT_MIN_ENTRIES = 1 << 20


def _near_chunk_bothways(
    phi: Array,
    pos: Array,
    sco: Array,
    leaf_slots: Array,
    leaf_mask: Array,
    rows_a: Array,
    rows_b: Array,
    live: Array,
    h: float | Float[Array, ""],
) -> Array:
    """Scatter both directions of one chunk of unordered near leaf pairs.

    Args:
        phi: Accumulator in sorted-slot order, shape ``(n, d)``.
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        leaf_slots: Padded per-leaf slot blocks, shape ``(L, ml)``.
        leaf_mask: Validity of ``leaf_slots``, shape ``(L, ml)``.
        rows_a: Leaf rows on one side of each pair, shape ``(chunk,)``.
        rows_b: Leaf rows on the other side, shape ``(chunk,)``.
        live: 1.0 for real pairs, 0.0 for the chunk's padding, ``(chunk,)``.
        h: Kernel bandwidth.

    Returns:
        ``phi`` with this chunk's contribution added.
    """
    slots_a = leaf_slots[rows_a]  # (chunk, ml)
    slots_b = leaf_slots[rows_b]
    mask_a = leaf_mask[rows_a] * live[:, None]
    mask_b = leaf_mask[rows_b] * live[:, None]
    x_a, x_b = pos[slots_a], pos[slots_b]  # (chunk, ml, d)
    s_a, s_b = sco[slots_a], sco[slots_b]

    diff = x_a[:, :, None, :] - x_b[:, None, :, :]  # (chunk, ml, ml, d)
    k = jnp.exp(-jnp.sum(diff * diff, axis=-1) / (2.0 * h**2))[..., None]
    repulsive = k * diff / (h**2)

    to_a = jnp.sum(
        (k * s_b[:, None, :, :] + repulsive) * mask_b[:, None, :, None], axis=2
    )
    to_b = jnp.sum(
        (k * s_a[:, :, None, :] - repulsive) * mask_a[:, :, None, None], axis=1
    )
    phi = phi.at[slots_a].add(to_a * mask_a[..., None])
    return phi.at[slots_b].add(to_b * mask_b[..., None])


def _near_chunk_to_target(
    pos: Array,
    sco: Array,
    leaf_slots: Array,
    leaf_mask: Array,
    rows_t: Array,
    rows_s: Array,
    live: Array,
    h: float | Float[Array, ""],
) -> Array:
    """Return one chunk of directed near pairs' contribution to their targets.

    One direction only, so the kernel is evaluated twice per unordered pair --
    the opposite trade to :func:`_near_chunk_bothways`, and the right one when
    the accumulation is a segmented reduction rather than a scatter.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        leaf_slots: Padded per-leaf slot blocks, shape ``(L, ml)``.
        leaf_mask: Validity of ``leaf_slots``, shape ``(L, ml)``.
        rows_t: Target leaf row of each pair, shape ``(chunk,)``.
        rows_s: Source leaf row of each pair, shape ``(chunk,)``.
        live: 1.0 for real pairs, 0.0 for the chunk's padding, ``(chunk,)``.
        h: Kernel bandwidth.

    Returns:
        Per-pair target contributions, shape ``(chunk, ml, d)``.
    """
    slots_t, slots_s = leaf_slots[rows_t], leaf_slots[rows_s]
    mask_s = leaf_mask[rows_s] * live[:, None]
    x_t, x_s = pos[slots_t], pos[slots_s]
    diff = x_t[:, :, None, :] - x_s[:, None, :, :]
    k = jnp.exp(-jnp.sum(diff * diff, axis=-1) / (2.0 * h**2))[..., None]
    terms = k * sco[slots_s][:, None, :, :] + k * diff / (h**2)
    return jnp.sum(terms * mask_s[:, None, :, None], axis=2)


def _accumulate_near_fused(
    pos: Array,
    sco: Array,
    topo: SvgdTopology,
    h: float | Float[Array, ""],
    backend: str,
) -> Array:
    """Accumulate the near field with the fused Pallas kernel.

    Replaces *both* halves of the pure-JAX near field -- the within-leaf block
    and the cross-leaf pairs -- because a leaf paired with itself is just one
    more entry in the kernel's source loop.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        h: Kernel bandwidth.
        backend: ``"pallas"`` or ``"interpret"``; the latter runs Pallas with
            CPU semantics, which is how the tests reach this path with no GPU.

    Returns:
        The whole near field in sorted-slot order, shape ``(n, d)``.
    """
    acc = nearfield_stein(
        pos[topo.leaf_slots],
        sco[topo.leaf_slots],
        topo.leaf_mask > 0,
        topo.near_dir_offsets,
        topo.near_dir_source,
        h,
        include_self=True,
        backend="pallas",
        interpret=backend == "interpret",
    )
    # Leaves tile [0, n) disjointly, so this is a permutation, not a scatter.
    return (
        jnp.zeros_like(pos)
        .at[topo.leaf_slots]
        .add(acc * topo.leaf_mask[..., None].astype(pos.dtype))
    )


def _accumulate_near_by_segment(
    pos: Array,
    sco: Array,
    topo: SvgdTopology,
    h: float | Float[Array, ""],
    chunk_pairs: int,
) -> Array:
    """Accumulate the near field with a segmented reduction, not a scatter.

    Leaves tile ``[0, n)`` disjointly, so summing each target leaf's directed
    pairs into a ``(L, ml, d)`` array and then *placing* it is a permutation --
    no index is written twice and no atomic is needed anywhere.

    This is the float32 path. ``.at[].add()`` costs 258 ms of the 279 ms float32
    near field at N = 1e5 against 6 ms of 29 ms in float64, because the scatter
    indices repeat ~62x on average and float32 lowers to contended
    ``atomicAdd``. Measured, near field only, N = 1e5 on an A100:

    ==========================================  =========  =========
    strategy                                      float64    float32
    ==========================================  =========  =========
    gather + arithmetic only (the floor)          22.60 ms   20.47 ms
    halved pairs, two scatters                    28.60 ms  278.90 ms
    directed pairs, segment_sum, chunked          56.20 ms   32.50 ms
    ==========================================  =========  =========

    Hence the split: float64 keeps the scatter, float32 comes here.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        h: Kernel bandwidth.
        chunk_pairs: Directed pairs per chunk.

    Returns:
        The near-field contribution in sorted-slot order, shape ``(n, d)``.
    """
    rows_t, rows_s = topo.near_dir_target, topo.near_dir_source
    num_pairs = int(rows_t.shape[0])
    num_leaves, max_leaf = topo.leaf_slots.shape
    chunk = min(int(chunk_pairs), num_pairs)
    num_chunks = -(-num_pairs // chunk)
    pad = num_chunks * chunk - num_pairs
    live = topo.near_dir_live.astype(pos.dtype)
    if pad:
        # Pad with the last leaf paired to itself and zero weight: the segment
        # ids stay non-decreasing, which is what makes the reduction segmented.
        tail = jnp.full((pad,), num_leaves - 1, dtype=rows_t.dtype)
        rows_t = jnp.concatenate([rows_t, tail])
        rows_s = jnp.concatenate([rows_s, tail])
        live = jnp.concatenate([live, jnp.zeros((pad,), dtype=pos.dtype)])

    @jax.checkpoint
    def _step(carry, xs):
        contrib = _near_chunk_to_target(
            pos, sco, topo.leaf_slots, topo.leaf_mask, xs[0], xs[1], xs[2], h
        )
        return (
            carry
            + jax.ops.segment_sum(
                contrib, xs[0], num_segments=num_leaves, indices_are_sorted=True
            ),
            None,
        )

    acc, _ = jax.lax.scan(
        _step,
        jnp.zeros((num_leaves, max_leaf, pos.shape[1]), dtype=pos.dtype),
        (
            rows_t.reshape(num_chunks, chunk),
            rows_s.reshape(num_chunks, chunk),
            live.reshape(num_chunks, chunk),
        ),
    )
    return jnp.zeros_like(pos).at[topo.leaf_slots].add(acc * topo.leaf_mask[..., None])


def svgd_phi_from_topology(
    particles: Float[Array, "n d"],
    scores: Float[Array, "n d"],
    h: float | Float[Array, ""],
    topo: SvgdTopology,
    chunk_pairs: int | None = None,
    accumulate: str = "scatter",
) -> Float[Array, "n d"]:
    """Tree-accelerated Stein update given a fixed partition (differentiable).

    Args:
        particles: Particle positions, shape ``(n, d)``.
        scores: Target score at each particle, shape ``(n, d)``.
        h: Kernel bandwidth.
        topo: Partition from :func:`build_svgd_topology`.
        chunk_pairs: Near leaf pairs per rematerialised chunk. ``None``
            (default) picks the largest chunk whose ``(chunk, ml, ml, d)``
            tensor stays under 256 MiB.
        accumulate: How the near field is summed. ``"scatter"`` (default) sums
            each unordered pair once and scatters both directions;
            ``"segment"`` sums directed pairs with a segmented reduction, which
            needs no atomics; ``"pallas"`` uses the fused kernel in
            :mod:`~yggdrax.applications.svgd.pallas_nearfield`, whose reverse is
            a hand-written rule so neither pass scatters; ``"interpret"`` is the
            same kernel under Pallas interpret mode, for testing without a GPU;
            ``"auto"`` picks ``"pallas"`` where
            :func:`~yggdrax.applications.svgd.pallas_nearfield.prefer_fused_nearfield`
            says it is worth launching, and otherwise ``"segment"`` for float32
            and ``"scatter"`` for float64.
            **The default is deliberately the one that is never worse under
            differentiation** -- see the note below.

    Returns:
        Update directions, shape ``(n, d)``.

    Note:
        ``"segment"`` is much faster *forward* in float32 -- 11.64 -> 2.93 ms at
        N = 1e4 and 586 -> 321 ms at 1e5 on an A100 -- because it replaces a
        contended ``atomicAdd`` with a segmented reduction. It is **slower under
        reverse mode** (fwd+grad 2392 -> 3172 ms at N = 1e5), because the
        transpose of a gather is a scatter: reverse mode reintroduces exactly
        the operation the forward pass removed, and doubles the arithmetic
        besides, since the directed list evaluates each kernel twice.

        So a caller that only ever runs forward should ask for ``"auto"`` --
        which is what :func:`svgd_phi` and :func:`run_tree_svgd` do -- and a
        caller that differentiates should leave the default alone.

    Raises:
        ValueError: If ``accumulate`` is not one of the three names.
    """
    if accumulate not in ("scatter", "segment", "pallas", "interpret", "auto"):
        raise ValueError(
            "accumulate must be 'scatter', 'segment', 'pallas', 'interpret' or "
            f"'auto'; got {accumulate!r}"
        )
    if accumulate == "auto" and prefer_fused_nearfield(topo.leaf_slots.shape[0]):
        accumulate = "pallas"
    if accumulate in ("pallas", "interpret"):
        pos_s, sco_s = particles[topo.order], scores[topo.order]
        return _finish(
            _accumulate_near_fused(pos_s, sco_s, topo, h, accumulate),
            pos_s,
            sco_s,
            topo,
            h,
            *particles.shape,
            far_backend=accumulate,
        )
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

    # cross-leaf near pairs. Two accumulations, and which one is faster is a
    # property of the dtype, not of the problem -- see _accumulate_near_by_segment
    # for the table. float64's scatter is cheap and its segmented reduction is
    # not; float32's scatter is 43x more expensive than float64's under index
    # contention, and the reduction wins 8.6x.
    num_pairs = int(topo.near_target_row.shape[0])
    if num_pairs > 0:
        max_leaf = int(topo.leaf_slots.shape[1])
        if chunk_pairs is None:
            per_pair = max(1, max_leaf * max_leaf * d * particles.dtype.itemsize)
            chunk_pairs = max(1, _NEAR_CHUNK_BYTES // per_pair)
        use_segment = accumulate == "segment" or (
            accumulate == "auto" and particles.dtype.itemsize <= 4
        )
        if use_segment:
            return _finish(
                phi + _accumulate_near_by_segment(pos, sco, topo, h, chunk_pairs),
                pos,
                sco,
                topo,
                h,
                n,
                d,
            )
        chunk = min(int(chunk_pairs), num_pairs)
        num_chunks = -(-num_pairs // chunk)
        pad = num_chunks * chunk - num_pairs
        rows_a = topo.near_target_row
        rows_b = topo.near_source_row
        live = topo.near_live.astype(pos.dtype)
        if pad:
            zeros_i = jnp.zeros((pad,), dtype=rows_a.dtype)
            rows_a = jnp.concatenate([rows_a, zeros_i])
            rows_b = jnp.concatenate([rows_b, zeros_i])
            live = jnp.concatenate([live, jnp.zeros((pad,), dtype=pos.dtype)])

        @jax.checkpoint
        def _step(carry, xs):
            return (
                _near_chunk_bothways(
                    carry,
                    pos,
                    sco,
                    topo.leaf_slots,
                    mask,
                    xs[0],
                    xs[1],
                    xs[2],
                    h,
                ),
                None,
            )

        phi, _ = jax.lax.scan(
            _step,
            phi,
            (
                rows_a.reshape(num_chunks, chunk),
                rows_b.reshape(num_chunks, chunk),
                live.reshape(num_chunks, chunk),
            ),
        )

    return _finish(phi, pos, sco, topo, h, n, d)


def _far_monopoles(
    pos: Array, sco: Array, topo: SvgdTopology, n: int, d: int
) -> tuple[Array, Array, Array]:
    """Summarise every distinct far source node by count, centre and score sum.

    One monopole per *node*, not per far entry: the prefix-sum gather is over
    ``F`` nodes -- thousands -- where the old per-particle expansion made it
    ``M``, which is 89,555,008 at N = 1e5.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        n: Particle count.
        d: Dimension.

    Returns:
        ``(count, com, sum_s)`` with shapes ``(F, 1)``, ``(F, d)``, ``(F, d)``.
    """
    zero_x = jnp.zeros((1, d), pos.dtype)
    pos_prefix = jnp.concatenate([zero_x, jnp.cumsum(pos, axis=0)])
    sco_prefix = jnp.concatenate([zero_x, jnp.cumsum(sco, axis=0)])
    cnt_prefix = jnp.arange(n + 1, dtype=pos.dtype)
    start, end = topo.far_node_start, topo.far_node_end
    count = (cnt_prefix[end + 1] - cnt_prefix[start])[:, None]
    sum_x = pos_prefix[end + 1] - pos_prefix[start]
    sum_s = sco_prefix[end + 1] - sco_prefix[start]
    return count, sum_x / count, sum_s


def _accumulate_far_fused(
    pos: Array,
    sco: Array,
    topo: SvgdTopology,
    h: float | Float[Array, ""],
    n: int,
    d: int,
    backend: str,
) -> Array:
    """Accumulate the far field with the fused Pallas kernel.

    The monopoles themselves stay in JAX: they are a prefix-sum gather over
    ``F`` *nodes*, so reverse mode's transpose of them is a scatter over ``F``
    entries -- thousands -- rather than over ``M``. Only the M2P evaluation,
    which is the ``M``-sized part, goes to the kernel.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        h: Kernel bandwidth.
        n: Particle count.
        d: Dimension.
        backend: ``"pallas"`` or ``"interpret"``.

    Returns:
        The far-field contribution in sorted-slot order, shape ``(n, d)``.
    """
    count, com, sum_s = _far_monopoles(pos, sco, topo, n, d)
    acc = farfield_stein(
        pos[topo.leaf_slots],
        topo.leaf_mask > 0,
        count[:, 0],
        com,
        sum_s,
        topo.far_leaf_offsets,
        topo.far_leaf_source,
        topo.far_node_offsets,
        topo.far_entry_perm,
        h,
        backend="pallas",
        interpret=backend == "interpret",
    )
    return (
        jnp.zeros_like(pos)
        .at[topo.leaf_slots]
        .add(acc * topo.leaf_mask[..., None].astype(pos.dtype))
    )


def _accumulate_far_by_leaf(
    pos: Array,
    sco: Array,
    topo: SvgdTopology,
    h: float | Float[Array, ""],
    n: int,
    d: int,
    chunk_entries: int | None = None,
) -> Array:
    """Accumulate the far field into leaves, so placing it is a permutation.

    The old form expanded every accepted pair to its target *particles* and
    scattered ``M`` contributions back. Those indices repeat 899 times per
    target at N = 1e5 -- against the near field's 62 -- so in float32 the
    scatter alone was 271.71 ms of the 269.49 ms far field, and the arithmetic
    was 25.57 ms. Here each pair is expanded to the *leaves* under its target
    node instead, summed per leaf with a segmented reduction, and placed; leaves
    tile ``[0, n)`` disjointly, so the placement writes each slot once.

    Args:
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        h: Kernel bandwidth.
        n: Particle count.
        d: Dimension.
        chunk_entries: Far entries per rematerialised chunk. ``None`` picks the
            largest whose ``(chunk, ml, d)`` tensor stays under 256 MiB.

    Returns:
        The far-field contribution in sorted-slot order, shape ``(n, d)``.
    """
    count, com, sum_s = _far_monopoles(pos, sco, topo, n, d)
    num_leaves, max_leaf = topo.leaf_slots.shape
    num_entries = int(topo.far_leaf_source.shape[0])
    if chunk_entries is None:
        per_entry = max(1, max_leaf * d * pos.dtype.itemsize)
        chunk_entries = max(1, _FAR_CHUNK_BYTES // per_entry)
    chunk = min(int(chunk_entries), num_entries)
    num_chunks = -(-num_entries // chunk)
    pad = num_chunks * chunk - num_entries

    rows = _far_target_rows(topo.far_leaf_offsets, num_entries)
    src = topo.far_leaf_source
    live = topo.far_leaf_live.astype(pos.dtype)
    if pad:
        rows = jnp.concatenate([rows, jnp.full((pad,), num_leaves - 1, rows.dtype)])
        src = jnp.concatenate([src, jnp.zeros((pad,), src.dtype)])
        live = jnp.concatenate([live, jnp.zeros((pad,), pos.dtype)])

    inv_h2 = 1.0 / (h * h)

    @jax.checkpoint
    def _step(carry, xs):
        row, source, alive = xs
        x_t = pos[topo.leaf_slots[row]]  # (chunk, ml, d)
        sep = x_t - com[source][:, None, :]
        kern = jnp.exp(-jnp.sum(sep * sep, axis=-1, keepdims=True) * (0.5 * inv_h2))
        contrib = kern * (
            sum_s[source][:, None, :] + count[source][:, None, :] * sep * inv_h2
        )
        contrib = contrib * alive[:, None, None]
        return (
            carry
            + jax.ops.segment_sum(
                contrib, row, num_segments=num_leaves, indices_are_sorted=True
            ),
            None,
        )

    acc, _ = jax.lax.scan(
        _step,
        jnp.zeros((num_leaves, max_leaf, d), dtype=pos.dtype),
        (
            rows.reshape(num_chunks, chunk),
            src.reshape(num_chunks, chunk),
            live.reshape(num_chunks, chunk),
        ),
    )
    return (
        jnp.zeros_like(pos)
        .at[topo.leaf_slots]
        .add(acc * topo.leaf_mask[..., None].astype(pos.dtype))
    )


def _far_target_rows(offsets: Array, num_entries: int) -> Array:
    """Return the target leaf row of each far CSR entry.

    Args:
        offsets: CSR offsets, shape ``(L + 1,)``, non-decreasing.
        num_entries: Length of the entry list, including capacity padding.

    Returns:
        Target leaf row per entry, shape ``(P,)``, non-decreasing. Padding past
        ``offsets[-1]`` maps to the last leaf, which keeps the sequence sorted
        for the segmented reduction; it is masked by ``far_leaf_live``.
    """
    positions = jnp.arange(num_entries, dtype=offsets.dtype)
    rows = jnp.searchsorted(offsets, positions, side="right") - 1
    return jnp.clip(rows, 0, offsets.shape[0] - 2)


def _finish(
    phi: Array,
    pos: Array,
    sco: Array,
    topo: SvgdTopology,
    h: float | Float[Array, ""],
    n: int,
    d: int,
    far_backend: str = "jax",
) -> Array:
    """Add the far field, average, and return to the caller's particle order.

    Shared by both near-field accumulations so they cannot drift apart.

    Args:
        phi: Near-field accumulator in sorted-slot order, shape ``(n, d)``.
        pos: Positions in sorted order, shape ``(n, d)``.
        sco: Scores in sorted order, shape ``(n, d)``.
        topo: The partition.
        h: Kernel bandwidth.
        n: Particle count.
        d: Dimension.
        far_backend: ``"jax"`` for the chunked segmented reduction,
            ``"pallas"`` / ``"interpret"`` for the fused kernel.

    Returns:
        Update directions in the caller's particle order, shape ``(n, d)``.
    """
    # --- far field: monopole (M2P), leaf-major ---
    # The entry list is the guard. The monopole table is keyed by node id and so
    # is never empty, and a padding entry under ``capacity="pow2"`` names node 0,
    # whose monopole exists and is then masked by ``far_leaf_live`` -- so unlike
    # the compacted table this once used, there is nothing here to fall off.
    if int(topo.far_leaf_source.shape[0]) > 0:
        if far_backend == "jax":
            phi = phi + _accumulate_far_by_leaf(pos, sco, topo, h, n, d)
        else:
            phi = phi + _accumulate_far_fused(pos, sco, topo, h, n, d, far_backend)

    phi = phi / n
    # phi is in sorted order; scatter back to original particle order.
    return jnp.zeros_like(phi).at[topo.order].set(phi)


# Fused, compiled accumulation. Given a (fixed) partition the Stein update is a
# pure array computation; jitting it collapses the eager per-op dispatch into one
# kernel (~1.5x faster per step even when the partition shapes vary a little).
_jit_svgd_phi_from_topology = jax.jit(
    svgd_phi_from_topology, static_argnames=("chunk_pairs", "accumulate")
)


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
    # svgd_phi is the forward-only entry point, so it takes the accumulation
    # that is fastest forward; svgd_phi_from_topology keeps the differentiable
    # default for callers that wrap it in grad.
    return _jit_svgd_phi_from_topology(particles, scores, h, topo, accumulate="auto")


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
    capacity: str | int = "pow2",
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
        capacity: Shape-padding policy, defaulting to ``"pow2"`` because this
            function rebuilds the partition every call; see
            :func:`assemble_svgd_topology`.

    Returns:
        Updated particles, shape ``(n, d)``.
    """
    scores = score_fn(particles)
    topo = build_svgd_topology(
        particles,
        theta=theta,
        leaf_size=leaf_size,
        backend=backend,
        traversal_config=traversal_config,
        kernel_cutoff=_cutoff_radius(cutoff_bandwidths, h),
        capacity=capacity,
    )
    phi = _jit_svgd_phi_from_topology(particles, scores, h, topo, accumulate="auto")
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
    capacity: str | int = "pow2",
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
        capacity: Shape-padding policy, defaulting to ``"pow2"``. This is the
            single most valuable setting in this module for a rebuilding
            sampler: 8 steps at N = 1e4 take **35.5 s** with ``"exact"`` and
            **2.2 s** with ``"pow2"``, because the partition's shapes otherwise
            change every step and both the traversal and the update recompile.

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
            capacity=capacity,
        )
    return p
