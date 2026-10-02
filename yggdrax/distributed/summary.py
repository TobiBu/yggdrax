"""The per-device tree SUMMARY the cross-domain exchange is addressed by.

A device that must send part of its tree to another does not have the receiver's
tree. The exchange therefore works against a small summary the receiver publishes: a
set of cells, each bounding a region of the receiver, against which the SENDER walks
its own tree and decides unilaterally what to send. A cell bounds every real target
inside it, so a MAC decision taken against the cell is conservative -- a node far from
the cell is far from each of its targets -- and the export can over-send but never
miss.

**The summary is a CUT of the receiver's own tree, chosen by occupancy**, not a set of
fixed-level Morton cells. That choice is measured, not aesthetic. The receiver's
evaluation cost is ``sum over cells of |list(cell)| x leaves_in_cell``, which a few
cells holding many leaves AND naming many nodes dominate. A fixed Morton level leaves
that product wildly unbalanced, and the imbalance grows with N: against the cost of a
summary made of individual leaves, a fixed level cost 7x the evaluation work at
N/device = 10^4 and **61-65x at 10^5**, while an occupancy cut stays at 1.1-1.3x
(jaccpot ``docs/multigpu_fused_2026-09.md``, Phase 3.5).

Cells being real tree nodes has a second payoff: they are subtrees by construction, so
each has a local root, which is what lets the receiver seed its evaluation per cell
rather than against one flat pool. A flat pool is not a cut -- it holds nodes together
with their own descendants, because different cells need different granularities --
and evaluating against it double-counts on every target leaf while leaving momentum
exact.

Nothing here allocates per level or loops over the tree: the cut is a pure predicate
on subtree leaf counts, which come from two ``searchsorted`` calls.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..dtypes import INDEX_DTYPE, as_index

__all__ = [
    "SummaryTree",
    "TreeSummary",
    "occupancy_cut",
    "subtree_leaf_counts",
    "summary_tree",
]


class TreeSummary(NamedTuple):
    """A capacity-padded cut of one device's tree.

    Attributes
    ----------
    cells:
        ``(capacity,)`` node indices of the cut, ``-1``-padded. Ascending.
    num_cells:
        Number of live entries in ``cells``.
    leaves_per_cell:
        ``(capacity,)`` live leaves under each cell, ``0`` in the padding. Sums to
        the tree's live leaf count -- the cut property, in one number.
    overflow:
        True when the cut did not fit ``capacity``. **Must be read**: a truncated
        summary silently drops part of the receiver from the exchange, which loses
        force rather than accuracy.
    """

    cells: Array
    num_cells: Array
    leaves_per_cell: Array
    overflow: Array


def subtree_leaf_counts(
    node_ranges: Array, num_internal: int, num_valid: Optional[Array] = None
) -> Array:
    """Live leaves under every node, from particle ranges alone.

    Leaves hold contiguous, ascending particle ranges, so the number of leaves under
    a node is the number of leaf STARTS inside that node's range -- two
    ``searchsorted`` calls over the leaf starts, with no per-level accumulation and
    no loop over the tree.

    Parameters
    ----------
    node_ranges:
        ``(total_nodes, 2)`` inclusive particle ranges. An empty node carries
        ``start > end``.
    num_internal:
        Nodes at or above this index are leaves.
    num_valid:
        Live particle count of a capacity-padded shard. Padding leaves sit at
        ``start >= num_valid`` and are excluded from the counts.

        This does NOT make a padded shard's cut identical to the unpadded one's: the
        two trees have different balanced structures over different numbers of leaf
        slots, so the cut lands on different nodes covering different particle
        blocks. What it guarantees is that the counts are over LIVE leaves only.

        **On the ranges convention ``_build_balanced_bucket_structure`` produces it
        is a NO-OP**, measured: its padding leaves carry ``start > end``, hold no leaf
        starts, and the ``sub > 0`` guard excludes them already. It is kept for
        interface consistency with the rest of yggdrax's ``num_valid`` threading and
        against the other convention this codebase has -- a padding leaf carrying
        ``start == end == n``, which reads as one particle rather than as empty --
        but no caller on the present path needs it.

    Returns
    -------
    Array
        ``(total_nodes,)`` live leaf counts; 0 for an empty node.
    """
    nr = jnp.asarray(node_ranges)
    starts = nr[num_internal:, 0]
    if num_valid is not None:
        # a padding leaf starts at or past the live count; push it past every
        # search key instead of dropping it, so the shape stays static
        big = jnp.asarray(jnp.iinfo(jnp.asarray(starts).dtype).max, starts.dtype)
        starts = jnp.where(starts < as_index(num_valid), starts, big)
    starts = jnp.sort(starts)
    lo = jnp.searchsorted(starts, nr[:, 0], side="left")
    hi = jnp.searchsorted(starts, nr[:, 1] + 1, side="left")
    return jnp.maximum(hi - lo, 0).astype(INDEX_DTYPE)


def occupancy_cut(
    parent: Array,
    node_ranges: Array,
    num_internal: int,
    *,
    max_leaves: int,
    capacity: int,
    num_valid: Optional[Array] = None,
    node_extent: Optional[Array] = None,
    max_extent: Optional[Array | float] = None,
) -> TreeSummary:
    """The shallowest cut whose cells each hold at most ``max_leaves`` live leaves.

    A node is in the cut exactly when it holds at most ``max_leaves`` live leaves and
    its parent holds more -- a pure predicate, evaluated on every node at once. The
    root is in the cut when the whole tree already fits.

    That predicate is what makes the result a cut: descending from the root, the first
    node that fits is taken and nothing below it is, so **every live leaf has exactly
    one ancestor-or-self in the cut**. Empty nodes hold no leaves and are excluded, so
    they cannot shadow a live sibling.

    Parameters
    ----------
    parent:
        ``(total_nodes,)`` parent indices; the root's is negative.
    node_ranges:
        ``(total_nodes, 2)`` inclusive particle ranges.
    num_internal:
        Nodes at or above this index are leaves.
    max_leaves:
        Cell occupancy bound, in leaves. ``1`` returns the live leaves themselves.
        Smaller cells cost more summary and less evaluation work; the measured
        operating band is 4 to 16.
    capacity:
        Static length of the returned arrays.
    num_valid:
        Live particle count of a capacity-padded shard; see
        :func:`subtree_leaf_counts`.
    node_extent, max_extent:
        Optional SIZE bound: an internal node is a cell only when it also has
        ``node_extent <= max_extent`` (leaves always qualify, so every live leaf
        stays covered). Use an axis-aligned extent: a node's box contains its
        children's, so the bound is monotone down the tree and the result is still
        a cut. Without it, a sparse node holding a few tiny far-apart leaves is one
        huge cell, and whatever a sender decides against that cell's bounding
        sphere -- near, typically, for an entire remote domain -- applies to all of
        it. ``None`` (default): the occupancy bound alone.

    Returns
    -------
    TreeSummary
        The cut, padded to ``capacity``. Read ``overflow``.

    Raises
    ------
    ValueError
        If ``max_leaves`` or ``capacity`` is not positive.
    """
    if int(max_leaves) < 1:
        raise ValueError(f"max_leaves must be positive, got {max_leaves}")
    if int(capacity) < 1:
        raise ValueError(f"capacity must be positive, got {capacity}")

    par = jnp.asarray(parent)
    sub = subtree_leaf_counts(node_ranges, num_internal, num_valid)
    m = as_index(int(max_leaves))
    is_root = par < 0
    # the root is in the cut only when the whole tree fits; otherwise a node is in
    # it when it fits and its parent does not
    fits = (sub > 0) & (sub <= m)
    if node_extent is not None and max_extent is not None:
        ext = jnp.asarray(node_extent)
        is_leaf = jnp.arange(par.shape[0], dtype=INDEX_DTYPE) >= as_index(num_internal)
        fits = fits & (is_leaf | (ext <= jnp.asarray(max_extent, ext.dtype)))
    parent_fits = fits[jnp.where(is_root, 0, par)]
    in_cut = fits & jnp.where(is_root, True, ~parent_fits)

    idx = jnp.arange(par.shape[0], dtype=INDEX_DTYPE)
    order = jnp.argsort(jnp.where(in_cut, idx, as_index(par.shape[0])), stable=True)
    n_cut = jnp.sum(in_cut.astype(INDEX_DTYPE), dtype=INDEX_DTYPE)
    take = order[:capacity]
    live = jnp.arange(capacity, dtype=INDEX_DTYPE) < n_cut
    return TreeSummary(
        cells=jnp.where(live, take, as_index(-1)),
        num_cells=jnp.minimum(n_cut, as_index(capacity)),
        leaves_per_cell=jnp.where(live, sub[take], as_index(0)),
        overflow=n_cut > as_index(capacity),
    )


class SummaryTree(NamedTuple):
    """The TOP of one device's tree: its summary cut plus every ancestor of the cut.

    Published instead of the bare cells so that a sender's export walk can refine the
    RECEIVER's side too: a large sender node far from a whole region of the receiver
    then pairs with one receiver ancestor instead of with every cell under it -- the
    FMM-style pairing the local mutual walk has, instead of a treecode-style list per
    cell.

    Index space: ``[0, num_nodes)``, compacted in ascending tree-node order except that
    **the root is always index 0**, so a sender seeds ``(0, its root)`` without the
    root having to travel. Children are summary indices.

    Attributes
    ----------
    nodes:
        ``(capacity,)`` the tree node behind each summary index, ``-1``-padded.
    left, right:
        ``(capacity,)`` summary indices of the children; ``-1`` at summary leaves
        and in the padding. An internal summary node always has BOTH children: a
        child holding no live leaf is kept as an inactive summary leaf, because the
        walks read ``left < 0`` as "leaf" and would otherwise stop at a node with
        one empty child.
    num_nodes:
        Live entries.
    is_cell:
        ``(capacity,)`` True on the cut cells -- the summary leaves that hold live
        leaves. Near pairs (particles) are only ever emitted against these.
    active:
        ``(capacity,)`` True where the node holds live leaves (cells and their
        ancestors); False on empty children and in the padding.
    overflow:
        True when the summary did not fit ``capacity`` or the cut it was built from
        had overflowed. **Must be read**: a truncated summary drops part of the
        receiver from the exchange.
    """

    nodes: Array
    left: Array
    right: Array
    num_nodes: Array
    is_cell: Array
    active: Array
    overflow: Array


def summary_tree(
    parent: Array,
    left_child: Array,
    right_child: Array,
    node_ranges: Array,
    num_internal: int,
    cut: TreeSummary,
    *,
    capacity: int,
    num_valid: Optional[Array] = None,
) -> SummaryTree:
    """The summary cut together with its ancestors, as a tree with child links.

    The top tree is every node in the cut or above it. Every live node is exactly
    one of in the cut, strictly below it, or strictly above it (the cut property),
    so "above" is "live, not in the cut, and no strict ancestor in the cut". The
    last is one pointer-doubling pass over ``parent`` -- ``ceil(log2(nodes))`` rounds
    of two gathers, no loop over levels.

    The children of an ancestor are ancestors, cells, or EMPTY (holding no live
    leaf): a live child strictly below the cut would need a cut node at or above
    its parent, and the parent is an ancestor. So the top tree is closed under
    children once the empty ones are kept, and it is a full binary tree whose leaves
    are the cut cells plus a few empty nodes -- about ``2 x cells`` nodes.

    Parameters
    ----------
    parent:
        ``(total_nodes,)`` parent indices; the root's is negative.
    left_child, right_child:
        ``(num_internal,)`` child node indices.
    node_ranges:
        ``(total_nodes, 2)`` inclusive particle ranges.
    num_internal:
        Nodes at or above this index are leaves.
    cut:
        The cut, from :func:`occupancy_cut` on the same tree (with or without its
        size bound).
    capacity:
        Static length of the returned arrays. ``2 x`` the cut capacity covers a
        full cut with room for the empty children.
    num_valid:
        Live particle count of a capacity-padded shard; see
        :func:`subtree_leaf_counts`.

    Returns
    -------
    SummaryTree
        The top tree, padded to ``capacity``. Read ``overflow``.

    Raises
    ------
    ValueError
        If ``capacity`` is not positive.
    """
    if int(capacity) < 1:
        raise ValueError(f"capacity must be positive, got {capacity}")
    par = jnp.asarray(parent)
    total = int(par.shape[0])
    node = jnp.arange(total, dtype=INDEX_DTYPE)
    neg = as_index(-1)

    sub = subtree_leaf_counts(node_ranges, num_internal, num_valid)
    live = sub > 0
    cells = as_index(cut.cells)
    in_cut = (
        jnp.zeros((total,), bool)
        .at[jnp.where(cells >= 0, cells, as_index(total))]
        .set(True, mode="drop")
    )

    # strictly below the cut: some STRICT ancestor is a cut node. Pointer doubling:
    # after round j, `below` covers the ancestors at distance 1 .. 2^j and `anc` is
    # the ancestor at distance 2^j (saturating at the root, whose flag stays False).
    is_root = par < 0
    # cast: three-argument `jnp.where` is always an Array; the stubs say
    # `Array | tuple` because of the one-argument (nonzero) form
    up = cast(Array, jnp.where(is_root, node, as_index(par)))
    below = cast(Array, jnp.where(is_root, False, in_cut[up]))
    anc = up
    for _ in range(max(1, (total - 1).bit_length())):
        below = below | below[anc]
        anc = anc[anc]

    internal = node < as_index(num_internal)
    ancestor = live & ~in_cut & ~below & internal
    full_left = jnp.concatenate(
        [as_index(left_child), jnp.full((total - num_internal,), neg)]
    )
    full_right = jnp.concatenate(
        [as_index(right_child), jnp.full((total - num_internal,), neg)]
    )
    # an ancestor's children, live or not: the walks need both of them
    kid_slot = jnp.concatenate(
        [
            cast(Array, jnp.where(ancestor, full_left, as_index(total))),
            cast(Array, jnp.where(ancestor, full_right, as_index(total))),
        ]
    )
    is_kid = jnp.zeros((total,), bool).at[kid_slot].set(True, mode="drop")
    member = ancestor | in_cut | is_kid

    # compact in node order with the root moved to the front
    root = jnp.argmin(par).astype(INDEX_DTYPE)
    key = cast(
        Array, jnp.where(member, jnp.where(node == root, as_index(-1), node), total)
    )
    order = jnp.argsort(key, stable=True)
    n_top = jnp.sum(member.astype(INDEX_DTYPE), dtype=INDEX_DTYPE)
    if capacity > total:
        order = jnp.concatenate([order, jnp.zeros((capacity - total,), order.dtype)])
    take = as_index(order[:capacity])
    slot = jnp.arange(capacity, dtype=INDEX_DTYPE)
    ok = slot < jnp.minimum(n_top, as_index(capacity))
    nodes = cast(Array, jnp.where(ok, take, neg))

    # node -> summary index (-1 for non-members and for members past capacity)
    sidx = (
        jnp.full((total,), neg)
        .at[jnp.where(ok, take, as_index(total))]
        .set(slot, mode="drop")
    )
    safe = jnp.where(ok, take, 0)
    expand = ok & ancestor[safe]
    left = cast(Array, jnp.where(expand, sidx[full_left[safe]], neg))
    right = cast(Array, jnp.where(expand, sidx[full_right[safe]], neg))
    # an ancestor whose child fell past capacity has lost part of the tree
    lost_kid = jnp.any(expand & ((left < 0) | (right < 0)))
    return SummaryTree(
        nodes=nodes,
        left=left,
        right=right,
        num_nodes=jnp.minimum(n_top, as_index(capacity)),
        is_cell=ok & in_cut[safe],
        active=ok & live[safe],
        overflow=jnp.asarray(cut.overflow) | (n_top > as_index(capacity)) | lost_kid,
    )
