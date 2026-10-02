"""The sender-side export walk: what each device must send, decided unilaterally.

The receiver does not have the sender's tree, so it publishes a small summary instead
(:func:`~yggdrax.distributed.summary.occupancy_cut`) and the SENDER walks its own full
tree against every receiver's cells. A cell bounds every real target inside it, so a
MAC decision taken against the cell is conservative -- a node far from the cell is far
from each of its targets -- and the export over-sends but never misses. One round, no
request round.

**All receivers are walked at once.** The cells of every device are laid out in one
block ahead of the local tree, so the combined index space is
``[ndev x max_cells cells ; local nodes]`` and a single
:func:`~yggdrax.interactions.dual_tree_walk_mutual` seeded with
``(cell_i, local_root)`` does the whole job. Three things fall out for free:

* the walk's ``(min, max)`` canonicalisation orders every emitted pair as
  ``(cell, local node)``, because every cell index is strictly below every local one;
* the cells carry no children, so ``split_b`` never fires on them and the walk refines
  only the sender's side, terminating there;
* a cell's owning device is ``cell_index // max_cells``, with no side table.

**The sender's own cells are masked out**, not skipped by index arithmetic. A device
that exported to itself would have its own particles counted twice -- once locally and
once as an import -- and that is the kind of error a momentum check cannot see.

What comes back is per-cell interaction LISTS, far and near kept apart because they are
paid for differently downstream: a far pair ships a multipole, a near pair ships
particles. They are lists and not a node set on purpose -- a flat pool of nodes is not
a cut, since different cells need different granularities, and evaluating against one
double-counts on every target leaf.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..dtypes import INDEX_DTYPE, as_index
from ..interactions import dual_tree_walk_mutual

__all__ = ["ExportLists", "SendBuffers", "build_send_buffers", "export_walk"]


class ExportLists(NamedTuple):
    """Per-cell interaction lists this device owes every other device.

    Attributes
    ----------
    far_cell, far_node:
        ``(far_cap,)`` pairs, ``-1``-padded: the local NODE whose multipole cell
        ``far_cell`` needs. The owning device of a cell is ``cell // max_cells``.
    far_count:
        Live entries in the far arrays.
    near_cell, near_node:
        ``(near_cap,)`` pairs, ``-1``-padded: the local LEAF whose particles cell
        ``near_cell`` needs.
    near_count:
        Live entries in the near arrays.
    far_overflow, near_overflow, queue_overflow:
        **Must be read.** A truncated export silently drops part of somebody's
        force, which shows up as a percent-level error and never as a conservation
        violation.
    """

    far_cell: Array
    far_node: Array
    far_count: Array
    near_cell: Array
    near_node: Array
    near_count: Array
    far_overflow: Array
    near_overflow: Array
    queue_overflow: Array


def export_walk(
    left_child_full: Array,
    right_child_full: Array,
    centers: Array,
    extents: Array,
    root: Array,
    cell_centers: Array,
    cell_radii: Array,
    cell_active: Array,
    theta: float,
    my_device: Array,
    *,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
    mac_type: str = "dehnen",
    node_active: Optional[Array] = None,
) -> ExportLists:
    """Walk this device's tree against every other device's summary cells.

    Parameters
    ----------
    left_child_full, right_child_full:
        ``(total_nodes,)`` child arrays of the LOCAL tree, ``-1`` at leaves.
    centers, extents:
        ``(total_nodes, 3)`` and ``(total_nodes,)`` local node centres and MAC
        extents. Caller-supplied, as :func:`dual_tree_walk_mutual` requires, so this
        stays agnostic about which extent convention is in force.
    root:
        Index of the local root, in the LOCAL index space.
    cell_centers, cell_radii:
        ``(ndev, max_cells, 3)`` and ``(ndev, max_cells)``, every device's published
        summary. The radii must BOUND their cells -- that is what makes the export
        conservative rather than approximate.
    cell_active:
        ``(ndev, max_cells)`` bool; False in the padding of each device's summary.
    theta:
        MAC parameter, the same one the receiver will evaluate with.
    my_device:
        This device's index along the mesh axis. Its own row of ``cell_active`` is
        forced False here, so a device can never export to itself.
    max_pair_queue, far_cap, near_cap:
        Static capacities. Read the overflow flags.
    mac_type:
        MAC variant, passed through. Static.
    node_active:
        Optional ``(total_nodes,)`` mask over the LOCAL tree, for a
        capacity-padded shard.

    Returns
    -------
    ExportLists
        The per-cell far and near lists, ``-1``-padded.

    Raises
    ------
    ValueError
        On inconsistent cell array shapes.
    """
    cc = jnp.asarray(cell_centers)
    cr = jnp.asarray(cell_radii)
    ca = jnp.asarray(cell_active, dtype=bool)
    if cc.ndim != 3 or cc.shape[2] != 3:
        raise ValueError(f"cell_centers must be (ndev, max_cells, 3), got {cc.shape}")
    if cr.shape != cc.shape[:2] or ca.shape != cc.shape[:2]:
        raise ValueError(
            "cell_radii and cell_active must be (ndev, max_cells) aligned with "
            f"cell_centers, got {cr.shape} and {ca.shape} against {cc.shape[:2]}"
        )
    ndev, max_cells = int(cc.shape[0]), int(cc.shape[1])
    n_cells = ndev * max_cells

    idx = jnp.asarray(left_child_full).dtype
    n_local = int(jnp.asarray(left_child_full).shape[0])

    # never export to myself: my particles are already in my own local field, and
    # counting them twice leaves momentum exact while inflating the force
    mine = jnp.arange(ndev, dtype=INDEX_DTYPE)[:, None] == as_index(my_device)
    ca = ca & ~mine

    cell_fill = jnp.full((n_cells,), -1, dtype=idx)
    left = jnp.concatenate([cell_fill, jnp.asarray(left_child_full, idx)])
    right = jnp.concatenate([cell_fill, jnp.asarray(right_child_full, idx)])
    shift = as_index(n_cells).astype(idx)
    left = jnp.where(left >= 0, left + shift, left).at[:n_cells].set(-1)
    right = jnp.where(right >= 0, right + shift, right).at[:n_cells].set(-1)

    dtype = jnp.asarray(centers).dtype
    all_centers = jnp.concatenate(
        [cc.reshape(n_cells, 3).astype(dtype), jnp.asarray(centers, dtype)]
    )
    all_extents = jnp.concatenate(
        [cr.reshape(n_cells).astype(dtype), jnp.asarray(extents, dtype)]
    )
    local_active = (
        jnp.ones((n_local,), bool)
        if node_active is None
        else jnp.asarray(node_active, bool)
    )
    active = jnp.concatenate([ca.reshape(n_cells), local_active])

    root_shifted = (jnp.asarray(root, idx) + shift).astype(idx)
    res = dual_tree_walk_mutual(
        left,
        right,
        all_centers,
        all_extents,
        theta,
        root_shifted,
        max_pair_queue=max_pair_queue,
        far_cap=far_cap,
        near_cap=near_cap,
        mac_type=mac_type,  # pyright: ignore[reportArgumentType]
        node_active=active,
        seed_a=jnp.arange(n_cells, dtype=idx),
        seed_b=jnp.full((n_cells,), root_shifted, dtype=idx),
    )

    def split(a, b, n):
        """(cell, local node) with the walk's padding preserved as -1."""
        live = jnp.arange(a.shape[0], dtype=idx) < n
        # cast: three-argument `jnp.where` is always an Array; the stubs say
        # `Array | tuple` because of the one-argument (nonzero) form
        return (
            cast(Array, jnp.where(live, a, as_index(-1).astype(idx))),
            cast(Array, jnp.where(live, b - shift, as_index(-1).astype(idx))),
        )

    far_cell, far_node = split(res.far_a, res.far_b, res.far_count)
    near_cell, near_node = split(res.near_a, res.near_b, res.near_count)
    return ExportLists(
        far_cell=far_cell,
        far_node=far_node,
        far_count=res.far_count,
        near_cell=near_cell,
        near_node=near_node,
        near_count=res.near_count,
        far_overflow=res.far_overflow,
        near_overflow=res.near_overflow,
        queue_overflow=res.queue_overflow,
    )


class SendBuffers(NamedTuple):
    """One export list, laid out for :func:`ragged_all_to_all_exchange`.

    Rows destined for device ``i`` form a contiguous block starting at
    ``exclusive_cumsum(sizes)[i]``, which is the layout that primitive requires.

    Attributes
    ----------
    node_rows:
        ``(node_capacity,)`` LOCAL node indices to ship, deduplicated per
        destination and grouped by it. ``-1`` in the padding.
    node_sizes:
        ``(ndev,)`` rows destined for each device.
    csr_cell:
        ``(csr_capacity,)`` the RECEIVER's own cell index, already rebased out of
        the global cell block, so the receiver reads it without knowing who sent it.
    csr_row:
        ``(csr_capacity,)`` which row of the block that receiver will receive holds
        this entry's node. Rebased the same way: an index into what the receiver
        gets, not into the sender's buffer.
    csr_sizes:
        ``(ndev,)`` CSR entries destined for each device.
    node_overflow, csr_overflow:
        **Must be read.** A truncated send drops somebody's force silently.
    """

    node_rows: Array
    node_sizes: Array
    csr_cell: Array
    csr_row: Array
    csr_sizes: Array
    node_overflow: Array
    csr_overflow: Array


def build_send_buffers(
    cell: Array,
    node: Array,
    count: Array,
    *,
    ndev: int,
    max_cells: int,
    num_nodes: int,
    node_capacity: int,
    csr_capacity: int,
) -> SendBuffers:
    """Group one export list by destination and deduplicate its payload.

    A node named by many of a receiver's cells is SHIPPED ONCE and referenced many
    times: that is the whole reason the payload and the CSR are separate objects. The
    measured duplication is 3-25 references per node, and a CSR entry is 4 bytes
    against 120-164 for a node payload, so this is what keeps the exchange one round
    instead of a per-pair broadcast.

    Everything is a sort and two scans -- no per-device loop, so the cost does not
    grow with the mesh.

    Parameters
    ----------
    cell, node, count:
        One list out of :class:`ExportLists` -- far or near, never both, because
        their payloads are different objects (a multipole against particles).
    ndev, max_cells:
        Mesh size and the common summary capacity; ``cell // max_cells`` is the
        destination and ``cell % max_cells`` the receiver's own cell index.
    num_nodes:
        Local node count, used only to key the sort. Static.
    node_capacity, csr_capacity:
        Static send-buffer lengths. Read the overflow flags.

    Returns
    -------
    SendBuffers
        Grouped, deduplicated, and rebased into the receiver's frame.
    """
    idx = jnp.asarray(node).dtype
    P = int(jnp.asarray(cell).shape[0])
    ndev_i, mc = as_index(ndev), as_index(max_cells)

    live = (jnp.arange(P, dtype=INDEX_DTYPE) < as_index(count)) & (
        jnp.asarray(cell) >= 0
    )
    dev = jnp.where(live, as_index(cell) // mc, ndev_i)
    # dead rows sort last, so the live prefix is contiguous and grouped by device
    key = dev * as_index(num_nodes + 1) + jnp.where(
        live, as_index(node), as_index(num_nodes)
    )
    order = jnp.argsort(key, stable=True)
    s_dev, s_node, s_cell = dev[order], jnp.asarray(node)[order], as_index(cell)[order]
    s_live = s_dev < ndev_i

    first = (
        jnp.concatenate(
            [
                jnp.ones((1,), bool),
                (s_dev[1:] != s_dev[:-1]) | (s_node[1:] != s_node[:-1]),
            ]
        )
        & s_live
    )
    # position of each kept node among all kept nodes, in destination order
    g_row = jnp.cumsum(first.astype(INDEX_DTYPE), dtype=INDEX_DTYPE) - 1

    node_sizes = jax.ops.segment_sum(
        first.astype(INDEX_DTYPE),
        jnp.where(s_live, s_dev, ndev_i),
        num_segments=ndev + 1,
    )[:ndev]
    node_offsets = jnp.concatenate(
        [jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(node_sizes, dtype=INDEX_DTYPE)[:-1]]
    )

    n_nodes_out = jnp.sum(node_sizes, dtype=INDEX_DTYPE)
    keep = first & (g_row < as_index(node_capacity))
    node_rows = (
        jnp.full((node_capacity,), -1, dtype=idx)
        .at[jnp.where(keep, g_row, as_index(node_capacity))]
        .set(s_node, mode="drop")
    )

    n_csr = jnp.sum(s_live.astype(INDEX_DTYPE), dtype=INDEX_DTYPE)
    pos = jnp.arange(P, dtype=INDEX_DTYPE)
    csr_keep = s_live & (pos < as_index(csr_capacity))
    safe = jnp.where(csr_keep, pos, as_index(csr_capacity))
    # rebase BOTH columns into the receiver's frame: it must not need to know the
    # sender's cell block or the sender's own buffer offsets to read this
    csr_cell = (
        jnp.full((csr_capacity,), -1, dtype=idx)
        .at[safe]
        .set((s_cell - s_dev * mc).astype(idx), mode="drop")
    )
    csr_row = (
        jnp.full((csr_capacity,), -1, dtype=idx)
        .at[safe]
        .set(
            (g_row - node_offsets[jnp.where(s_live, s_dev, 0)]).astype(idx), mode="drop"
        )
    )
    csr_sizes = jax.ops.segment_sum(
        s_live.astype(INDEX_DTYPE),
        jnp.where(s_live, s_dev, ndev_i),
        num_segments=ndev + 1,
    )[:ndev]

    return SendBuffers(
        node_rows=node_rows,
        node_sizes=node_sizes,
        csr_cell=csr_cell,
        csr_row=csr_row,
        csr_sizes=csr_sizes,
        node_overflow=n_nodes_out > as_index(node_capacity),
        csr_overflow=n_csr > as_index(csr_capacity),
    )
