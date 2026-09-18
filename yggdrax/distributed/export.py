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

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array

from ..dtypes import INDEX_DTYPE, as_index
from ..interactions import dual_tree_walk_mutual

__all__ = ["ExportLists", "export_walk"]


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
        return (
            jnp.where(live, a, as_index(-1).astype(idx)),
            jnp.where(live, b - shift, as_index(-1).astype(idx)),
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
