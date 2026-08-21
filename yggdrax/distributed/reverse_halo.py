"""Return the ``-f`` half of a cross-domain mutual pair to the domain that owns it.

The forward halo (:func:`~yggdrax.distributed.let.import_near_halo`) runs the other
way: it *imports* remote sources so a local target can gather from them. A mutual FMM
needs the opposite direction as well, and that direction is not a convenience -- it is
the momentum-conservation mechanism at multi-GPU scale.

Within one device the ``+f``/``-f`` antisymmetry is free: both endpoints of a
canonical pair are local, so one kernel writes both halves under a single rounding
regime and ``sum_i m_i a_i`` cancels structurally. Across a boundary the evaluating
device owns only one endpoint, so the other half has to be sent back. Drop it or
double-count it and the force is wrong at the percent level while every device's
*local* momentum sum still looks perfect -- see the note on
``cross_walk.cross_pair_owner``.

This is a scatter-with-accumulate, not a gather: several contributions can land on
the same remote index and must sum. It reuses
:func:`~yggdrax.distributed.comm.ragged_all_to_all_exchange` rather than growing a
second communication pattern.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from yggdrax.dtypes import INDEX_DTYPE, as_index

from .comm import ragged_all_to_all_exchange
from .sharding import AXIS_NAME

__all__ = [
    "ReverseHaloResult",
    "apply_reverse_halo",
    "export_reverse_halo",
    "group_by_destination",
]


class ReverseHaloResult(NamedTuple):
    """What came back from other devices, and whether anything was dropped.

    ``target_index`` and ``values`` are the contributions this device must apply to
    its OWN particles or nodes; ``n_received`` is how many are live. ``overflow`` is true
    if the receive capacity was too small, which is reported rather than silently
    truncated because a dropped ``-f`` is invisible in a per-device momentum check.
    """

    target_index: Array
    values: Array
    n_received: Array
    overflow: Array


def group_by_destination(owner: Array, ndev: int) -> tuple[Array, Array]:
    """Order entries by destination device and count how many go to each.

    ``ragged_all_to_all_exchange`` expects each device's outgoing rows laid out
    contiguously per destination, so the payload has to be permuted before it is
    sent. Padding (``owner < 0``) is pushed past every real destination so it lands
    outside the send counts.

    Parameters
    ----------
    owner:
        ``(cap,)`` destination device per entry, negative for padding.
    ndev:
        Number of devices.

    Returns
    -------
    tuple[Array, Array]
        ``(order, send_sizes)``: the permutation putting entries in destination
        order, and the per-destination counts.
    """
    live = owner >= as_index(0)
    # Sorting on a key that maps padding to `ndev` keeps it after every real
    # destination, so `send_sizes` never counts it.
    key = jnp.where(live, owner.astype(INDEX_DTYPE), as_index(ndev))
    order = jnp.argsort(key, stable=True)
    send_sizes = jnp.bincount(
        jnp.where(live, owner.astype(INDEX_DTYPE), as_index(ndev)),
        length=ndev + 1,
    )[:ndev].astype(INDEX_DTYPE)
    return order, send_sizes


def export_reverse_halo(
    owner: Array,
    target_index: Array,
    values: Array,
    ndev: int,
    *,
    recv_capacity: int,
    axis_name: str = AXIS_NAME,
) -> ReverseHaloResult:
    """Send each ``-f`` contribution to the device that owns its endpoint.

    One ragged all-to-all. The payload is ``(target_index, values)`` per
    contribution: the index is carried rather than re-derived on arrival, because the
    receiving device has no way to reconstruct which of its own nodes a remote
    device was talking about.

    Parameters
    ----------
    owner:
        ``(cap,)`` destination device per contribution, negative for padding. This is
        ``far_owner``/``near_owner`` from
        :func:`~yggdrax.distributed.cross_walk.dual_tree_walk_cross_mutual`.
    target_index:
        ``(cap,)`` index on the DESTINATION device to accumulate into.
    values:
        ``(cap, k)`` contributions, e.g. ``(cap, 3)`` forces.
    ndev:
        Number of devices.
    recv_capacity:
        Rows this device can receive. Static.
    axis_name:
        Mesh axis to exchange over.

    Returns
    -------
    ReverseHaloResult
        The contributions destined for this device, and an overflow flag.
    """
    order, send_sizes = group_by_destination(owner, int(ndev))
    idx_sorted = target_index[order].astype(values.dtype)
    val_sorted = values[order]
    payload = jnp.concatenate([idx_sorted[:, None], val_sorted], axis=1)
    recv, recv_sizes, _offsets = ragged_all_to_all_exchange(
        payload,
        send_sizes,
        output_capacity=int(recv_capacity),
        axis_name=axis_name,
    )
    total = jnp.sum(recv_sizes).astype(INDEX_DTYPE)
    return ReverseHaloResult(
        target_index=recv[:, 0].astype(INDEX_DTYPE),
        values=recv[:, 1:],
        n_received=total,
        overflow=total > as_index(int(recv_capacity)),
    )


def apply_reverse_halo(into: Array, result: ReverseHaloResult) -> Array:
    """Accumulate received contributions into a local array.

    Additive, not assignment: several remote devices can send a contribution for the
    same local index, and a mutual force requires all of them to sum. Rows past
    ``n_received`` are dropped rather than added, so a partially filled receive buffer
    does not inject zeros-with-an-index.

    Parameters
    ----------
    into:
        ``(n, k)`` local array to accumulate into.
    result:
        What :func:`export_reverse_halo` returned.

    Returns
    -------
    Array
        ``into`` with the contributions added.
    """
    rows = jnp.arange(result.target_index.shape[0], dtype=INDEX_DTYPE)
    live = rows < result.n_received
    # `.astype(INDEX_DTYPE)`: a scatter whose index dtype is wider than the buffer's
    # is a FutureWarning now and an error in later JAX.
    idx = jnp.where(live, result.target_index, as_index(into.shape[0])).astype(
        INDEX_DTYPE
    )
    return into.at[idx].add(
        jnp.where(live[:, None], result.values, jnp.zeros_like(result.values)),
        mode="drop",
    )
