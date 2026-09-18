"""The cross-domain exchange itself: two ragged rounds, and what the receiver gets.

The sender has already decided everything (:mod:`.export`), so this moves bytes and
nothing more. Per export list -- far and near are exchanged separately, because a far
pair ships a multipole and a near pair ships particles -- two rounds go out:

1. the **payload**, one row per distinct node a given receiver needs;
2. the **CSR**, one entry per (receiver cell, node) reference.

Both are already grouped by destination by
:func:`~yggdrax.distributed.export.build_send_buffers`, and the CSR's two columns are
already expressed in the receiver's own frame, so the receiver reads what arrives
without knowing anything about the sender's buffers.

**The receive side has to re-offset the CSR, and this is the trap.** Each sender
numbered its ``csr_row`` from zero within the block it sent. Those blocks land
end-to-end in one receive buffer, so an entry from sender ``s`` must be shifted by
where ``s``'s payload block starts. Skip that and every entry from every sender after
the first silently reads the wrong node -- plausible values, right shapes, wrong force,
and momentum stays exact because nothing is created or destroyed. :func:`rebase_csr`
exists to make that step a named, tested thing rather than an inline addition.

Sizes travel with the data: ``ragged_all_to_all_exchange`` returns ``recv_sizes`` per
sender, which is exactly what the re-offsetting needs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..dtypes import INDEX_DTYPE, as_index
from .comm import ragged_all_to_all_exchange
from .sharding import AXIS_NAME

__all__ = ["ImportedCells", "exchange_export_list", "rebase_csr"]


class ImportedCells(NamedTuple):
    """What one device receives for one export list.

    Attributes
    ----------
    payload:
        ``(payload_capacity, *feat)`` the imported node payloads, blocks laid
        end-to-end in sender order.
    num_payload:
        Total live payload rows.
    csr_cell:
        ``(csr_capacity,)`` this device's OWN cell index for each reference.
    csr_row:
        ``(csr_capacity,)`` row of ``payload``, already re-offset across senders.
    num_csr:
        Total live CSR entries.
    payload_sizes, csr_sizes:
        ``(ndev,)`` rows received from each sender, for diagnostics and for
        anything that needs to attribute an import to its origin.
    """

    payload: Array
    num_payload: Array
    csr_cell: Array
    csr_row: Array
    num_csr: Array
    payload_sizes: Array
    csr_sizes: Array


def rebase_csr(
    csr_row: Array, csr_sizes: Array, payload_sizes: Array, *, ndev: int
) -> Array:
    """Shift each sender's ``csr_row`` to where that sender's payload block landed.

    Every sender numbers its rows from zero within its own block. The blocks arrive
    concatenated in sender order, so entries from sender ``s`` must be offset by
    ``exclusive_cumsum(payload_sizes)[s]``.

    Parameters
    ----------
    csr_row:
        ``(csr_capacity,)`` as received, each sender's numbering from zero.
    csr_sizes:
        ``(ndev,)`` CSR entries received from each sender -- this is what says which
        entry came from whom.
    payload_sizes:
        ``(ndev,)`` payload rows received from each sender.
    ndev:
        Mesh size.

    Returns
    -------
    Array
        ``csr_row`` in the receive buffer's own frame; padding stays ``-1``.
    """
    cap = int(jnp.asarray(csr_row).shape[0])
    p_off = jnp.concatenate(
        [
            jnp.zeros((1,), INDEX_DTYPE),
            jnp.cumsum(as_index(payload_sizes), dtype=INDEX_DTYPE),
        ]
    )
    c_off = jnp.concatenate(
        [
            jnp.zeros((1,), INDEX_DTYPE),
            jnp.cumsum(as_index(csr_sizes), dtype=INDEX_DTYPE),
        ]
    )
    pos = jnp.arange(cap, dtype=INDEX_DTYPE)
    # which sender each slot came from: how many block boundaries it is past
    sender = jnp.sum(pos[:, None] >= c_off[None, 1 : ndev + 1], axis=1)
    live = (jnp.asarray(csr_row) >= 0) & (pos < c_off[ndev])
    shifted = as_index(csr_row) + p_off[jnp.minimum(sender, ndev)]
    return jnp.where(live, shifted, as_index(-1)).astype(jnp.asarray(csr_row).dtype)


def exchange_export_list(
    payload_rows: Array,
    node_sizes: Array,
    csr_cell: Array,
    csr_row: Array,
    csr_sizes: Array,
    *,
    payload_capacity: int,
    csr_capacity: int,
    ndev: int,
    axis_name: str = AXIS_NAME,
    method: str = "auto",
) -> ImportedCells:
    """Two ragged rounds for one export list, returning the receiver's view.

    Parameters
    ----------
    payload_rows:
        ``(n, *feat)`` the payload to send, already grouped by destination -- the
        gathered multipoles or particles, not node indices.
    node_sizes:
        ``(ndev,)`` payload rows per destination.
    csr_cell, csr_row, csr_sizes:
        The CSR to send, grouped by destination, in the receiver's frame.
    payload_capacity, csr_capacity:
        Static receive-buffer lengths. Over-allocate: what arrives depends on the
        other devices, so these cannot be derived locally.
    ndev:
        Mesh size.
    axis_name:
        Mesh axis; must match the enclosing ``shard_map``.
    method:
        Passed to :func:`ragged_all_to_all_exchange`; ``"auto"`` avoids the
        ``ragged_all_to_all`` corruption on jax below 0.9.1.

    Returns
    -------
    ImportedCells
        With ``csr_row`` already re-offset across senders.
    """
    payload, payload_sizes, _p_off = ragged_all_to_all_exchange(
        jnp.asarray(payload_rows),
        as_index(node_sizes),
        output_capacity=payload_capacity,
        axis_name=axis_name,
        method=method,
    )
    both = jnp.stack(
        [as_index(csr_cell), as_index(csr_row)], axis=-1
    )  # one round for two columns: they are always needed together
    recv, csr_recv_sizes, _c_off = ragged_all_to_all_exchange(
        both,
        as_index(csr_sizes),
        output_capacity=csr_capacity,
        axis_name=axis_name,
        fill_value=-1.0,
        method=method,
    )
    in_cell, in_row = recv[..., 0], recv[..., 1]
    return ImportedCells(
        payload=payload,
        num_payload=jnp.sum(as_index(payload_sizes), dtype=INDEX_DTYPE),
        csr_cell=in_cell,
        csr_row=rebase_csr(in_row, csr_recv_sizes, payload_sizes, ndev=ndev),
        num_csr=jnp.sum(as_index(csr_recv_sizes), dtype=INDEX_DTYPE),
        payload_sizes=as_index(payload_sizes),
        csr_sizes=as_index(csr_recv_sizes),
    )
