"""Collective communication primitives for Yggdrax multi-GPU execution.

Phase 0 of the multi-GPU roadmap. Mirrors the design of jztree's ``comm.py``:
the central primitive is a *ragged* all-to-all (variable per-destination row
counts) used to exchange boundary particles and locally-essential-tree nodes
between GPU domains. Everything is expressed with ``jax.lax`` collectives and
is meant to run inside a ``jax.shard_map`` over the mesh axis built in
:mod:`yggdrax.distributed.sharding` (``AXIS_NAME``).

The functions here operate on the *per-device local view*: inside ``shard_map``
each call sees only this device's shard, and the collectives move data across
devices. All buffers are statically shaped (padded to a capacity) so the whole
thing stays ``jit``-able -- the number of *valid* rows is carried alongside as a
dynamic count, exactly like jztree's padded ``Pos`` container.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .sharding import AXIS_NAME

# Offsets/sizes handed to ``ragged_all_to_all`` must be an integer type; int32
# is plenty for per-device row counts and matches XLA's expectations.
_COUNT_DTYPE = jnp.int32

#: First JAX release whose XLA:GPU ``ragged_all_to_all`` survives buffer movement.
#:
#: Below it, ``RaggedAllToAllStartThunk::Initialize`` caches the peer output-buffer
#: device addresses from the FIRST execution. Once the allocator moves those
#: buffers -- **input donation is enough; no gradient is required** -- the one-shot
#: kernel writes to stale addresses and the real output keeps its ``fill_value``.
#: Measured 2026-09-04 on jax 0.9.0 / 2xA100 with ``jit(shard_map(ragged_all_to_all))``
#: and nothing else (jaccpot ``bench/repro_jax_ragged_all_to_all_forward.py``):
#: 36/40 calls corrupt under donation, 3/40 under allocator churn alone, 0/40 with
#: identical buffers -- so a repeat-the-same-input reproducibility check passes on a
#: broken build. Clean on 0.9.1 and 0.10.2 (XLA ``4e0cc7e356``, shipped in the 0.9.1
#: GPU plugin -- check ``jax_plugins.xla_cuda12.__file__`` is the one you think).
#: In a LET this silently drops the entire cross-domain near field: a ~45 % force
#: error on a 17.8M-particle disc across 4 GPUs while angular momentum, energy and
#: every overflow flag stayed healthy.
RAGGED_NATIVE_FIXED_JAX = (0, 9, 1)


def _jax_version_tuple() -> tuple[int, ...]:
    """``jax.__version__`` as comparable ints, tolerant of dev/rc suffixes."""
    parts: list[int] = []
    for piece in jax.__version__.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def resolve_ragged_method(method: str = "auto") -> str:
    """Decide which ragged exchange ``"auto"`` means on this JAX and backend.

    ``"native"`` (:func:`jax.lax.ragged_all_to_all`) only when BOTH hold: the default
    backend is GPU or TPU (XLA:CPU has no lowering -- ``UNIMPLEMENTED: HLO opcode
    'ragged-all-to-all'``), and ``jax >= `` :data:`RAGGED_NATIVE_FIXED_JAX`. Otherwise
    ``"buf"``, the all_gather-based exchange, which computes the identical result at
    O(ndev^2 x capacity) bandwidth instead of O(actual halo). An explicit
    ``"native"`` / ``"buf"`` is never overridden -- that is how the defect is
    reproduced on purpose.
    """
    if method not in ("auto", "native", "buf"):
        raise ValueError(f"method must be 'auto', 'native' or 'buf'; got {method!r}")
    if method != "auto":
        return method
    if jax.default_backend() not in ("gpu", "tpu"):
        return "buf"
    return "native" if _jax_version_tuple() >= RAGGED_NATIVE_FIXED_JAX else "buf"


def _exclusive_cumsum(x: Array, axis: int = 0) -> Array:
    """Exclusive prefix sum (``out[i] = sum(x[:i])``) along ``axis``."""

    return jnp.cumsum(x, axis=axis) - x


def _ragged_native(
    operand, send_sizes, full, me, output_capacity, axis_name, fill_value
):
    """XLA ``ragged_all_to_all`` path (GPU/TPU)."""

    input_offsets = _exclusive_cumsum(send_sizes)
    recv_sizes = full[:, me]
    # output_offsets[i] = offset into device i's output our block lands at =
    # number of rows device i receives from sources ranked before us.
    output_offsets = _exclusive_cumsum(full)[me]

    feat = operand.shape[1:]
    output = jnp.full((output_capacity, *feat), fill_value, dtype=operand.dtype)
    output = jax.lax.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )
    return output


def _ragged_through_buf(
    operand, send_sizes, full, me, output_capacity, axis_name, fill_value
):
    """all_gather-based fallback for backends without ``ragged_all_to_all``.

    Gathers every device's send buffer (small at Phase-0 scale) and gathers the
    rows destined for this device with a vectorised index computation. Same
    result as the native path; used on CPU (XLA:CPU lacks the ragged op) and as
    a cross-check.
    """

    ndev, c_in = full.shape[0], operand.shape[0]
    feat = operand.shape[1:]
    ops = jax.lax.all_gather(operand, axis_name, tiled=False)  # [ndev, c_in, *feat]
    flat = ops.reshape((ndev * c_in, *feat))

    recv_sizes = full[:, me]
    recv_end = jnp.cumsum(recv_sizes)
    total = recv_end[-1]
    # per-source offset within the source's own send buffer for our column.
    src_in_off = _exclusive_cumsum(full, axis=1)[:, me]  # [ndev]

    r = jnp.arange(output_capacity)
    src = jnp.searchsorted(recv_end, r, side="right")
    src = jnp.minimum(src, ndev - 1)
    within = r - (recv_end[src] - recv_sizes[src])
    gidx = src * c_in + src_in_off[src] + within
    gathered = flat[gidx]
    valid = (r < total).reshape((output_capacity,) + (1,) * len(feat))
    output = jnp.where(valid, gathered, jnp.asarray(fill_value, operand.dtype))
    return output


def ragged_all_to_all_exchange(
    operand: Array,
    send_sizes: Array,
    *,
    output_capacity: int,
    axis_name: str = AXIS_NAME,
    fill_value: float = 0.0,
    method: str = "auto",
) -> tuple[Array, Array, Array]:
    """Exchange variable-length row blocks between devices (the core primitive).

    The caller lays ``operand`` out so the rows destined for device ``i`` form a
    contiguous block at ``exclusive_cumsum(send_sizes)[i]``. On return, rows
    received from device ``s`` form a contiguous block in the output at
    ``exclusive_cumsum(recv_sizes)[s]``.

    Parameters
    ----------
    operand:
        Padded local send buffer, shape ``[C_in, *feat]``. Rows are grouped by
        destination device in device order.
    send_sizes:
        Integer array ``[ndev]``; ``send_sizes[i]`` rows go to device ``i``.
    output_capacity:
        Static leading dimension of the receive buffer. Must be large enough to
        hold this device's total received rows (over-allocate with slack, per
        the padding-based load-balancing strategy).
    axis_name:
        Mesh axis the exchange runs over; must match the enclosing ``shard_map``.
    fill_value:
        Value the unused (padding) tail of the output buffer is initialised to.
    method:
        ``"native"`` uses ``jax.lax.ragged_all_to_all`` (GPU/TPU); ``"buf"`` uses
        the all_gather fallback (works everywhere, incl. XLA:CPU); ``"auto"``
        (default) picks native on gpu/tpu **on jax >=** :data:`RAGGED_NATIVE_FIXED_JAX`
        and buf otherwise. Below that version the native collective silently returns
        ``fill_value`` once its buffers move (donation is enough) -- see the constant
        for the measurement. Do not select ``"native"`` there except to reproduce it.

    Returns
    -------
    (output, recv_sizes, recv_offsets)
        ``output`` shape ``[output_capacity, *feat]``; ``recv_sizes[s]`` rows
        received from device ``s`` at ``recv_offsets[s]``.
    """

    send_sizes = send_sizes.astype(_COUNT_DTYPE)
    # Gather the full ndev x ndev size matrix so every offset is computed
    # locally with no further communication. full[s, i] = rows s -> i.
    full = jax.lax.all_gather(send_sizes, axis_name, tiled=False)
    me = jax.lax.axis_index(axis_name)

    method = resolve_ragged_method(method)
    impl = _ragged_native if method == "native" else _ragged_through_buf
    output = impl(operand, send_sizes, full, me, output_capacity, axis_name, fill_value)

    recv_sizes = full[:, me]
    recv_offsets = _exclusive_cumsum(recv_sizes)
    return output, recv_sizes, recv_offsets


__all__ = [
    "ragged_all_to_all_exchange",
]
