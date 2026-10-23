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

import os

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

#: XLA flag that selects the NCCL send/recv ragged fallback when ``false``.
_ONE_SHOT_FLAG = "xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel"
#: XLA flag the NCCL fallback cannot run with. Its default is ``true`` from jax
#: :data:`_BARRIER_DEFAULT_ON_JAX` on (XLA 91888df6), ``false`` before.
_BARRIER_FLAG = "xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl"
_BARRIER_DEFAULT_ON_JAX = (0, 11)


def _xla_flag_values(xla_flags: str) -> dict[str, str]:
    """The ``--name=value`` tokens of ``XLA_FLAGS``; a later one wins, as in XLA."""
    values: dict[str, str] = {}
    for token in xla_flags.split():
        if token.startswith("--"):
            name, _, value = token[2:].partition("=")
            values[name] = value.lower() if value else "true"
    return values


def ragged_native_flag_hazard(xla_flags: str | None = None) -> str | None:
    """Say why ``XLA_FLAGS`` breaks the native ``ragged_all_to_all``, else ``None``.

    The NCCL send/recv fallback (``one_shot_kernel=false``) is broken while
    ``use_barrier_with_nccl`` is on. From about 512 rows of 56 float32 per message up,
    the first call returns only each device's block to itself (every cross-device row
    keeps ``fill_value``), and the second call never returns, with both GPUs spinning.
    Measured 2026-10-12 on 2x A100-PCIE with pure JAX (jaccpot
    ``bench/repro_jax_ragged_all_to_all_nccl_fallback.py``): broken on jax 0.11.2 at
    its default and on 0.10.2 with the flag set explicitly; clean on both with it off.
    jax 0.11 turned the flag on by default, so the one-flag NCCL setting that was safe
    on 0.10 hangs on 0.11.

    Parameters
    ----------
    xla_flags:
        The flag string to check; ``None`` reads ``XLA_FLAGS``. XLA reads its flags
        once, when the backend starts, so a later change is not what XLA runs with.

    Returns
    -------
    str or None
        The reason, ready for an error message, or ``None`` when the flags are safe.
    """
    if xla_flags is None:
        xla_flags = os.environ.get("XLA_FLAGS", "")
    values = _xla_flag_values(xla_flags)
    if values.get(_ONE_SHOT_FLAG, "true") not in ("false", "0"):
        return None
    default = "true" if _jax_version_tuple() >= _BARRIER_DEFAULT_ON_JAX else "false"
    if values.get(_BARRIER_FLAG, default) in ("false", "0"):
        return None
    return (
        f"XLA_FLAGS selects the NCCL ragged_all_to_all (--{_ONE_SHOT_FLAG}=false) "
        f"with --{_BARRIER_FLAG} on (its default from jax 0.11). That combination "
        "delivers no cross-device rows and deadlocks on the next call. Add "
        f"--{_BARRIER_FLAG}=false, or drop the one-shot flag for XLA's default kernel."
    )


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


def _clamped_transfer(
    full: Array, c_in: int, output_capacity: int
) -> tuple[Array, Array, Array]:
    """The block of every (sender, receiver) pair that stays inside both buffers.

    Blocks sit where the TRUE sizes put them: sender ``s``'s block for ``i`` starts at
    row ``in_off[s, i]`` of its operand and lands at row ``out_off[s, i]`` of ``i``'s
    output, both exclusive prefix sums of ``full``. A block is cut where it would read
    past the operand's ``c_in`` rows or write past ``output_capacity``; a block that
    starts beyond either moves nothing. Computed from the all-gathered ``full`` alone,
    so every device derives the identical matrix and a sender and its receiver agree
    on every block. When every row sum of ``full`` fits ``c_in`` and every column sum
    fits ``output_capacity`` the clamp is the identity.

    Parameters
    ----------
    full:
        ``[ndev, ndev]`` requested sizes, ``full[s, i]`` = rows ``s`` -> ``i``.
    c_in:
        Rows of every device's operand (shard_map gives each the same shape).
    output_capacity:
        Rows of every device's output.

    Returns
    -------
    (moved, in_off, out_off)
        ``[ndev, ndev]`` each: the rows actually moved, and each block's input and
        output offset, clipped into ``[0, c_in]`` / ``[0, output_capacity]`` (only a
        block that moves nothing is ever clipped).
    """

    in_off = _exclusive_cumsum(full, axis=1)
    out_off = _exclusive_cumsum(full, axis=0)
    room = jnp.minimum(c_in - in_off, output_capacity - out_off)
    moved = jnp.clip(jnp.minimum(full, room), 0, None)
    return (
        moved,
        jnp.clip(in_off, 0, c_in),
        jnp.clip(out_off, 0, output_capacity),
    )


def _ragged_native(operand, full, me, output_capacity, axis_name, fill_value):
    """XLA ``ragged_all_to_all`` path (GPU/TPU).

    The op does no bounds check: handed the requested sizes of an overflowing
    exchange it reads past the operand and writes past the peer's output on the GPU
    (measured 2026-10-22, 2x A100, NCCL fallback). It gets the clamped blocks.
    """

    moved, in_off, out_off = _clamped_transfer(full, operand.shape[0], output_capacity)
    feat = operand.shape[1:]
    output = jnp.full((output_capacity, *feat), fill_value, dtype=operand.dtype)
    # output_offsets[i] = row of device i's output our block lands at.
    output = jax.lax.ragged_all_to_all(
        operand,
        output,
        in_off[me],
        moved[me],
        out_off[me],
        moved[:, me],
        axis_name=axis_name,
    )
    return output


def _ragged_through_buf(operand, full, me, output_capacity, axis_name, fill_value):
    """all_gather-based fallback for backends without ``ragged_all_to_all``.

    Gathers every device's send buffer (small at Phase-0 scale) and gathers the
    rows destined for this device with a vectorised index computation. Same
    result as the native path, overflow included (it delivers the same clamped
    blocks); used on CPU (XLA:CPU lacks the ragged op) and as a cross-check.
    """

    ndev, c_in = full.shape[0], operand.shape[0]
    feat = operand.shape[1:]
    ops = jax.lax.all_gather(operand, axis_name, tiled=False)  # [ndev, c_in, *feat]
    flat = ops.reshape((ndev * c_in, *feat))
    moved, in_off, _ = _clamped_transfer(full, c_in, output_capacity)

    recv_sizes = full[:, me]
    recv_end = jnp.cumsum(recv_sizes)
    r = jnp.arange(output_capacity)
    src = jnp.searchsorted(recv_end, r, side="right")
    src = jnp.minimum(src, ndev - 1)
    within = r - (recv_end[src] - recv_sizes[src])
    # a row is real when it lies in the moved part of its block; rows past the
    # last block (src clipped to ndev - 1) have within >= its size, so they fail too.
    # A discarded row's index may point anywhere: the gather clamps it, never reads
    # out of bounds, and on an exchange that fits every index is the one used before
    # the clamp existed (in_off is clipped only for blocks that move nothing).
    valid = within < moved[src, me]
    gidx = src * c_in + in_off[src, me] + within
    gathered = flat[gidx]
    valid = valid.reshape((output_capacity,) + (1,) * len(feat))
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
        Static leading dimension of the receive buffer. Should hold this device's
        total received rows (over-allocate with slack, per the padding-based
        load-balancing strategy); see *Overflow* below for what happens when not.
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
        On GPU, a native exchange raises ``RuntimeError`` when ``XLA_FLAGS`` selects the
        NCCL fallback in a way that deadlocks; see :func:`ragged_native_flag_hazard`.

    Returns
    -------
    (output, recv_sizes, recv_offsets)
        ``output`` shape ``[output_capacity, *feat]``; ``recv_sizes[s]`` rows
        received from device ``s`` at ``recv_offsets[s]``. Both follow the sizes the
        senders REQUESTED, also on an overflow, so a caller can detect one from them.

    Notes
    -----
    *Overflow.* The exchange never reads or writes out of bounds. A block that would
    read past a sender's ``C_in`` rows or write past ``output_capacity`` is cut there,
    identically on both paths: rows of a block keep their offsets, rows past the
    sender's operand arrive as ``fill_value``, rows past ``output_capacity`` are
    dropped. The output is then incomplete, and the call cannot tell its caller so
    except through the sizes: treat ``sum(recv_sizes) > output_capacity`` on a
    receiver, and ``sum(send_sizes) > C_in`` on a sender, as an overflow. An exchange
    that fits both buffers is unaffected (the clamp is the identity).
    """

    send_sizes = send_sizes.astype(_COUNT_DTYPE)
    # Gather the full ndev x ndev size matrix so every offset is computed
    # locally with no further communication. full[s, i] = rows s -> i.
    full = jax.lax.all_gather(send_sizes, axis_name, tiled=False)
    me = jax.lax.axis_index(axis_name)

    method = resolve_ragged_method(method)
    if method == "native" and jax.default_backend() == "gpu":
        hazard = ragged_native_flag_hazard()
        if hazard is not None:
            raise RuntimeError(hazard)
    impl = _ragged_native if method == "native" else _ragged_through_buf
    output = impl(operand, full, me, output_capacity, axis_name, fill_value)

    recv_sizes = full[:, me]
    recv_offsets = _exclusive_cumsum(recv_sizes)
    return output, recv_sizes, recv_offsets


__all__ = [
    "ragged_all_to_all_exchange",
    "ragged_native_flag_hazard",
]
