"""Multi-device collective tests for ``yggdrax.distributed``.

Run on forced host CPU devices so they need no GPU:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed -q

They also run unchanged on a real multi-GPU node (they just pick up whatever
devices JAX exposes).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:  # stable location across recent JAX versions
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed import (
    device_count,
    make_mesh,
    ragged_all_to_all_exchange,
)
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="multi-device collectives need >= 2 devices"
)


def _mesh(n=None):
    n = min(4, device_count()) if n is None else n
    return make_mesh(n)


def _size_matrix(ndev):
    """A small deterministic ndev x ndev send-size matrix; full[s, i] = s->i."""
    s = np.arange(ndev)[:, None]
    i = np.arange(ndev)[None, :]
    return (1 + ((s + i) % 3)).astype(np.int32)  # each block in [1, 3]


def test_make_mesh_and_device_count():
    mesh = _mesh()
    assert mesh.axis_names == (AXIS_NAME,)
    assert mesh.size == min(4, device_count())


def test_ragged_all_to_all_round_trip():
    mesh = _mesh()
    ndev = mesh.size
    full = _size_matrix(ndev)  # full[s, i] rows sent s -> i
    feat = 2
    c_in = int(full.sum(axis=1).max()) + 2  # send-buffer capacity (padded)
    c_out = int(full.sum(axis=0).max()) + 2  # recv-buffer capacity (padded)

    # Build each source's send buffer: rows for destination i are a contiguous
    # block at exclusive_cumsum(send_sizes)[i], every row tagged with the source
    # rank so we can verify routing. Padding tail = -1.
    operand = np.full((ndev, c_in, feat), -1, dtype=np.float32)
    for s in range(ndev):
        off = 0
        for i in range(ndev):
            n = full[s, i]
            operand[s, off : off + n, :] = float(s)
            off += n
    # Flatten the device axis into axis 0 so shard_map hands each device its slice.
    send_sizes = jnp.asarray(full.reshape(ndev * ndev))
    operand_j = jnp.asarray(operand.reshape(ndev * c_in, feat))

    def fn(op, ss):
        out, recv_sizes, recv_offsets = ragged_all_to_all_exchange(
            op, ss, output_capacity=c_out, fill_value=-1.0
        )
        return out, recv_sizes, recv_offsets

    out, recv_sizes, recv_offsets = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)),
    )(operand_j, send_sizes)

    out = np.asarray(out).reshape(ndev, c_out, feat)
    recv_sizes = np.asarray(recv_sizes).reshape(ndev, ndev)
    recv_offsets = np.asarray(recv_offsets).reshape(ndev, ndev)

    for i in range(ndev):
        # device i receives full[s, i] rows from each source s
        np.testing.assert_array_equal(recv_sizes[i], full[:, i])
        expected_off = np.concatenate([[0], np.cumsum(full[:, i])[:-1]])
        np.testing.assert_array_equal(recv_offsets[i], expected_off)
        # every received block carries its source rank
        for s in range(ndev):
            n = full[s, i]
            o = expected_off[s]
            block = out[i, o : o + n, :]
            np.testing.assert_array_equal(block, np.full((n, feat), float(s)))
        total = int(full[:, i].sum())
        # padding tail is untouched
        assert np.all(out[i, total:, :] == -1.0)


def test_ragged_conserves_total_rows():
    """Global row count is conserved by the exchange (nothing lost/created)."""
    mesh = _mesh()
    ndev = mesh.size
    full = _size_matrix(ndev)
    feat = 1
    c_in = int(full.sum(axis=1).max()) + 1
    c_out = int(full.sum(axis=0).max()) + 1

    operand = np.zeros((ndev, c_in, feat), dtype=np.float32)
    for s in range(ndev):
        operand[s, : full[s].sum(), 0] = 1.0  # mark valid rows
    send_sizes = jnp.asarray(full.reshape(ndev * ndev))
    operand_j = jnp.asarray(operand.reshape(ndev * c_in, feat))

    def fn(op, ss):
        out, recv_sizes, _ = ragged_all_to_all_exchange(op, ss, output_capacity=c_out)
        return jnp.sum(recv_sizes)[None]

    recv_totals = shard_map(
        fn, mesh=mesh, in_specs=(P(AXIS_NAME), P(AXIS_NAME)), out_specs=P(AXIS_NAME)
    )(operand_j, send_sizes)

    assert int(np.asarray(recv_totals).sum()) == int(full.sum())


# --------------------------------------------------------------------------------------
# Overflow: an exchange whose sizes do not fit its buffers stays in bounds.
#
# jaccpot's cross exchanges hand over sizes that overflow by design on a capacity miss
# and flag it afterwards from the returned sizes. The native collective does no bounds
# check (measured on 2x A100: rows written past the output, read past the operand), so
# both paths move one clamped block matrix and still return the REQUESTED sizes.


def _overflow_case(ndev, kind):
    """``(full, c_in, cap)``: the size matrix and buffers that overflow as named."""
    full = 2 * _size_matrix(ndev)  # blocks of 2..6 rows
    row, col = int(full.sum(axis=1).max()), int(full.sum(axis=0).max())
    c_in, cap = {
        "fits": (row, col),
        "recv": (row, col - 3),  # the fullest receivers drop rows
        "send": (row - 3, col),  # the fullest senders' last blocks run off the operand
        # a receiver still has room for some of the rows a sender reads past its
        # operand, so a wrong read shows up (on 2 devices col - 3 would hide it)
        "both": (row - 3, col - 1),
    }[kind]
    return full, c_in, cap


def _operands(ndev, c_in, feat, seed=0):
    """Random float32 send buffers ``[ndev, c_in, feat]`` (some ``-0.0``)."""
    rng = np.random.default_rng(seed)
    ops = rng.standard_normal((ndev, c_in, feat)).astype(np.float32)
    ops[:, ::5, 0] = -0.0
    return ops


def _expected_output(full, ops, cap, fill):
    """The overflow contract, row by row: every block at its REQUESTED offset; a row
    past the sender's operand stays ``fill``, a row past ``cap`` is dropped."""
    ndev, c_in = ops.shape[:2]
    out = np.full((ndev, cap) + ops.shape[2:], fill, dtype=ops.dtype)
    for i in range(ndev):
        at = 0
        for s in range(ndev):
            start = int(full[s, :i].sum())
            for k in range(int(full[s, i])):
                if start + k < c_in and at + k < cap:
                    out[i, at + k] = ops[s, start + k]
            at += int(full[s, i])
    return out


def _run_exchange(mesh, full, ops, cap, fill=-1.0, method="auto"):
    ndev, c_in, feat = ops.shape

    def fn(op, ss):
        return ragged_all_to_all_exchange(
            op, ss, output_capacity=cap, fill_value=fill, method=method
        )

    out, recv_sizes, recv_offsets = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)),
    )(jnp.asarray(ops.reshape(ndev * c_in, feat)), jnp.asarray(full.reshape(-1)))
    return (
        np.asarray(out).reshape(ndev, cap, feat),
        np.asarray(recv_sizes).reshape(ndev, ndev),
        np.asarray(recv_offsets).reshape(ndev, ndev),
    )


@pytest.mark.parametrize("kind", ["recv", "send", "both"])
def test_ragged_overflow_is_clamped(kind):
    """An overflowing exchange delivers exactly the in-bounds rows and reports the
    requested sizes. On the base, ``send`` and ``both`` failed: rows past a sender's
    operand were read from the next device's buffer (or clamped to the last row).
    ``recv`` alone passed there: the gather never wrote past the output."""
    mesh = _mesh()
    ndev = mesh.size
    full, c_in, cap = _overflow_case(ndev, kind)
    assert full.sum(axis=1).max() > c_in or full.sum(axis=0).max() > cap
    ops = _operands(ndev, c_in, feat=3)
    out, recv_sizes, recv_offsets = _run_exchange(mesh, full, ops, cap)

    np.testing.assert_array_equal(out, _expected_output(full, ops, cap, -1.0))
    for i in range(ndev):
        # the sizes the senders asked for, overflow or not: callers detect it from them
        np.testing.assert_array_equal(recv_sizes[i], full[:, i])
        np.testing.assert_array_equal(
            recv_offsets[i], np.cumsum(full[:, i]) - full[:, i]
        )
    overflowing = full.sum(axis=0) > cap
    if kind != "send":
        assert overflowing.any() and not overflowing.all()  # one that fits, too


def test_ragged_fitting_exchange_is_bitwise_unchanged():
    """The clamp is the identity on an exchange that fits: every row bit for bit
    (random data with signed zeros), exactly at the round-trip capacities."""
    mesh = _mesh()
    ndev = mesh.size
    full, c_in, cap = _overflow_case(ndev, "fits")
    ops = _operands(ndev, c_in, feat=3, seed=1)
    out, recv_sizes, _ = _run_exchange(mesh, full, ops, cap, fill=0.0)
    expect = _expected_output(full, ops, cap, 0.0)
    np.testing.assert_array_equal(out.view(np.uint32), expect.view(np.uint32))
    np.testing.assert_array_equal(recv_sizes, full.T)


class _NativeEmulator:
    """``jax.lax.ragged_all_to_all`` semantics on CPU, recording what it was handed.

    XLA:CPU has no ragged lowering, so the native path cannot run here. This stand-in
    applies the op's semantics (sender ``s``'s block for ``i`` is
    ``operand[input_offsets[i]:][:send_sizes[i]]``, written to ``i``'s output at
    ``s``'s ``output_offsets[i]``) and records, per device, whether every block it
    reads and every block written into its output lies in bounds, and whether its
    ``recv_sizes`` match what the senders send. The real op on GPU checks none of it.
    """

    def __init__(self):
        self.checks = []
        self.args = []

    def __call__(
        self,
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        *,
        axis_name,
    ):
        import jax.lax as lax

        me = lax.axis_index(axis_name)
        c_in, cap = operand.shape[0], output.shape[0]
        ops = lax.all_gather(operand, axis_name)
        in_off = lax.all_gather(input_offsets, axis_name)[:, me]
        snd = lax.all_gather(send_sizes, axis_name)[:, me]
        out_off = lax.all_gather(output_offsets, axis_name)[:, me]
        read_ok = jnp.all((input_offsets >= 0) & (input_offsets + send_sizes <= c_in))
        write_ok = jnp.all((out_off >= 0) & (out_off + snd <= cap))
        agree = jnp.all(recv_sizes == snd)
        self.checks.append(jnp.stack([read_ok, write_ok, agree])[None])
        self.args.append(
            jnp.stack([input_offsets, send_sizes, output_offsets, recv_sizes])[None]
        )
        k = jnp.arange(c_in)
        for s in range(ops.shape[0]):
            live = k < snd[s]
            dst = jnp.where(live, out_off[s] + k, cap)
            src = jnp.clip(in_off[s] + k, 0, c_in - 1)
            output = output.at[dst].set(ops[s][src], mode="drop")
        return output


@pytest.mark.parametrize("kind", ["fits", "recv", "send", "both"])
def test_ragged_native_path_stays_in_bounds(monkeypatch, kind):
    """The native path hands the collective in-bounds, mutually consistent blocks
    and delivers the contract's rows; on an exchange that fits it hands over exactly
    the unclamped offsets and sizes, so the op's result is unchanged. On the base,
    every overflowing kind failed the bounds check."""
    import jax

    emulator = _NativeEmulator()
    monkeypatch.setattr(jax.lax, "ragged_all_to_all", emulator)
    mesh = _mesh()
    ndev = mesh.size
    full, c_in, cap = _overflow_case(ndev, kind)
    ops = _operands(ndev, c_in, feat=2, seed=2)
    feat = ops.shape[2]

    def fn(op, ss):
        out, recv_sizes, _ = ragged_all_to_all_exchange(
            op, ss, output_capacity=cap, fill_value=-1.0, method="native"
        )
        return out, recv_sizes, emulator.checks.pop(), emulator.args.pop()

    out, recv_sizes, checks, args = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME),) * 4,
    )(jnp.asarray(ops.reshape(ndev * c_in, feat)), jnp.asarray(full.reshape(-1)))
    checks = np.asarray(checks)
    assert checks.shape == (ndev, 3)
    assert checks[:, 0].all(), f"read past the operand on devices {~checks[:, 0]}"
    assert checks[:, 1].all(), f"wrote past the output on devices {~checks[:, 1]}"
    assert checks[:, 2].all(), "a receiver expects other sizes than its senders send"
    out = np.asarray(out).reshape(ndev, cap, feat)
    np.testing.assert_array_equal(out, _expected_output(full, ops, cap, -1.0))
    np.testing.assert_array_equal(np.asarray(recv_sizes).reshape(ndev, ndev), full.T)
    if kind == "fits":
        # (input_offsets, send_sizes, output_offsets, recv_sizes) per device, as
        # computed before the clamp existed
        unclamped = np.stack(
            [
                np.cumsum(full, axis=1) - full,
                full,
                (np.cumsum(full, axis=0) - full),
                full.T,
            ],
            axis=1,
        )
        np.testing.assert_array_equal(np.asarray(args), unclamped)
