"""Multi-device collective tests for ``yggdrax.distributed``.

Run on forced host CPU devices so they need no GPU:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed -q

They also run unchanged on a real multi-GPU node (they just pick up whatever
devices JAX exposes).
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:  # stable location across recent JAX versions
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed import (
    ShardedArray,
    all_to_all_dense,
    device_count,
    exchange_pytree,
    make_mesh,
    ragged_all_to_all_exchange,
)
from yggdrax.distributed.comm import exchange_sizes
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


def test_sharded_array_is_pytree():
    sa = ShardedArray(data=jnp.zeros((8, 3)), count=jnp.asarray(5))
    leaves, treedef = jax.tree_util.tree_flatten(sa)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.capacity == 8
    # survives a jit round-trip (identity)
    out = jax.jit(lambda x: x)(sa)
    assert int(out.count) == 5
    assert out.data.shape == (8, 3)


def test_exchange_sizes_transposes_matrix():
    mesh = _mesh()
    ndev = mesh.size
    full = _size_matrix(ndev)
    # Flat global buffer: axis-0 sharding gives device s its row full[s] ([ndev]).
    send = jnp.asarray(full.reshape(ndev * ndev))

    def fn(send_local):
        return exchange_sizes(send_local)

    recv = shard_map(
        fn, mesh=mesh, in_specs=P(AXIS_NAME), out_specs=P(AXIS_NAME)
    )(send)
    recv = np.asarray(recv).reshape(ndev, ndev)  # recv[i] = sizes i got per source
    # device i receives from source s exactly full[s, i]
    np.testing.assert_array_equal(recv, full.T)


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
        out, recv_sizes, _ = ragged_all_to_all_exchange(
            op, ss, output_capacity=c_out
        )
        return jnp.sum(recv_sizes)[None]

    recv_totals = shard_map(
        fn, mesh=mesh, in_specs=(P(AXIS_NAME), P(AXIS_NAME)), out_specs=P(AXIS_NAME)
    )(operand_j, send_sizes)

    assert int(np.asarray(recv_totals).sum()) == int(full.sum())


def test_all_to_all_dense_fallback():
    mesh = _mesh()
    ndev = mesh.size
    full = _size_matrix(ndev)  # full[s, i] valid rows sent s -> i
    cap = int(full.max()) + 1
    feat = 2

    # operand[s, i] : block device s sends to device i, first full[s,i] rows = s.
    operand = np.full((ndev, ndev, cap, feat), -1, dtype=np.float32)
    for s in range(ndev):
        for i in range(ndev):
            operand[s, i, : full[s, i], :] = float(s)
    operand_j = jnp.asarray(operand.reshape(ndev * ndev, cap, feat))
    send_counts = jnp.asarray(full.reshape(ndev * ndev))

    def fn(op, sc):
        received, recv_counts = all_to_all_dense(op, sc)
        return received, recv_counts

    received, recv_counts = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME), P(AXIS_NAME)),
    )(operand_j, send_counts)

    received = np.asarray(received).reshape(ndev, ndev, cap, feat)
    recv_counts = np.asarray(recv_counts).reshape(ndev, ndev)
    for i in range(ndev):
        np.testing.assert_array_equal(recv_counts[i], full[:, i])
        for s in range(ndev):
            n = full[s, i]
            np.testing.assert_array_equal(
                received[i, s, :n, :], np.full((n, feat), float(s))
            )


def test_exchange_pytree_routes_all_particle_leaves():
    mesh = _mesh()
    ndev = mesh.size
    full = _size_matrix(ndev)
    c_in = int(full.sum(axis=1).max()) + 2
    c_out = int(full.sum(axis=0).max()) + 2

    # Two per-particle leaves (pos [c_in,3], mass [c_in]) + one pass-through leaf.
    pos = np.full((ndev, c_in, 3), -1.0, dtype=np.float32)
    mass = np.full((ndev, c_in), -1.0, dtype=np.float32)
    for s in range(ndev):
        off = 0
        for i in range(ndev):
            n = full[s, i]
            pos[s, off : off + n, :] = float(s)
            mass[s, off : off + n] = float(s)
            off += n

    pos_j = jnp.asarray(pos.reshape(ndev * c_in, 3))
    mass_j = jnp.asarray(mass.reshape(ndev * c_in))
    send_sizes = jnp.asarray(full.reshape(ndev * ndev))
    passthrough = jnp.arange(5)  # leading dim != c_in -> untouched

    def fn(p, m, ss):
        tree = {"pos": p, "mass": m, "tag": passthrough}
        new_tree, recv_sizes, _ = exchange_pytree(
            tree, ss, output_capacity=c_out, n_local=c_in
        )
        return new_tree["pos"], new_tree["mass"], new_tree["tag"], recv_sizes

    out_pos, out_mass, out_tag, recv_sizes = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)),
    )(pos_j, mass_j, send_sizes)

    out_pos = np.asarray(out_pos).reshape(ndev, c_out, 3)
    out_mass = np.asarray(out_mass).reshape(ndev, c_out)
    recv_sizes = np.asarray(recv_sizes).reshape(ndev, ndev)
    # pass-through leaf is unchanged (replicated per device)
    np.testing.assert_array_equal(np.asarray(out_tag).reshape(ndev, 5)[0], np.arange(5))
    for i in range(ndev):
        np.testing.assert_array_equal(recv_sizes[i], full[:, i])
        off = 0
        for s in range(ndev):
            n = full[s, i]
            np.testing.assert_array_equal(out_pos[i, off : off + n, :], float(s))
            np.testing.assert_array_equal(out_mass[i, off : off + n], float(s))
            off += n
