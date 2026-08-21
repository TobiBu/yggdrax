"""Returning the ``-f`` half of a cross-domain pair to its owner.

Run with several devices, as the repo's other distributed tests do::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_reverse_halo.py

This is the momentum-conservation mechanism at multi-GPU scale, not a convenience:
within one device ``+f``/``-f`` cancels structurally because both endpoints are
local, and across a boundary the other half has to be sent back. So the test that
matters is a GLOBAL sum -- a per-device sum stays exact under both failure modes
(dropping and double-counting), which is precisely what makes them dangerous.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.reverse_halo import (
    apply_reverse_halo,
    export_reverse_halo,
    group_by_destination,
)
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the reverse exchange needs >= 2 devices"
)

try:
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
except ImportError:  # pragma: no cover - older jax
    shard_map = None


def _ndev():
    return min(4, device_count())


def test_group_by_destination_orders_and_counts():
    """Padding must land past every real destination, not inside the counts."""
    owner = jnp.asarray([2, 0, -1, 1, 0, -1, 2], dtype=jnp.int32)
    order, sizes = group_by_destination(owner, 3)
    o = np.asarray(owner)[np.asarray(order)]
    live = o[o >= 0]
    assert list(live) == sorted(live), "live entries not in destination order"
    assert list(np.asarray(sizes)) == [2, 1, 2], np.asarray(sizes)
    assert int(np.asarray(sizes).sum()) == 5, "padding was counted as sendable"


def test_apply_reverse_halo_accumulates_rather_than_assigns():
    """Several remote devices can target the same local index; all must sum."""
    from yggdrax.distributed.reverse_halo import ReverseHaloResult

    res = ReverseHaloResult(
        target_index=jnp.asarray([1, 1, 2, 0], dtype=jnp.int32),
        values=jnp.asarray([[1.0, 0, 0], [2.0, 0, 0], [5.0, 0, 0], [7.0, 0, 0]]),
        n_received=jnp.asarray(3, dtype=jnp.int32),
        overflow=jnp.asarray(False),
    )
    out = apply_reverse_halo(jnp.zeros((3, 3)), res)
    got = np.asarray(out)[:, 0]
    assert got[1] == pytest.approx(3.0), "two contributions to index 1 did not sum"
    assert got[2] == pytest.approx(5.0)
    assert got[0] == pytest.approx(0.0), "a row past `n_received` was applied anyway"


@pytest.mark.skipif(shard_map is None, reason="needs jax.experimental.shard_map")
def test_every_contribution_reaches_its_owner_and_nothing_is_lost():
    """The global property: what is sent equals what is received, summed.

    Each device sends one contribution to every device (itself included), tagged so
    the receiver can check provenance. A dropped or duplicated row shows up in the
    global sum but NOT in any per-device one.
    """
    nd = _ndev()
    mesh = make_mesh(nd)
    cap = nd

    def body(dummy):
        me = jax.lax.axis_index(AXIS_NAME)
        owner = jnp.arange(cap, dtype=jnp.int32)
        target = jnp.zeros((cap,), dtype=jnp.int32)
        # value encodes the sender, so the received total is predictable
        vals = jnp.full((cap, 3), 1.0) * (me + 1).astype(jnp.float32)
        res = export_reverse_halo(owner, target, vals, nd, recv_capacity=cap * nd)
        got = apply_reverse_halo(jnp.zeros((1, 3)), res)
        return got + 0.0 * dummy

    out = shard_map(
        body,
        mesh=mesh,
        in_specs=(P(AXIS_NAME),),
        out_specs=P(AXIS_NAME),
        check_rep=False,
    )(jnp.zeros((nd,)))
    per_device = np.asarray(out).reshape(nd, 3)
    # every device receives one row from each sender: sum of (sender+1)
    expected = sum(range(1, nd + 1))
    for d in range(nd):
        assert per_device[d, 0] == pytest.approx(
            expected
        ), f"device {d} got {per_device[d, 0]}, expected {expected}"


@pytest.mark.skipif(shard_map is None, reason="needs jax.experimental.shard_map")
def test_momentum_cancels_globally_but_not_per_device():
    """The property the whole reverse halo exists for.

    Each device applies ``+f`` locally and exports ``-f`` to a neighbour. The GLOBAL
    sum must vanish; the per-device sums must NOT, since that is exactly the
    asymmetry a boundary creates and the reason a per-device assertion would pass
    while the force was wrong.
    """
    nd = _ndev()
    mesh = make_mesh(nd)

    def body(dummy):
        me = jax.lax.axis_index(AXIS_NAME)
        nxt = (me + 1) % nd
        f = jnp.asarray([[1.0, 2.0, -0.5]]) * (me + 1).astype(jnp.float32)
        local = f  # +f applied here
        res = export_reverse_halo(
            nxt[None].astype(jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            -f,  # -f owed to the neighbour
            nd,
            recv_capacity=nd,
        )
        total = apply_reverse_halo(local, res)
        return total + 0.0 * dummy

    out = shard_map(
        body,
        mesh=mesh,
        in_specs=(P(AXIS_NAME),),
        out_specs=P(AXIS_NAME),
        check_rep=False,
    )(jnp.zeros((nd,)))
    per_device = np.asarray(out).reshape(nd, 3)
    total = per_device.sum(axis=0)
    assert np.allclose(
        total, 0.0, atol=1e-6
    ), f"global momentum did not cancel: {total}"
    # and the guard against a vacuous pass: no single device is individually zero,
    # so the cancellation is genuinely global rather than trivially local
    assert not np.allclose(
        per_device, 0.0, atol=1e-6
    ), "per-device sums are all zero -- the test would pass without any exchange"


def test_halo_return_addresses_maps_coarse_positions_to_owner_local_indices():
    """The glue between the LET and the reverse exchange.

    A cross pair names its remote side by a position in the MERGED remote coarse
    tree, which has its own numbering. The `-f` has to land on a particle in the
    owning domain's own array, and the receiver cannot reconstruct which one -- so the
    address is derived where the tags are in scope. Getting this wrong would apply a
    correct force to the wrong particle: momentum would still conserve globally, and
    only a force comparison would notice.
    """
    from yggdrax.distributed.reverse_halo import halo_return_addresses

    # coarse position -> (origin domain, origin particle range)
    tag_domain = jnp.asarray([2, 2, 5, 7], dtype=jnp.int32)
    tag_range = jnp.asarray([[0, 4], [4, 8], [16, 20], [8, 12]], dtype=jnp.int32)

    coarse_pos = jnp.asarray([0, 1, 2, 3, 0], dtype=jnp.int32)
    within = jnp.asarray([0, 3, 1, 2, 9], dtype=jnp.int32)
    valid = jnp.asarray([True, True, True, True, False])

    owner, index = halo_return_addresses(
        coarse_pos, within, tag_domain, tag_range, valid
    )
    assert list(np.asarray(owner)) == [2, 2, 5, 7, -1]
    # index = tag_range[p, 0] + within_leaf, in the OWNER's local numbering
    assert list(np.asarray(index))[:4] == [0, 7, 17, 10]
    assert int(np.asarray(owner)[4]) == -1, "padding must be droppable"


def test_halo_return_addresses_round_trips_through_the_exchange():
    """An address derived from the tags must survive the exchange unchanged."""
    from yggdrax.distributed.reverse_halo import (
        ReverseHaloResult,
        apply_reverse_halo,
        halo_return_addresses,
    )

    tag_domain = jnp.asarray([0, 0], dtype=jnp.int32)
    tag_range = jnp.asarray([[2, 4], [6, 8]], dtype=jnp.int32)
    owner, index = halo_return_addresses(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([1, 0], dtype=jnp.int32),
        tag_domain,
        tag_range,
        jnp.asarray([True, True]),
    )
    # feed the derived addresses straight into the accumulate step
    res = ReverseHaloResult(
        target_index=index,
        values=jnp.asarray([[1.0, 0, 0], [2.0, 0, 0]]),
        n_received=jnp.asarray(2, dtype=jnp.int32),
        overflow=jnp.asarray(False),
    )
    out = np.asarray(apply_reverse_halo(jnp.zeros((8, 3)), res))[:, 0]
    assert out[3] == pytest.approx(1.0), "range start 2 + offset 1 -> index 3"
    assert out[6] == pytest.approx(2.0), "range start 6 + offset 0 -> index 6"
    assert out.sum() == pytest.approx(3.0), "a contribution landed somewhere else"
