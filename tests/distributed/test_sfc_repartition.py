"""``sfc_repartition``: re-partition a padded shard, checking every receiver's capacity first.

Four properties, each with a control that would fail if the test were vacuous:

* a PYTREE payload (velocities + int32 ids) travels with its particle and keeps its
  dtype -- ids above 2^24 stay exact, which a float32 column would not;
* a repartition that would overfill a device is DECLINED on every device and moves
  nothing, while the same particles with good sampling do move;
* the balance bound ``max count <= ceil(N/ndev) * (1 + 2*ndev/num_samples)`` holds;
* a device with no live particles does not drag the pivots to the padding sentinel.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_sfc_repartition.py -q --no-cov
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.partition import sfc_repartition
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the partition needs >= 2 devices"
)

ID_OFFSET = 2**30  # far beyond float32's exact-integer range (2^24)
BOX = (jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.float32))


def _plummer01(n, seed=0):
    """A Plummer sphere squeezed into the unit box (so BOX bounds it)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    r = np.minimum(r, 20.0) / 40.0
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return 0.5 + np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _shards(per_device, cap):
    """Flat padded arrays from a list of per-device (positions, velocities) blocks."""
    ndev = len(per_device)
    pos = np.zeros((ndev * cap, 3), np.float32)
    vel = np.zeros((ndev * cap, 3), np.float32)
    ids = np.full((ndev * cap,), -1, np.int32)
    counts = np.zeros(ndev, np.int32)
    nxt = 0
    for d, (p, v) in enumerate(per_device):
        k = len(p)
        assert k <= cap
        pos[d * cap : d * cap + k] = p
        vel[d * cap : d * cap + k] = v
        ids[d * cap : d * cap + k] = ID_OFFSET + np.arange(nxt, nxt + k)
        # padding sits on a live row of its own device, as the rollout puts it
        if k:
            pos[d * cap + k : (d + 1) * cap] = p[0]
        counts[d] = k
        nxt += k
    return pos, vel, ids, counts


def _repartition(pos, vel, ids, counts, cap, num_samples):
    ndev = len(counts)
    mesh = make_mesh(ndev)

    @jax.jit
    def go(pos, vel, ids, counts):
        def body(pos, vel, ids, counts):
            res = sfc_repartition(
                pos,
                jnp.where(jnp.arange(cap) < counts[0], 1.0, 0.0).astype(jnp.float32),
                counts[0],
                ndev,
                output_capacity=cap,
                bounds=BOX,
                num_samples=num_samples,
                axis_name=AXIS_NAME,
                payload={"v": vel, "id": ids},
                payload_fill={"v": 0.0, "id": -1},
            )
            return (
                res.positions[None],
                res.payload["v"][None],
                res.payload["id"][None],
                res.live_count[None],
                res.declined[None],
                res.recv_counts[None],
                res.sent_off_device[None],
            )

        return shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME),) * 4,
            out_specs=(P(AXIS_NAME),) * 7,
            check_vma=False,
        )(pos, vel, ids, counts)

    out = go(jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(ids), jnp.asarray(counts))
    return [np.asarray(o) for o in out]


@pytest.mark.parametrize("ndev", [2, 4])
def test_a_pytree_payload_travels_with_its_particle_and_stays_exact(ndev):
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    n = 2000
    pts = _plummer01(n)
    vel = np.random.default_rng(1).normal(size=(n, 3)).astype(np.float32)
    cap = int(math.ceil(1.3 * n / ndev))
    blocks = [
        (p, v) for p, v in zip(np.array_split(pts, ndev), np.array_split(vel, ndev))
    ]
    pos, v_in, ids, counts = _shards(blocks, cap)
    p_out, v_out, id_out, cnt, declined, _recv, sent = _repartition(
        pos, v_in, ids, counts, cap, 256
    )
    assert not declined.any()
    assert int(sent.sum()) > 0, "nothing moved, so nothing was tested"
    assert int(cnt.sum()) == n
    by_id_pos = {int(i): pos[k] for k, i in enumerate(ids) if i >= 0}
    by_id_vel = {int(i): v_in[k] for k, i in enumerate(ids) if i >= 0}
    seen = []
    for d in range(ndev):
        k = int(cnt[d])
        for row in range(k):
            gid = int(id_out[d, row])
            seen.append(gid)
            np.testing.assert_array_equal(p_out[d, row], by_id_pos[gid])
            np.testing.assert_array_equal(v_out[d, row], by_id_vel[gid])
        # padding carries the per-leaf fill, not a stale particle
        assert np.all(id_out[d, k:] == -1)
        assert np.all(v_out[d, k:] == 0.0)
    assert sorted(seen) == list(ID_OFFSET + np.arange(n))
    assert id_out.dtype == np.int32
    # NEGATIVE CONTROL: packing the same ids into a float32 column would lose them
    assert np.float32(ID_OFFSET + 1) == np.float32(ID_OFFSET)


def _skewed_line(n0, n1):
    """Particles on an x-line (Morton order = x order) so the pivot can be steered:
    device 0 spans x in (0, 1), device 1 sits just above x = 0.1."""
    x0 = np.linspace(0.01, 0.99, n0)
    x1 = np.linspace(0.10, 0.12, n1)
    line = lambda x: np.stack([x, np.full_like(x, 0.5), np.full_like(x, 0.5)], 1)
    zeros = lambda k: np.zeros((k, 3), np.float32)
    return [(line(x0), zeros(n0)), (line(x1), zeros(n1))]


def test_a_repartition_that_would_overflow_is_declined_everywhere():
    """With ONE sample per device the pivot is device 1's smallest code (x ~ 0.1), so
    device 1 would receive ~90 of device 0's rows plus its own 20 > cap 100."""
    cap = 100
    pos, vel, ids, counts = _shards(_skewed_line(100, 20), cap)
    p_out, v_out, id_out, cnt, declined, recv, sent = _repartition(
        pos, vel, ids, counts, cap, 1
    )
    assert declined.all(), "the decline must be the same on every device"
    assert recv[0].max() > cap, "the proposal must really overflow, or this is vacuous"
    assert int(sent.sum()) == 0
    np.testing.assert_array_equal(cnt, counts)
    for d in range(2):
        before = set(ids[d * cap : d * cap + counts[d]].tolist())
        after = set(id_out[d, : cnt[d]].tolist())
        assert (
            before == after
        ), "a declined repartition must keep every device's own set"


def test_the_same_particles_route_when_the_sampling_is_good():
    """CONTROL for the decline: enough samples balance the same input within cap."""
    cap = 100
    pos, vel, ids, counts = _shards(_skewed_line(100, 20), cap)
    _p, _v, id_out, cnt, declined, recv, sent = _repartition(
        pos, vel, ids, counts, cap, 256
    )
    assert not declined.any()
    assert recv[0].max() <= cap
    assert int(sent.sum()) > 0
    assert sorted(
        np.concatenate([id_out[d, : cnt[d]] for d in range(2)]).tolist()
    ) == sorted(ids[ids >= 0].tolist())


@pytest.mark.parametrize("ndev", [2, 4])
def test_the_balance_bound_holds(ndev):
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    n = 8000
    num_samples = max(256, 64 * ndev)
    pts = _plummer01(n, seed=3)
    cap = int(math.ceil(1.15 * n / ndev))
    zeros = np.zeros((n, 3), np.float32)
    blocks = list(zip(np.array_split(pts, ndev), np.array_split(zeros, ndev)))
    pos, vel, ids, counts = _shards(blocks, cap)
    _p, _v, _id, cnt, declined, _r, _s = _repartition(
        pos, vel, ids, counts, cap, num_samples
    )
    assert not declined.any()
    bound = math.ceil(n / ndev) * (1 + 2 * ndev / num_samples)
    assert cnt.max() <= bound, (cnt, bound)


def test_an_empty_device_does_not_skew_the_pivots():
    """All particles start on device 0. Device 1's samples are its padding sentinel;
    counted as samples they would put the pivot at the sentinel and send everything
    back to device 0. Excluded, the two halves come out balanced."""
    n, cap = 1000, 1000
    pts = _plummer01(n, seed=5)
    pos, vel, ids, counts = _shards(
        [(pts, np.zeros((n, 3), np.float32)), (np.zeros((0, 3)), np.zeros((0, 3)))], cap
    )
    _p, _v, _id, cnt, declined, _r, sent = _repartition(pos, vel, ids, counts, cap, 256)
    assert not declined.any()
    assert int(sent.sum()) > 0
    assert cnt.min() >= n // 2 * 0.9, cnt
