"""The repartition cadence: one verdict for the whole mesh, or it deadlocks.

``sfc_partition`` contains collectives. If one device repartitions and another does
not, the first blocks on a collective the second never enters. A test cannot observe a
deadlock without hanging, so what is tested is the property that prevents it: the
predicate returns the SAME value on every device, including when the devices disagree
about their own local state.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_repartition_cadence.py -q
"""

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
from yggdrax.distributed.partition import (
    global_bounds,
    maybe_repartition,
    repartition_due,
)
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the cadence predicate needs >= 2 devices"
)

_CAP = 100


def _verdicts(counts, step, interval=16, headroom=0.9):
    """`repartition_due` on every device, given per-device counts."""
    ndev = len(counts)
    mesh = make_mesh(ndev)

    @jax.jit
    def go(c):
        def body(c):
            return repartition_due(
                c[0],
                _CAP,
                jnp.asarray(step),
                interval=interval,
                headroom=headroom,
                axis_name=AXIS_NAME,
            )[None]

        return shard_map(
            body, mesh=mesh, in_specs=(P(AXIS_NAME),), out_specs=P(AXIS_NAME)
        )(c)

    return np.asarray(go(jnp.asarray(counts, jnp.int32)))


@pytest.mark.parametrize("ndev", [2, 4])
def test_one_device_near_capacity_makes_every_device_agree(ndev):
    """THE test. Only device 0 is full; all of them must still say yes.

    A predicate built on the LOCAL count would return True on device 0 and False
    everywhere else -- and then device 0 enters `sfc_partition`'s all_gather alone.
    """
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    counts = [95] + [10] * (ndev - 1)  # 0.95 of capacity against 0.10
    v = _verdicts(counts, step=3)  # not a scheduled step
    assert v.all(), f"drift guard must be mesh-wide, got {v}"
    assert len(set(v.tolist())) == 1


@pytest.mark.parametrize("ndev", [2, 4])
def test_no_device_near_capacity_and_not_due_means_nobody_repartitions(ndev):
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    v = _verdicts([10] * ndev, step=3)
    assert not v.any()
    assert len(set(v.tolist())) == 1


def test_the_schedule_fires_on_the_interval_and_agrees():
    if device_count() < 2:
        pytest.skip("needs 2 devices")
    for step, want in ((0, True), (1, False), (15, False), (16, True), (32, True)):
        v = _verdicts([10, 10], step=step, interval=16)
        assert bool(v[0]) is want, f"step {step}"
        assert len(set(v.tolist())) == 1


def test_headroom_is_the_boundary_and_is_read_from_the_worst_device():
    if device_count() < 2:
        pytest.skip("needs 2 devices")
    assert not _verdicts([89, 10], step=3, headroom=0.9).any()
    assert _verdicts([91, 10], step=3, headroom=0.9).all()
    # and it is the MAX that counts, not the mean: 91 and 10 average to 0.505
    assert _verdicts([10, 91], step=3, headroom=0.9).all()


def _shard(ndev, per, cap):
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(ndev, cap, 3)).astype(np.float32)
    mass = np.ones((ndev, cap), np.float32)
    gid = np.full((ndev, cap), -1, np.int32)
    counts = np.full(ndev, per, np.int32)
    for d in range(ndev):
        gid[d, :per] = np.arange(d * per, (d + 1) * per)
        pos[d, per:] = 0.0
    return (jnp.asarray(pos), jnp.asarray(mass), jnp.asarray(gid), jnp.asarray(counts))


@pytest.mark.parametrize("should", [False, True])
def test_maybe_repartition_keeps_the_shard_untouched_when_it_declines(should):
    """Both branches must produce the same shapes, and 'no' must mean no."""
    if device_count() < 2:
        pytest.skip("needs 2 devices")
    ndev, per, cap = 2, 40, 96
    pos, mass, gid, counts = _shard(ndev, per, cap)
    mesh = make_mesh(ndev)

    @jax.jit
    def go(pos, mass, gid, counts):
        def body(pos, mass, gid, counts):
            f = lambda x: x[0]
            b = global_bounds(f(pos), axis_name=AXIS_NAME)
            p, m, c, n, g = maybe_repartition(
                f(pos),
                f(mass),
                f(counts),
                jnp.asarray(should),
                ndev,
                output_capacity=cap,
                bounds=b,
                axis_name=AXIS_NAME,
                payload=f(gid),
            )
            return p[None], m[None], n[None], g[None]

        return shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME),) * 4,
            out_specs=(P(AXIS_NAME),) * 4,
        )(pos, mass, gid, counts)

    p, m, n, g = map(np.asarray, go(pos, mass, gid, counts))
    assert p.shape == (ndev, cap, 3) and g.shape == (ndev, cap)
    ids = np.concatenate([g[d, : int(n[d])] for d in range(ndev)])
    assert np.array_equal(
        np.sort(ids), np.arange(ndev * per)
    ), "every particle survives exactly once either way"
    if not should:
        assert np.array_equal(p, np.asarray(pos)), "declining must change nothing"
        assert np.array_equal(g, np.asarray(gid))
