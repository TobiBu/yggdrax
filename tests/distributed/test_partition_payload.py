"""``sfc_partition(payload=...)``: a particle's identity must travel WITH the particle.

The partition permutes twice -- a local Morton sort, then a ragged route, then a
re-sort -- and pads to a static capacity. A host-side id array indexed by row matches
the shard only while ``capacity == count``; once it does not, it is silently wrong.
`docs/distributed_padding_force_defect.md` records what that looks like: "plausible,
smooth, and wrong by tens of percent".

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_partition_payload.py -q
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
from yggdrax.distributed.partition import global_bounds, sfc_partition
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the partition needs >= 2 devices"
)


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _run(ndev, n, capacity, align_level=None):
    mesh = make_mesh(ndev)
    pos = jnp.asarray(_plummer(n), jnp.float32)
    mass = jnp.asarray(np.linspace(1.0, 2.0, n), jnp.float32)
    gid = jnp.arange(n, dtype=jnp.int32)

    @jax.jit
    def go(pos, mass, gid):
        def body(pos, mass, gid):
            b = global_bounds(pos, axis_name=AXIS_NAME)
            p, m, c, cnt, g = sfc_partition(
                pos,
                mass,
                ndev,
                output_capacity=capacity,
                bounds=b,
                align_level=align_level,
                axis_name=AXIS_NAME,
                payload=gid,
            )
            return p[None], m[None], cnt[None], g[None]

        return shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)),
            out_specs=(P(AXIS_NAME),) * 4,
        )(pos, mass, gid)

    p, m, cnt, g = map(np.asarray, go(pos, mass, gid))
    return np.asarray(pos), np.asarray(mass), p, m, cnt, g


@pytest.mark.parametrize("ndev", [2, 4])
def test_the_routed_id_identifies_the_particle_it_arrived_with(ndev):
    """Every shard row's id must name the particle whose position and mass it holds.

    This is the check a host-side id array fails: it would have to reproduce a local
    sort, a ragged route and a re-sort, and it cannot, because particles sharing a
    Morton code are not distinguishable from the codes alone.
    """
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    n, cap = 4 * ndev * 40, 4 * 40 * 3
    pos0, mass0, p, m, cnt, g = _run(ndev, n, cap)
    seen = []
    for d in range(ndev):
        k = int(cnt[d])
        ids = g[d, :k]
        assert np.all(ids >= 0) and np.all(ids < n)
        assert np.allclose(p[d, :k], pos0[ids], atol=0), "id does not match position"
        assert np.allclose(m[d, :k], mass0[ids], atol=0), "id does not match mass"
        seen.append(ids)
    allids = np.concatenate(seen)
    assert np.array_equal(
        np.sort(allids), np.arange(n)
    ), "every particle exactly once, across all devices"


@pytest.mark.parametrize("ndev", [2, 4])
def test_the_padding_carries_no_identity(ndev):
    if device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    n, cap = 4 * ndev * 40, 4 * 40 * 3
    _p0, _m0, _p, _m, cnt, g = _run(ndev, n, cap)
    for d in range(ndev):
        assert np.all(g[d, int(cnt[d]) :] == -1)


def test_capacity_far_above_count_is_where_a_host_side_id_would_break():
    """The regime the argument exists for, made explicit.

    With capacity much larger than the count the shard row index and the input row
    index have nothing to do with each other, so `gid[row]` on the host is nonsense
    while the routed id is still exact.
    """
    if device_count() < 2:
        pytest.skip("needs 2 devices")
    n, cap = 200, 400  # capacity 4x the per-device count
    pos0, mass0, p, m, cnt, g = _run(2, n, cap)
    for d in range(2):
        k = int(cnt[d])
        assert k < cap // 2, "the test needs capacity well above count to be honest"
        assert np.allclose(p[d, :k], pos0[g[d, :k]], atol=0)
        naive = np.arange(k)  # what a host-side gid array would have said
        assert not np.array_equal(
            naive, g[d, :k]
        ), "if these agreed the test would prove nothing"


def test_align_level_still_partitions_every_particle_once():
    if device_count() < 2:
        pytest.skip("needs 2 devices")
    n, cap = 4 * 2 * 40, 4 * 40 * 3
    _p0, _m0, _p, _m, cnt, g = _run(2, n, cap, align_level=3)
    ids = np.concatenate([g[d, : int(cnt[d])] for d in range(2)])
    assert np.array_equal(np.sort(ids), np.arange(n))
