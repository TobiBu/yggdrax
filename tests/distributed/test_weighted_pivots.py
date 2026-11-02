"""``_choose_pivots``: samples weighted by the particles they stand for.

Each device samples ``num_samples`` codes at even ranks of its live prefix. Until
2026-11 every sample weighed the same, so on nearly exact Morton ranges the pivot sat
at the boundary between the devices' sample sets whatever their counts, and the
imbalance grew with every repartition (two forced CPU devices, Plummer 2e5 drifting
8 KDK steps of 5e-3 between repartitions: +0.14 % -> +1.34 % of the mean over 40;
four devices, 4e5: +0.67 % -> +14.9 %, then every repartition declined at 1.15x
capacity). Now device ``d``'s samples weigh ``count_d``. Four properties:

* equal live counts (any number of empty devices, any ``align_level``) and
  ``count=None`` give pivots BIT-IDENTICAL to the unweighted rule, kept below
  verbatim as the reference -- so balanced decompositions keep their bits;
* a repartition of exact ranges at 60 / 40 % moves the cut back to within one sample
  block of N / 2 (the CONTROL: the unweighted rule moves nothing);
* every device empty still yields the padding sentinel, as before;
* the pivots do not depend on which device holds which counts' samples first
  (ties between equal codes cannot change a pivot's value).

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_weighted_pivots.py -q --no-cov
"""

from typing import Optional

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
from yggdrax.distributed import partition as part
from yggdrax.distributed.comm import _COUNT_DTYPE
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the pivots need >= 2 devices"
)

SENTINEL = np.uint64((1 << 64) - 1)


def _unweighted_pivots(
    codes_sorted, ndev, num_samples, axis_name, align_level, count=None
):
    """yggdrax cfc0e73's ``_choose_pivots`` (equal weight per sample), verbatim."""
    n = codes_sorted.shape[0] if count is None else count
    idx = (jnp.arange(num_samples) * jnp.asarray(n)) // num_samples
    samples = codes_sorted[idx]
    total = num_samples * ndev
    if count is not None:
        live_dev = (jnp.asarray(count) > 0).astype(_COUNT_DTYPE)
        samples = jnp.where(live_dev > 0, samples, part._CODE_SENTINEL)
        total = num_samples * jnp.maximum(jax.lax.psum(live_dev, axis_name), 1)
    all_samples = jnp.sort(jax.lax.all_gather(samples, axis_name, tiled=True))
    piv_idx = (jnp.arange(1, ndev) * total) // ndev
    return part._align_pivots(all_samples[piv_idx], align_level)


def _pivots(fn, codes, counts, num_samples, align_level=None, with_count=True):
    """Run a pivot rule on every device of a mesh; the replicated pivots of device 0."""
    ndev = len(counts)
    mesh = make_mesh(ndev)

    def body(c, cnt):
        piv = fn(
            c,
            ndev,
            num_samples,
            AXIS_NAME,
            align_level,
            count=cnt[0] if with_count else None,
        )
        return piv[None]

    out = jax.jit(
        shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
            out_specs=P(AXIS_NAME),
            check_vma=False,
        )
    )(jnp.asarray(codes), jnp.asarray(np.asarray(counts, np.int32)))
    out = np.asarray(out)
    assert all(np.array_equal(out[0], o) for o in out), "pivots must be replicated"
    return out[0]


def _shards(counts, cap, rng, *, ranges: Optional[list] = None):
    """Flat padded code shards: each device's live codes sorted, sentinel tail."""
    codes = np.full((len(counts) * cap,), SENTINEL, np.uint64)
    for d, k in enumerate(counts):
        lo, hi = (0, 1 << 62) if ranges is None else ranges[d]
        live = np.sort(rng.integers(lo, hi, size=k, dtype=np.uint64))
        codes[d * cap : d * cap + k] = live
    return codes


def _ndevs():
    return [d for d in (2, 3, 4) if device_count() >= d]


@pytest.mark.parametrize("num_samples", [1, 7, 64, 256])
@pytest.mark.parametrize("align_level", [None, 3])
def test_equal_counts_give_the_unweighted_pivots_bitwise(num_samples, align_level):
    rng = np.random.default_rng(11)
    for ndev in _ndevs():
        for n in (1, 5, 1000, 4097):
            cap = n + 13
            codes = _shards([n] * ndev, cap, rng)
            want = _pivots(
                _unweighted_pivots, codes, [n] * ndev, num_samples, align_level
            )
            got = _pivots(
                part._choose_pivots, codes, [n] * ndev, num_samples, align_level
            )
            np.testing.assert_array_equal(got, want)
            assert got.dtype == np.uint64


@pytest.mark.parametrize("num_samples", [7, 256])
def test_equal_counts_with_empty_devices_give_the_unweighted_pivots_bitwise(
    num_samples,
):
    rng = np.random.default_rng(12)
    for ndev in _ndevs():
        n, cap = 777, 800
        for empty in range(1, ndev):
            counts = [n] * (ndev - empty) + [0] * empty
            codes = _shards(counts, cap, rng)
            want = _pivots(_unweighted_pivots, codes, counts, num_samples)
            got = _pivots(part._choose_pivots, codes, counts, num_samples)
            np.testing.assert_array_equal(got, want)


def test_count_none_gives_the_unweighted_pivots_bitwise():
    rng = np.random.default_rng(13)
    for ndev in _ndevs():
        cap = 512
        codes = _shards([cap] * ndev, cap, rng)
        for num_samples in (8, 256):
            want = _pivots(
                _unweighted_pivots, codes, [cap] * ndev, num_samples, with_count=False
            )
            got = _pivots(
                part._choose_pivots, codes, [cap] * ndev, num_samples, with_count=False
            )
            np.testing.assert_array_equal(got, want)


def test_every_device_empty_still_yields_the_sentinel():
    for ndev in _ndevs():
        codes = np.full((ndev * 16,), SENTINEL, np.uint64)
        got = _pivots(part._choose_pivots, codes, [0] * ndev, 8)
        want = _pivots(_unweighted_pivots, codes, [0] * ndev, 8)
        np.testing.assert_array_equal(got, want)
        assert np.all(got == SENTINEL)


def test_unequal_exact_ranges_are_rebalanced():
    """Device 0 owns the lower 60 % of the codes, device 1 the upper 40 %: exact
    ranges, unequal counts. The weighted cut lands within one sample block of the
    global median; the CONTROL (the unweighted rule) keeps the old cut, 20 % off."""
    rng = np.random.default_rng(14)
    n0, n1, cap, s = 6000, 4000, 7000, 256
    codes = _shards([n0, n1], cap, rng, ranges=[(0, 1 << 40), (1 << 40, 1 << 41)])
    live = np.concatenate([codes[:n0], codes[cap : cap + n1]])
    block = max(n0, n1) // s + 1

    def left(piv):
        # sfc_repartition's rule: dest = searchsorted(pivots, code, 'right')
        return int(np.sum(np.searchsorted(piv, live, side="right") == 0))

    got = left(_pivots(part._choose_pivots, codes, [n0, n1], s))
    old = left(_pivots(_unweighted_pivots, codes, [n0, n1], s))
    assert abs(got - (n0 + n1) // 2) <= 2 * block, got
    assert old == n0, "the control must reproduce the unweighted rule's stuck cut"


def test_the_pivots_do_not_depend_on_which_device_samples_first():
    """Swap the two devices' shards: the pooled samples and weights are the same
    multiset, so the pivots must be too -- including codes duplicated across devices."""
    rng = np.random.default_rng(15)
    cap = 3000
    shared = np.sort(rng.integers(0, 1 << 20, size=500, dtype=np.uint64))
    a = np.sort(np.concatenate([shared, rng.integers(0, 1 << 20, 2000, np.uint64)]))
    b = np.sort(np.concatenate([shared, rng.integers(0, 1 << 20, 300, np.uint64)]))

    def pack(first, second):
        codes = np.full((2 * cap,), SENTINEL, np.uint64)
        codes[: len(first)] = first
        codes[cap : cap + len(second)] = second
        return codes, [len(first), len(second)]

    for s in (7, 64):
        p_ab = _pivots(part._choose_pivots, *pack(a, b), s)
        p_ba = _pivots(part._choose_pivots, *pack(b, a), s)
        np.testing.assert_array_equal(p_ab, p_ba)
