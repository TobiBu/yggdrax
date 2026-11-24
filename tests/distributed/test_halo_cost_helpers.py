"""The halo's live-sized replacements equal what they replace, exactly.

``prefix_argsort`` (send buffers), ``_sorted_counts`` (partition),
``_sort_nearly_sorted_rows`` (kNN rows) and ``_true_rows`` (fast kNN buckets)
exist only to make a stage cost what is live instead of what is allocated; each
must return the same array as the general operation it stands in for.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed.export import prefix_argsort
from yggdrax.distributed.partition import _sorted_counts
from yggdrax.distributed.stats import _sort_nearly_sorted_rows
from yggdrax.fast_knn import _true_rows


@pytest.mark.parametrize(
    "P, count",
    [
        (100, 37),
        (5000, 0),
        (5000, 4096),
        (5000, 4097),
        (70000, 16000),
        (70000, 70000),
        (70000, 90000),
    ],
)
def test_prefix_argsort_equals_stable_argsort(P, count):
    rng = np.random.default_rng(P + count)
    big = 1_000_000
    live = min(count, P)
    key = np.full(P, big, np.int32)
    key[:live] = rng.integers(0, 300, live)  # many ties: stability matters
    key[:live:7] = big  # dead rows inside the live prefix too
    got = jax.jit(prefix_argsort)(jnp.asarray(key), jnp.asarray(count))
    np.testing.assert_array_equal(np.asarray(got), np.argsort(key, kind="stable"))


def test_sorted_counts_equals_bincount():
    rng = np.random.default_rng(0)
    ndev = 6
    dest = np.sort(rng.integers(0, ndev + 1, 1000)).astype(np.int32)  # ndev = dead
    got = np.asarray(_sorted_counts(jnp.asarray(dest), ndev))
    np.testing.assert_array_equal(got, np.bincount(dest, minlength=ndev + 1)[:ndev])


@pytest.mark.parametrize("k", [1, 2, 7, 8, 32])
def test_nearly_sorted_rows_equals_stable_argsort(k):
    rng = np.random.default_rng(k)
    n = 500
    d = np.sort(rng.uniform(0, 1, (n, k)), axis=1)
    # near-ties swapped, a few rows reversed, infs and exact ties
    swap = rng.uniform(size=n) < 0.3
    if k > 1:
        d[swap, :2] = d[swap, 1::-1]
        d[:5] = d[:5, ::-1]
        d[10:20, k // 2 :] = np.inf
        d[20:30, 0] = d[20:30, -1]
    i = np.tile(np.arange(k), (n, 1)) + 100 * np.arange(n)[:, None]
    gd, gi = jax.jit(_sort_nearly_sorted_rows)(jnp.asarray(d), jnp.asarray(i))
    order = np.argsort(d, axis=1, kind="stable")
    np.testing.assert_array_equal(np.asarray(gd), np.take_along_axis(d, order, 1))
    np.testing.assert_array_equal(np.asarray(gi), np.take_along_axis(i, order, 1))


@pytest.mark.parametrize("size", [1, 50, 400, 2000])
def test_true_rows_equals_nonzero(size):
    rng = np.random.default_rng(size)
    flag = rng.uniform(size=1000) < 0.3
    got = np.asarray(_true_rows(jnp.asarray(flag), size, 1000))
    ref = np.asarray(jnp.nonzero(jnp.asarray(flag), size=size, fill_value=1000)[0])
    np.testing.assert_array_equal(got, ref)
