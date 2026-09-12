"""Adaptive Morton-cell leaf partition with static shapes (``yggdrax._cell_partition``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import (
    MORTON_LEVELS,
    adaptive_cell_leaf_partition,
    adaptive_cell_leaf_partition_numpy,
)
from yggdrax.bounds import infer_bounds
from yggdrax.morton import morton_encode


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _sorted_codes(n, seed=0, duplicates=0):
    pos = _plummer(n, seed)
    if duplicates:
        pos[:duplicates] = pos[0]  # coincident particles: cells never split them
    P = jnp.asarray(pos, jnp.float32)
    codes = morton_encode(P, infer_bounds(P))
    return jnp.sort(codes)


def _check_partition(
    codes_np, starts, ends, depths, leaf_size, max_level=MORTON_LEVELS
):
    n = codes_np.shape[0]
    assert starts[0] == 0 and ends[-1] == n
    assert np.all(ends[:-1] == starts[1:]), "leaves tile the particles"
    occ = ends - starts
    assert np.all(occ >= 1)
    for s, e, d in zip(starts, ends, depths):
        shift = np.uint64(3 * (MORTON_LEVELS - d))
        cells = codes_np[s:e] >> shift
        assert np.all(cells == cells[0]), "a leaf lies within one cell of its depth"
        if d < max_level:
            assert e - s <= leaf_size
            # coarsest: the parent cell holds more than leaf_size particles
            if d > 0:
                pshift = np.uint64(3 * (MORTON_LEVELS - d + 1))
                pcell = codes_np[s] >> pshift
                assert np.sum((codes_np >> pshift) == pcell) > leaf_size


@pytest.mark.parametrize("leaf_size", [8, 32])
def test_numpy_reference_is_a_valid_coarsest_cell_partition(leaf_size):
    codes = np.asarray(_sorted_codes(4096, seed=1)).astype(np.uint64)
    starts, ends, depths = adaptive_cell_leaf_partition_numpy(
        codes, leaf_size=leaf_size
    )
    _check_partition(codes, starts, ends, depths, leaf_size)


@pytest.mark.parametrize("leaf_size", [8, 32])
def test_device_partition_matches_numpy_and_pads(leaf_size):
    codes = _sorted_codes(4096, seed=2)
    codes_np = np.asarray(codes).astype(np.uint64)
    s_ref, e_ref, d_ref = adaptive_cell_leaf_partition_numpy(
        codes_np, leaf_size=leaf_size
    )
    cap = int(s_ref.size) + 37
    part = adaptive_cell_leaf_partition(codes, leaf_size=leaf_size, capacity=cap)
    k = int(part.num_leaves)
    assert k == s_ref.size and not bool(part.overflow)
    assert np.array_equal(np.asarray(part.leaf_starts)[:k], s_ref)
    assert np.array_equal(np.asarray(part.leaf_ends)[:k], e_ref)
    assert np.array_equal(np.asarray(part.leaf_depths)[:k], d_ref)
    n = codes_np.shape[0]
    assert np.all(np.asarray(part.leaf_starts)[k:] == n)
    assert np.all(np.asarray(part.leaf_ends)[k:] == n)
    assert np.all(np.asarray(part.leaf_depths)[k:] == -1)


def test_overflow_is_flagged_not_silent():
    codes = _sorted_codes(2048, seed=3)
    s_ref, _, _ = adaptive_cell_leaf_partition_numpy(
        np.asarray(codes).astype(np.uint64), leaf_size=8
    )
    part = adaptive_cell_leaf_partition(
        codes, leaf_size=8, capacity=int(s_ref.size) // 2
    )
    assert bool(part.overflow)
    assert int(part.num_leaves) == s_ref.size


def test_coincident_particles_stop_at_the_deepest_level():
    codes = _sorted_codes(1024, seed=4, duplicates=40)
    codes_np = np.asarray(codes).astype(np.uint64)
    starts, ends, depths = adaptive_cell_leaf_partition_numpy(codes_np, leaf_size=8)
    occ = ends - starts
    assert occ.max() >= 40 and depths[np.argmax(occ)] == MORTON_LEVELS
    part = adaptive_cell_leaf_partition(codes, leaf_size=8, capacity=starts.size + 4)
    k = int(part.num_leaves)
    assert np.array_equal(np.asarray(part.leaf_starts)[:k], starts)


def test_partition_is_jittable_with_static_capacity():
    codes = _sorted_codes(1500, seed=5)
    f = jax.jit(lambda c: adaptive_cell_leaf_partition(c, leaf_size=16, capacity=512))
    part = f(codes)
    ref = adaptive_cell_leaf_partition(codes, leaf_size=16, capacity=512)
    for a, b in zip(part, ref):
        assert np.array_equal(np.asarray(a), np.asarray(b))
