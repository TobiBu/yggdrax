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


# --- capacity-padded shards (the distributed lane) ------------------------------

#: Dead rows sort last only if they carry the maximal Morton code.
_SENTINEL = np.uint64(2**63 - 1)


def _pad_with_sentinel(codes, rows):
    """Capacity-pad sorted codes the way a distributed shard is padded."""
    codes = np.asarray(codes, np.uint64)
    return jnp.asarray(
        np.concatenate([codes, np.full(rows - codes.size, _SENTINEL, np.uint64)])
    )


def _live_triples(part):
    k = int(part.num_leaves)
    return set(
        zip(
            np.asarray(part.leaf_starts)[:k].tolist(),
            np.asarray(part.leaf_ends)[:k].tolist(),
            np.asarray(part.leaf_depths)[:k].tolist(),
        )
    )


@pytest.mark.parametrize("n_live, rows", [(2000, 4096), (5000, 8192), (50, 4096)])
def test_num_valid_reproduces_the_unpadded_partition(n_live, rows):
    """A padded shard partitions exactly like the same particles unpadded.

    This is the contract the distributed lane needs: one device's shard is a
    capacity-padded array, and its tree must not depend on how much padding
    happens to follow the live rows.
    """
    codes = _sorted_codes(n_live)
    padded = _pad_with_sentinel(codes, rows)

    unpadded = adaptive_cell_leaf_partition(codes, leaf_size=64, capacity=1024)
    shard = adaptive_cell_leaf_partition(
        padded, leaf_size=64, capacity=1024, num_valid=n_live
    )

    assert not bool(shard.overflow)
    assert int(shard.num_leaves) == int(unpadded.num_leaves)
    assert _live_triples(shard) == _live_triples(unpadded)


def test_the_sentinel_alone_builds_one_oversized_padding_leaf():
    """Why ``num_valid`` exists, pinned as a negative control.

    Sorting the dead rows last is necessary but not sufficient: they all carry
    the *same* code, so no Morton level ever splits them and they collapse into
    a single leaf far wider than ``leaf_size``. Under trace that leaf truncates
    into a live, mass-0, radius-0 leaf at a real position, which fails the MAC
    against everything near it. If this test ever starts failing because the
    naive partition became well behaved, ``num_valid`` may be reconsidered --
    until then it is load bearing.
    """
    leaf_size, n_live, rows = 64, 5000, 8192
    padded = _pad_with_sentinel(_sorted_codes(n_live), rows)

    naive = adaptive_cell_leaf_partition(padded, leaf_size=leaf_size, capacity=1024)
    widest = int(np.max(np.asarray(naive.leaf_ends) - np.asarray(naive.leaf_starts)))
    assert widest > leaf_size, "expected the padding run to exceed one leaf"
    assert widest >= rows - n_live

    guarded = adaptive_cell_leaf_partition(
        padded, leaf_size=leaf_size, capacity=1024, num_valid=n_live
    )
    k = int(guarded.num_leaves)
    occ = np.asarray(guarded.leaf_ends)[:k] - np.asarray(guarded.leaf_starts)[:k]
    assert int(occ.max()) <= leaf_size
    assert int(np.asarray(guarded.leaf_ends)[:k].max()) == n_live


def test_num_valid_is_traceable_and_default_is_unchanged():
    """``num_valid`` arrives as a tracer under ``shard_map``; ``None`` is a no-op."""
    n_live, rows = 5000, 8192
    padded = _pad_with_sentinel(_sorted_codes(n_live), rows)

    fn = jax.jit(
        lambda c, nv: adaptive_cell_leaf_partition(
            c, leaf_size=64, capacity=1024, num_valid=nv
        )
    )
    traced = fn(padded, jnp.asarray(n_live, jnp.int32))
    eager = adaptive_cell_leaf_partition(
        padded, leaf_size=64, capacity=1024, num_valid=n_live
    )
    for field in eager._fields:
        assert jnp.array_equal(getattr(traced, field), getattr(eager, field))

    # Fewer live rows through the SAME compiled program must give fewer leaves.
    assert int(fn(padded, jnp.asarray(2000, jnp.int32)).num_leaves) < int(
        traced.num_leaves
    )

    codes = _sorted_codes(n_live)
    omitted = adaptive_cell_leaf_partition(codes, leaf_size=64, capacity=1024)
    explicit = adaptive_cell_leaf_partition(
        codes, leaf_size=64, capacity=1024, num_valid=codes.shape[0]
    )
    for field in omitted._fields:
        assert jnp.array_equal(getattr(omitted, field), getattr(explicit, field))
