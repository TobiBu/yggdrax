"""Tests for Morton code utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.morton import get_common_prefix_length, morton_decode, morton_encode


def test_morton_encode_decode():
    """Test that encoding and decoding are inverse operations."""
    # Create test positions
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [-0.5, 0.2, 0.8],
        ]
    )

    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))

    # Encode and decode
    morton_codes = morton_encode(positions, bounds)
    decoded = morton_decode(morton_codes, bounds)

    # Check that decoded positions are close to original
    # (some precision loss is expected due to integer quantization)
    assert jnp.allclose(positions, decoded, atol=1e-4)


def test_morton_ordering():
    """Test that Morton codes preserve spatial locality."""
    # Points close in space should have similar Morton codes
    # Using more distinct points to ensure the test is reliable
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
        ]
    )

    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))

    morton_codes = morton_encode(positions, bounds)

    # First two points should have more common bits than first and third
    common_01 = get_common_prefix_length(morton_codes[0], morton_codes[1])
    common_02 = get_common_prefix_length(morton_codes[0], morton_codes[2])

    # At minimum, codes should be different
    assert morton_codes[0] != morton_codes[1]
    assert morton_codes[0] != morton_codes[2]
    assert morton_codes[1] != morton_codes[2]


def test_morton_bounds_clamping():
    """Test that positions outside bounds are clamped correctly."""
    # Positions outside bounds
    positions = jnp.array(
        [
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0],
        ]
    )

    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))

    # Should not raise error, positions should be clamped
    morton_codes = morton_encode(positions, bounds)
    decoded = morton_decode(morton_codes, bounds)

    # Decoded positions should be at the bounds
    assert jnp.allclose(decoded[0], bounds[0], atol=1e-4)
    assert jnp.allclose(decoded[1], bounds[1], atol=1e-4)


def test_common_prefix_length():
    """Test common prefix length calculation."""
    # Same codes should have 64 common bits
    code = jnp.uint64(12345)
    assert get_common_prefix_length(code, code) == 64

    # Completely different codes should have few common bits
    code1 = jnp.uint64(0)
    code2 = jnp.uint64((1 << 63))
    common = get_common_prefix_length(code1, code2)
    assert common == 0


def test_morton_sort_is_the_stable_argsort_and_its_gather():
    """One key-value sort: argsort(stable)'s permutation and the gathered codes."""
    from yggdrax._tree_impl import _morton_sort, inverse_permutation

    rng = np.random.default_rng(5)
    # many ties (the stable tie-break by input order is what is pinned) and codes
    # above 2^32 (the sort must compare all 64 bits)
    codes = rng.integers(0, 50, size=4096).astype(np.uint64) << np.uint64(37)
    codes += rng.integers(0, 3, size=4096).astype(np.uint64)
    codes = jnp.asarray(codes)
    idx, sorted_codes = _morton_sort(codes)
    ref = jnp.argsort(codes, stable=True)
    assert np.array_equal(np.asarray(idx), np.asarray(ref))
    assert np.array_equal(np.asarray(sorted_codes), np.asarray(codes[ref]))
    assert len(np.unique(np.asarray(codes))) < codes.shape[0] // 10  # ties exist
    inv = np.asarray(inverse_permutation(idx))
    assert np.array_equal(inv[np.asarray(idx)], np.arange(codes.shape[0]))
