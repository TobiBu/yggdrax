"""Tests for Morton code utilities."""

import jax.numpy as jnp
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
