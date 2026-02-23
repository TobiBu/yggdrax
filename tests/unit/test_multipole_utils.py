"""Tests for multipole tensor utility helpers."""

import jax.numpy as jnp
import pytest

from yggdrax.multipole_utils import (
    level_offset,
    level_size,
    multi_index_factorial,
    multi_index_tuples,
    multi_power,
    pack_tensor,
    total_coefficients,
    triangular_index,
    triangular_indices,
    unpack_tensor,
)


def test_multi_index_tuples_and_factorial():
    combos = multi_index_tuples(3)
    assert len(combos) == level_size(3)
    assert all(sum(combo) == 3 for combo in combos)
    assert multi_index_factorial((2, 1, 0)) == 2


def test_multi_index_tuples_raises_for_negative_level():
    with pytest.raises(ValueError):
        multi_index_tuples(-1)


def test_multi_power_matches_direct_product():
    vec = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)
    value = multi_power(vec, (2, 1, 0))
    assert jnp.isclose(value, 12.0)


def test_level_offsets_and_totals_are_consistent():
    assert level_size(0) == 1
    assert level_size(1) == 3
    assert level_size(2) == 6
    assert level_offset(0) == 0
    assert level_offset(1) == 1
    assert level_offset(2) == 4
    assert total_coefficients(2) == 10


def test_triangular_index_and_indices_match():
    idx = triangular_indices(3)
    for row, (i_val, j_val, _k_val) in enumerate(idx.tolist()):
        assert triangular_index(3, int(i_val), int(j_val)) == row


@pytest.mark.parametrize(
    "args",
    [
        (2, -1, 0),
        (2, 0, -1),
        (2, 2, 2),
    ],
)
def test_triangular_index_raises_on_invalid_input(args):
    with pytest.raises(ValueError):
        triangular_index(*args)


def test_pack_and_unpack_round_trip():
    level = 2
    tensor = jnp.zeros((level + 1, level + 1, level + 1), dtype=jnp.float64)
    idx = triangular_indices(level)
    for n, (i_val, j_val, k_val) in enumerate(idx.tolist()):
        tensor = tensor.at[int(i_val), int(j_val), int(k_val)].set(float(n + 1))

    packed = pack_tensor(level, tensor)
    restored = unpack_tensor(level, packed)

    assert packed.shape == (level_size(level),)
    assert jnp.allclose(restored, tensor)


def test_pack_tensor_raises_for_shape_mismatch():
    with pytest.raises(ValueError):
        pack_tensor(2, jnp.zeros((2, 2, 2), dtype=jnp.float32))


def test_unpack_tensor_raises_for_invalid_length():
    with pytest.raises(ValueError):
        unpack_tensor(2, jnp.zeros((5,), dtype=jnp.float32))
