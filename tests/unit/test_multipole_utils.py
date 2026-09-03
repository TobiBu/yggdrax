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


# `multi_power` and the public M2M above it both document a length-3 vector and neither
# enforced it. JAX CLAMPS an out-of-bounds index, so `vec[2]` on a 2-vector silently
# returns `vec[-1]` and every violation is a valid gather afterwards -- no body-level
# check can fire. Measured before the annotations:
#
#     multi_power([2, 3], (1, 1, 1))                  -> 18.0, where [2, 3, 5] gives 30.0
#     multi_power([[2, 3, 5]], (1, 1, 1))             -> [8, 27, 125], a VECTOR
#     translate_packed_moments(packed, [.5, .25], 2)  -> a valid-looking (10,) result that
#                                                        is EXACTLY the answer for
#                                                        (x, y, y), i.e. the clamp


def test_multi_power_requires_a_length_three_vector():
    """A 2- or 4-vector has no correct monomial, and clamping returned one anyway."""
    from jaxtyping import TypeCheckError

    vec = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)
    assert jnp.isclose(multi_power(vec, (1, 1, 1)), 30.0)

    for bad in (vec[:2], jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)):
        with pytest.raises(TypeCheckError):
            multi_power(bad, (1, 1, 1))


def test_multi_power_requires_rank_one():
    """``[[2, 3, 5]]`` returned each component cubed -- an array where a scalar belongs."""
    from jaxtyping import TypeCheckError

    vec = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)
    for bad in (vec[None, :], vec[:, None]):
        with pytest.raises(TypeCheckError):
            multi_power(bad, (1, 1, 1))


def test_multi_power_still_accepts_a_complex_vector():
    """``Inexact`` and not ``Float``: ``jnp.array(1.0, dtype=vec.dtype)`` allows complex.

    ``translate_packed_moments`` casts ``delta`` to the coefficient dtype, so a complex
    packed expansion hands ``multi_power`` a complex vector. Narrowing to ``Float`` would
    break that path; every corruption measured above is a shape, not a dtype.
    """
    vec = jnp.array([2.0 + 1.0j, 3.0, 5.0])
    assert jnp.isclose(multi_power(vec, (1, 0, 0)), 2.0 + 1.0j)


def test_translate_packed_moments_requires_a_length_three_delta():
    """The public M2M, where the clamp produced a plausible wrong translation.

    A ``(2,)`` delta came back as a well-shaped ``(10,)`` expansion identical to the one
    for ``(x, y, y)``. A ``(4,)`` delta silently ignored the surplus component. Both are
    now rejected; the rank cases already raised a ``TypeError`` from a concatenate.
    """
    from jaxtyping import TypeCheckError

    from yggdrax.tree_moments import translate_packed_moments

    packed = jnp.arange(1, total_coefficients(2) + 1, dtype=jnp.float64)
    good = jnp.array([0.5, 0.25, 0.125], dtype=jnp.float64)
    # Non-vacuity: the documented call must still work.
    assert translate_packed_moments(packed, good, 2).shape == (10,)

    for bad in (good[:2], jnp.array([0.5, 0.25, 0.125, 0.0625], dtype=jnp.float64)):
        with pytest.raises(TypeCheckError):
            translate_packed_moments(packed, bad, 2)
