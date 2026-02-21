"""Utilities for handling symmetric multipole tensors.

This module provides helpers for working with the packed triangular
representation discussed in recent FMM optimisation work.  The packed
layout stores all Cartesian symmetric tensor components of order ``l``
contiguously using a triangular indexing scheme, which avoids redundant
entries while remaining friendly to vectorised JAX code.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE

MAX_MULTIPOLE_ORDER = 4


def multi_index_tuples(level: int) -> tuple[tuple[int, int, int], ...]:
    """Return tuples ``(i, j, k)`` satisfying ``i + j + k = level``."""

    lvl = int(level)
    if lvl < 0:
        raise ValueError("level must be >= 0")
    combos = []
    for i in range(lvl + 1):
        for j in range(lvl + 1 - i):
            combos.append((i, j, lvl - i - j))
    return tuple(combos)


def multi_index_factorial(combo: tuple[int, int, int]) -> int:
    """Return ``i! * j! * k!`` for a multi-index tuple."""

    i, j, k = combo
    return math.factorial(i) * math.factorial(j) * math.factorial(k)


@jaxtyped(typechecker=beartype)
def multi_power(vec: Array, combo: tuple[int, int, int]) -> Array:
    """Return ``vec[0]^i * vec[1]^j * vec[2]^k`` for ``combo = (i, j, k)``."""

    value = jnp.array(1.0, dtype=vec.dtype)
    if combo[0]:
        value = value * vec[0] ** combo[0]
    if combo[1]:
        value = value * vec[1] ** combo[1]
    if combo[2]:
        value = value * vec[2] ** combo[2]
    return value


@jaxtyped(typechecker=beartype)
def level_size(level: int) -> int:
    """Return coefficient count for a symmetric tensor of order ``level``."""

    level_int = int(level)
    return (level_int + 1) * (level_int + 2) // 2


@jaxtyped(typechecker=beartype)
def level_offset(level: int) -> int:
    """Return the packed offset for order ``level``.

    Offsets accumulate contributions from lower orders using the closed
    form ``level(level+1)(level+2)/6``.
    """

    level_int = int(level)
    return (level_int * (level_int + 1) * (level_int + 2)) // 6


@jaxtyped(typechecker=beartype)
def total_coefficients(max_order: int) -> int:
    """Return total packed length for orders ``0..max_order`` inclusive."""

    order = int(max_order)
    return (order + 1) * (order + 2) * (order + 3) // 6


@jaxtyped(typechecker=beartype)
def triangular_index(level: int, i: int, j: int) -> int:
    """Map Cartesian indices to the packed triangular index.

    The mapping assumes ``i >= 0``, ``j >= 0`` and ``i + j <= level``.  The
    remaining index is ``k = level - i - j``.
    """

    lvl = int(level)
    ii = int(i)
    jj = int(j)
    if ii < 0 or jj < 0 or ii + jj > lvl:
        raise ValueError("Invalid triangular indices for given level")
    prefix = ii * (lvl + 1) - (ii * (ii - 1)) // 2
    return prefix + jj


@jaxtyped(typechecker=beartype)
def triangular_indices(level: int) -> Array:
    """Enumerate all ``(i, j, k)`` tuples for a given tensor order."""

    lvl = int(level)
    grid_i = jnp.arange(lvl + 1, dtype=INDEX_DTYPE)
    grid_j = jnp.arange(lvl + 1, dtype=INDEX_DTYPE)
    ii, jj = jnp.meshgrid(grid_i, grid_j, indexing="ij")
    mask = ii + jj <= lvl
    i_vals = ii[mask]
    j_vals = jj[mask]
    k_vals = lvl - i_vals - j_vals
    return jnp.stack([i_vals, j_vals, k_vals], axis=1)


@jaxtyped(typechecker=beartype)
def pack_tensor(level: int, tensor: Array) -> Array:
    """Pack a symmetric Cartesian tensor of order ``level``.

    Parameters
    ----------
    level:
        Tensor order ``l``.
    tensor:
        Array of shape ``(l + 1, l + 1, l + 1)`` containing Cartesian
        components.  Only entries with ``i + j + k = l`` are read.

    Returns
    -------
    Array
        1-D flattened packed representation with length ``level_size(level)``.
    """

    lvl = int(level)
    if tensor.shape != (lvl + 1, lvl + 1, lvl + 1):
        raise ValueError("tensor must have shape (level+1, level+1, level+1)")
    idx = triangular_indices(lvl)
    i_vals, j_vals, k_vals = idx[:, 0], idx[:, 1], idx[:, 2]
    return tensor[i_vals, j_vals, k_vals]


@jaxtyped(typechecker=beartype)
def unpack_tensor(level: int, data: Array) -> Array:
    """Unpack a packed triangular buffer back into Cartesian components."""

    lvl = int(level)
    expected = level_size(lvl)
    if data.shape[-1] != expected:
        raise ValueError(f"Packed data length {data.shape[-1]} != expected {expected}")
    tensor = jnp.zeros((lvl + 1, lvl + 1, lvl + 1), dtype=data.dtype)
    idx = triangular_indices(lvl)
    i_vals, j_vals, k_vals = idx[:, 0], idx[:, 1], idx[:, 2]
    tensor = tensor.at[i_vals, j_vals, k_vals].set(data)
    return tensor


LOCAL_LEVEL_COMBOS = {
    level: multi_index_tuples(level) for level in range(MAX_MULTIPOLE_ORDER + 1)
}


LOCAL_COMBO_INV_FACTORIAL = {
    combo: 1.0 / multi_index_factorial(combo)
    for combos in LOCAL_LEVEL_COMBOS.values()
    for combo in combos
}


__all__ = [
    "MAX_MULTIPOLE_ORDER",
    "LOCAL_COMBO_INV_FACTORIAL",
    "LOCAL_LEVEL_COMBOS",
    "multi_index_tuples",
    "multi_index_factorial",
    "multi_power",
    "level_size",
    "level_offset",
    "total_coefficients",
    "triangular_index",
    "triangular_indices",
    "pack_tensor",
    "unpack_tensor",
]
