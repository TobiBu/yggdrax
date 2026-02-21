"""Morton code utilities for 3D spatial ordering."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

U64 = jnp.uint64

MASK_21 = U64(0x1FFFFF)
M1 = U64(0x1F00000000FFFF)
M2 = U64(0x1F0000FF0000FF)
M3 = U64(0x100F00F00F00F00F)
M4 = U64(0x10C30C30C30C30C3)
M5 = U64(0x1249249249249249)

Bounds = tuple[Float[Array, "3"], Float[Array, "3"]]


def _spread3_u64(x_u64: jnp.ndarray) -> jnp.ndarray:
    x = jnp.bitwise_and(x_u64, MASK_21)
    x = (x | (x << U64(32))) & M1
    x = (x | (x << U64(16))) & M2
    x = (x | (x << U64(8))) & M3
    x = (x | (x << U64(4))) & M4
    x = (x | (x << U64(2))) & M5
    return x


def _compact3_u64(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.bitwise_and(x, M5)
    x = (x | (x >> U64(2))) & M4
    x = (x | (x >> U64(4))) & M3
    x = (x | (x >> U64(8))) & M2
    x = (x | (x >> U64(16))) & M1
    x = (x | (x >> U64(32))) & MASK_21
    return x


@jax.jit
@jaxtyped(typechecker=beartype)
def morton_encode(
    positions: Array,
    bounds: Bounds,
) -> Array:
    min_c, max_c = bounds
    norm = (positions - min_c) / (max_c - min_c)
    norm = jnp.clip(norm, min=0.0, max=jnp.nextafter(1.0, 0.0))
    max_val = (1 << 21) - 1
    coords = jnp.rint(norm * max_val).astype(jnp.uint64)

    x = _spread3_u64(coords[:, 0])
    y = _spread3_u64(coords[:, 1])
    z = _spread3_u64(coords[:, 2])
    return x | (y << U64(1)) | (z << U64(2))


@jax.jit
@jaxtyped(typechecker=beartype)
def morton_decode(
    codes: Array,
    bounds: Bounds,
) -> Array:
    min_c, max_c = bounds
    x = _compact3_u64(codes)
    y = _compact3_u64(codes >> U64(1))
    z = _compact3_u64(codes >> U64(2))
    coords = jnp.stack([x, y, z], axis=1).astype(jnp.float32)

    max_val = (1 << 21) - 1
    norm = coords / max_val
    return norm * (max_c - min_c) + min_c


def get_common_prefix_length(code1: jnp.ndarray, code2: jnp.ndarray) -> int:
    """Return number of common leading bits between two Morton codes."""
    c1, c2 = int(code1), int(code2)
    if c1 == c2:
        return 64
    return 64 - (c1 ^ c2).bit_length()


def sort_by_morton(codes: jnp.ndarray) -> jnp.ndarray:
    """Return indices that sort points by Morton code."""
    return jnp.argsort(codes)


__all__ = [
    "Bounds",
    "get_common_prefix_length",
    "morton_decode",
    "morton_encode",
    "sort_by_morton",
]
