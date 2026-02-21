"""Local dtype policy for Yggdrasil contracts."""

import jax.numpy as jnp

# Keep tree/index contracts consistent across yggdrasil artifacts.
INDEX_DTYPE = jnp.int64


def as_index(x):
    """Convert a scalar/array to yggdrasil index dtype."""
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def complex_dtype_for_real(real_dtype):
    """Return complex dtype paired with a real floating dtype."""
    dtype = jnp.asarray(0, dtype=real_dtype).dtype
    if dtype == jnp.float64:
        return jnp.complex128
    return jnp.complex64


__all__ = ["INDEX_DTYPE", "as_index", "complex_dtype_for_real"]
