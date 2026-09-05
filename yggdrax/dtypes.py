"""Local dtype policy for Yggdrax contracts."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike


def _resolve_index_dtype() -> DTypeLike:
    """Resolve index dtype from environment.

    Supported values:
    - ``YGGDRAX_INDEX_PRECISION=int32``
    - ``YGGDRAX_INDEX_PRECISION=int64``

    For convenience, fall back to ``JACCPOT_INDEX_PRECISION`` so both
    workspaces can share the same notebook/session setting.
    """

    raw = os.environ.get("YGGDRAX_INDEX_PRECISION")
    if raw is None:
        raw = os.environ.get("JACCPOT_INDEX_PRECISION", "int64")
    raw_norm = str(raw).strip().lower()
    if raw_norm in ("int32", "i32", "32"):
        return jnp.int32
    if raw_norm in ("int64", "i64", "64"):
        return jnp.int64
    return jnp.int64


# Keep tree/index contracts consistent across yggdrax artifacts.
INDEX_DTYPE: DTypeLike = _resolve_index_dtype()


def as_index(x: ArrayLike) -> jax.Array:
    """Convert a scalar/array to yggdrax index dtype."""
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def complex_dtype_for_real(real_dtype: DTypeLike) -> DTypeLike:
    """Return complex dtype paired with a real floating dtype."""
    dtype = jnp.asarray(0, dtype=real_dtype).dtype
    if dtype == jnp.float64:
        return jnp.complex128
    return jnp.complex64


__all__ = ["INDEX_DTYPE", "as_index", "complex_dtype_for_real"]


def is_traced(value: object) -> bool:
    """Whether ``value`` is a JAX tracer rather than a concrete array.

    Used to decide whether a helper may jit itself: under a trace it must not,
    because the caller's own transformation owns the shapes and jitting inside
    would quietly build a different tree.

    ``jax.core.Tracer`` is the runtime spelling and there is no other, but
    ``jax.core`` is absent from JAX's type stubs, so a direct
    ``isinstance(x, jax.core.Tracer)`` is a type error at every call site. This
    puts the stub gap in one place and names what the check is for.

    Args:
        value: Any object.

    Returns:
        ``True`` when ``value`` is a tracer.
    """
    tracer = getattr(getattr(jax, "core", None), "Tracer", None)
    return tracer is not None and isinstance(value, tracer)
