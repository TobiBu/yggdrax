"""Local dtype policy for Yggdrax contracts."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike


def _resolve_index_dtype() -> DTypeLike:
    """Resolve index dtype from environment.

    Supported values:
    - ``YGGDRAX_INDEX_PRECISION=int32`` (default since 2026-10)
    - ``YGGDRAX_INDEX_PRECISION=int64`` (opt-in, index spaces past 2^31)

    For convenience, fall back to ``JACCPOT_INDEX_PRECISION`` so both
    workspaces can share the same notebook/session setting. Unrecognised
    values give the default.
    """

    raw = os.environ.get("YGGDRAX_INDEX_PRECISION")
    if raw is None:
        raw = os.environ.get("JACCPOT_INDEX_PRECISION", "int32")
    raw_norm = str(raw).strip().lower()
    if raw_norm in ("int64", "i64", "64"):
        return jnp.int64
    return jnp.int32


# Keep tree/index contracts consistent across yggdrax artifacts.
INDEX_DTYPE: DTypeLike = _resolve_index_dtype()


def as_index(x: ArrayLike) -> jax.Array:
    """Convert a scalar/array to yggdrax index dtype."""
    if INDEX_DTYPE == jnp.int64:
        require_x64("YGGDRAX_INDEX_PRECISION=int64")
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def require_x64(what: str) -> None:
    """Raise unless ``jax_enable_x64`` is on, because ``what`` needs a 64-bit type.

    yggdrax used to switch x64 on at import; it no longer changes JAX's configuration.
    With x64 off, JAX narrows every ``uint64`` / ``int64`` / ``float64`` request to
    32 bits with at most a warning: the 63-bit Morton codes lose their top bits, and
    the trees built from them are wrong without any error (the forces of the jaccpot
    lanes came out 19 % off when this was tried). So every helper that makes a 64-bit
    array calls this first. The flag is read when the caller runs or is traced; the
    traced program does not change.

    Args:
        what: What needs the 64-bit type, named in the message.

    Raises:
        RuntimeError: If ``jax_enable_x64`` is off.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"yggdrax needs jax_enable_x64=True ({what}). Enable it at the top of "
            "your program, before creating arrays: "
            'jax.config.update("jax_enable_x64", True), or set JAX_ENABLE_X64=1. '
            "The input dtype still sets the working precision: float32 inputs stay "
            "float32."
        )


__all__ = ["INDEX_DTYPE", "as_index", "require_x64"]


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
