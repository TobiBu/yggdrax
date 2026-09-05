"""Pallas API differences across the JAX versions yggdrax supports.

``pyproject.toml`` declares ``jax>=0.8`` at runtime and pins ``jax==0.8.3`` for
the type-check baseline, while the GPU machines these kernels are measured on
run 0.10.x. Two things moved in between:

* **How the GPU lowering is chosen.** Up to 0.9.0.x it was
  ``pallas_call(backend="triton")``; from 0.9.1 that keyword is gone and the
  choice is implied by the *type* of ``compiler_params``. Passing ``backend=``
  to 0.9.1 raises ``TypeError``.
* **What Triton's compiler-params class is called.** ``TritonCompilerParams`` on
  the older API, ``CompilerParams`` on the newer one.

Naming Triton rather than taking the default is deliberate: ``JAX_PALLAS_USE_MOSAIC_GPU``
flips the default lowering for any ``pallas_call`` that does not name a backend,
and Mosaic-GPU cannot express the small per-leaf tiles these kernels use.

This module is the one place either difference is handled, so the kernels
themselves carry no version branches.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeAlias

try:  # pragma: no cover - import is environment-dependent
    from jax.experimental.pallas import pallas_call as _pallas_call
    from jax.experimental.pallas import triton as _plgpu
except Exception:  # pragma: no cover - Pallas is optional
    _pallas_call = None
    _plgpu = None

#: A mutable block reference inside a Pallas kernel body -- what ``pallas_call``
#: hands the body for each input, output and scratch operand, read and written
#: with ``ref[...]`` rather than used as a value.
#:
#: Aliased here because JAX moves this name around and there is no ``pallas.Ref``:
#: the public spelling is ``jax.Ref`` on the versions these kernels are measured
#: on, and if it moves again this line changes and the annotations do not. To a
#: type checker it is plain ``Any`` -- the runtime value exists so the annotations
#: are honest documentation, and because pydoclint will not let a parameter be
#: *documented* until it is annotated.
if TYPE_CHECKING:
    KernelRef: TypeAlias = Any
else:  # pragma: no cover - the public spelling has moved between releases
    import jax as _jax

    KernelRef = getattr(_jax, "Ref", Any)

#: Whether the installed JAX still accepts ``pallas_call(backend=...)``
#: (True up to 0.9.0.x, False from 0.9.1 on).
PALLAS_CALL_TAKES_BACKEND = _pallas_call is not None and (
    "backend" in inspect.signature(_pallas_call).parameters
)

__all__ = ["KernelRef", "PALLAS_CALL_TAKES_BACKEND", "pallas_backend_kwargs"]


def _triton_compiler_params(num_warps: int | None, num_stages: int | None) -> Any:
    """Return Triton compiler params under whichever name this JAX exposes.

    Args:
        num_warps: Warps per program, or ``None`` for Triton's default.
        num_stages: Pipeline stages, or ``None`` for Triton's default.

    Returns:
        The params object.

    Raises:
        RuntimeError: If the Triton backend module is unavailable, or exposes
            neither known params class.
    """
    if _plgpu is None:
        raise RuntimeError("jax.experimental.pallas.triton is not available")
    params_cls = getattr(_plgpu, "CompilerParams", None) or getattr(
        _plgpu, "TritonCompilerParams", None
    )
    if params_cls is None:  # pragma: no cover - would mean a third rename
        raise RuntimeError(
            "jax.experimental.pallas.triton exposes neither CompilerParams nor "
            "TritonCompilerParams; the Pallas compat shim needs updating"
        )
    kwargs: dict[str, Any] = {}
    if num_warps is not None:
        kwargs["num_warps"] = int(num_warps)
    if num_stages is not None:
        kwargs["num_stages"] = int(num_stages)
    return params_cls(**kwargs)


def pallas_backend_kwargs(
    backend: str | None,
    *,
    interpret: bool = False,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> dict[str, Any]:
    """Return the ``pallas_call`` kwargs selecting ``backend`` on either API.

    Args:
        backend: ``"triton"``, or ``None`` to leave the choice to JAX. Only
            Triton is supported here -- see the module docstring.
        interpret: Whether the caller passes ``interpret=True``. Interpret mode
            runs CPU semantics with no lowering, so no backend kwarg is emitted
            and no Triton tuning is meaningful.
        num_warps: Warps per program, or ``None`` for Triton's default.
        num_stages: Pipeline stages, or ``None`` for Triton's default.

    Returns:
        Kwargs to splat into ``pallas_call``: ``{}`` under interpret, and
        otherwise ``compiler_params`` plus, on the old API, ``backend``.

    Raises:
        NotImplementedError: If a non-Triton backend is requested, where it
            could not be honoured on the new API.
    """
    if interpret:
        return {}
    if backend not in (None, "triton"):
        raise NotImplementedError(
            f"backend={backend!r} cannot be selected: pallas_call no longer takes "
            "a `backend` kwarg (removed in JAX 0.9.1) and only Triton is "
            "supported by yggdrax's Pallas kernels. Pass backend='triton' or None."
        )
    kwargs: dict[str, Any] = {
        "compiler_params": _triton_compiler_params(num_warps, num_stages)
    }
    if PALLAS_CALL_TAKES_BACKEND:
        kwargs["backend"] = backend
    return kwargs
