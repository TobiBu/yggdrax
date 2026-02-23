"""Shared backend fixtures for conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from yggdrax import build_tree


@dataclass(frozen=True)
class BackendAdapter:
    """Minimal adapter descriptor for backend conformance checks."""

    name: str
    build_fn: Callable


def conformance_adapters() -> tuple[BackendAdapter, ...]:
    """Return the set of backends that must pass conformance checks."""

    return (BackendAdapter(name="radix", build_fn=build_tree),)


__all__ = ["BackendAdapter", "conformance_adapters"]
