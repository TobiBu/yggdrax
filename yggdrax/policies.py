"""Traversal policy helpers for Yggdrax."""

from __future__ import annotations

from dataclasses import dataclass

from .interactions import DualTreeTraversalConfig


@dataclass(frozen=True)
class TraversalPolicy:
    """Top-level traversal policy contract."""

    config: DualTreeTraversalConfig
    description: str = "Explicit dual-tree traversal capacities."


__all__ = ["TraversalPolicy"]
