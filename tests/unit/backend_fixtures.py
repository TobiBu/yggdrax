"""Shared backend fixtures for conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from yggdrax import Tree, build_octree, build_tree


@dataclass(frozen=True)
class BackendAdapter:
    """Minimal adapter descriptor for backend conformance checks."""

    name: str
    build_fn: Callable


def _build_kdtree_adapter(positions, masses, *, leaf_size=8, return_reordered=False):
    """Signature-compatible KD-tree builder for conformance checks.

    The low-level :func:`yggdrax.build_kdtree` takes points only and returns a
    :class:`KDTree`. The conformance harness expects the ``build_tree`` /
    ``build_octree`` calling convention ``(positions, masses, *, leaf_size,
    return_reordered)`` returning ``(tree, positions_sorted, masses_sorted,
    inverse_permutation)``. Route through the unified ``Tree.from_particles``
    entry point, which produces a reordered KD particle tree exposing the same
    FMM-core topology fields as the radix/octree backends.
    """

    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="kdtree",
        build_mode="adaptive",
        leaf_size=leaf_size,
        return_reordered=True,
    )
    if not return_reordered:
        return tree
    return tree, tree.positions_sorted, tree.masses_sorted, tree.inverse_permutation


def conformance_adapters() -> tuple[BackendAdapter, ...]:
    """Return the set of backends that must pass conformance checks."""

    return (
        BackendAdapter(name="radix", build_fn=build_tree),
        BackendAdapter(name="octree", build_fn=build_octree),
        BackendAdapter(name="kdtree", build_fn=_build_kdtree_adapter),
    )


__all__ = ["BackendAdapter", "conformance_adapters"]
