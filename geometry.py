"""Public geometry API for Yggdrasil."""

from __future__ import annotations

from beartype import beartype
from jaxtyping import Array, jaxtyped

from . import _geometry_impl
from .tree import RadixTree

TreeGeometry = _geometry_impl.TreeGeometry
LevelMajorTreeGeometry = _geometry_impl.LevelMajorTreeGeometry
_MAX_MORTON_LEVEL = _geometry_impl._MAX_MORTON_LEVEL


@jaxtyped(typechecker=beartype)
def compute_tree_geometry(
    tree: RadixTree,
    positions_sorted: Array,
) -> TreeGeometry:
    """Compute per-node geometric bounds and helper radii."""

    return _geometry_impl.compute_tree_geometry(tree, positions_sorted)


@jaxtyped(typechecker=beartype)
def geometry_to_level_major(
    tree: RadixTree,
    geometry: TreeGeometry,
) -> LevelMajorTreeGeometry:
    """Convert geometry to padded level-major buffers."""

    return _geometry_impl.geometry_to_level_major(tree, geometry)


__all__ = [
    "LevelMajorTreeGeometry",
    "TreeGeometry",
    "_MAX_MORTON_LEVEL",
    "compute_tree_geometry",
    "geometry_to_level_major",
]
