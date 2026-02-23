"""Structural protocols for tree/topology capabilities."""

from __future__ import annotations

from typing import Protocol

from jaxtyping import Array


class TreeStructureProtocol(Protocol):
    """Minimal structure needed for parent/child traversal."""

    parent: Array
    left_child: Array
    right_child: Array


class TreeRangesProtocol(TreeStructureProtocol, Protocol):
    """Adds node-to-particle range metadata."""

    node_ranges: Array
    num_particles: Array | int


class TreeLevelIndexProtocol(TreeStructureProtocol, Protocol):
    """Adds explicit level-order indexing metadata."""

    node_level: Array
    nodes_by_level: Array
    level_offsets: Array
    num_levels: Array | int


class MortonLeafBoundsProtocol(TreeRangesProtocol, Protocol):
    """Adds Morton-derived leaf box information for geometry construction."""

    use_morton_geometry: Array | bool
    bounds_min: Array
    bounds_max: Array
    leaf_codes: Array
    leaf_depths: Array


class TopologyContainerProtocol(Protocol):
    """Container exposing a concrete topology payload."""

    topology: object


__all__ = [
    "MortonLeafBoundsProtocol",
    "TopologyContainerProtocol",
    "TreeLevelIndexProtocol",
    "TreeRangesProtocol",
    "TreeStructureProtocol",
]
