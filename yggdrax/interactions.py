"""Public interaction/traversal API for Yggdrax."""

from __future__ import annotations

from ._interactions_impl import (
    DEFAULT_PAIR_QUEUE_MULTIPLIER,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    _compute_effective_extents,
    _compute_leaf_effective_extents,
    _interaction_capacity_candidates,
    build_grouped_interactions_from_pairs,
    build_interactions_and_neighbors,
    build_leaf_neighbor_lists,
    build_well_separated_interactions,
    diagnose_leaf_neighbor_growth,
    interactions_for_node,
    neighbors_for_leaf,
)

__all__ = [
    "DEFAULT_PAIR_QUEUE_MULTIPLIER",
    "MACType",
    "DualTreeRetryEvent",
    "DualTreeTraversalConfig",
    "DualTreeWalkResult",
    "NodeInteractionList",
    "NodeNeighborList",
    "build_interactions_and_neighbors",
    "build_leaf_neighbor_lists",
    "build_grouped_interactions_from_pairs",
    "build_well_separated_interactions",
    "diagnose_leaf_neighbor_growth",
    "interactions_for_node",
    "neighbors_for_leaf",
    "_interaction_capacity_candidates",
    "_compute_effective_extents",
    "_compute_leaf_effective_extents",
]
