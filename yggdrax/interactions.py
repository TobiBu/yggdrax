"""Public interaction/traversal API for Yggdrax."""

from __future__ import annotations

from typing import Callable

from jaxtyping import Array

from . import _interactions_impl
from ._interactions_impl import (
    DEFAULT_PAIR_QUEUE_MULTIPLIER,
    CompactTaggedFarPairs,
    CompactTaggedOctreeFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    OctreeNativeNeighborList,
    PairPolicy,
    _compute_effective_extents,
    _compute_leaf_effective_extents,
    _interaction_capacity_candidates,
    interactions_for_node,
    neighbors_for_leaf,
)
from .geometry import TreeGeometry
from .tree import resolve_tree_topology


def _call_with_topology(func, tree: object, /, *args, **kwargs):
    """Resolve tree containers to topology payloads before dispatch."""

    return func(resolve_tree_topology(tree), *args, **kwargs)


def build_well_separated_interactions(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
) -> NodeInteractionList:
    """Construct only the far-field interaction list from a dual-tree walk.

    Like :func:`build_interactions_and_neighbors` but returns the sparse
    far-field (M2L) list alone, skipping near-field neighbor collection.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_interactions_per_node
        Capacity of the per-node far-interaction buffer; auto-sized when ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    NodeInteractionList
        Sparse far-field interaction list.
    """

    return _call_with_topology(
        _interactions_impl.build_well_separated_interactions,
        tree,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_compact_far_pairs(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
) -> CompactTaggedFarPairs:
    """Construct exact-length tagged far pairs from the dual-tree walk.

    Returns far-field pairs packed to their exact count (rather than the padded
    per-node layout of :func:`build_well_separated_interactions`).

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_interactions_per_node
        Per-node far-interaction capacity; auto-sized when ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    CompactTaggedFarPairs
        Exact-length far pairs with per-pair tags.
    """

    return _call_with_topology(
        _interactions_impl.build_compact_far_pairs,
        tree,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_compact_far_pairs_and_leaf_neighbor_lists(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
    timing_callback: Callable[..., object] | None = None,
    compact_far_pair_capacity: int | None = None,
) -> tuple[CompactTaggedFarPairs, NodeNeighborList]:
    """Construct compact far pairs and near neighbors from one count walk.

    Produces both exact-length far pairs and the near-field neighbor list while
    sharing a single bounded count pass (cheaper than two separate walks).

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_neighbors_per_leaf
        Per-leaf near-neighbor capacity.
    max_interactions_per_node
        Per-node far-interaction capacity; auto-sized when ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.
    timing_callback
        Optional callable invoked with per-stage timing diagnostics.
    compact_far_pair_capacity
        Optional explicit capacity for the compact far-pair buffer.

    Returns
    -------
    tuple
        ``(compact_far_pairs, neighbors)`` as
        (:class:`CompactTaggedFarPairs`, :class:`NodeNeighborList`).
    """

    return _call_with_topology(
        _interactions_impl.build_compact_far_pairs_and_leaf_neighbor_lists,
        tree,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
        timing_callback=timing_callback,
        compact_far_pair_capacity=compact_far_pair_capacity,
    )


def build_leaf_neighbor_lists(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
) -> NodeNeighborList:
    """Construct only the near-field neighbor list from a dual-tree walk.

    Like :func:`build_interactions_and_neighbors` but returns the leaf-leaf
    near-field (P2P) neighbor list alone, skipping far-field collection.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_neighbors_per_leaf
        Per-leaf near-neighbor capacity.
    max_interactions_per_node
        Per-node far-interaction capacity used during the walk; auto-sized when
        ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    NodeNeighborList
        Near-field leaf neighbor list.
    """

    return _call_with_topology(
        _interactions_impl.build_leaf_neighbor_lists,
        tree,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_interactions_and_neighbors(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    mac_type: MACType = "bh",
    dehnen_radius_scale: float = 1.0,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    *,
    return_result: bool = False,
    return_compact_far_pairs: bool = False,
    return_interactions: bool = True,
    return_grouped: bool = False,
) -> tuple:
    """Construct both far-field interactions and near-field neighbors.

    Runs a single dual-tree traversal that classifies node/leaf pairs into
    well-separated far-field interactions (M2L) and near-field neighbor pairs
    (P2P), using the multipole acceptance criterion selected by ``mac_type``
    (or a custom ``pair_policy``).

    When ``pair_policy`` is provided it overrides the built-in MAC decision for
    each candidate pair and may attach integer tags to accepted far pairs.
    Policies are evaluated in both directions; a pair is accepted only when
    both directed decisions accept, and the directed tags are exposed on
    ``DualTreeWalkResult.interaction_tags`` when ``return_result=True``.

    Fixed-capacity buffers are grown automatically (subject to internal caps)
    when the traversal reports overflow, so the capacity arguments are hints
    rather than hard limits unless they are pinned via ``traversal_config``.

    Parameters
    ----------
    tree
        Tree container exposing FMM-core topology (radix, octree, or kd-tree).
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion. Smaller
        values accept fewer far pairs (more accurate, more work).
    max_interactions_per_node
        Capacity of the per-node far-interaction buffer. ``None`` auto-sizes it
        and grows on overflow.
    max_neighbors_per_leaf
        Capacity of the per-leaf near-neighbor buffer.
    max_pair_queue
        Capacity of the traversal wavefront queue. ``None`` auto-sizes it.
    process_block
        Number of pairs processed per traversal iteration. ``None`` auto-sizes.
    traversal_config
        Bundled capacity/queue/block settings. When given, it takes precedence
        over the equivalent individual keyword arguments.
    retry_logger
        Optional callable invoked with a :class:`DualTreeRetryEvent` on each
        capacity-driven retry (for diagnostics/tuning).
    mac_type
        Multipole acceptance criterion: ``"bh"`` (Barnes-Hut opening angle),
        ``"dehnen"``, or ``"engblom"``.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC decision.
    policy_state
        Opaque state threaded to ``pair_policy`` on every evaluation.
    return_result
        If ``True``, also return the raw :class:`DualTreeWalkResult` (including
        ``interaction_tags``).
    return_compact_far_pairs
        If ``True``, also return exact-length :class:`CompactTaggedFarPairs`.
    return_interactions
        If ``True`` (default), include the sparse :class:`NodeInteractionList`
        in the result.
    return_grouped
        If ``True``, also return displacement-grouped interaction buffers.

    Returns
    -------
    tuple
        By default ``(interactions, neighbors)`` where ``interactions`` is a
        :class:`NodeInteractionList` and ``neighbors`` is a
        :class:`NodeNeighborList`. Additional elements are appended, in
        declaration order, for each enabled ``return_*`` flag
        (``return_compact_far_pairs``, ``return_grouped``, ``return_result``).
    """

    return _call_with_topology(
        _interactions_impl.build_interactions_and_neighbors,
        tree,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        mac_type=mac_type,
        dehnen_radius_scale=dehnen_radius_scale,
        pair_policy=pair_policy,
        policy_state=policy_state,
        return_result=return_result,
        return_compact_far_pairs=return_compact_far_pairs,
        return_interactions=return_interactions,
        return_grouped=return_grouped,
    )


def build_interactions_and_neighbors_split(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    mac_type: MACType = "bh",
    dehnen_radius_scale: float = 1.0,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
) -> tuple:
    """Construct far and near products using two separate dual-tree walks.

    Same outputs as :func:`build_interactions_and_neighbors` but runs the
    far-field and near-field collection as independent walks instead of one
    shared traversal.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_interactions_per_node
        Per-node far-interaction capacity; auto-sized when ``None``.
    max_neighbors_per_leaf
        Per-leaf near-neighbor capacity.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.

    Returns
    -------
    tuple
        ``(interactions, neighbors)`` as
        (:class:`NodeInteractionList`, :class:`NodeNeighborList`).
    """

    return _call_with_topology(
        _interactions_impl.build_interactions_and_neighbors_split,
        tree,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        mac_type=mac_type,
        dehnen_radius_scale=dehnen_radius_scale,
        pair_policy=pair_policy,
        policy_state=policy_state,
    )


def build_octree_native_far_pairs(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
) -> CompactTaggedOctreeFarPairs:
    """Construct exact-length far-field pairs in explicit octree node space.

    Like :func:`build_compact_far_pairs` but emits pairs indexed in the explicit
    octree cell space (requires an octree-augmented ``tree``).

    Parameters
    ----------
    tree
        Octree-augmented tree exposing the explicit ``oct_*`` buffers.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_interactions_per_node
        Per-node far-interaction capacity; auto-sized when ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    CompactTaggedOctreeFarPairs
        Exact-length far pairs in octree node space.
    """

    return _call_with_topology(
        _interactions_impl.build_octree_native_far_pairs,
        tree,
        geometry,
        theta=theta,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_octree_native_neighbor_lists(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    max_interactions_per_node: int | None = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: PairPolicy | None = None,
    policy_state: object = None,
    max_pair_queue: int | None = None,
    process_block: int | None = None,
    traversal_config: DualTreeTraversalConfig | None = None,
    retry_logger: Callable[[DualTreeRetryEvent], object] | None = None,
    dehnen_radius_scale: float = 1.0,
) -> OctreeNativeNeighborList:
    """Construct exact-length near neighbors in explicit octree leaf space.

    Like :func:`build_leaf_neighbor_lists` but emits neighbor pairs indexed in
    the explicit octree leaf space (requires an octree-augmented ``tree``).

    Parameters
    ----------
    tree
        Octree-augmented tree exposing the explicit ``oct_*`` buffers.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_neighbors_per_leaf
        Per-leaf near-neighbor capacity.
    max_interactions_per_node
        Per-node far-interaction capacity used during the walk; auto-sized when
        ``None``.
    mac_type
        MAC variant: ``"bh"``, ``"dehnen"``, or ``"engblom"``.
    pair_policy
        Optional JAX-traceable callable overriding the built-in MAC.
    policy_state
        Opaque state threaded to ``pair_policy``.
    max_pair_queue
        Traversal wavefront capacity; auto-sized when ``None``.
    process_block
        Pairs processed per traversal iteration; auto-sized when ``None``.
    traversal_config
        Bundled traversal settings; overrides the individual arguments.
    retry_logger
        Optional callable invoked on each capacity-driven retry.
    dehnen_radius_scale
        Effective-radius scale applied for the Dehnen MAC.

    Returns
    -------
    OctreeNativeNeighborList
        Exact-length near neighbors in octree leaf space.
    """

    return _call_with_topology(
        _interactions_impl.build_octree_native_neighbor_lists,
        tree,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_interactions_per_node=max_interactions_per_node,
        mac_type=mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )


def build_grouped_interactions_from_pairs(
    tree: object,
    geometry: TreeGeometry,
    interaction_sources: Array,
    interaction_targets: Array,
    *,
    level_offsets: Array | None = None,
):
    """Group far-field pairs into displacement classes for class-major M2L.

    Thin wrapper over
    :func:`yggdrax.grouped_interactions.build_grouped_interactions_from_pairs`
    that first resolves ``tree`` to its topology payload.

    Parameters
    ----------
    tree
        Tree container or topology exposing level-order metadata.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    interaction_sources
        Far-pair source node indices, shape ``(num_pairs,)``.
    interaction_targets
        Far-pair target node indices, shape ``(num_pairs,)``.
    level_offsets
        Optional precomputed level offsets passed through for diagnostics;
        derived from ``tree`` when omitted.

    Returns
    -------
    GroupedInteractionBuffers
        Class keys, per-class displacements, and CSR-style class offsets/ids.
    """

    return _call_with_topology(
        _interactions_impl.build_grouped_interactions_from_pairs,
        tree,
        geometry,
        interaction_sources,
        interaction_targets,
        level_offsets=level_offsets,
    )


def diagnose_leaf_neighbor_growth(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    *,
    max_neighbors_per_leaf: int = _interactions_impl._DEFAULT_MAX_NEIGHBORS,
    top_k: int = 10,
    sample_neighbors: int = 20,
) -> dict:
    """Return a compact report of the highest per-leaf neighbor counts.

    Diagnostic helper for tuning ``max_neighbors_per_leaf``: runs the near-field
    walk and summarizes which leaves accumulate the most neighbors.

    Parameters
    ----------
    tree
        Tree container or topology exposing the FMM-core contract.
    geometry
        Per-node geometry from :func:`compute_tree_geometry` for ``tree``.
    theta
        Opening-angle parameter of the multipole acceptance criterion.
    max_neighbors_per_leaf
        Per-leaf near-neighbor capacity used for the probe walk.
    top_k
        Number of highest-count leaves to report.
    sample_neighbors
        Number of sample neighbor ids to include per reported leaf.

    Returns
    -------
    dict
        Report with the top per-leaf neighbor counts and sampled neighbor ids.
    """

    return _call_with_topology(
        _interactions_impl.diagnose_leaf_neighbor_growth,
        tree,
        geometry,
        theta=theta,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        top_k=top_k,
        sample_neighbors=sample_neighbors,
    )


__all__ = [
    "DEFAULT_PAIR_QUEUE_MULTIPLIER",
    "MACType",
    "PairPolicy",
    "DualTreeRetryEvent",
    "DualTreeTraversalConfig",
    "DualTreeWalkResult",
    "CompactTaggedFarPairs",
    "CompactTaggedOctreeFarPairs",
    "OctreeNativeNeighborList",
    "NodeInteractionList",
    "NodeNeighborList",
    "build_interactions_and_neighbors",
    "build_interactions_and_neighbors_split",
    "build_compact_far_pairs",
    "build_compact_far_pairs_and_leaf_neighbor_lists",
    "build_octree_native_far_pairs",
    "build_octree_native_neighbor_lists",
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
