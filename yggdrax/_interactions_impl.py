"""Interaction list construction for FMM downward sweep.

This module implements tree-based traversal routines that build the
multipole-to-local (far-field) and particle-to-particle (near-field)
interaction lists needed by the Fast Multipole Method.  The new
implementations rely exclusively on JAX primitives so they can be JIT
compiled and executed on accelerators while keeping memory usage linear
in the number of tree nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import List, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jax import core as jax_core
from jax import lax
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index
from .geometry import TreeGeometry
from .grouped_interactions import (
    GroupedInteractionBuffers,
    build_grouped_interactions_from_pairs,
)
from .tree import (
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
)

# Each node only needs to interact with a bounded number of well-separated
# partners (189 in the classic 3D MAC stencil). Keeping the default well above
# that limit grants headroom for irregular trees without forcing enormous
# row-major buffers.
_DEFAULT_MAX_INTERACTIONS = 512
_DEFAULT_MAX_NEIGHBORS = 2048
DEFAULT_PAIR_QUEUE_MULTIPLIER = 128
_DEFAULT_PAIR_BATCH = 32
_MAX_REFINEMENT_PAIRS = 4
_DEFAULT_KDTREE_DEHNEN_RADIUS_SCALE = 1.2

_AUTO_INTERACTION_MIN = 2048
_AUTO_INTERACTION_MAX = 65536
_AUTO_INTERACTION_SCALE = 4


# Module-level helper constants and functions used by the traversal refiners.
# These are defined at module scope so they are picklable/visible to JAX's
# tracer during jitting of nested traversal functions.
_FILLER_PAIR = jnp.asarray([-1, -1], dtype=INDEX_DTYPE)
_EMPTY_REFINEMENT_PAIRS = jnp.tile(_FILLER_PAIR[None, :], (_MAX_REFINEMENT_PAIRS, 1))


def _count_sorted_pair(a: Array, b: Array) -> Array:
    lo = jnp.minimum(a, b)
    hi = jnp.maximum(a, b)
    return jnp.stack([lo, hi], axis=0)


def _count_refine_pairs_single(
    tgt: Array,
    src: Array,
    same: Array,
    split_both_flag: Array,
    split_target_flag: Array,
    split_source_flag: Array,
    tgt_left: Array,
    tgt_right: Array,
    src_left: Array,
    src_right: Array,
) -> Array:
    """Branchless refine-pair helper for the count pass.

    Computes all 5 possible outcomes and selects with ``jnp.where``
    instead of nested ``lax.cond``, which under ``vmap`` evaluates
    all branches anyway.
    """
    # Case 1: split_both & same
    p_bs_0 = _count_sorted_pair(tgt_left, tgt_left)
    p_bs_1 = _count_sorted_pair(tgt_left, tgt_right)
    p_bs_2 = _count_sorted_pair(tgt_right, tgt_right)
    both_same = jnp.stack([p_bs_0, p_bs_1, p_bs_2, _FILLER_PAIR], axis=0)

    # Case 2: split_both & ~same
    p_bc_0 = _count_sorted_pair(tgt_left, src_left)
    p_bc_1 = _count_sorted_pair(tgt_left, src_right)
    p_bc_2 = _count_sorted_pair(tgt_right, src_left)
    p_bc_3 = _count_sorted_pair(tgt_right, src_right)
    both_cross = jnp.stack([p_bc_0, p_bc_1, p_bc_2, p_bc_3], axis=0)

    # Case 3: split_target only
    p_st_0 = _count_sorted_pair(tgt_left, src)
    p_st_1 = _count_sorted_pair(tgt_right, src)
    split_tgt = jnp.stack([p_st_0, p_st_1, _FILLER_PAIR, _FILLER_PAIR], axis=0)

    # Case 4: split_source only
    p_ss_0 = _count_sorted_pair(tgt, src_left)
    p_ss_1 = _count_sorted_pair(tgt, src_right)
    split_src = jnp.stack([p_ss_0, p_ss_1, _FILLER_PAIR, _FILLER_PAIR], axis=0)

    # Select: priority order split_both > split_target > split_source > empty
    result = _EMPTY_REFINEMENT_PAIRS
    result = jnp.where(split_source_flag, split_src, result)
    result = jnp.where(split_target_flag, split_tgt, result)
    result = jnp.where(split_both_flag & ~same, both_cross, result)
    result = jnp.where(split_both_flag & same, both_same, result)
    return result


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class DualTreeTraversalConfig:
    """Fixed traversal parameters for the dual-tree walk."""

    max_pair_queue: int
    process_block: int
    max_interactions_per_node: int
    max_neighbors_per_leaf: int = _DEFAULT_MAX_NEIGHBORS


_GLOBAL_DUAL_TREE_CONFIG: Optional[DualTreeTraversalConfig] = None


def set_default_dual_tree_config(
    config: Optional[DualTreeTraversalConfig],
) -> None:
    """Set the module-level fallback configuration for dual-tree walks."""

    if config is not None:
        if config.max_pair_queue < 1:
            raise ValueError("max_pair_queue must be >= 1")
        if config.process_block < 1:
            raise ValueError("process_block must be >= 1")
        if config.max_interactions_per_node < 1:
            raise ValueError("max_interactions_per_node must be >= 1")
        if config.max_neighbors_per_leaf < 1:
            raise ValueError("max_neighbors_per_leaf must be >= 1")

    global _GLOBAL_DUAL_TREE_CONFIG
    _GLOBAL_DUAL_TREE_CONFIG = config


def _resolve_dual_tree_config(
    config: Optional[DualTreeTraversalConfig],
) -> Optional[DualTreeTraversalConfig]:
    if config is not None:
        return config
    return _GLOBAL_DUAL_TREE_CONFIG


_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2


PairPolicy = Callable[..., tuple[Array, Array]]


# -----------------------------------------------------------------------------
# Multipole acceptance criteria (MAC)
# -----------------------------------------------------------------------------

MACType = Literal[
    "bh",  # Barnes-Hut / opening-angle based (current default)
    "engblom",  # Engblom 2010 / jaxFMM-style criterion
    "dehnen",  # Dehnen 2014 Eq. (6): (rho_zA + rho_sB) / r < theta_crit
]


def _compute_mac_ok(
    *,
    mac_type: MACType,
    theta_sq: Array,
    dist_sq: Array,
    extent_target: Array,
    extent_source: Array,
    valid_pairs: Array,
    different_nodes: Array,
) -> Array:
    """Return per-pair acceptance decisions for the configured MAC.

    All inputs are vectorized over the pair batch.

    Args:
        mac_type: Which criterion to use.
        theta_sq: $\theta^2$ parameter (scalar broadcast to batch).
        dist_sq: Squared distance between node centers.
        extent_target: Effective extent/radius of target node.
        extent_source: Effective extent/radius of source node.
        valid_pairs: Boolean mask for valid pairs.
        different_nodes: Boolean mask for target != source.

    Returns:
        Boolean array matching ``dist_sq`` shape.
    """

    # Common validity guards.
    valid = valid_pairs & different_nodes & (dist_sq > 0.0)

    if mac_type == "bh":
        # Symmetric opening angle: (r_t + r_s)^2 <= theta^2 * d^2
        extent_sum = extent_target + extent_source
        return valid & (extent_sum * extent_sum <= theta_sq * dist_sq)

    if mac_type == "dehnen":
        # Dehnen (2014) arXiv:1405.2255, equation (6):
        #   theta = (rho_z,A + rho_s,B) / r
        # accept if theta < theta_crit.
        #
        # In this code, the per-node extents are conservative radii derived
        # from node bounding boxes (TreeGeometry.max_extent) and propagated /
        # padded as needed. These play the role of rho_z and rho_s.
        extent_sum = extent_target + extent_source
        return valid & (extent_sum * extent_sum <= theta_sq * dist_sq)

    if mac_type == "engblom":
        # Engblom 2010 (as used in jaxFMM):
        #   R + theta r <= theta d
        # where R = max(r_t, r_s), r = min(r_t, r_s), d = |c_t - c_s|.
        # We square both sides to avoid a sqrt in the inner loop.
        radius_big = jnp.maximum(extent_target, extent_source)
        radius_small = jnp.minimum(extent_target, extent_source)
        theta = jnp.sqrt(theta_sq)
        lhs = radius_big + theta * radius_small
        return valid & (lhs * lhs <= theta_sq * dist_sq)

    raise ValueError(f"Unknown mac_type: {mac_type}")


logger = logging.getLogger(__name__)


class DualTreeRetryEvent(NamedTuple):
    """Metadata describing a single dual-tree retry attempt."""

    attempt: int
    queue_capacity: int
    interaction_capacity: int
    status: str
    far_pair_count: int
    near_pair_count: int


def log_retry_event(
    event: DualTreeRetryEvent,
    *,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log a retry event using the provided (or module) logger."""

    target_logger = logger or logging.getLogger(__name__)
    target_logger.log(
        level,
        (
            "Dual-tree %s (attempt %d): queue=%d, inter_cap=%d, "
            "far_pairs=%d, near_pairs=%d"
        ),
        event.status,
        event.attempt,
        event.queue_capacity,
        event.interaction_capacity,
        event.far_pair_count,
        event.near_pair_count,
    )


def _auto_pair_queue_candidates(
    total_nodes: int,
    num_internal: int,
) -> List[int]:
    """Heuristic queue capacities to try for the dual-tree walk.

    We start from a value proportional to the number of internal nodes (pairs
    that can spawn refinements) and progressively double until we reach the
    conservative default derived from ``DEFAULT_PAIR_QUEUE_MULTIPLIER``.  This
    avoids compiling the large walker with multi-million entry stacks unless it
    is truly necessary (common for deep fixed-depth trees), substantially
    reducing both compile and run time.
    """

    conservative = DEFAULT_PAIR_QUEUE_MULTIPLIER * max(4, total_nodes)
    # Start with a smaller buffer that still scales with tree fan-out.
    start = max(4096, num_internal * 2)
    start = min(start, conservative)
    if start >= conservative:
        return [int(conservative)]

    candidates: List[int] = []
    capacity = start
    while capacity < conservative:
        candidates.append(int(capacity))
        capacity = max(capacity * 2, capacity + 1024)
    candidates.append(int(conservative))

    # Remove potential duplicates while preserving order.
    seen = set()
    unique_candidates = []
    for cap in candidates:
        if cap not in seen:
            seen.add(cap)
            unique_candidates.append(cap)
    return unique_candidates


def _resolve_process_block(
    queue_capacity: int,
    requested_block: Optional[int],
) -> int:
    """Choose a process_block that balances vectorization and stack usage."""

    if requested_block is not None:
        if requested_block < 1:
            raise ValueError("process_block must be >= 1")
        return int(min(queue_capacity, requested_block))

    if queue_capacity >= 65536:
        auto = 1024
    elif queue_capacity >= 16384:
        auto = 512
    elif queue_capacity >= 4096:
        auto = 256
    elif queue_capacity >= 1024:
        auto = 128
    else:
        auto = _DEFAULT_PAIR_BATCH
    return int(min(queue_capacity, auto))


def _interaction_capacity_candidates(
    requested: Optional[int],
    total_nodes: int,
) -> tuple[list[int], bool]:
    if requested is not None:
        if requested < 1:
            raise ValueError("max_interactions_per_node must be >= 1")
        return [int(requested)], False

    # Single-pass heuristic: scale with node count, keep within practical
    # bounds, and hand the exact value to the walker. The user can override
    # via ``max_interactions_per_node`` if this estimate proves insufficient.
    scaled = max(total_nodes // _AUTO_INTERACTION_SCALE, _AUTO_INTERACTION_MIN)
    scaled = min(scaled, _AUTO_INTERACTION_MAX)
    capacity = int(
        min(
            _AUTO_INTERACTION_MAX,
            max(_AUTO_INTERACTION_MIN, _next_power_of_two(scaled)),
        )
    )
    return [capacity], False


class NodeInteractionList(NamedTuple):
    """Compressed far-field interaction list for all tree nodes.

    Attributes:
        offsets: Start index for each node within ``sources``/``targets``;
            combine with ``counts`` to recover the slice length per node.
        sources: Source node indices for every far-field pair, written in
            level-major order.
        targets: Target node indices matching ``sources``.
        counts: Interaction counts per node (number of entries for each
            target).
        level_offsets: Prefix offsets delimiting the interaction ranges for
            each tree level (length ``num_levels + 1``).
        target_levels: Tree level for each pair (monotonically
            non-decreasing).
    """

    offsets: Array
    sources: Array
    targets: Array
    counts: Array
    level_offsets: Array
    target_levels: Array


class NodeNeighborList(NamedTuple):
    """Compressed near-field neighbor list for leaf nodes."""

    offsets: Array
    neighbors: Array
    leaf_indices: Array
    counts: Array


class DualTreeWalkResult(NamedTuple):
    """Container for far-field and near-field results from a dual walk."""

    interaction_offsets: Array
    interaction_sources: Array
    interaction_targets: Array
    interaction_tags: Array
    interaction_counts: Array
    neighbor_offsets: Array
    neighbor_indices: Array
    neighbor_counts: Array
    leaf_indices: Array
    far_pair_count: Array
    near_pair_count: Array
    queue_overflow: Array
    far_overflow: Array
    near_overflow: Array
    accept_decisions: Array
    near_decisions: Array
    refine_decisions: Array


def _default_pair_actions(
    *,
    mac_ok: Array,
    valid_pairs: Array,
    different_nodes: Array,
    target_leaf: Array,
    source_leaf: Array,
) -> tuple[Array, Array]:
    """Return default traversal actions and placeholder tags."""

    should_accept = mac_ok
    should_near = valid_pairs & (~mac_ok) & target_leaf & source_leaf & different_nodes
    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=INDEX_DTYPE)
    actions = jnp.where(should_accept, as_index(_ACTION_ACCEPT), actions)
    actions = jnp.where(should_near, as_index(_ACTION_NEAR), actions)
    tags = jnp.full(valid_pairs.shape, -1, dtype=INDEX_DTYPE)
    return actions, tags


def _resolve_pair_actions(
    *,
    pair_policy: Optional[PairPolicy],
    policy_state: object,
    valid_pairs: Array,
    mac_ok: Array,
    different_nodes: Array,
    target_leaf: Array,
    source_leaf: Array,
    same_node: Array,
    target_nodes: Array,
    source_nodes: Array,
    center_target: Array,
    center_source: Array,
    dist_sq: Array,
    extent_target: Array,
    extent_source: Array,
) -> tuple[Array, Array, Array]:
    """Resolve per-pair traversal actions and optional integer tags."""

    if pair_policy is None:
        actions, tags = _default_pair_actions(
            mac_ok=mac_ok,
            valid_pairs=valid_pairs,
            different_nodes=different_nodes,
            target_leaf=target_leaf,
            source_leaf=source_leaf,
        )
        return actions, tags, tags

    forward_actions, forward_tags = pair_policy(
        policy_state,
        valid_pairs=valid_pairs,
        mac_ok=mac_ok,
        different_nodes=different_nodes,
        target_leaf=target_leaf,
        source_leaf=source_leaf,
        same_node=same_node,
        target_nodes=target_nodes,
        source_nodes=source_nodes,
        center_target=center_target,
        center_source=center_source,
        dist_sq=dist_sq,
        extent_target=extent_target,
        extent_source=extent_source,
    )
    reverse_actions, reverse_tags = pair_policy(
        policy_state,
        valid_pairs=valid_pairs,
        mac_ok=mac_ok,
        different_nodes=different_nodes,
        target_leaf=source_leaf,
        source_leaf=target_leaf,
        same_node=same_node,
        target_nodes=source_nodes,
        source_nodes=target_nodes,
        center_target=center_source,
        center_source=center_target,
        dist_sq=dist_sq,
        extent_target=extent_source,
        extent_source=extent_target,
    )
    forward_actions = jnp.asarray(forward_actions, dtype=INDEX_DTYPE)
    reverse_actions = jnp.asarray(reverse_actions, dtype=INDEX_DTYPE)
    forward_tags = jnp.asarray(forward_tags, dtype=INDEX_DTYPE)
    reverse_tags = jnp.asarray(reverse_tags, dtype=INDEX_DTYPE)
    forward_actions = jnp.where(valid_pairs, forward_actions, as_index(_ACTION_REFINE))
    reverse_actions = jnp.where(valid_pairs, reverse_actions, as_index(_ACTION_REFINE))
    accept_both = (forward_actions == as_index(_ACTION_ACCEPT)) & (
        reverse_actions == as_index(_ACTION_ACCEPT)
    )
    near_both = (forward_actions == as_index(_ACTION_NEAR)) & (
        reverse_actions == as_index(_ACTION_NEAR)
    )
    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=INDEX_DTYPE)
    actions = jnp.where(accept_both, as_index(_ACTION_ACCEPT), actions)
    actions = jnp.where(near_both, as_index(_ACTION_NEAR), actions)
    forward_tags = jnp.where(accept_both, forward_tags, as_index(-1))
    reverse_tags = jnp.where(accept_both, reverse_tags, as_index(-1))
    return actions, forward_tags, reverse_tags


def _per_key_prefix(keys: Array, mask: Array, num_segments: int = 0) -> Array:
    """Exclusive prefix count per key among masked entries.

    For each masked position *i*, count how many earlier masked positions
    share the same key.  Uses ``argsort`` for *O(B log B)* complexity
    instead of the *O(B²)* pairwise comparison matrix, enabling
    efficient processing of large batches (wavefront traversal).

    Parameters
    ----------
    keys : Array
        Per-entry key values, shape ``(B,)``.
    mask : Array
        Boolean mask, shape ``(B,)``.  Only masked entries participate.
    num_segments : int
        Unused; kept for call-site compatibility.

    Returns
    -------
    Array
        Shape ``(B,)`` with exclusive prefix counts.  Unmasked entries
        are zero.
    """
    sentinel_high = jnp.array(jnp.iinfo(INDEX_DTYPE).max, dtype=INDEX_DTYPE)
    sentinel_low = jnp.array(jnp.iinfo(INDEX_DTYPE).min, dtype=INDEX_DTYPE)
    effective_keys = jnp.where(mask, keys, sentinel_high)

    # Stable sort groups same-key entries together while preserving
    # their original relative order within each group.
    order = jnp.argsort(effective_keys, stable=True)
    sorted_keys = effective_keys[order]
    sorted_mask = mask[order]

    # Detect group boundaries (where the key changes).
    prev_keys = jnp.concatenate(
        [jnp.full((1,), sentinel_low, dtype=INDEX_DTYPE), sorted_keys[:-1]]
    )
    is_group_start = sorted_keys != prev_keys

    # Inclusive cumulative count of masked entries.
    cumsum = jnp.cumsum(sorted_mask.astype(INDEX_DTYPE))

    # At each group boundary, record how many masked entries preceded
    # this group.  Forward-fill via prefix-maximum (cumsum is monotone,
    # so boundary values are non-decreasing).
    group_base = jnp.where(
        is_group_start,
        cumsum - sorted_mask.astype(INDEX_DTYPE),
        as_index(0),
    )
    group_base = lax.associative_scan(jnp.maximum, group_base)

    # 0-based position within each key group.
    prefix_sorted = cumsum - group_base - sorted_mask.astype(INDEX_DTYPE)

    # Un-sort back to original positions.
    result = jnp.zeros(keys.shape[0], dtype=INDEX_DTYPE)
    result = result.at[order].set(prefix_sorted)

    return jnp.where(mask, result, as_index(0))


def _propagate_extents(parent: Array, extents: Array) -> Array:
    """Propagate zero extents up the tree to obtain conservative sizes.

    Uses pointer doubling: each node maintains a "shortcut" pointer
    that doubles its reach each round.  On round *k* the shortcut
    jumps 2^k ancestors, so convergence takes O(log depth) rounds
    instead of O(depth).
    """
    n = extents.shape[0]
    # shortcut[i] starts as parent[i] (or i for the root).
    shortcut = jnp.where(parent >= 0, parent, jnp.arange(n, dtype=parent.dtype))
    result = extents.copy()

    def cond_fn(state):
        _shortcut, _result, changed = state
        return changed

    def body_fn(state):
        sc, res, _changed = state
        # Pointer doubling: jump through the shortcut itself.
        new_sc = sc[sc]
        # Adopt the extent of the shortcut target when own is zero.
        candidate = extents[new_sc]
        new_res = jnp.where(res <= 0.0, candidate, res)
        changed = jnp.any(new_res != res) | jnp.any(new_sc != sc)
        return new_sc, new_res, changed

    _, result, _ = lax.while_loop(cond_fn, body_fn, (shortcut, result, jnp.bool_(True)))
    return result


def _compute_node_depths(parent: Array) -> Array:
    """Return the depth of every node (root depth = 0).

    Uses pointer doubling for O(log depth) convergence.  Each node
    keeps a *depth-to-root* counter and a shortcut pointer.  On each
    round the shortcut doubles its reach and accumulated depth
    contributions are propagated.
    """
    total_nodes = parent.shape[0]
    is_root = parent < 0
    # dist[i] = accumulated distance along the shortcut chain.
    # Initially 1 for non-root nodes (edge to parent), 0 for root.
    dist = jnp.where(is_root, as_index(0), as_index(1))
    # shortcut[i] = parent[i] for non-root, i for root.
    shortcut = jnp.where(
        is_root,
        jnp.arange(total_nodes, dtype=parent.dtype),
        parent,
    )

    def cond_fn(state):
        _sc, _d, changed = state
        return changed

    def body_fn(state):
        sc, d, _changed = state
        # Pointer doubling: add distance of shortcut target.
        new_d = d + d[sc]
        new_sc = sc[sc]
        changed = jnp.any(new_sc != sc)
        return new_sc, new_d, changed

    _, depth, _ = lax.while_loop(cond_fn, body_fn, (shortcut, dist, jnp.bool_(True)))
    return depth


def _compute_effective_extents(parent: Array, extents: Array) -> Array:
    """Baseline effective extents used for far-field interactions."""

    return _propagate_extents(parent, extents)


def _compute_leaf_effective_extents(
    parent: Array,
    extents: Array,
    num_internal: int,
) -> Array:
    """Effective extents with depth-based padding applied to leaves."""

    propagated = _propagate_extents(parent, extents)
    depths = _compute_node_depths(parent)
    # Root is the unique node whose parent is -1.
    root_idx = as_index(jnp.argmin(parent))
    root_extent = propagated[root_idx]
    depth_scaling = root_extent / (2.0 ** (depths.astype(extents.dtype) + 1.0))

    indices = jnp.arange(extents.shape[0], dtype=INDEX_DTYPE)
    leaf_mask = indices >= as_index(num_internal)

    return jnp.where(
        leaf_mask & (extents <= 0.0),
        depth_scaling,
        propagated,
    )


def _exclusive_cumsum(values: Array) -> Array:
    zeros = jnp.zeros((1,), dtype=values.dtype)
    cumsum = jnp.cumsum(values, axis=0, dtype=values.dtype)
    return jnp.concatenate([zeros, cumsum])


def _raise_if_true(flag, message: str) -> None:
    if isinstance(flag, jax.core.Tracer):

        def _callback(value):
            if bool(value):
                raise RuntimeError(message)

        jax.debug.callback(_callback, flag)
    else:
        if bool(flag):
            raise RuntimeError(message)


@partial(
    jax.jit,
    static_argnames=(
        "max_interactions_per_node",
        "max_neighbors_per_leaf",
        "max_pair_queue",
        "mac_type",
        "pair_policy",
        "collect_far",
        "collect_near",
        "process_block",
    ),
)
def _dual_tree_walk_impl(
    tree: object,
    geometry: TreeGeometry,
    nodes_by_level: Array,
    theta: float,
    *,
    mac_type: MACType = "bh",
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    dehnen_radius_scale: float = 1.0,
    max_interactions_per_node: int,
    max_neighbors_per_leaf: int,
    max_pair_queue: int,
    collect_far: bool = True,
    collect_near: bool = True,
    process_block: int = _DEFAULT_PAIR_BATCH,
) -> DualTreeWalkResult:
    parent = tree.parent
    left_child = tree.left_child
    right_child = tree.right_child
    total_nodes = parent.shape[0]
    num_internal = left_child.shape[0]

    if num_internal == 0:
        leaf_indices = jnp.arange(total_nodes, dtype=INDEX_DTYPE)
        leaf_count = leaf_indices.shape[0]
        zero_nodes = jnp.zeros((total_nodes,), dtype=INDEX_DTYPE)
        zero_leaves = jnp.zeros((leaf_count,), dtype=INDEX_DTYPE)
        return DualTreeWalkResult(
            interaction_offsets=jnp.zeros((total_nodes + 1,), dtype=INDEX_DTYPE),
            interaction_sources=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_targets=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_tags=jnp.zeros((0,), dtype=INDEX_DTYPE),
            interaction_counts=zero_nodes,
            neighbor_offsets=jnp.zeros((leaf_count + 1,), dtype=INDEX_DTYPE),
            neighbor_indices=jnp.zeros((0,), dtype=INDEX_DTYPE),
            neighbor_counts=zero_leaves,
            leaf_indices=leaf_indices,
            far_pair_count=as_index(0),
            near_pair_count=as_index(0),
            queue_overflow=jnp.bool_(False),
            far_overflow=jnp.bool_(False),
            near_overflow=jnp.bool_(False),
            accept_decisions=as_index(0),
            near_decisions=as_index(0),
            refine_decisions=as_index(0),
        )

    centers = jnp.asarray(geometry.center)
    # NOTE: We maintain two extent measures:
    # - max_extent: conservative box half-extent (L_inf radius)
    # - radius: bounding-sphere radius (L2 norm of half-extents)
    # Different MAC variants prefer different measures.
    extents_box = jnp.asarray(geometry.max_extent)
    extents_sphere = jnp.asarray(geometry.radius)

    theta_sq = (jnp.asarray(theta, dtype=centers.dtype)) ** 2
    # Far/leaf effective extents are built from a per-node base extent.
    effective_extents_box_far = _compute_effective_extents(parent, extents_box)
    effective_extents_box_leaf = _compute_leaf_effective_extents(
        parent,
        extents_box,
        num_internal,
    )
    effective_extents_sphere_far = _compute_effective_extents(parent, extents_sphere)
    effective_extents_sphere_leaf = _compute_leaf_effective_extents(
        parent,
        extents_sphere,
        num_internal,
    )
    # Root is the unique node whose parent is -1.
    root_idx = as_index(jnp.argmin(parent))

    # Leaves are the nodes with indices >= num_internal. Construct the
    # leaf index array without calling jnp.nonzero/jnp.where to remain
    # JIT-traceable (same approach as the count-only pass).
    leaf_indices = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    num_leaves = total_nodes - num_internal
    # Map node index -> leaf ordinal (0..num_leaves-1) or -1 for non-leaves
    positions = jnp.arange(total_nodes, dtype=INDEX_DTYPE) - as_index(num_internal)
    leaf_position = jnp.where(positions >= as_index(0), positions, as_index(-1))

    stack_capacity = max(int(max_pair_queue), 4)

    pair_stack_target = jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
    pair_stack_source = jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
    pair_stack_target = pair_stack_target.at[0].set(root_idx)
    pair_stack_source = pair_stack_source.at[0].set(root_idx)
    stack_size = as_index(1)

    far_counts = jnp.zeros((total_nodes,), dtype=INDEX_DTYPE)
    if collect_far:
        max_total_far_pairs = max(total_nodes * max_interactions_per_node, 1)
        far_buffer = jnp.full(
            (total_nodes, max_interactions_per_node),
            -1,
            dtype=INDEX_DTYPE,
        )
        far_tag_buffer = jnp.full(
            (total_nodes, max_interactions_per_node),
            -1,
            dtype=INDEX_DTYPE,
        )
    else:
        max_total_far_pairs = 0
        far_buffer = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        far_tag_buffer = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
    far_pair_total = as_index(0)

    near_counts = jnp.zeros((num_leaves,), dtype=INDEX_DTYPE)
    if collect_near:
        max_total_near_pairs = max(num_leaves * max_neighbors_per_leaf, 1)
        neighbor_buffer = jnp.full(
            (num_leaves, max_neighbors_per_leaf),
            -1,
            dtype=INDEX_DTYPE,
        )
    else:
        max_total_near_pairs = 0
        neighbor_buffer = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
    near_pair_total = as_index(0)

    overflow_wf = jnp.bool_(False)
    far_overflow = jnp.bool_(False)
    near_overflow = jnp.bool_(False)

    num_internal_val = as_index(num_internal)
    node_indices = jnp.arange(total_nodes, dtype=INDEX_DTYPE)
    # Choose the geometric proxy required by the MAC.
    # - bh: historically uses box max half-extent
    # - dehnen/engblom: use bounding-sphere radii
    use_sphere = (mac_type == "dehnen") | (mac_type == "engblom")
    extents_far = jnp.where(
        use_sphere,
        effective_extents_sphere_far,
        effective_extents_box_far,
    )
    extents_leaf = jnp.where(
        use_sphere,
        effective_extents_sphere_leaf,
        effective_extents_box_leaf,
    )
    dehnen_scale = jnp.asarray(dehnen_radius_scale, dtype=centers.dtype)
    is_dehnen = jnp.asarray(mac_type == "dehnen")
    extents_far = jnp.where(is_dehnen, dehnen_scale * extents_far, extents_far)
    extents_leaf = jnp.where(is_dehnen, dehnen_scale * extents_leaf, extents_leaf)
    mac_extents = jnp.where(
        node_indices >= num_internal_val,
        extents_leaf,
        extents_far,
    )

    leaf_fill = jnp.full((total_nodes - num_internal,), -1, dtype=INDEX_DTYPE)
    left_child_full = jnp.concatenate([left_child, leaf_fill], axis=0)
    right_child_full = jnp.concatenate([right_child, leaf_fill], axis=0)

    filler_pair = jnp.asarray([-1, -1], dtype=INDEX_DTYPE)
    empty_pairs = jnp.tile(filler_pair[None, :], (_MAX_REFINEMENT_PAIRS, 1))

    def _sorted_pair(a: Array, b: Array) -> Array:
        lo = jnp.minimum(a, b)
        hi = jnp.maximum(a, b)
        return jnp.stack([lo, hi], axis=0)

    def _refine_pairs_branchless(
        tgt: Array,
        src: Array,
        same: Array,
        split_both_flag: Array,
        split_target_flag: Array,
        split_source_flag: Array,
        tgt_left: Array,
        tgt_right: Array,
        src_left: Array,
        src_right: Array,
    ) -> Array:
        """Branchless refine-pair computation.

        Computes all 5 possible outcomes using ``jnp.where`` and selects
        the correct one.  Under ``vmap`` this avoids the overhead of
        nested ``lax.cond`` which evaluates all branches anyway.
        """
        # Pre-compute all candidate pairs using _sorted_pair.
        # Case 1: split_both & same → (tL,tL), (tL,tR), (tR,tR), filler
        p_bs_0 = _sorted_pair(tgt_left, tgt_left)
        p_bs_1 = _sorted_pair(tgt_left, tgt_right)
        p_bs_2 = _sorted_pair(tgt_right, tgt_right)
        both_same = jnp.stack([p_bs_0, p_bs_1, p_bs_2, filler_pair], axis=0)

        # Case 2: split_both & ~same → (tL,sL), (tL,sR), (tR,sL), (tR,sR)
        p_bc_0 = _sorted_pair(tgt_left, src_left)
        p_bc_1 = _sorted_pair(tgt_left, src_right)
        p_bc_2 = _sorted_pair(tgt_right, src_left)
        p_bc_3 = _sorted_pair(tgt_right, src_right)
        both_cross = jnp.stack([p_bc_0, p_bc_1, p_bc_2, p_bc_3], axis=0)

        # Case 3: split_target only → (tL,src), (tR,src), filler, filler
        p_st_0 = _sorted_pair(tgt_left, src)
        p_st_1 = _sorted_pair(tgt_right, src)
        split_tgt = jnp.stack([p_st_0, p_st_1, filler_pair, filler_pair], axis=0)

        # Case 4: split_source only → (tgt,sL), (tgt,sR), filler, filler
        p_ss_0 = _sorted_pair(tgt, src_left)
        p_ss_1 = _sorted_pair(tgt, src_right)
        split_src = jnp.stack([p_ss_0, p_ss_1, filler_pair, filler_pair], axis=0)

        # Select: priority order split_both > split_target > split_source > empty
        result = empty_pairs
        result = jnp.where(split_source_flag, split_src, result)
        result = jnp.where(split_target_flag, split_tgt, result)
        result = jnp.where(split_both_flag & ~same, both_cross, result)
        result = jnp.where(split_both_flag & same, both_same, result)
        return result

    refine_vm = jax.vmap(
        _refine_pairs_branchless,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

    # --- Wavefront-based traversal ---
    # Instead of a LIFO stack that processes ``process_block`` pairs per
    # iteration, use a wavefront buffer that holds ALL pairs for the
    # current "generation".  Each round evaluates the full wavefront in
    # parallel, records accept/near results, and collects refined child
    # pairs into a *new* wavefront.  This reduces the number of
    # while_loop rounds from O(total_pairs / process_block) to
    # O(tree_depth) while maximising data-level parallelism per round.
    wavefront_capacity = stack_capacity
    wf_indices = jnp.arange(wavefront_capacity, dtype=INDEX_DTYPE)

    def cond_fun(state):
        (
            _wf_target,
            _wf_source,
            current_size,
            _far_buf,
            _far_tag_buf,
            _far_cnts,
            _far_ptr,
            _nbr_buf,
            _nbr_cnts,
            _nbr_ptr,
            wf_over,
            far_over,
            near_over,
            _accept_decisions,
            _near_decisions,
            _refine_decisions,
        ) = state
        return (current_size > 0) & (~wf_over) & (~far_over) & (~near_over)

    def body_fun(state):
        (
            wf_target,
            wf_source,
            wf_size,
            far_buffer,
            far_tag_buffer,
            far_counts,
            far_pair_total,
            neighbor_buffer,
            near_counts,
            near_pair_total,
            wf_overflow,
            far_overflow,
            near_overflow,
            accept_decisions,
            near_decisions,
            refine_decisions,
        ) = state

        # --- Read entire wavefront ---
        valid_mask = wf_indices < wf_size
        targets = jnp.where(valid_mask, wf_target, as_index(-1))
        sources = jnp.where(valid_mask, wf_source, as_index(-1))

        valid_pairs = valid_mask & (targets >= 0) & (sources >= 0)
        valid_pairs_bool = valid_pairs.astype(jnp.bool_)
        safe_targets = jnp.where(valid_pairs, targets, as_index(0))
        safe_sources = jnp.where(valid_pairs, sources, as_index(0))

        center_target = centers[safe_targets]
        center_source = centers[safe_sources]
        delta = center_target - center_source
        valid_scale = valid_pairs[:, None].astype(delta.dtype)
        delta = delta * valid_scale
        dist_sq = jnp.sum(delta * delta, axis=1)

        different_nodes = targets != sources
        extent_mac_target = mac_extents[safe_targets]
        extent_mac_source = mac_extents[safe_sources]
        mac_ok = _compute_mac_ok(
            mac_type=mac_type,
            theta_sq=theta_sq,
            dist_sq=dist_sq,
            extent_target=extent_mac_target,
            extent_source=extent_mac_source,
            valid_pairs=valid_pairs_bool,
            different_nodes=different_nodes,
        )

        target_internal = valid_pairs_bool & (targets < num_internal_val)
        source_internal = valid_pairs_bool & (sources < num_internal_val)
        target_leaf = valid_pairs_bool & (~target_internal)
        source_leaf = valid_pairs_bool & (~source_internal)
        same_node = valid_pairs_bool & (targets == sources)
        pair_actions, pair_tags, pair_tags_rev = _resolve_pair_actions(
            pair_policy=pair_policy,
            policy_state=policy_state,
            valid_pairs=valid_pairs_bool,
            mac_ok=mac_ok,
            different_nodes=different_nodes,
            target_leaf=target_leaf,
            source_leaf=source_leaf,
            same_node=same_node,
            target_nodes=targets,
            source_nodes=sources,
            center_target=center_target,
            center_source=center_source,
            dist_sq=dist_sq,
            extent_target=extent_mac_target,
            extent_source=extent_mac_source,
        )

        should_accept = valid_pairs_bool & (pair_actions == as_index(_ACTION_ACCEPT))
        should_near = valid_pairs_bool & (pair_actions == as_index(_ACTION_NEAR))
        do_refine = valid_pairs_bool & (pair_actions == as_index(_ACTION_REFINE))

        batch_accept = jnp.sum(should_accept.astype(INDEX_DTYPE))
        batch_near = jnp.sum(should_near.astype(INDEX_DTYPE))
        batch_refine = jnp.sum(do_refine.astype(INDEX_DTYPE))

        extent_target = mac_extents[safe_targets]
        extent_source = mac_extents[safe_sources]

        split_target = (
            do_refine
            & target_internal
            & (same_node | (~source_internal) | (extent_target >= extent_source))
        )
        split_source = (
            do_refine
            & source_internal
            & (same_node | (~target_internal) | (extent_source > extent_target))
        )
        split_both = split_target & split_source

        target_left = left_child_full[safe_targets]
        target_right = right_child_full[safe_targets]
        source_left = left_child_full[safe_sources]
        source_right = right_child_full[safe_sources]

        leaf_pos_target = leaf_position[safe_targets]
        leaf_pos_source = leaf_position[safe_sources]

        if collect_far:
            # Vectorised accept recording.  For each accepted pair
            # (tgt, src) we record src in far_buffer[tgt, ...] AND tgt in
            # far_buffer[src, ...].  To avoid the sequential while_loop we
            # compute a per-key exclusive prefix sum that assigns each
            # entry its slot within the node's buffer row.
            accept_mask = should_accept
            accept_targets_a = jnp.where(accept_mask, targets, as_index(-1))
            accept_sources_a = jnp.where(accept_mask, sources, as_index(-1))
            accept_tags_fwd = jnp.where(accept_mask, pair_tags, as_index(-1))
            accept_tags_bwd = jnp.where(accept_mask, pair_tags_rev, as_index(-1))

            # --- forward direction: record src at far_buffer[tgt, ...] ---
            fwd_prefix = _per_key_prefix(accept_targets_a, accept_mask, total_nodes)
            fwd_slot = far_counts[safe_targets] + fwd_prefix

            # Check capacity: any slot >= max_interactions_per_node → overflow
            fwd_ok = accept_mask & (fwd_slot < as_index(max_interactions_per_node))
            far_overflow = far_overflow | jnp.any(
                accept_mask & (fwd_slot >= as_index(max_interactions_per_node))
            )

            # Scatter into far_buffer and far_tag_buffer
            oob_far = as_index(total_nodes)  # out-of-bounds node index
            fwd_node = jnp.where(fwd_ok, safe_targets, oob_far)
            fwd_col = jnp.where(
                fwd_ok, fwd_slot, as_index(max_interactions_per_node - 1)
            )
            far_buffer = far_buffer.at[fwd_node, fwd_col].set(
                jnp.where(fwd_ok, safe_sources, as_index(-1)), mode="drop"
            )
            far_tag_buffer = far_tag_buffer.at[fwd_node, fwd_col].set(
                jnp.where(fwd_ok, accept_tags_fwd, as_index(-1)), mode="drop"
            )

            # --- backward direction: record tgt at far_buffer[src, ...] ---
            # The forward writes already updated far_counts conceptually;
            # we need the updated counts for the backward pass.
            fwd_incr = jax.ops.segment_sum(
                fwd_ok.astype(INDEX_DTYPE), safe_targets, num_segments=total_nodes
            )
            far_counts_after_fwd = far_counts + fwd_incr

            bwd_prefix = _per_key_prefix(accept_sources_a, accept_mask, total_nodes)
            bwd_slot = far_counts_after_fwd[safe_sources] + bwd_prefix
            bwd_ok = accept_mask & (bwd_slot < as_index(max_interactions_per_node))
            far_overflow = far_overflow | jnp.any(
                accept_mask & (bwd_slot >= as_index(max_interactions_per_node))
            )

            bwd_node = jnp.where(bwd_ok, safe_sources, oob_far)
            bwd_col = jnp.where(
                bwd_ok, bwd_slot, as_index(max_interactions_per_node - 1)
            )
            far_buffer = far_buffer.at[bwd_node, bwd_col].set(
                jnp.where(bwd_ok, safe_targets, as_index(-1)), mode="drop"
            )
            far_tag_buffer = far_tag_buffer.at[bwd_node, bwd_col].set(
                jnp.where(bwd_ok, accept_tags_bwd, as_index(-1)), mode="drop"
            )

            bwd_incr = jax.ops.segment_sum(
                bwd_ok.astype(INDEX_DTYPE),
                safe_sources,
                num_segments=total_nodes,
            )
            far_counts = far_counts_after_fwd + bwd_incr
            far_pair_total = (
                far_pair_total
                + jnp.sum(fwd_ok.astype(INDEX_DTYPE))
                + jnp.sum(bwd_ok.astype(INDEX_DTYPE))
            )

        if collect_near:
            # Vectorised near recording.  For each near pair (tgt, src)
            # both directions are recorded: src at neighbor_buffer[pos_t, ...]
            # and tgt at neighbor_buffer[pos_s, ...].
            near_mask = should_near
            near_pos_t = jnp.where(near_mask, leaf_pos_target, as_index(-1))
            near_pos_s = jnp.where(near_mask, leaf_pos_source, as_index(-1))
            near_tgt_nodes = jnp.where(near_mask, targets, as_index(-1))
            near_src_nodes = jnp.where(near_mask, sources, as_index(-1))

            # Forward: record src at neighbor_buffer[pos_t, ...]
            safe_pos_t = jnp.where(near_mask, near_pos_t, as_index(0))
            safe_pos_s = jnp.where(near_mask, near_pos_s, as_index(0))

            fwd_near_prefix = _per_key_prefix(safe_pos_t, near_mask, num_leaves)
            fwd_near_slot = near_counts[safe_pos_t] + fwd_near_prefix
            fwd_near_ok = near_mask & (fwd_near_slot < as_index(max_neighbors_per_leaf))
            near_overflow = near_overflow | jnp.any(
                near_mask & (fwd_near_slot >= as_index(max_neighbors_per_leaf))
            )

            oob_leaf = as_index(num_leaves)
            fwd_near_row = jnp.where(fwd_near_ok, safe_pos_t, oob_leaf)
            fwd_near_col = jnp.where(
                fwd_near_ok,
                fwd_near_slot,
                as_index(max_neighbors_per_leaf - 1),
            )
            neighbor_buffer = neighbor_buffer.at[fwd_near_row, fwd_near_col].set(
                jnp.where(fwd_near_ok, near_src_nodes, as_index(-1)),
                mode="drop",
            )

            fwd_near_incr = jax.ops.segment_sum(
                fwd_near_ok.astype(INDEX_DTYPE),
                safe_pos_t,
                num_segments=num_leaves,
            )
            near_counts_after_fwd = near_counts + fwd_near_incr

            # Backward: record tgt at neighbor_buffer[pos_s, ...]
            bwd_near_prefix = _per_key_prefix(safe_pos_s, near_mask, num_leaves)
            bwd_near_slot = near_counts_after_fwd[safe_pos_s] + bwd_near_prefix
            bwd_near_ok = near_mask & (bwd_near_slot < as_index(max_neighbors_per_leaf))
            near_overflow = near_overflow | jnp.any(
                near_mask & (bwd_near_slot >= as_index(max_neighbors_per_leaf))
            )

            bwd_near_row = jnp.where(bwd_near_ok, safe_pos_s, oob_leaf)
            bwd_near_col = jnp.where(
                bwd_near_ok,
                bwd_near_slot,
                as_index(max_neighbors_per_leaf - 1),
            )
            neighbor_buffer = neighbor_buffer.at[bwd_near_row, bwd_near_col].set(
                jnp.where(bwd_near_ok, near_tgt_nodes, as_index(-1)),
                mode="drop",
            )

            bwd_near_incr = jax.ops.segment_sum(
                bwd_near_ok.astype(INDEX_DTYPE),
                safe_pos_s,
                num_segments=num_leaves,
            )
            near_counts = near_counts_after_fwd + bwd_near_incr
            near_pair_total = (
                near_pair_total
                + jnp.sum(fwd_near_ok.astype(INDEX_DTYPE))
                + jnp.sum(bwd_near_ok.astype(INDEX_DTYPE))
            )

        refine_pairs = refine_vm(
            targets,
            sources,
            same_node,
            split_both.astype(jnp.bool_),
            split_target.astype(jnp.bool_),
            split_source.astype(jnp.bool_),
            target_left,
            target_right,
            source_left,
            source_right,
        )

        # --- Build new wavefront from refined pairs ---
        refine_targets = refine_pairs[..., 0].reshape(
            (wavefront_capacity * _MAX_REFINEMENT_PAIRS,)
        )
        refine_sources = refine_pairs[..., 1].reshape(
            (wavefront_capacity * _MAX_REFINEMENT_PAIRS,)
        )

        valid_push = (refine_targets >= 0) & (refine_sources >= 0)

        # Exclusive prefix sum gives each valid pair a unique slot
        # in the new wavefront.
        push_prefix = jnp.cumsum(valid_push.astype(INDEX_DTYPE)) - valid_push.astype(
            INDEX_DTYPE
        )

        # Capacity check.
        push_ok = valid_push & (push_prefix < as_index(wavefront_capacity))
        wf_overflow = wf_overflow | jnp.any(
            valid_push & (push_prefix >= as_index(wavefront_capacity))
        )

        # Sorted push: lo = min(tgt, src), hi = max(tgt, src).
        push_lo = jnp.minimum(refine_targets, refine_sources)
        push_hi = jnp.maximum(refine_targets, refine_sources)

        oob_wf = as_index(wavefront_capacity)
        safe_slot = jnp.where(push_ok, push_prefix, oob_wf)
        new_wf_target = (
            jnp.full((wavefront_capacity,), -1, dtype=INDEX_DTYPE)
            .at[safe_slot]
            .set(jnp.where(push_ok, push_lo, as_index(-1)), mode="drop")
        )
        new_wf_source = (
            jnp.full((wavefront_capacity,), -1, dtype=INDEX_DTYPE)
            .at[safe_slot]
            .set(jnp.where(push_ok, push_hi, as_index(-1)), mode="drop")
        )
        new_wf_size = jnp.sum(push_ok.astype(INDEX_DTYPE))

        return (
            new_wf_target,
            new_wf_source,
            new_wf_size,
            far_buffer,
            far_tag_buffer,
            far_counts,
            far_pair_total,
            neighbor_buffer,
            near_counts,
            near_pair_total,
            wf_overflow,
            far_overflow,
            near_overflow,
            accept_decisions + batch_accept,
            near_decisions + batch_near,
            refine_decisions + batch_refine,
        )

    (
        _wf_target_out,
        _wf_source_out,
        _wf_size_out,
        far_buffer,
        far_tag_buffer,
        far_counts,
        far_pair_total,
        neighbor_buffer,
        near_counts,
        near_pair_total,
        overflow_wf,
        far_overflow,
        near_overflow,
        accept_decisions,
        near_decisions,
        refine_decisions,
    ) = lax.while_loop(
        cond_fun,
        body_fun,
        (
            pair_stack_target,
            pair_stack_source,
            stack_size,
            far_buffer,
            far_tag_buffer,
            far_counts,
            far_pair_total,
            neighbor_buffer,
            near_counts,
            near_pair_total,
            overflow_wf,
            far_overflow,
            near_overflow,
            as_index(0),
            as_index(0),
            as_index(0),
        ),
    )

    interaction_offsets = _exclusive_cumsum(far_counts)
    neighbor_offsets = _exclusive_cumsum(near_counts)

    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    num_nodes_level = nodes_by_level.shape[0]

    if collect_far:
        # Vectorised compression: flatten the 2D far_buffer into sorted 1D
        # arrays using the precomputed per-node counts and level ordering.
        #
        # 1. Compute per-node write offsets in level order.
        level_counts = far_counts[nodes_by_level]
        level_cumsum = jnp.cumsum(level_counts, dtype=INDEX_DTYPE)
        level_offsets_write = jnp.concatenate(
            [jnp.zeros((1,), dtype=INDEX_DTYPE), level_cumsum]
        )

        # 2. Build flat (num_nodes_level * K) index arrays where K is
        #    max_interactions_per_node.
        K = max_interactions_per_node
        node_rep = jnp.repeat(
            jnp.arange(num_nodes_level, dtype=INDEX_DTYPE), K
        )  # which level-order position
        slot_rep = jnp.tile(
            jnp.arange(K, dtype=INDEX_DTYPE), num_nodes_level
        )  # which slot within that node
        level_node_ids = nodes_by_level[node_rep]  # actual node id
        per_node_count = far_counts[level_node_ids]
        valid = slot_rep < per_node_count

        # 3. Compute write position: level_offsets_write[node_rep] + slot_rep
        write_pos = level_offsets_write[node_rep] + slot_rep

        # 4. Gather source values and tags from the 2D buffers.
        src_vals = far_buffer[level_node_ids, slot_rep]
        tag_vals = far_tag_buffer[level_node_ids, slot_rep]

        # 5. Scatter into flat output arrays.  Invalid entries get an
        #    out-of-bounds write position that ``mode="drop"`` silently
        #    discards, so they never clobber valid data.
        oob = as_index(max_total_far_pairs)  # guaranteed out-of-bounds
        safe_write = jnp.where(valid, write_pos, oob)

        interaction_sources = (
            jnp.zeros((max_total_far_pairs,), dtype=INDEX_DTYPE)
            .at[safe_write]
            .set(src_vals, mode="drop")
        )
        interaction_targets = (
            jnp.zeros((max_total_far_pairs,), dtype=INDEX_DTYPE)
            .at[safe_write]
            .set(level_node_ids, mode="drop")
        )
        interaction_tags = (
            jnp.full((max_total_far_pairs,), -1, dtype=INDEX_DTYPE)
            .at[safe_write]
            .set(tag_vals, mode="drop")
        )
    else:
        interaction_sources = jnp.zeros((0,), dtype=INDEX_DTYPE)
        interaction_targets = jnp.zeros((0,), dtype=INDEX_DTYPE)
        interaction_tags = jnp.zeros((0,), dtype=INDEX_DTYPE)

    if collect_near:
        # Vectorised compression for near-field buffer.
        K_near = max_neighbors_per_leaf
        near_node_rep = jnp.repeat(jnp.arange(num_leaves, dtype=INDEX_DTYPE), K_near)
        near_slot_rep = jnp.tile(jnp.arange(K_near, dtype=INDEX_DTYPE), num_leaves)
        near_per_leaf_count = near_counts[near_node_rep]
        near_valid = near_slot_rep < near_per_leaf_count

        # Per-leaf write offsets (leaves are already in order 0..num_leaves-1).
        near_cumsum = jnp.cumsum(near_counts, dtype=INDEX_DTYPE)
        near_offsets_write = jnp.concatenate(
            [jnp.zeros((1,), dtype=INDEX_DTYPE), near_cumsum]
        )
        near_write_pos = near_offsets_write[near_node_rep] + near_slot_rep

        near_vals = neighbor_buffer[near_node_rep, near_slot_rep]
        oob_near = as_index(max_total_near_pairs)
        safe_near_write = jnp.where(near_valid, near_write_pos, oob_near)

        neighbor_indices = (
            jnp.zeros((max_total_near_pairs,), dtype=INDEX_DTYPE)
            .at[safe_near_write]
            .set(near_vals, mode="drop")
        )
    else:
        neighbor_indices = jnp.zeros((0,), dtype=INDEX_DTYPE)

    far_pair_count = jnp.sum(far_counts, dtype=INDEX_DTYPE)
    near_pair_count = jnp.sum(near_counts, dtype=INDEX_DTYPE)

    return DualTreeWalkResult(
        interaction_offsets=interaction_offsets,
        interaction_sources=interaction_sources,
        interaction_targets=interaction_targets,
        interaction_tags=interaction_tags,
        interaction_counts=far_counts,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        neighbor_counts=near_counts,
        leaf_indices=leaf_indices,
        far_pair_count=far_pair_count,
        near_pair_count=near_pair_count,
        queue_overflow=overflow_wf,
        far_overflow=far_overflow,
        near_overflow=near_overflow,
        accept_decisions=accept_decisions,
        near_decisions=near_decisions,
        refine_decisions=refine_decisions,
    )


@partial(
    jax.jit,
    static_argnames=(
        "mac_type",
        "pair_policy",
        "collect_far",
        "collect_near",
        "process_block",
    ),
)
def _dual_tree_walk_count_impl(
    tree: object,
    geometry: TreeGeometry,
    theta: float,
    *,
    mac_type: MACType = "bh",
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    dehnen_radius_scale: float = 1.0,
    collect_far: bool = True,
    collect_near: bool = True,
    process_block: int = _DEFAULT_PAIR_BATCH,
) -> tuple[Array, Array, Array]:
    """Count-only dual-tree walk.

    Returns (far_counts, near_counts, max_wf_usage).
    This function performs the same traversal logic but only accumulates
    per-node/per-leaf counts and tracks the maximum stack size observed.
    It avoids building buffers so it cannot overflow; allocate safe
    stack capacity equal to total_nodes to guarantee correctness.
    """
    parent = tree.parent
    left_child = tree.left_child
    right_child = tree.right_child
    total_nodes = parent.shape[0]
    num_internal = left_child.shape[0]

    if num_internal == 0:
        leaf_count = total_nodes
        return (
            jnp.zeros((total_nodes,), dtype=INDEX_DTYPE),
            jnp.zeros((leaf_count,), dtype=INDEX_DTYPE),
            as_index(1),
        )

    centers = jnp.asarray(geometry.center)
    extents_box = jnp.asarray(geometry.max_extent)
    extents_sphere = jnp.asarray(geometry.radius)
    theta_sq = (jnp.asarray(theta, dtype=centers.dtype)) ** 2
    effective_extents_box_far = _compute_effective_extents(parent, extents_box)
    effective_extents_box_leaf = _compute_leaf_effective_extents(
        parent,
        extents_box,
        num_internal,
    )
    effective_extents_sphere_far = _compute_effective_extents(parent, extents_sphere)
    effective_extents_sphere_leaf = _compute_leaf_effective_extents(
        parent,
        extents_sphere,
        num_internal,
    )

    root_idx = as_index(jnp.argmax(parent < 0))

    # Leaves are the nodes with indices >= num_internal. Use a static
    # computation that avoids jnp.nonzero/nonzero (which requires a
    # statically-known size) so this function remains JIT-traceable.
    num_leaves = total_nodes - num_internal
    # Map node index -> leaf ordinal (0..num_leaves-1) or -1 for non-leaves
    positions = jnp.arange(total_nodes, dtype=INDEX_DTYPE) - as_index(num_internal)
    leaf_position = jnp.where(positions >= as_index(0), positions, as_index(-1))

    # Wavefront capacity for the count-only walk.  Each pair can produce
    # up to _MAX_REFINEMENT_PAIRS refined pairs per round, and the peak
    # wavefront can grow substantially before the accept/near paths start
    # removing pairs.  Use a generous multiplier so we never silently
    # drop pairs (which would undercount and lead to overflow on the
    # sized fill pass).  For the full walk the caller controls the
    # capacity via *max_pair_queue*; here we must be self-sufficient.
    #
    # Also ensure we use at least the same capacity the full walk would
    # pick via _auto_pair_queue_candidates, so the two walks process the
    # same pairs per round and produce identical counts.
    #
    # The peak wavefront can be much larger than total_nodes * 16 for
    # trees with deep traversals (e.g. KD-trees).  Use the conservative
    # auto-config ceiling to guarantee no silent pair dropping.
    auto_base = _auto_pair_queue_candidates(total_nodes, num_internal)[-1]
    stack_capacity = max(
        int(total_nodes) * DEFAULT_PAIR_QUEUE_MULTIPLIER,
        int(auto_base),
    )
    wf_indices = jnp.arange(stack_capacity, dtype=INDEX_DTYPE)

    # Use distinct local names for the counting-pass wavefront.
    count_wf_t = jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
    count_wf_s = jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
    count_wf_t = count_wf_t.at[0].set(root_idx)
    count_wf_s = count_wf_s.at[0].set(root_idx)
    count_wf_size = as_index(1)

    far_counts = jnp.zeros((total_nodes,), dtype=INDEX_DTYPE)
    near_counts = jnp.zeros((num_leaves,), dtype=INDEX_DTYPE)
    max_wf = as_index(1)

    num_internal_val = as_index(num_internal)
    node_indices = jnp.arange(total_nodes, dtype=INDEX_DTYPE)
    use_sphere = (mac_type == "dehnen") | (mac_type == "engblom")
    extents_far = jnp.where(
        use_sphere,
        effective_extents_sphere_far,
        effective_extents_box_far,
    )
    extents_leaf = jnp.where(
        use_sphere,
        effective_extents_sphere_leaf,
        effective_extents_box_leaf,
    )
    dehnen_scale = jnp.asarray(dehnen_radius_scale, dtype=centers.dtype)
    is_dehnen = jnp.asarray(mac_type == "dehnen")
    extents_far = jnp.where(is_dehnen, dehnen_scale * extents_far, extents_far)
    extents_leaf = jnp.where(is_dehnen, dehnen_scale * extents_leaf, extents_leaf)
    mac_extents = jnp.where(
        node_indices >= num_internal_val,
        extents_leaf,
        extents_far,
    )

    leaf_fill = jnp.full((total_nodes - num_internal,), -1, dtype=INDEX_DTYPE)
    left_child_full = jnp.concatenate([left_child, leaf_fill], axis=0)
    right_child_full = jnp.concatenate([right_child, leaf_fill], axis=0)

    def cond_fun(state):
        _wf_t, _wf_s, current_size, _far_c, _near_c, _max_wf = state
        return current_size > 0

    def body_fun(state):
        (
            wf_target,
            wf_source,
            wf_size,
            far_counts,
            near_counts,
            max_wf,
        ) = state

        # --- Read entire wavefront ---
        valid_mask = wf_indices < wf_size
        targets = jnp.where(valid_mask, wf_target, as_index(-1))
        sources = jnp.where(valid_mask, wf_source, as_index(-1))

        valid_pairs = valid_mask & (targets >= 0) & (sources >= 0)
        valid_pairs_bool = valid_pairs.astype(jnp.bool_)
        safe_targets = jnp.where(valid_pairs, targets, as_index(0))
        safe_sources = jnp.where(valid_pairs, sources, as_index(0))

        center_target = centers[safe_targets]
        center_source = centers[safe_sources]
        delta = center_target - center_source
        valid_scale = valid_pairs[:, None].astype(delta.dtype)
        delta = delta * valid_scale
        dist_sq = jnp.sum(delta * delta, axis=1)

        different_nodes = targets != sources
        extent_mac_target = mac_extents[safe_targets]
        extent_mac_source = mac_extents[safe_sources]
        mac_ok = _compute_mac_ok(
            mac_type=mac_type,
            theta_sq=theta_sq,
            dist_sq=dist_sq,
            extent_target=extent_mac_target,
            extent_source=extent_mac_source,
            valid_pairs=valid_pairs_bool,
            different_nodes=different_nodes,
        )

        target_internal = valid_pairs_bool & (targets < num_internal_val)
        source_internal = valid_pairs_bool & (sources < num_internal_val)
        target_leaf = valid_pairs_bool & (~target_internal)
        source_leaf = valid_pairs_bool & (~source_internal)
        same_node = valid_pairs_bool & (targets == sources)
        pair_actions, _pair_tags, _pair_tags_rev = _resolve_pair_actions(
            pair_policy=pair_policy,
            policy_state=policy_state,
            valid_pairs=valid_pairs_bool,
            mac_ok=mac_ok,
            different_nodes=different_nodes,
            target_leaf=target_leaf,
            source_leaf=source_leaf,
            same_node=same_node,
            target_nodes=targets,
            source_nodes=sources,
            center_target=center_target,
            center_source=center_source,
            dist_sq=dist_sq,
            extent_target=extent_mac_target,
            extent_source=extent_mac_source,
        )

        should_accept = valid_pairs_bool & (pair_actions == as_index(_ACTION_ACCEPT))
        should_near = valid_pairs_bool & (pair_actions == as_index(_ACTION_NEAR))
        do_refine = valid_pairs_bool & (pair_actions == as_index(_ACTION_REFINE))

        extent_target = extents_leaf[safe_targets]
        extent_source = extents_leaf[safe_sources]

        split_target = (
            do_refine
            & target_internal
            & (same_node | (~source_internal) | (extent_target >= extent_source))
        )
        split_source = (
            do_refine
            & source_internal
            & (same_node | (~target_internal) | (extent_source > extent_target))
        )
        split_both = split_target & split_source

        target_left = left_child_full[safe_targets]
        target_right = right_child_full[safe_targets]
        source_left = left_child_full[safe_sources]
        source_right = right_child_full[safe_sources]

        leaf_pos_target = leaf_position[safe_targets]
        leaf_pos_source = leaf_position[safe_sources]

        # process accepted far pairs: increment counts
        if collect_far:
            # Vectorised accumulation across the batch: compute per-node increments
            # for accepted far pairs using a segment-sum instead of scanning with
            # argmax (which repeated the same index and produced incorrect counts).
            mask = should_accept
            ones = mask.astype(INDEX_DTYPE)
            safe_tgt = jnp.where(mask, targets, as_index(0))
            safe_src = jnp.where(mask, sources, as_index(0))
            # Accumulate contributions per node across this block
            incr_t = jax.ops.segment_sum(ones, safe_tgt, num_segments=total_nodes)
            incr_s = jax.ops.segment_sum(ones, safe_src, num_segments=total_nodes)
            far_counts = far_counts + incr_t + incr_s

        if collect_near:
            # Vectorised neighbor-count accumulation for this batch. Use
            # segment_sum to avoid tracer-unsafe nonzero and avoid repeated
            # selection via argmax which led to pathological over-counting.
            mask = should_near
            ones = mask.astype(INDEX_DTYPE)
            safe_pos_t = jnp.where(mask, leaf_pos_target, as_index(0))
            safe_pos_s = jnp.where(mask, leaf_pos_source, as_index(0))
            incr_t = jax.ops.segment_sum(ones, safe_pos_t, num_segments=num_leaves)
            incr_s = jax.ops.segment_sum(ones, safe_pos_s, num_segments=num_leaves)
            near_counts = near_counts + incr_t + incr_s

        # refine: push refined pairs onto stack
        refine_vm = jax.vmap(
            _count_refine_pairs_single,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

        refine_pairs = refine_vm(
            targets,
            sources,
            same_node,
            split_both.astype(jnp.bool_),
            split_target.astype(jnp.bool_),
            split_source.astype(jnp.bool_),
            target_left,
            target_right,
            source_left,
            source_right,
        )

        refine_targets = refine_pairs[..., 0].reshape(
            (stack_capacity * _MAX_REFINEMENT_PAIRS,)
        )
        refine_sources = refine_pairs[..., 1].reshape(
            (stack_capacity * _MAX_REFINEMENT_PAIRS,)
        )

        # --- Build new wavefront from refined pairs ---
        valid_push = (refine_targets >= 0) & (refine_sources >= 0)
        push_prefix = jnp.cumsum(valid_push.astype(INDEX_DTYPE)) - valid_push.astype(
            INDEX_DTYPE
        )

        push_ok = valid_push & (push_prefix < as_index(stack_capacity))
        push_lo = jnp.minimum(refine_targets, refine_sources)
        push_hi = jnp.maximum(refine_targets, refine_sources)

        oob_wf = as_index(stack_capacity)
        safe_slot = jnp.where(push_ok, push_prefix, oob_wf)
        new_wf_target = (
            jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
            .at[safe_slot]
            .set(jnp.where(push_ok, push_lo, as_index(-1)), mode="drop")
        )
        new_wf_source = (
            jnp.full((stack_capacity,), -1, dtype=INDEX_DTYPE)
            .at[safe_slot]
            .set(jnp.where(push_ok, push_hi, as_index(-1)), mode="drop")
        )
        new_wf_size = jnp.sum(push_ok.astype(INDEX_DTYPE))

        max_wf = jnp.maximum(max_wf, new_wf_size)

        return (
            new_wf_target,
            new_wf_source,
            new_wf_size,
            far_counts,
            near_counts,
            max_wf,
        )

    (
        _wf_out_t,
        _wf_out_s,
        _wf_size_out,
        far_counts,
        near_counts,
        max_wf,
    ) = lax.while_loop(
        cond_fun,
        body_fun,
        (
            count_wf_t,
            count_wf_s,
            jnp.asarray(count_wf_size, dtype=INDEX_DTYPE),
            far_counts,
            near_counts,
            jnp.asarray(max_wf, dtype=INDEX_DTYPE),
        ),
    )

    return far_counts, near_counts, max_wf


def _result_to_interactions(
    result: DualTreeWalkResult,
    tree: object,
) -> NodeInteractionList:
    total_nodes = int(tree.parent.shape[0])
    counts = jnp.asarray(result.interaction_counts)
    far_pair_count = jnp.asarray(result.far_pair_count, dtype=INDEX_DTYPE)
    traced_total = isinstance(result.far_pair_count, jax_core.Tracer)

    node_levels_all = get_node_levels(tree)
    if hasattr(tree, "level_offsets"):
        level_offsets_all = jnp.asarray(tree.level_offsets, dtype=INDEX_DTYPE)
        if hasattr(tree, "num_levels"):
            num_levels_raw = tree.num_levels
            if isinstance(num_levels_raw, jax_core.Tracer):
                num_levels = int(level_offsets_all.shape[0] - 1)
            else:
                num_levels = int(num_levels_raw)
            level_offsets_all = level_offsets_all[: num_levels + 1]
        else:
            num_levels = int(level_offsets_all.shape[0] - 1)
    else:
        level_offsets_all = get_level_offsets(tree, node_levels=node_levels_all)
        num_levels = int(level_offsets_all.shape[0] - 1)
    level_indices = jnp.asarray(level_offsets_all[: num_levels + 1])

    if traced_total:
        sources_sorted = jnp.asarray(result.interaction_sources)
        targets_sorted = jnp.asarray(result.interaction_targets)
    else:
        far_total = int(far_pair_count)
        if far_total == 0:
            zero_offsets = jnp.zeros((total_nodes + 1,), dtype=INDEX_DTYPE)
            zero_levels = jnp.zeros((num_levels + 1,), dtype=INDEX_DTYPE)
            return NodeInteractionList(
                offsets=zero_offsets,
                sources=jnp.zeros((0,), dtype=INDEX_DTYPE),
                targets=jnp.zeros((0,), dtype=INDEX_DTYPE),
                counts=counts,
                level_offsets=zero_levels,
                target_levels=jnp.zeros((0,), dtype=INDEX_DTYPE),
            )
        sources_sorted = jax.device_put(result.interaction_sources[:far_total])
        targets_sorted = jax.device_put(result.interaction_targets[:far_total])
    levels_sorted = node_levels_all[targets_sorted]

    nodes_by_level = get_nodes_by_level(tree, node_levels=node_levels_all)
    counts_by_level = counts[nodes_by_level]
    cumulative_counts = jnp.cumsum(counts_by_level, dtype=INDEX_DTYPE)
    offsets_by_level = jnp.concatenate(
        [jnp.zeros((1,), dtype=INDEX_DTYPE), cumulative_counts]
    )

    node_offsets = jnp.zeros((total_nodes,), dtype=INDEX_DTYPE)
    node_offsets = node_offsets.at[nodes_by_level].set(offsets_by_level[:-1])
    node_offsets = jnp.concatenate([node_offsets, far_pair_count[None]])

    level_offsets = offsets_by_level[level_indices]

    return NodeInteractionList(
        offsets=node_offsets,
        sources=sources_sorted,
        targets=targets_sorted,
        counts=counts,
        level_offsets=level_offsets,
        target_levels=levels_sorted,
    )


def _result_to_neighbors(result: DualTreeWalkResult) -> NodeNeighborList:
    traced_total = isinstance(result.near_pair_count, jax_core.Tracer)
    neighbor_offsets = jnp.asarray(result.neighbor_offsets)
    neighbor_counts = jnp.asarray(result.neighbor_counts)
    if traced_total:
        neighbor_indices = jnp.asarray(result.neighbor_indices)
    else:
        near_total = int(result.near_pair_count)
        neighbor_indices = jax.device_put(result.neighbor_indices[:near_total])
    return NodeNeighborList(
        offsets=neighbor_offsets,
        neighbors=neighbor_indices,
        leaf_indices=jnp.asarray(result.leaf_indices),
        counts=neighbor_counts,
    )


def _run_dual_tree_walk(
    tree: object,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType = "bh",
    *,
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    max_interactions_per_node: Optional[int],
    max_neighbors_per_leaf: int,
    max_pair_queue: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    collect_far: bool,
    collect_near: bool,
    dehnen_radius_scale: float = 1.0,
    process_block: Optional[int] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
) -> DualTreeWalkResult:
    # Public APIs may pass the wrapper ``yggdrax.tree.RadixTree``.
    # Jitted kernels require the underlying topology pytree.
    tree = tree.topology if hasattr(tree, "topology") else tree

    theta_val = float(theta)
    dehnen_scale_val = float(dehnen_radius_scale)
    is_kdtree_topology = (
        hasattr(tree, "split_dim")
        and hasattr(tree, "points")
        and hasattr(tree, "leaf_size")
    )
    if (
        mac_type == "dehnen"
        and is_kdtree_topology
        and abs(dehnen_scale_val - 1.0) <= 1e-12
    ):
        dehnen_scale_val = _DEFAULT_KDTREE_DEHNEN_RADIUS_SCALE
    if dehnen_scale_val <= 0.0:
        raise ValueError("dehnen_radius_scale must be > 0")
    total_nodes = int(tree.parent.shape[0])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    nodes_by_level = get_nodes_by_level(tree)

    config = _resolve_dual_tree_config(traversal_config)

    if config is not None:
        queue_candidates = [max(4, int(config.max_pair_queue))]
        interaction_candidates = [int(config.max_interactions_per_node)]
        allow_auto_interactions = False
        neighbors_limit = int(config.max_neighbors_per_leaf)
        process_override = int(config.process_block)
        user_supplied_queue = True
    else:
        user_supplied_queue = max_pair_queue is not None
        if user_supplied_queue:
            queue_candidates = [max(4, int(max_pair_queue))]
        else:
            queue_candidates = _auto_pair_queue_candidates(
                total_nodes,
                num_internal,
            )

        (
            interaction_candidates,
            allow_auto_interactions,
        ) = _interaction_capacity_candidates(
            max_interactions_per_node,
            total_nodes,
        )
        neighbors_limit = max_neighbors_per_leaf
        process_override = None

    # If the user did not supply capacities (and no traversal_config override),
    # run a jitted count-only traversal to determine per-node and per-leaf
    # counts plus the maximum stack usage observed. We then size buffers from
    # those counts and run the full fill pass once. This reduces host/kernel
    # round trips while keeping memory usage tight.
    use_count_pass = (
        max_interactions_per_node is None
        and max_pair_queue is None
        and traversal_config is None
    )
    if use_count_pass:
        # The count-pass auto-sizing path converts observed counts to Python
        # ints for host-side buffer sizing. Under an outer jit those counts are
        # traced, so fall back to the existing static candidate capacities
        # instead of forcing concretization.
        traced_count_pass = isinstance(tree.parent, jax_core.Tracer) or isinstance(
            geometry.center, jax_core.Tracer
        )
        if traced_count_pass:
            use_count_pass = False

    if use_count_pass:
        # Choose a process block for the count pass. Prefer an explicit
        # override, else use the provided block size or the default.
        resolved_block = (
            process_override if process_override is not None else process_block
        )
        if resolved_block is None:
            count_process_block = _DEFAULT_PAIR_BATCH
        else:
            count_process_block = int(resolved_block)

        far_counts, near_counts, max_wf = _dual_tree_walk_count_impl(
            tree,
            geometry,
            theta_val,
            mac_type=mac_type,
            pair_policy=pair_policy,
            policy_state=policy_state,
            dehnen_radius_scale=dehnen_scale_val,
            collect_far=collect_far,
            collect_near=collect_near,
            process_block=count_process_block,
        )

        # Compute suggested capacities from observed counts.
        if collect_far:
            max_interactions_suggest = int(jnp.max(far_counts))
        else:
            max_interactions_suggest = 0

        if collect_near:
            max_neighbors_suggest = int(jnp.max(near_counts))
        else:
            max_neighbors_suggest = max_neighbors_per_leaf

        # Base suggestion from observed stack usage.  The count-only pass
        # reports the peak wavefront it actually observed, but the sized
        # fill pass may need extra headroom because:
        #   1. The fill pass carries larger per-pair state, which can
        #      change the JIT schedule relative to the count pass.
        #   2. Borderline pairs (near the MAC threshold) that happen to
        #      be evaluated in a different order can temporarily inflate
        #      the wavefront.
        # A 50 % safety margin handles observed cases (count peak ~93 K,
        # fill minimum ~125 K for KD-trees).
        #
        # We do NOT floor at the conservative auto_base here: the whole
        # point of the count pass is to replace the heuristic with real
        # data, so we size the fill pass tightly around the observation.
        observed_wf = int(max_wf)
        queue_suggest = max(
            4,
            observed_wf + observed_wf // 2,  # 1.5× observed peak
        )

        # Replace candidate lists so the main loop performs a single
        # sized pass. Add a defensive guard to avoid producing host-side
        # allocation sizes that overflow 32-bit integers or are otherwise
        # unrealistic (this previously caused an OverflowError when the
        # derived total pairs exceeded int32 limits).
        queue_candidates = [queue_suggest]
        interaction_candidates = [max(1, max_interactions_suggest)]

        # Clamp per-leaf neighbor suggestion so the estimated total
        # neighbour pair count (num_leaves * per_leaf) fits in signed
        # 32-bit and is safe to allocate on the host. If clamping occurs
        # we log a warning recommending the caller enable local
        # refinement or increase capacities explicitly.
        suggested_per_leaf = max(1, int(max_neighbors_suggest))
        # avoid division by zero
        safe_num_leaves = max(1, int(total_nodes - num_internal))
        int32_max = (1 << 31) - 1
        max_per_leaf_from_int32 = max(1, int(int32_max // safe_num_leaves))
        # If the caller's observed suggestion would overflow a signed 32-bit
        # integer when multiplied by the number of leaves, fail early with a
        # clear message rather than silently producing a too-small buffer
        # that will cause the fill pass to overflow. This encourages the
        # user to enable local refinement or explicitly provide capacities.
        if suggested_per_leaf > max_per_leaf_from_int32:
            raise RuntimeError(
                "Suggested per-leaf neighbor capacity is too large to "
                "safely allocate on the host (would overflow int32). "
                "Enable local refinement or pass explicit capacities "
                "(max_neighbors_per_leaf / max_interactions_per_node)."
            )
        neighbors_limit = max(1, int(suggested_per_leaf))
        allow_auto_interactions = False
        user_supplied_queue = True

    result: Optional[DualTreeWalkResult] = None
    attempt_counter = 0

    queue_error_msg = (
        "Pair queue capacity exceeded; increase max_pair_queue and rebuild."
        if user_supplied_queue
        else (
            "Pair queue capacity exceeded; pass a larger max_pair_queue "
            "value to override auto-scaling."
        )
    )

    far_error_msg = (
        "Interaction list capacity exceeded; increase "
        "max_interactions_per_node and rebuild."
    )

    success = False

    def _emit_retry_event(
        status: str,
        *,
        attempt: int,
        queue_capacity: int,
        interaction_capacity: int,
        walk_result: DualTreeWalkResult,
    ) -> None:
        if retry_logger is None:
            return
        event = DualTreeRetryEvent(
            attempt=int(attempt),
            queue_capacity=int(queue_capacity),
            interaction_capacity=int(interaction_capacity),
            status=status,
            far_pair_count=int(walk_result.far_pair_count),
            near_pair_count=int(walk_result.near_pair_count),
        )
        try:
            retry_logger(event)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("retry_logger raised", exc_info=True)

    for queue_capacity in queue_candidates:
        resolved_block = (
            process_override if process_override is not None else process_block
        )
        block_size = _resolve_process_block(queue_capacity, resolved_block)
        queue_retry = False
        for interaction_capacity in interaction_candidates:
            attempt_counter += 1
            attempt_idx = attempt_counter
            result = _dual_tree_walk_impl(
                tree,
                geometry,
                nodes_by_level,
                theta_val,
                mac_type=mac_type,
                pair_policy=pair_policy,
                policy_state=policy_state,
                dehnen_radius_scale=dehnen_scale_val,
                max_interactions_per_node=interaction_capacity,
                max_neighbors_per_leaf=neighbors_limit,
                max_pair_queue=queue_capacity,
                collect_far=collect_far,
                collect_near=collect_near,
                process_block=block_size,
            )

            overflow_queue = result.queue_overflow
            overflow_far = result.far_overflow

            if isinstance(overflow_queue, jax_core.Tracer) or isinstance(
                overflow_far, jax_core.Tracer
            ):
                success = True
                break

            if bool(overflow_queue):
                # Need a larger queue; try next candidate (outer loop).
                _emit_retry_event(
                    "queue_overflow",
                    attempt=attempt_idx,
                    queue_capacity=queue_capacity,
                    interaction_capacity=interaction_capacity,
                    walk_result=result,
                )
                queue_retry = True
                break

            if bool(overflow_far):
                if allow_auto_interactions:
                    # Re-run with a larger per-node interaction cap.
                    _emit_retry_event(
                        "interaction_overflow",
                        attempt=attempt_idx,
                        queue_capacity=queue_capacity,
                        interaction_capacity=interaction_capacity,
                        walk_result=result,
                    )
                    continue
                _raise_if_true(overflow_far, far_error_msg)

            # No queue overflow and far capacity sufficient.
            success = True
            if attempt_idx > 1:
                _emit_retry_event(
                    "success",
                    attempt=attempt_idx,
                    queue_capacity=queue_capacity,
                    interaction_capacity=interaction_capacity,
                    walk_result=result,
                )
            break
        else:
            # Ran out of interaction capacities without success.
            if allow_auto_interactions and result is not None:
                _raise_if_true(result.far_overflow, far_error_msg)
            continue

        if queue_retry and not success:
            continue

        if success:
            break
    else:
        assert result is not None
        _raise_if_true(result.queue_overflow, queue_error_msg)

    if not success:
        assert result is not None
        _raise_if_true(result.far_overflow, far_error_msg)

    assert result is not None

    if not allow_auto_interactions and isinstance(result.far_overflow, bool):
        _raise_if_true(result.far_overflow, far_error_msg)

    if allow_auto_interactions and bool(result.far_overflow):
        _raise_if_true(result.far_overflow, far_error_msg)

    if collect_far:
        _raise_if_true(result.far_overflow, far_error_msg)
    if collect_near:
        _raise_if_true(
            result.near_overflow,
            (
                "Neighbor list capacity exceeded; increase "
                "max_neighbors_per_leaf and rebuild."
            ),
        )

    return result


@jaxtyped(typechecker=beartype)
def build_well_separated_interactions(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: Optional[int] = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    max_pair_queue: Optional[int] = None,
    process_block: Optional[int] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    dehnen_radius_scale: float = 1.0,
) -> NodeInteractionList:
    """Construct multipole-to-local interaction lists using a MAC walk.

    When ``max_interactions_per_node`` is ``None`` the builder auto-scales the
    per-node capacity, retrying with progressively larger buffers only if the
    interaction list overflows. That keeps the default footprint small without
    sacrificing correctness for adversarial inputs.
    """

    result = _run_dual_tree_walk(
        tree,
        geometry,
        theta,
        mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=_DEFAULT_MAX_NEIGHBORS,
        max_pair_queue=max_pair_queue,
        traversal_config=traversal_config,
        collect_far=True,
        collect_near=False,
        dehnen_radius_scale=dehnen_radius_scale,
        process_block=process_block,
        retry_logger=retry_logger,
    )
    return _result_to_interactions(result, tree)


@jaxtyped(typechecker=beartype)
def build_leaf_neighbor_lists(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_neighbors_per_leaf: int = _DEFAULT_MAX_NEIGHBORS,
    max_interactions_per_node: Optional[int] = None,
    mac_type: MACType = "bh",
    *,
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    max_pair_queue: Optional[int] = None,
    process_block: Optional[int] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    dehnen_radius_scale: float = 1.0,
) -> NodeNeighborList:
    """Construct near-field adjacency for leaf nodes via a MAC walk."""

    result = _run_dual_tree_walk(
        tree,
        geometry,
        theta,
        mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        traversal_config=traversal_config,
        collect_far=False,
        collect_near=True,
        dehnen_radius_scale=dehnen_radius_scale,
        process_block=process_block,
        retry_logger=retry_logger,
    )
    return _result_to_neighbors(result)


@jaxtyped(typechecker=beartype)
def build_interactions_and_neighbors(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    max_interactions_per_node: Optional[int] = None,
    max_neighbors_per_leaf: int = _DEFAULT_MAX_NEIGHBORS,
    max_pair_queue: Optional[int] = None,
    process_block: Optional[int] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    mac_type: MACType = "bh",
    dehnen_radius_scale: float = 1.0,
    pair_policy: Optional[PairPolicy] = None,
    policy_state: object = None,
    *,
    return_result: bool = False,
    return_grouped: bool = False,
) -> Union[
    tuple[NodeInteractionList, NodeNeighborList],
    tuple[NodeInteractionList, NodeNeighborList, DualTreeWalkResult],
    tuple[NodeInteractionList, NodeNeighborList, GroupedInteractionBuffers],
    tuple[
        NodeInteractionList,
        NodeNeighborList,
        DualTreeWalkResult,
        GroupedInteractionBuffers,
    ],
]:
    """Return both far-field interactions and near-field neighbour lists.

    The dual tree walk visits source/target pairs once, populating both
    multipole (far-field) and neighbour (near-field) lists while respecting the
    existing compressed-row storage layout. Capacity arguments mirror the
    standalone builders; ``max_pair_queue`` controls the traversal queue size
    and defaults to ``DEFAULT_PAIR_QUEUE_MULTIPLIER * num_nodes``.
    """

    result = _run_dual_tree_walk(
        tree,
        geometry,
        theta,
        mac_type,
        pair_policy=pair_policy,
        policy_state=policy_state,
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        traversal_config=traversal_config,
        collect_far=True,
        collect_near=True,
        dehnen_radius_scale=dehnen_radius_scale,
        process_block=process_block,
        retry_logger=retry_logger,
    )

    interactions = _result_to_interactions(result, tree)
    neighbors = _result_to_neighbors(result)
    grouped = None
    if return_grouped:
        far_total = int(result.far_pair_count)
        grouped = build_grouped_interactions_from_pairs(
            tree,
            geometry,
            result.interaction_sources[:far_total],
            result.interaction_targets[:far_total],
            level_offsets=interactions.level_offsets,
        )

    if return_result and return_grouped:
        return interactions, neighbors, result, grouped
    if return_result:
        return interactions, neighbors, result
    if return_grouped:
        return interactions, neighbors, grouped
    return interactions, neighbors


@jaxtyped(typechecker=beartype)
def interactions_for_node(data: NodeInteractionList, node: int) -> Array:
    """Return source node indices interacting with ``node``."""

    start = int(data.offsets[node])
    count = int(data.counts[node])
    end = start + count
    return data.sources[start:end]


@jaxtyped(typechecker=beartype)
def neighbors_for_leaf(data: NodeNeighborList, leaf_node: int) -> Array:
    """Return neighbouring leaf nodes interacting with ``leaf_node``."""

    leaf_indices = jnp.asarray(data.leaf_indices)
    pos = int(jnp.searchsorted(leaf_indices, leaf_node, side="left"))
    if pos >= leaf_indices.shape[0] or int(leaf_indices[pos]) != leaf_node:
        raise ValueError("leaf_node not present in neighbor list")

    start = int(data.offsets[pos])
    count = int(data.counts[pos])
    end = start + count
    return data.neighbors[start:end]


def diagnose_leaf_neighbor_growth(
    tree: object,
    geometry: TreeGeometry,
    theta: float = 0.5,
    *,
    max_neighbors_per_leaf: int = _DEFAULT_MAX_NEIGHBORS,
    top_k: int = 10,
    sample_neighbors: int = 20,
) -> dict:
    """Return a compact report of the top-k leaf neighbour counts.

    The report contains simple statistics (num_leaves, max, mean, median)
    and for each of the top-k leaves a small sample of their neighbour
    lists so we can inspect which nodes contributed to large expansions.

    This helper is intended for debugging and local inspection; it returns
    standard Python types (ints, lists) to make printing/serialization
    straightforward.
    """

    # Build neighbor lists using the existing builder.
    nl = build_leaf_neighbor_lists(
        tree, geometry, theta, max_neighbors_per_leaf=max_neighbors_per_leaf
    )

    counts = jnp.asarray(nl.counts)
    leaf_indices = jnp.asarray(nl.leaf_indices)
    offsets = jnp.asarray(nl.offsets)
    neighbors_flat = jnp.asarray(nl.neighbors)

    num_leaves = int(leaf_indices.shape[0])
    if num_leaves == 0:
        return {
            "stats": {
                "num_leaves": 0,
                "max_count": 0,
                "mean": 0.0,
                "median": 0,
            },
            "top": [],
        }

    # Compute basic statistics
    max_count = int(jnp.max(counts))
    mean = float(jnp.mean(counts))
    # approximate median via sort (small arrays)
    sorted_counts = jnp.sort(counts)
    median = int(sorted_counts[num_leaves // 2])

    # Select top-k leaves by count
    order = jnp.argsort(-counts)
    k = int(min(top_k, num_leaves))
    top_positions = [int(x) for x in order[:k]]

    top = []
    for pos in top_positions:
        leaf_node = int(leaf_indices[pos])
        cnt = int(counts[pos])
        start = int(offsets[pos])
        end = start + cnt
        neigh = neighbors_flat[start:end]
        sample = [int(x) for x in neigh[:sample_neighbors]]
        top.append(
            {
                "leaf_node": leaf_node,
                "count": cnt,
                "neighbors_sample": sample,
            }
        )

    return {
        "stats": {
            "num_leaves": num_leaves,
            "max_count": max_count,
            "mean": mean,
            "median": median,
        },
        "top": top,
    }


def set_default_pair_queue_multiplier(multiplier: int) -> None:
    """Configure the fallback multiplier used for pair queue sizing."""

    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")

    global DEFAULT_PAIR_QUEUE_MULTIPLIER
    DEFAULT_PAIR_QUEUE_MULTIPLIER = int(multiplier)


__all__ = [
    "DEFAULT_PAIR_QUEUE_MULTIPLIER",
    "DualTreeRetryEvent",
    "DualTreeTraversalConfig",
    "NodeInteractionList",
    "NodeNeighborList",
    "DualTreeWalkResult",
    "build_well_separated_interactions",
    "build_leaf_neighbor_lists",
    "build_interactions_and_neighbors",
    "interactions_for_node",
    "neighbors_for_leaf",
    "log_retry_event",
    "set_default_dual_tree_config",
    "set_default_pair_queue_multiplier",
]
