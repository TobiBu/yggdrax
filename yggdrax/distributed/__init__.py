"""Multi-GPU foundations for Yggdrax (Phase 0).

Device mesh construction (:mod:`.sharding`) and collective communication
primitives (:mod:`.comm`) used to shard tree build, traversal, and FMM force
evaluation across GPUs. The design follows jztree: a per-device local view
under ``jax.shard_map``, space-filling-curve domain decomposition, and a ragged
all-to-all as the central communication primitive.
"""

from __future__ import annotations

from .comm import (
    ShardedArray,
    all_to_all_dense,
    exchange_pytree,
    exchange_sizes,
    ragged_all_to_all_exchange,
)
from .cross_walk import dual_tree_walk_cross, dual_tree_walk_cross_impl
from .let import (
    ClassifyMetrics,
    CoarseFrontier,
    CoarseTreeMetrics,
    GlobalCoarseTree,
    build_coarse_frontier,
    build_distributed_coarse_tree,
    build_remote_coarse_tree,
    classify_against_remote,
    gather_global_coarse_tree,
)
from .local_tree import (
    DistributedTreeMoments,
    build_local_moments,
    distributed_tree_moments,
    sanitize_padding,
)
from .partition import (
    ShardedDomain,
    equalize_domain,
    global_bounds,
    sfc_decompose,
    sfc_partition,
)
from .sharding import AXIS_NAME, available_devices, device_count, make_mesh

__all__ = [
    "AXIS_NAME",
    "ClassifyMetrics",
    "CoarseFrontier",
    "CoarseTreeMetrics",
    "DistributedTreeMoments",
    "GlobalCoarseTree",
    "ShardedArray",
    "ShardedDomain",
    "all_to_all_dense",
    "available_devices",
    "build_coarse_frontier",
    "build_distributed_coarse_tree",
    "build_local_moments",
    "build_remote_coarse_tree",
    "classify_against_remote",
    "device_count",
    "distributed_tree_moments",
    "dual_tree_walk_cross",
    "dual_tree_walk_cross_impl",
    "equalize_domain",
    "gather_global_coarse_tree",
    "exchange_pytree",
    "exchange_sizes",
    "global_bounds",
    "make_mesh",
    "ragged_all_to_all_exchange",
    "sanitize_padding",
    "sfc_decompose",
    "sfc_partition",
]
