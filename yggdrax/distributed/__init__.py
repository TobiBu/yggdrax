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
    "DistributedTreeMoments",
    "ShardedArray",
    "ShardedDomain",
    "all_to_all_dense",
    "available_devices",
    "build_local_moments",
    "device_count",
    "distributed_tree_moments",
    "equalize_domain",
    "exchange_pytree",
    "exchange_sizes",
    "global_bounds",
    "make_mesh",
    "ragged_all_to_all_exchange",
    "sanitize_padding",
    "sfc_decompose",
    "sfc_partition",
]
