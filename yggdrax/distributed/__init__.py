"""Multi-GPU foundations for Yggdrax (Phase 0).

Device mesh construction (:mod:`.sharding`) and collective communication
primitives (:mod:`.comm`) used to shard tree build, traversal, and FMM force
evaluation across GPUs. The design follows jztree: a per-device local view
under ``jax.shard_map``, space-filling-curve domain decomposition, and a ragged
all-to-all as the central communication primitive.
"""

from __future__ import annotations

from .comm import (
    RAGGED_NATIVE_FIXED_JAX,
    ragged_all_to_all_exchange,
    ragged_native_flag_hazard,
    resolve_ragged_method,
)
from .cross_walk import dual_tree_walk_cross_impl
from .export import (
    ExportLists,
    SendBuffers,
    build_send_buffers,
    export_walk,
    export_walk_two_sided,
)
from .import_cells import (
    ImportedCells,
    ReceiverLists,
    exchange_export_list,
    rebase_csr,
    receiver_interaction_lists,
)
from .let import (
    CoarseFrontier,
    GlobalCoarseTree,
    HaloImport,
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from .local_tree import (
    sanitize_padding,
)
from .partition import (
    RepartitionResult,
    global_bounds,
    sfc_partition,
    sfc_repartition,
)
from .sharding import AXIS_NAME, available_devices, device_count, make_mesh
from .summary import (
    SummaryTree,
    TreeSummary,
    occupancy_cut,
    subtree_leaf_counts,
    summary_tree,
)

__all__ = [
    "AXIS_NAME",
    "CoarseFrontier",
    "GlobalCoarseTree",
    "HaloImport",
    "available_devices",
    "build_coarse_frontier",
    "build_remote_coarse_tree",
    "device_count",
    "ExportLists",
    "ImportedCells",
    "ReceiverLists",
    "SendBuffers",
    "build_send_buffers",
    "exchange_export_list",
    "export_walk",
    "export_walk_two_sided",
    "rebase_csr",
    "receiver_interaction_lists",
    "SummaryTree",
    "TreeSummary",
    "occupancy_cut",
    "subtree_leaf_counts",
    "summary_tree",
    "dual_tree_walk_cross_impl",
    "import_near_halo",
    "global_bounds",
    "make_mesh",
    "ragged_all_to_all_exchange",
    "ragged_native_flag_hazard",
    "resolve_ragged_method",
    "RAGGED_NATIVE_FIXED_JAX",
    "sanitize_padding",
    "sfc_partition",
    "sfc_repartition",
    "RepartitionResult",
]
