"""Yggdrax: tree generation, geometry, traversal, and interaction APIs."""

from jax import config as _jax_config

# Morton codes rely on uint64 bit arithmetic.
_jax_config.update("jax_enable_x64", True)

from .bounds import infer_bounds
from .dtypes import INDEX_DTYPE, as_index
from .geometry import (
    LevelMajorTreeGeometry,
    TreeGeometry,
    compute_tree_geometry,
    geometry_to_level_major,
)
from .interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    build_interactions_and_neighbors,
    build_leaf_neighbor_lists,
)
from .kdtree import (
    KDTree,
    build_and_query,
    build_kdtree,
    count_neighbors,
    query_neighbors,
)
from .morton import (
    get_common_prefix_length,
    morton_decode,
    morton_encode,
    sort_by_morton,
)
from .protocols import (
    MortonLeafBoundsProtocol,
    TopologyContainerProtocol,
    TreeLevelIndexProtocol,
    TreeRangesProtocol,
    TreeStructureProtocol,
)
from .traversal import build_prepared_tree_artifacts
from .tree import (
    MAX_TREE_LEVELS,
    FixedDepthTreeBuildConfig,
    KDParticleTree,
    RadixTree,
    RadixTreeWorkspace,
    Tree,
    TreeBuildConfig,
    TreeBuilder,
    TreeBuildMode,
    TreeBuildRequest,
    TreeType,
    available_tree_types,
    build_fixed_depth_tree,
    build_fixed_depth_tree_jit,
    build_tree,
    build_tree_jit,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
    get_num_internal_nodes,
    get_num_levels,
    register_tree_builder,
    resolve_tree_topology,
)
from .types import PreparedTreeArtifacts, TraversalArtifacts, TraversalResult

__all__ = [
    "MAX_TREE_LEVELS",
    "MACType",
    "DualTreeRetryEvent",
    "DualTreeTraversalConfig",
    "DualTreeWalkResult",
    "FixedDepthTreeBuildConfig",
    "INDEX_DTYPE",
    "LevelMajorTreeGeometry",
    "NodeInteractionList",
    "NodeNeighborList",
    "MortonLeafBoundsProtocol",
    "PreparedTreeArtifacts",
    "KDParticleTree",
    "RadixTree",
    "RadixTreeWorkspace",
    "Tree",
    "TreeBuildRequest",
    "TreeBuildConfig",
    "TreeBuilder",
    "TreeBuildMode",
    "TreeType",
    "TopologyContainerProtocol",
    "TraversalArtifacts",
    "TraversalResult",
    "TreeGeometry",
    "KDTree",
    "build_and_query",
    "build_kdtree",
    "count_neighbors",
    "query_neighbors",
    "build_fixed_depth_tree",
    "build_fixed_depth_tree_jit",
    "available_tree_types",
    "build_interactions_and_neighbors",
    "build_leaf_neighbor_lists",
    "build_prepared_tree_artifacts",
    "build_tree",
    "build_tree_jit",
    "get_level_offsets",
    "get_node_levels",
    "get_nodes_by_level",
    "get_num_internal_nodes",
    "get_num_levels",
    "compute_tree_geometry",
    "geometry_to_level_major",
    "get_common_prefix_length",
    "infer_bounds",
    "morton_decode",
    "morton_encode",
    "resolve_tree_topology",
    "register_tree_builder",
    "TreeLevelIndexProtocol",
    "TreeRangesProtocol",
    "TreeStructureProtocol",
    "sort_by_morton",
    "as_index",
]
