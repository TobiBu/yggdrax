"""Yggdrasil: tree generation, geometry, traversal, and interaction APIs."""

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
)
from .morton import (
    get_common_prefix_length,
    morton_decode,
    morton_encode,
    sort_by_morton,
)
from .traversal import build_prepared_tree_artifacts
from .tree import (
    MAX_TREE_LEVELS,
    FixedDepthTreeBuildConfig,
    RadixTree,
    RadixTreeWorkspace,
    TreeBuildConfig,
    build_fixed_depth_tree,
    build_fixed_depth_tree_jit,
    build_tree,
    build_tree_jit,
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
    "PreparedTreeArtifacts",
    "RadixTree",
    "RadixTreeWorkspace",
    "TreeBuildConfig",
    "TraversalArtifacts",
    "TraversalResult",
    "TreeGeometry",
    "build_fixed_depth_tree",
    "build_fixed_depth_tree_jit",
    "build_interactions_and_neighbors",
    "build_prepared_tree_artifacts",
    "build_tree",
    "build_tree_jit",
    "compute_tree_geometry",
    "geometry_to_level_major",
    "get_common_prefix_length",
    "infer_bounds",
    "morton_decode",
    "morton_encode",
    "sort_by_morton",
    "as_index",
]
