"""Public tree-building API for Yggdrasil."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jaxtyping import Array

from . import _tree_impl

from .bounds import infer_bounds

MAX_TREE_LEVELS = _tree_impl.MAX_TREE_LEVELS
RadixTree = _tree_impl.RadixTree
RadixTreeWorkspace = _tree_impl.RadixTreeWorkspace
reorder_particles_by_indices = _tree_impl.reorder_particles_by_indices


@dataclass(frozen=True)
class TreeBuildConfig:
    """Resolved options for standard LBVH tree construction."""

    leaf_size: int = 8
    return_reordered: bool = False
    workspace: Optional[RadixTreeWorkspace] = None
    return_workspace: bool = False


@dataclass(frozen=True)
class FixedDepthTreeBuildConfig:
    """Resolved options for fixed-depth tree construction."""

    target_leaf_particles: int = 32
    return_reordered: bool = False
    workspace: Optional[RadixTreeWorkspace] = None
    return_workspace: bool = False
    max_depth: Optional[int] = None
    refine_local: bool = True
    max_refine_levels: int = 2
    aspect_threshold: float = 8.0
    min_refined_leaf_particles: int = 2


def build_tree(
    positions: Array,
    masses: Array,
    bounds: Optional[Tuple[Array, Array]] = None,
    *,
    return_reordered: bool = False,
    leaf_size: int = 8,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
    config: Optional[TreeBuildConfig] = None,
):
    """Build an LBVH tree, inferring bounds when not provided."""

    cfg = config or TreeBuildConfig()
    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    return _tree_impl.build_tree(
        positions,
        masses,
        bounds_resolved,
        return_reordered=cfg.return_reordered if config else return_reordered,
        leaf_size=cfg.leaf_size if config else leaf_size,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


def build_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Optional[Tuple[Array, Array]] = None,
    *,
    return_reordered: bool = False,
    leaf_size: int = 8,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
    config: Optional[TreeBuildConfig] = None,
):
    """JIT build for an LBVH tree, inferring bounds when not provided."""

    cfg = config or TreeBuildConfig()
    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    return _tree_impl.build_tree_jit(
        positions,
        masses,
        bounds_resolved,
        return_reordered=cfg.return_reordered if config else return_reordered,
        leaf_size=cfg.leaf_size if config else leaf_size,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


def build_fixed_depth_tree(
    positions: Array,
    masses: Array,
    bounds: Optional[Tuple[Array, Array]] = None,
    *,
    target_leaf_particles: int = 32,
    return_reordered: bool = False,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
    max_depth: Optional[int] = None,
    refine_local: bool = True,
    max_refine_levels: int = 2,
    aspect_threshold: float = 8.0,
    min_refined_leaf_particles: int = 2,
    config: Optional[FixedDepthTreeBuildConfig] = None,
):
    """Build a fixed-depth tree, inferring bounds when not provided."""

    cfg = config or FixedDepthTreeBuildConfig()
    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    return _tree_impl.build_fixed_depth_tree(
        positions,
        masses,
        bounds_resolved,
        target_leaf_particles=(
            cfg.target_leaf_particles if config else target_leaf_particles
        ),
        return_reordered=cfg.return_reordered if config else return_reordered,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
        max_depth=cfg.max_depth if config else max_depth,
        refine_local=cfg.refine_local if config else refine_local,
        max_refine_levels=cfg.max_refine_levels if config else max_refine_levels,
        aspect_threshold=cfg.aspect_threshold if config else aspect_threshold,
        min_refined_leaf_particles=(
            cfg.min_refined_leaf_particles if config else min_refined_leaf_particles
        ),
    )


def build_fixed_depth_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Optional[Tuple[Array, Array]] = None,
    *,
    target_leaf_particles: int = 32,
    return_reordered: bool = False,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
    max_depth: Optional[int] = None,
    refine_local: bool = True,
    max_refine_levels: int = 2,
    aspect_threshold: float = 8.0,
    min_refined_leaf_particles: int = 2,
    config: Optional[FixedDepthTreeBuildConfig] = None,
):
    """JIT build for a fixed-depth tree, inferring bounds when not provided."""

    cfg = config or FixedDepthTreeBuildConfig()
    bounds_resolved = infer_bounds(positions) if bounds is None else bounds
    return _tree_impl.build_fixed_depth_tree_jit(
        positions,
        masses,
        bounds_resolved,
        target_leaf_particles=(
            cfg.target_leaf_particles if config else target_leaf_particles
        ),
        return_reordered=cfg.return_reordered if config else return_reordered,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
        max_depth=cfg.max_depth if config else max_depth,
        refine_local=cfg.refine_local if config else refine_local,
        max_refine_levels=cfg.max_refine_levels if config else max_refine_levels,
        aspect_threshold=cfg.aspect_threshold if config else aspect_threshold,
        min_refined_leaf_particles=(
            cfg.min_refined_leaf_particles if config else min_refined_leaf_particles
        ),
    )


__all__ = [
    "MAX_TREE_LEVELS",
    "RadixTree",
    "RadixTreeWorkspace",
    "TreeBuildConfig",
    "FixedDepthTreeBuildConfig",
    "build_fixed_depth_tree",
    "build_fixed_depth_tree_jit",
    "build_tree",
    "build_tree_jit",
    "reorder_particles_by_indices",
]
