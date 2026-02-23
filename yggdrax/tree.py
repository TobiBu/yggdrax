"""Public tree-building API for Yggdrax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax
from beartype import beartype
from jaxtyping import Array, jaxtyped

from . import _tree_impl
from .bounds import infer_bounds

MAX_TREE_LEVELS = _tree_impl.MAX_TREE_LEVELS
RadixTreeTopology = _tree_impl.RadixTree
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


TreeType = Literal["radix"]
TreeBuildMode = Literal["adaptive", "fixed_depth"]


@dataclass(frozen=True)
class Tree:
    """Public base class for concrete tree containers."""

    @property
    def num_nodes(self) -> int:
        """Return number of nodes in the concrete topology."""

        return int(self.topology.parent.shape[0])

    @property
    def num_particles(self) -> int:
        """Return number of particles represented by this tree."""

        return int(self.topology.num_particles)

    @property
    def num_leaves(self) -> int:
        """Return number of leaf nodes represented by this tree."""

        return int(self.topology.parent.shape[0]) - int(
            self.topology.num_internal_nodes
        )

    def __getattr__(self, name):
        """Delegate missing attributes to the concrete topology object."""

        return getattr(self.topology, name)

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_particles(
        cls,
        positions: Array,
        masses: Array,
        *,
        tree_type: str = "radix",
        build_mode: str = "adaptive",
        bounds: Optional[tuple[Array, Array]] = None,
        return_reordered: bool = True,
        workspace: Optional[RadixTreeWorkspace] = None,
        return_workspace: bool = False,
        leaf_size: int = 8,
        target_leaf_particles: int = 32,
        max_depth: Optional[int] = None,
        refine_local: bool = True,
        max_refine_levels: int = 2,
        aspect_threshold: float = 8.0,
        min_refined_leaf_particles: int = 2,
    ) -> "Tree":
        """Build and return a concrete tree instance selected by keywords."""

        if tree_type != "radix":
            raise ValueError(
                f"Unsupported tree_type '{tree_type}'. Supported: ('radix',)"
            )
        return RadixTree.from_particles(
            positions,
            masses,
            build_mode=build_mode,
            bounds=bounds,
            return_reordered=return_reordered,
            workspace=workspace,
            return_workspace=return_workspace,
            leaf_size=leaf_size,
            target_leaf_particles=target_leaf_particles,
            max_depth=max_depth,
            refine_local=refine_local,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
            min_refined_leaf_particles=min_refined_leaf_particles,
        )


@dataclass(frozen=True)
class RadixTree(Tree):
    """Concrete radix-tree container implementing the generic Tree contract."""

    topology: RadixTreeTopology
    build_mode: TreeBuildMode = "adaptive"
    positions_sorted: Optional[Array] = None
    masses_sorted: Optional[Array] = None
    inverse_permutation: Optional[Array] = None
    workspace: Optional[RadixTreeWorkspace] = None

    @property
    def tree_type(self) -> TreeType:
        """Tree-family identifier for this concrete tree."""

        return "radix"

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_particles(
        cls,
        positions: Array,
        masses: Array,
        *,
        build_mode: str = "adaptive",
        bounds: Optional[tuple[Array, Array]] = None,
        return_reordered: bool = True,
        workspace: Optional[RadixTreeWorkspace] = None,
        return_workspace: bool = False,
        leaf_size: int = 8,
        target_leaf_particles: int = 32,
        max_depth: Optional[int] = None,
        refine_local: bool = True,
        max_refine_levels: int = 2,
        aspect_threshold: float = 8.0,
        min_refined_leaf_particles: int = 2,
    ) -> "RadixTree":
        """Build a radix tree from particles using a selected build mode."""

        bounds_resolved = infer_bounds(positions) if bounds is None else bounds
        if build_mode == "adaptive":
            result = _tree_impl.build_tree(
                positions,
                masses,
                bounds_resolved,
                return_reordered=return_reordered,
                leaf_size=leaf_size,
                workspace=workspace,
                return_workspace=return_workspace,
            )
        elif build_mode == "fixed_depth":
            result = _tree_impl.build_fixed_depth_tree(
                positions,
                masses,
                bounds_resolved,
                target_leaf_particles=target_leaf_particles,
                return_reordered=return_reordered,
                workspace=workspace,
                return_workspace=return_workspace,
                max_depth=max_depth,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
                min_refined_leaf_particles=min_refined_leaf_particles,
            )
        else:
            raise ValueError(
                "Unsupported build_mode "
                f"'{build_mode}'. Supported: ('adaptive', 'fixed_depth')"
            )

        return cls._from_build_result(
            result=result,
            build_mode=build_mode,
            return_reordered=return_reordered,
            return_workspace=return_workspace,
        )

    @classmethod
    def _from_build_result(
        cls,
        *,
        result,
        build_mode: str,
        return_reordered: bool,
        return_workspace: bool,
    ) -> "RadixTree":
        if return_reordered and return_workspace:
            topology, pos_sorted, mass_sorted, inv, workspace = result
            return cls(
                topology=topology,
                build_mode=build_mode,  # type: ignore[arg-type]
                positions_sorted=pos_sorted,
                masses_sorted=mass_sorted,
                inverse_permutation=inv,
                workspace=workspace,
            )
        if return_reordered:
            topology, pos_sorted, mass_sorted, inv = result
            return cls(
                topology=topology,
                build_mode=build_mode,  # type: ignore[arg-type]
                positions_sorted=pos_sorted,
                masses_sorted=mass_sorted,
                inverse_permutation=inv,
            )
        if return_workspace:
            topology, workspace = result
            return cls(
                topology=topology,
                build_mode=build_mode,  # type: ignore[arg-type]
                workspace=workspace,
            )
        return cls(topology=result, build_mode=build_mode)  # type: ignore[arg-type]


def _register_radix_tree_pytree() -> None:
    if getattr(RadixTree, "_yggdrax_pytree_registered", False):
        return

    def flatten(tree: RadixTree):
        topology = tree.topology
        topology_fields = tuple(getattr(topology, name) for name in topology._fields)
        children = topology_fields + (
            tree.positions_sorted,
            tree.masses_sorted,
            tree.inverse_permutation,
        )
        aux = (type(topology), topology._fields, tree.build_mode)
        return children, aux

    def unflatten(aux, children):
        topology_type, topology_fields, build_mode = aux
        n_topo = len(topology_fields)
        topology_values = children[:n_topo]
        positions_sorted, masses_sorted, inverse_permutation = children[n_topo:]
        topology = topology_type(*topology_values)
        return RadixTree(
            topology=topology,
            build_mode=build_mode,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse_permutation,
            workspace=None,
        )

    jax.tree_util.register_pytree_node(RadixTree, flatten, unflatten)
    setattr(RadixTree, "_yggdrax_pytree_registered", True)


_register_radix_tree_pytree()


def _wrap_radix_public_result(
    *,
    result,
    build_mode: str,
    return_reordered: bool,
    return_workspace: bool,
):
    """Wrap low-level tree_impl outputs while preserving public tuple conventions."""

    if return_reordered and return_workspace:
        topology, pos_sorted, mass_sorted, inv, workspace = result
        tree = RadixTree(
            topology=topology,
            build_mode=build_mode,  # type: ignore[arg-type]
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=inv,
            workspace=workspace,
        )
        return tree, pos_sorted, mass_sorted, inv, workspace
    if return_reordered:
        topology, pos_sorted, mass_sorted, inv = result
        tree = RadixTree(
            topology=topology,
            build_mode=build_mode,  # type: ignore[arg-type]
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=inv,
        )
        return tree, pos_sorted, mass_sorted, inv
    if return_workspace:
        topology, workspace = result
        tree = RadixTree(
            topology=topology,
            build_mode=build_mode,  # type: ignore[arg-type]
            workspace=workspace,
        )
        return tree, workspace
    return RadixTree(
        topology=result,
        build_mode=build_mode,  # type: ignore[arg-type]
    )


@jaxtyped(typechecker=beartype)
def build_tree(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
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
    result = _tree_impl.build_tree(
        positions,
        masses,
        bounds_resolved,
        return_reordered=cfg.return_reordered if config else return_reordered,
        leaf_size=cfg.leaf_size if config else leaf_size,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )
    return _wrap_radix_public_result(
        result=result,
        build_mode="adaptive",
        return_reordered=cfg.return_reordered if config else return_reordered,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


@jaxtyped(typechecker=beartype)
def build_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
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
    result = _tree_impl.build_tree_jit(
        positions,
        masses,
        bounds_resolved,
        return_reordered=cfg.return_reordered if config else return_reordered,
        leaf_size=cfg.leaf_size if config else leaf_size,
        workspace=cfg.workspace if config else workspace,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )
    return _wrap_radix_public_result(
        result=result,
        build_mode="adaptive",
        return_reordered=cfg.return_reordered if config else return_reordered,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


@jaxtyped(typechecker=beartype)
def build_fixed_depth_tree(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
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
    result = _tree_impl.build_fixed_depth_tree(
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
    return _wrap_radix_public_result(
        result=result,
        build_mode="fixed_depth",
        return_reordered=cfg.return_reordered if config else return_reordered,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


@jaxtyped(typechecker=beartype)
def build_fixed_depth_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Optional[tuple[Array, Array]] = None,
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
    result = _tree_impl.build_fixed_depth_tree_jit(
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
    return _wrap_radix_public_result(
        result=result,
        build_mode="fixed_depth",
        return_reordered=cfg.return_reordered if config else return_reordered,
        return_workspace=cfg.return_workspace if config else return_workspace,
    )


__all__ = [
    "MAX_TREE_LEVELS",
    "Tree",
    "TreeType",
    "TreeBuildMode",
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
