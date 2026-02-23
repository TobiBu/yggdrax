"""Public tree-building API for Yggdrax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from . import _tree_impl
from .bounds import infer_bounds
from .dtypes import INDEX_DTYPE, as_index
from .kdtree import KDTree as KDTreeTopology
from .kdtree import build_kdtree

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


TreeType = Literal["radix", "kdtree"]
TreeBuildMode = Literal["adaptive", "fixed_depth"]


@dataclass(frozen=True)
class TreeBuildRequest:
    """Common request object passed to registered tree builders."""

    positions: Array
    masses: Array
    build_mode: str
    bounds: Optional[tuple[Array, Array]]
    return_reordered: bool
    workspace: Optional[RadixTreeWorkspace]
    return_workspace: bool
    leaf_size: int
    target_leaf_particles: int
    max_depth: Optional[int]
    refine_local: bool
    max_refine_levels: int
    aspect_threshold: float
    min_refined_leaf_particles: int


TreeBuilder = Callable[[TreeBuildRequest], "Tree"]

FMM_CORE_REQUIRED_FIELDS: tuple[str, ...] = (
    "parent",
    "left_child",
    "right_child",
    "node_ranges",
    "num_particles",
    "use_morton_geometry",
)

MORTON_TOPOLOGY_REQUIRED_FIELDS: tuple[str, ...] = (
    "bounds_min",
    "bounds_max",
    "leaf_codes",
    "leaf_depths",
)

# Backward-compatible alias used by previous checks.
FMM_TOPOLOGY_REQUIRED_FIELDS: tuple[str, ...] = (
    FMM_CORE_REQUIRED_FIELDS + MORTON_TOPOLOGY_REQUIRED_FIELDS
)


@dataclass(frozen=True)
class Tree:
    """Public base class for concrete tree containers."""

    @property
    def num_nodes(self) -> int:
        """Return number of nodes in the concrete topology."""

        topology = self.topology
        if hasattr(topology, "parent"):
            return int(topology.parent.shape[0])
        if hasattr(topology, "node_start"):
            return int(topology.node_start.shape[0])
        raise AttributeError("topology does not expose parent or node_start")

    @property
    def num_particles(self) -> int:
        """Return number of particles represented by this tree."""

        topology = self.topology
        if hasattr(topology, "num_particles"):
            return int(topology.num_particles)
        if hasattr(topology, "points"):
            return int(topology.points.shape[0])
        raise AttributeError("topology does not expose num_particles or points")

    @property
    def num_leaves(self) -> int:
        """Return number of leaf nodes represented by this tree."""

        topology = self.topology
        if hasattr(topology, "leaf_nodes"):
            return int(topology.leaf_nodes.shape[0])
        if hasattr(topology, "parent") and hasattr(topology, "num_internal_nodes"):
            return int(topology.parent.shape[0]) - int(topology.num_internal_nodes)
        raise AttributeError("topology does not expose leaf metadata")

    def __getattr__(self, name):
        """Delegate missing attributes to the concrete topology object."""

        return getattr(self.topology, name)

    @property
    def missing_fmm_topology_fields(self) -> tuple[str, ...]:
        """Return missing topology fields required by FMM core APIs."""

        return missing_fmm_topology_fields(self)

    @property
    def supports_fmm_topology(self) -> bool:
        """Whether this tree exposes topology needed by FMM core routines."""

        return len(self.missing_fmm_topology_fields) == 0

    def require_fmm_topology(self) -> None:
        """Raise when this tree cannot satisfy FMM core topology requirements."""

        require_fmm_topology(self)

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

        request = TreeBuildRequest(
            positions=positions,
            masses=masses,
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

        builder = _TREE_BUILDERS.get(tree_type)
        if builder is None:
            supported = ", ".join(f"'{name}'" for name in sorted(_TREE_BUILDERS))
            raise ValueError(
                f"Unsupported tree_type '{tree_type}'. Supported: ({supported})"
            )
        return builder(request)


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


@dataclass(frozen=True)
class KDParticleTree(Tree):
    """Concrete KD-tree container implementing the generic Tree contract."""

    topology: KDTreeTopology
    build_mode: Literal["adaptive"] = "adaptive"
    positions_sorted: Optional[Array] = None
    masses_sorted: Optional[Array] = None
    inverse_permutation: Optional[Array] = None
    workspace: Optional[RadixTreeWorkspace] = None

    @property
    def tree_type(self) -> TreeType:
        return "kdtree"

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
    ) -> "KDParticleTree":
        del (
            bounds,
            workspace,
            return_workspace,
            target_leaf_particles,
            max_depth,
            refine_local,
            max_refine_levels,
            aspect_threshold,
            min_refined_leaf_particles,
        )
        if build_mode != "adaptive":
            raise ValueError(
                "Unsupported build_mode "
                f"'{build_mode}' for kdtree. Supported: ('adaptive',)"
            )

        topology = build_kdtree(positions, leaf_size=leaf_size)
        if return_reordered:
            idx = jnp.asarray(topology.particle_indices, dtype=INDEX_DTYPE)
            pos_sorted = positions[idx]
            mass_sorted = masses[idx]
            inv = jnp.empty_like(idx)
            inv = inv.at[idx].set(jnp.arange(idx.shape[0], dtype=idx.dtype))
            return cls(
                topology=topology,
                positions_sorted=pos_sorted,
                masses_sorted=mass_sorted,
                inverse_permutation=inv,
            )
        return cls(topology=topology)


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


def _register_kdtree_tree_pytree() -> None:
    if getattr(KDParticleTree, "_yggdrax_pytree_registered", False):
        return

    def flatten(tree: KDParticleTree):
        children = (
            tree.topology,
            tree.positions_sorted,
            tree.masses_sorted,
            tree.inverse_permutation,
        )
        aux = ("adaptive",)
        return children, aux

    def unflatten(aux, children):
        (build_mode,) = aux
        topology, positions_sorted, masses_sorted, inverse_permutation = children
        return KDParticleTree(
            topology=topology,
            build_mode=build_mode,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse_permutation,
            workspace=None,
        )

    jax.tree_util.register_pytree_node(KDParticleTree, flatten, unflatten)
    setattr(KDParticleTree, "_yggdrax_pytree_registered", True)


_register_kdtree_tree_pytree()


def _build_radix_tree_from_request(request: TreeBuildRequest) -> Tree:
    return RadixTree.from_particles(
        request.positions,
        request.masses,
        build_mode=request.build_mode,
        bounds=request.bounds,
        return_reordered=request.return_reordered,
        workspace=request.workspace,
        return_workspace=request.return_workspace,
        leaf_size=request.leaf_size,
        target_leaf_particles=request.target_leaf_particles,
        max_depth=request.max_depth,
        refine_local=request.refine_local,
        max_refine_levels=request.max_refine_levels,
        aspect_threshold=request.aspect_threshold,
        min_refined_leaf_particles=request.min_refined_leaf_particles,
    )


def _build_kdtree_from_request(request: TreeBuildRequest) -> Tree:
    return KDParticleTree.from_particles(
        request.positions,
        request.masses,
        build_mode=request.build_mode,
        return_reordered=request.return_reordered,
        leaf_size=request.leaf_size,
    )


_TREE_BUILDERS: dict[str, TreeBuilder] = {
    "radix": _build_radix_tree_from_request,
    "kdtree": _build_kdtree_from_request,
}


def resolve_tree_topology(tree_or_topology: object) -> object:
    """Return a topology payload from a tree container or topology object."""

    topology = getattr(tree_or_topology, "topology", None)
    return tree_or_topology if topology is None else topology


def missing_fmm_core_topology_fields(tree_or_topology: object) -> tuple[str, ...]:
    """Return FMM-core required topology fields missing on the provided object."""

    topology = resolve_tree_topology(tree_or_topology)
    return tuple(
        name for name in FMM_CORE_REQUIRED_FIELDS if not hasattr(topology, name)
    )


def missing_morton_topology_fields(tree_or_topology: object) -> tuple[str, ...]:
    """Return Morton-geometry required fields missing on the provided object."""

    topology = resolve_tree_topology(tree_or_topology)
    return tuple(
        name
        for name in MORTON_TOPOLOGY_REQUIRED_FIELDS
        if not hasattr(topology, name)
    )


def has_fmm_core_topology(tree_or_topology: object) -> bool:
    """Return ``True`` when all FMM-core fields are available."""

    return len(missing_fmm_core_topology_fields(tree_or_topology)) == 0


def has_morton_topology(tree_or_topology: object) -> bool:
    """Return ``True`` when all Morton-geometry fields are available."""

    return len(missing_morton_topology_fields(tree_or_topology)) == 0


def require_fmm_core_topology(tree_or_topology: object) -> None:
    """Raise ``ValueError`` when FMM-core topology fields are missing."""

    missing = missing_fmm_core_topology_fields(tree_or_topology)
    if not missing:
        return
    tree_type = getattr(tree_or_topology, "tree_type", None)
    prefix = f"tree_type='{tree_type}' " if tree_type is not None else ""
    missing_txt = ", ".join(missing)
    raise ValueError(
        f"{prefix}topology is missing FMM-core-required fields: {missing_txt}"
    )


def require_morton_topology(tree_or_topology: object) -> None:
    """Raise ``ValueError`` when Morton-geometry fields are missing."""

    missing = missing_morton_topology_fields(tree_or_topology)
    if not missing:
        return
    tree_type = getattr(tree_or_topology, "tree_type", None)
    prefix = f"tree_type='{tree_type}' " if tree_type is not None else ""
    missing_txt = ", ".join(missing)
    raise ValueError(
        f"{prefix}topology is missing Morton-geometry-required fields: {missing_txt}"
    )


# Backward-compatible aliases
def missing_fmm_topology_fields(tree_or_topology: object) -> tuple[str, ...]:
    """Alias of ``missing_fmm_core_topology_fields`` for compatibility."""

    return missing_fmm_core_topology_fields(tree_or_topology)


def has_fmm_topology(tree_or_topology: object) -> bool:
    """Alias of ``has_fmm_core_topology`` for compatibility."""

    return has_fmm_core_topology(tree_or_topology)


def require_fmm_topology(tree_or_topology: object) -> None:
    """Alias of ``require_fmm_core_topology`` for compatibility."""

    require_fmm_core_topology(tree_or_topology)


def get_num_internal_nodes(tree: object) -> int:
    """Return number of internal nodes, deriving it from child buffers when needed."""

    if hasattr(tree, "num_internal_nodes"):
        return int(getattr(tree, "num_internal_nodes"))
    return int(jnp.asarray(tree.left_child).shape[0])


def get_node_levels(tree: object) -> Array:
    """Return per-node depth levels, deriving from parent links when missing."""

    if hasattr(tree, "node_level"):
        return jnp.asarray(getattr(tree, "node_level"), dtype=INDEX_DTYPE)

    parent = jnp.asarray(tree.parent, dtype=INDEX_DTYPE)
    num_nodes = int(parent.shape[0])
    if num_nodes == 0:
        return jnp.zeros((0,), dtype=INDEX_DTYPE)

    levels = jnp.zeros((num_nodes,), dtype=INDEX_DTYPE)
    parent_safe = jnp.where(parent >= 0, parent, as_index(0))
    for _ in range(max(num_nodes - 1, 0)):
        candidate = jnp.where(
            parent >= 0,
            levels[parent_safe] + as_index(1),
            as_index(0),
        )
        levels = jnp.maximum(levels, candidate)
    return levels


def get_num_levels(tree: object, *, node_levels: Optional[Array] = None) -> int:
    """Return tree depth count, deriving from node levels when needed."""

    if hasattr(tree, "num_levels"):
        return int(getattr(tree, "num_levels"))
    levels = get_node_levels(tree) if node_levels is None else jnp.asarray(node_levels)
    if int(levels.shape[0]) == 0:
        return 0
    return int(jnp.max(levels)) + 1


def get_level_offsets(tree: object, *, node_levels: Optional[Array] = None) -> Array:
    """Return level offsets, deriving compact level partitions when absent."""

    if hasattr(tree, "level_offsets"):
        return jnp.asarray(getattr(tree, "level_offsets"), dtype=INDEX_DTYPE)

    levels = get_node_levels(tree) if node_levels is None else jnp.asarray(node_levels)
    num_levels = get_num_levels(tree, node_levels=levels)
    counts = jnp.bincount(levels, length=num_levels)
    return jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(counts, dtype=INDEX_DTYPE),
        ],
        axis=0,
    )


def get_nodes_by_level(tree: object, *, node_levels: Optional[Array] = None) -> Array:
    """Return nodes sorted by level (stable by node index within each level)."""

    if hasattr(tree, "nodes_by_level"):
        return jnp.asarray(getattr(tree, "nodes_by_level"), dtype=INDEX_DTYPE)

    levels = get_node_levels(tree) if node_levels is None else jnp.asarray(node_levels)
    node_ids = jnp.arange(levels.shape[0], dtype=INDEX_DTYPE)
    order = jnp.lexsort((node_ids, levels))
    return jnp.asarray(order, dtype=INDEX_DTYPE)


def available_tree_types() -> tuple[str, ...]:
    """Return registered public tree-type identifiers."""

    return tuple(sorted(_TREE_BUILDERS.keys()))


def register_tree_builder(
    tree_type: str, builder: TreeBuilder, *, overwrite: bool = False
) -> None:
    """Register a new tree builder for ``Tree.from_particles`` dispatch."""

    normalized = tree_type.strip()
    if not normalized:
        raise ValueError("tree_type must be a non-empty string")
    if (normalized in _TREE_BUILDERS) and (not overwrite):
        raise ValueError(
            f"tree_type '{normalized}' is already registered; "
            "pass overwrite=True to replace it"
        )
    _TREE_BUILDERS[normalized] = builder


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
    "FMM_CORE_REQUIRED_FIELDS",
    "MORTON_TOPOLOGY_REQUIRED_FIELDS",
    "FMM_TOPOLOGY_REQUIRED_FIELDS",
    "Tree",
    "TreeBuilder",
    "TreeBuildRequest",
    "TreeType",
    "TreeBuildMode",
    "RadixTree",
    "KDParticleTree",
    "RadixTreeWorkspace",
    "TreeBuildConfig",
    "FixedDepthTreeBuildConfig",
    "build_fixed_depth_tree",
    "build_fixed_depth_tree_jit",
    "build_tree",
    "build_tree_jit",
    "available_tree_types",
    "get_level_offsets",
    "get_node_levels",
    "get_nodes_by_level",
    "get_num_internal_nodes",
    "get_num_levels",
    "has_fmm_core_topology",
    "has_fmm_topology",
    "has_morton_topology",
    "missing_fmm_core_topology_fields",
    "missing_fmm_topology_fields",
    "missing_morton_topology_fields",
    "resolve_tree_topology",
    "require_fmm_core_topology",
    "require_fmm_topology",
    "require_morton_topology",
    "register_tree_builder",
    "reorder_particles_by_indices",
]
