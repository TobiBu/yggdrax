"""
Radix tree (binary tree) construction for hierarchical particle organization.

The radix tree is built from sorted Morton codes and represents the
hierarchical octree structure implicitly through the Morton code ordering.
"""

import math
from collections import deque
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, jaxtyped

from .dtypes import INDEX_DTYPE, as_index


class RadixTree(NamedTuple):
    """
    Radix tree representation using parallel arrays.

    Attributes:
        parent: Parent node index for each node
        left_child: Left child index for each internal node (-1 for leaves)
        right_child: Right child index for each internal node (-1 for leaves)
        left_is_leaf: Boolean flag for whether the left child is a leaf
        right_is_leaf: Boolean flag for whether the right child is a leaf
        particle_indices: Sorted particle indices by Morton code
        morton_codes: Sorted Morton codes
        node_ranges: (start, end) inclusive range in Morton-sorted order
        num_particles: Total number of particles
        num_internal_nodes: Number of internal nodes (num_leaves - 1)
        node_level: Per-node depth from the root in level order
        level_offsets: Prefix offsets into ``nodes_by_level`` for each level
        nodes_by_level: Node indices arranged contiguously by level depth
        num_levels: Total populated levels (scalar ``int32`` array)
    bounds_min: Minimum domain corner used during Morton encoding
    bounds_max: Maximum domain corner used during Morton encoding
    leaf_codes: Morton codes representing each leaf's spatial cell
    leaf_depths: Morton depth for each leaf (``-1`` when unspecified)
        use_morton_geometry: Flag indicating whether Morton-derived geometry
            should be used when computing node extents
    """

    parent: jnp.ndarray
    left_child: jnp.ndarray
    right_child: jnp.ndarray
    left_is_leaf: jnp.ndarray
    right_is_leaf: jnp.ndarray
    particle_indices: jnp.ndarray
    morton_codes: jnp.ndarray
    node_ranges: jnp.ndarray
    num_particles: int
    num_internal_nodes: int
    node_level: jnp.ndarray
    level_offsets: jnp.ndarray
    nodes_by_level: jnp.ndarray
    num_levels: jnp.ndarray
    bounds_min: jnp.ndarray
    bounds_max: jnp.ndarray
    leaf_codes: jnp.ndarray
    leaf_depths: jnp.ndarray
    use_morton_geometry: jnp.ndarray


class RadixTreeWorkspace(NamedTuple):
    """Reusable buffers for radix tree construction."""

    parent: jnp.ndarray
    left_child: jnp.ndarray
    right_child: jnp.ndarray
    left_is_leaf: jnp.ndarray
    right_is_leaf: jnp.ndarray
    node_ranges: jnp.ndarray


Bounds = tuple[Float[Array, "3"], Float[Array, "3"]]

MAX_TREE_LEVELS = 64
_MAX_MORTON_LEVEL = 21  # 21 * 3 = 63 bits of Morton depth


# ---------------------------------------------
# JIT-safe helpers for LBVH (Karras 2012)
# ---------------------------------------------


def _clz_u64(x: jnp.ndarray) -> jnp.ndarray:
    """
    Count leading zeros for uint64 values (returns int32).

    Implemented with a fixed sequence of shifts and conditional adds to
    avoid Python control flow. Works on CPU/GPU/TPU without custom calls.
    """
    x = x.astype(jnp.uint64)
    n = as_index(0)

    y = x >> jnp.uint64(32)
    add = jnp.where(y == 0, as_index(32), as_index(0))
    n = n + add
    x = jnp.where(y == 0, x << jnp.uint64(32), x)

    y = x >> jnp.uint64(48)
    add = jnp.where(y == 0, as_index(16), as_index(0))
    n = n + add
    x = jnp.where(y == 0, x << jnp.uint64(16), x)

    y = x >> jnp.uint64(56)
    add = jnp.where(y == 0, as_index(8), as_index(0))
    n = n + add
    x = jnp.where(y == 0, x << jnp.uint64(8), x)

    y = x >> jnp.uint64(60)
    add = jnp.where(y == 0, as_index(4), as_index(0))
    n = n + add
    x = jnp.where(y == 0, x << jnp.uint64(4), x)

    y = x >> jnp.uint64(62)
    add = jnp.where(y == 0, as_index(2), as_index(0))
    n = n + add
    x = jnp.where(y == 0, x << jnp.uint64(2), x)

    y = x >> jnp.uint64(63)
    add = jnp.where(y == 0, as_index(1), as_index(0))
    n = n + add

    # If original input was 0, return 64.
    return jnp.where(x == 0, as_index(64), n)


def _lcp_codes_only(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Leading common prefix length for two uint64 Morton codes
    (no index tie-break). Returns int32 in [0, 64].
    """
    xor = jnp.bitwise_xor(a, b)
    return jnp.where(xor == jnp.uint64(0), as_index(64), _clz_u64(xor))


# ---------------------------------------------
# Tree construction (LBVH)
# ---------------------------------------------


@jaxtyped(typechecker=beartype)
def build_tree(
    positions: Array,
    masses: Array,
    bounds: Bounds,
    *,
    return_reordered: bool = False,
    leaf_size: int = 8,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
):
    """
    Build a radix tree (LBVH) from particle positions using Karras (2012).

    - Leaves correspond to contiguous Morton-sorted ranges (size <= leaf_size)
    - Internal nodes are indexed 0..num_leaves-2
    - Leaf node indices in the combined array are num_internal + leaf_idx
    - Children entries store combined node indices (internal or leaf)

    Args:
        positions: (N,3) particle positions
        masses: (N,) particle masses
        bounds: (min,max) domain bounds
        return_reordered: if True, also return
            (positions_sorted, masses_sorted, inverse_perm)
        leaf_size: maximum number of particles per leaf node (>= 1)
        workspace: optional reusable buffers to avoid reallocations
        return_workspace: if True, return updated workspace as final item

    Returns:
        RadixTree, and when return_reordered=True also the reordered arrays.
    """
    from .morton import morton_encode  # local import to avoid circulars

    n = positions.shape[0]
    assert n >= 1, "Need at least one particle"

    if leaf_size < 1:
        raise ValueError("leaf_size must be >= 1")

    # Encode and sort by Morton codes (stable tie-break by original index)
    morton_codes = morton_encode(positions, bounds)
    orig_idx = jnp.arange(n, dtype=INDEX_DTYPE)
    # jnp.lexsort sorts by last key first; provide (idx, codes) to break ties
    sorted_indices = jnp.lexsort((orig_idx, morton_codes))
    sorted_codes = morton_codes[sorted_indices]

    # Determine leaf groups in Morton order
    leaf_starts = jnp.arange(0, n, leaf_size, dtype=INDEX_DTYPE)
    leaf_ends = jnp.minimum(leaf_starts + leaf_size, n)

    return _build_tree_from_leaf_partitions(
        positions,
        masses,
        sorted_indices,
        sorted_codes,
        leaf_starts,
        leaf_ends,
        bounds,
        return_reordered=return_reordered,
        workspace=workspace,
        return_workspace=return_workspace,
    )


@jaxtyped(typechecker=beartype)
def build_fixed_depth_tree(
    positions: Array,
    masses: Array,
    bounds: Bounds,
    *,
    target_leaf_particles: int = 32,
    return_reordered: bool = False,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
    max_depth: Optional[int] = None,
    # Local Morton refinement options
    refine_local: bool = True,
    max_refine_levels: int = 2,
    aspect_threshold: float = 8.0,
    min_refined_leaf_particles: int = 2,
):
    """Build a tree whose leaves all share the same Morton depth.

    When local refinement kicks in, the helper will not split leaves below
    ``min_refined_leaf_particles`` particles so that near-field pairs remain
    anchored to reproducible leaf extents.
    """

    from .morton import morton_encode  # local import to avoid circulars

    n = positions.shape[0]
    if n < 1:
        raise ValueError("Need at least one particle")

    if target_leaf_particles < 1:
        raise ValueError("target_leaf_particles must be >= 1")

    morton_codes = morton_encode(positions, bounds)
    orig_idx = jnp.arange(n, dtype=INDEX_DTYPE)
    sorted_indices = jnp.lexsort((orig_idx, morton_codes))
    sorted_codes = morton_codes[sorted_indices]

    max_allowed_depth = min(MAX_TREE_LEVELS - 1, _MAX_MORTON_LEVEL)
    if max_depth is not None:
        max_allowed_depth = min(max_allowed_depth, int(max_depth))

    resolved_depth = _resolve_fixed_depth_level(
        n,
        target_leaf_particles,
        max_allowed_depth=max_allowed_depth,
    )

    (
        leaf_starts,
        leaf_ends,
        leaf_codes,
        leaf_depths,
    ) = _fixed_depth_leaf_partitions(
        sorted_codes,
        resolved_depth,
        n,
    )

    # Post-process leaf partitions to avoid pathological aspect ratios by
    # locally refining Morton buckets where leaves are highly elongated.
    # This preserves Morton contiguity and keeps downstream LBVH logic intact.
    # Prepare JAX-side reordered slices; convert to host NumPy only when
    # running the host-based refinement helper (and when not tracing).
    pos_sorted_jax = positions[sorted_indices]
    sorted_codes_jax = sorted_codes
    leaf_starts_jax = leaf_starts
    leaf_ends_jax = leaf_ends
    leaf_codes_jax = leaf_codes
    leaf_depths_jax = leaf_depths

    def _refine_leaf_partitions_by_aspect(
        pos_sorted_np: np.ndarray,
        sorted_codes_np: np.ndarray,
        leaf_starts_np: np.ndarray,
        leaf_ends_np: np.ndarray,
        leaf_codes_np: np.ndarray,
        leaf_depths_np: np.ndarray,
        current_depth: int,
        max_refine_levels: int = 2,
        aspect_threshold: float = 8.0,
        min_leaf_particles: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Refine Morton leaf partitions locally where aspect ratio is high.

        Parameters are NumPy arrays on host. Returns refined (starts, ends,
        codes, depths) arrays also as NumPy arrays.
        """

        codes_all = sorted_codes_np

        new_starts: list[int] = []
        new_ends: list[int] = []
        new_codes: list[np.uint64] = []
        new_depths: list[int] = []

        # Queue items are tuples (s, e, base_code, depth)
        queue = deque[tuple[int, int, np.uint64, int]]()
        for i in range(leaf_starts_np.shape[0]):
            s = int(leaf_starts_np[i])
            e = int(leaf_ends_np[i])
            base_code = np.uint64(leaf_codes_np[i])
            depth_val = (
                int(leaf_depths_np[i])
                if leaf_depths_np is not None
                else int(current_depth)
            )
            queue.append((s, e, base_code, depth_val))

        eps = 1e-12
        while queue:
            s, e, base_code, depth_level = queue.popleft()
            if e <= s:
                # Keep zero-particle leaves so downstream consumers retain
                # the original Morton layout and near-field neighbour lists
                # continue to include placeholder entries for empty cells.
                new_starts.append(s)
                new_ends.append(e)
                new_codes.append(base_code)
                new_depths.append(depth_level)
                continue
            pts = pos_sorted_np[s:e]
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            extents = maxs - mins
            min_axis = float(max(extents.min(), eps))
            max_axis = float(extents.max())
            aspect = max_axis / min_axis if min_axis > 0 else float("inf")

            # If aspect is acceptable or we've exhausted refinement levels,
            # keep this partition.
            exceeded_levels = (depth_level - current_depth) >= max_refine_levels
            if aspect <= aspect_threshold or exceeded_levels:
                new_starts.append(s)
                new_ends.append(e)
                new_codes.append(base_code)
                new_depths.append(depth_level)
                continue

            # Attempt one level of Morton subdivision for this leaf.
            entry_shift = max(0, 64 - depth_level * 3)
            child_depth = depth_level + 1
            child_shift = entry_shift - 3
            if child_shift < 0:
                # can't subdivide further in Morton bits; keep as-is
                new_starts.append(s)
                new_ends.append(e)
                new_codes.append(base_code)
                new_depths.append(depth_level)
                continue

            # compute 3-bit sub-id for each particle within this leaf
            sub_ids = (
                (codes_all[s:e] >> np.uint64(child_shift)) & np.uint64(0x7)
            ).astype(np.int64)
            # bincount (length 8) gives counts for child subcells
            # in Morton order
            counts = np.bincount(sub_ids, minlength=8)
            # If subdivision fails to distribute particles across multiple
            # child buckets, keep the original partition to avoid inventing
            # deeper Morton depths that do not correspond to actual tree
            # nodes. This prevents downstream geometry from assigning
            # overly tight bounds and retains consistency with the radix
            # tree structure.
            if int(np.count_nonzero(counts)) <= 1:
                new_starts.append(s)
                new_ends.append(e)
                new_codes.append(base_code)
                new_depths.append(depth_level)
                continue
            # Build child ranges for non-empty bins
            offset = s
            for sub in range(8):
                cnt = int(counts[sub])
                if cnt == 0:
                    continue
                child_s = offset
                child_e = offset + cnt
                child_code = np.uint64(
                    base_code | (np.uint64(sub) << np.uint64(child_shift))
                )
                if cnt <= min_leaf_particles:
                    new_starts.append(child_s)
                    new_ends.append(child_e)
                    new_codes.append(child_code)
                    new_depths.append(child_depth)
                    offset = child_e
                    continue
                # push child back into queue for further refinement if needed
                queue.append((child_s, child_e, child_code, child_depth))
                offset = child_e

        if len(new_starts) == 0:
            # fallback to original partitions
            return leaf_starts_np, leaf_ends_np, leaf_codes_np, leaf_depths_np

        starts_arr = np.asarray(new_starts, dtype=np.int64)
        ends_arr = np.asarray(new_ends, dtype=np.int64)
        codes_arr = np.asarray(new_codes, dtype=np.uint64)
        depths_arr = np.asarray(new_depths, dtype=np.int64)
        order = np.argsort(starts_arr)

        return (
            starts_arr[order],
            ends_arr[order],
            codes_arr[order],
            depths_arr[order],
        )

    # Run the local refinement pass. The refinement helper uses NumPy and
    # performs host-side array manipulation, so avoid invoking it when this
    # function is being traced by JAX (e.g., when JIT-compiling). In traced
    # contexts we simply skip refinement to keep the function JAX-traceable.
    if refine_local:
        # Detect whether `positions` is a JAX Tracer; if so, skip host
        # NumPy conversion and refinement.
        _is_tracing = isinstance(positions, jax.core.Tracer)

        if _is_tracing:
            # Under tracing/JIT we cannot call np.asarray on traced arrays.
            # Skip the local refinement and use the original JAX partitions.
            refined_starts_np, refined_ends_np, refined_codes_np = (
                leaf_starts_jax,
                leaf_ends_jax,
                leaf_codes_jax,
            )
            refined_depths_np = leaf_depths_jax
        else:
            (
                refined_starts_np,
                refined_ends_np,
                refined_codes_np,
                refined_depths_np,
            ) = (
                leaf_starts_jax,
                leaf_ends_jax,
                leaf_codes_jax,
                leaf_depths_jax,
            )
            try:
                # Convert required arrays to host NumPy and run the refinement
                pos_sorted = np.asarray(pos_sorted_jax)
                sorted_codes_np = np.asarray(sorted_codes_jax, dtype=np.uint64)
                leaf_starts_np = np.asarray(leaf_starts_jax, dtype=np.int64)
                leaf_ends_np = np.asarray(leaf_ends_jax, dtype=np.int64)
                leaf_codes_np = np.asarray(leaf_codes_jax, dtype=np.uint64)
                leaf_depths_np = np.asarray(leaf_depths_jax, dtype=np.int64)

                (
                    refined_starts_np,
                    refined_ends_np,
                    refined_codes_np,
                    refined_depths_np,
                ) = _refine_leaf_partitions_by_aspect(
                    pos_sorted,
                    sorted_codes_np,
                    leaf_starts_np,
                    leaf_ends_np,
                    leaf_codes_np,
                    leaf_depths_np,
                    resolved_depth,
                    max_refine_levels=max_refine_levels,
                    aspect_threshold=aspect_threshold,
                    min_leaf_particles=min_refined_leaf_particles,
                )
            except (TypeError, ValueError):
                # If any unexpected failure occurs, fall back to original
                # partitions
                pass
    else:
        # No refinement requested; use the original JAX-side partitions so
        # later conversions to JAX arrays work consistently whether we ran
        # the host-based refinement or not.
        (
            refined_starts_np,
            refined_ends_np,
            refined_codes_np,
            refined_depths_np,
        ) = (
            leaf_starts_jax,
            leaf_ends_jax,
            leaf_codes_jax,
            leaf_depths_jax,
        )

    # Convert back to jnp arrays for downstream processing
    leaf_starts = jnp.asarray(refined_starts_np, dtype=INDEX_DTYPE)
    leaf_ends = jnp.asarray(refined_ends_np, dtype=INDEX_DTYPE)
    leaf_codes = jnp.asarray(refined_codes_np, dtype=jnp.uint64)
    leaf_depths = jnp.asarray(refined_depths_np, dtype=INDEX_DTYPE)

    return _build_tree_from_leaf_partitions(
        positions,
        masses,
        sorted_indices,
        sorted_codes,
        leaf_starts,
        leaf_ends,
        bounds,
        use_morton_geometry=True,
        return_reordered=return_reordered,
        workspace=workspace,
        return_workspace=return_workspace,
        leaf_codes_override=leaf_codes,
        leaf_depths_override=leaf_depths,
    )


# ---------------------------------------------
# Reordering helpers for memory locality
# ---------------------------------------------


@jaxtyped(typechecker=beartype)
def inverse_permutation(sorted_indices: Array) -> Array:
    """
    Compute inverse permutation for `sorted_indices`.

    If `sorted_indices[k] = i` (k is the position of original index i in
    the sorted order), then `inv[i] = k`.
    """
    n = sorted_indices.shape[0]
    inv = jnp.empty((n,), dtype=INDEX_DTYPE)
    return inv.at[sorted_indices].set(jnp.arange(n, dtype=INDEX_DTYPE))


@jaxtyped(typechecker=beartype)
def reorder_particles_by_indices(
    positions: Array,
    masses: Array,
    sorted_indices: Array,
) -> tuple[Array, Array, Array]:
    """
    Reorder particle arrays to Morton order using `sorted_indices` and
    return (positions_sorted, masses_sorted, inverse_perm).
    """
    pos_sorted = positions[sorted_indices]
    mass_sorted = masses[sorted_indices]
    inv = inverse_permutation(sorted_indices)
    return pos_sorted, mass_sorted, inv


@partial(
    jax.jit,
    static_argnames=("return_reordered", "leaf_size", "return_workspace"),
)
@jaxtyped(typechecker=beartype)
def build_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Bounds,
    *,
    return_reordered: bool = False,
    leaf_size: int = 8,
    workspace: Optional[RadixTreeWorkspace] = None,
    return_workspace: bool = False,
):
    """JIT-compiled wrapper around :func:`build_tree`.

    Parameters mirror :func:`build_tree`; see that docstring for details.
    JIT compilation specialises on the boolean flags and ``leaf_size`` to
    maximise performance when constructing many trees of fixed shape.
    """

    return build_tree(
        positions,
        masses,
        bounds,
        return_reordered=return_reordered,
        leaf_size=leaf_size,
        workspace=workspace,
        return_workspace=return_workspace,
    )


@partial(
    jax.jit,
    static_argnames=(
        "return_reordered",
        "target_leaf_particles",
        "return_workspace",
        "max_depth",
        "refine_local",
        "max_refine_levels",
        "aspect_threshold",
        "min_refined_leaf_particles",
    ),
)
@jaxtyped(typechecker=beartype)
def build_fixed_depth_tree_jit(
    positions: Array,
    masses: Array,
    bounds: Bounds,
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
):
    """JIT-compiled wrapper for :func:`build_fixed_depth_tree`."""

    return build_fixed_depth_tree(
        positions,
        masses,
        bounds,
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


def _build_tree_from_leaf_partitions(
    positions: Array,
    masses: Array,
    sorted_indices: Array,
    sorted_codes: Array,
    leaf_starts: Array,
    leaf_ends_exclusive: Array,
    bounds: Bounds,
    *,
    use_morton_geometry: bool = False,
    return_reordered: bool,
    workspace: Optional[RadixTreeWorkspace],
    return_workspace: bool,
    leaf_codes_override: Optional[Array] = None,
    leaf_depths_override: Optional[Array] = None,
):
    n = positions.shape[0]
    num_leaves = int(leaf_starts.shape[0])

    if num_leaves == 0:
        raise ValueError("tree construction requires at least one leaf")

    leaf_starts = leaf_starts.astype(INDEX_DTYPE)
    leaf_ends_exclusive = leaf_ends_exclusive.astype(INDEX_DTYPE)
    if leaf_codes_override is not None:
        leaf_codes = jnp.asarray(leaf_codes_override, dtype=jnp.uint64)
    else:
        leaf_codes = sorted_codes[leaf_starts]
    if leaf_depths_override is not None:
        leaf_depths = jnp.asarray(leaf_depths_override, dtype=INDEX_DTYPE)
    else:
        leaf_depths = jnp.full((num_leaves,), -1, dtype=INDEX_DTYPE)

    bounds_min = jnp.asarray(bounds[0], dtype=positions.dtype)
    bounds_max = jnp.asarray(bounds[1], dtype=positions.dtype)
    morton_geometry_flag = jnp.asarray(use_morton_geometry, dtype=jnp.bool_)

    if return_reordered:
        pos_sorted = positions[sorted_indices]
        mass_sorted = masses[sorted_indices]
        inv = inverse_permutation(sorted_indices)

    if num_leaves == 1:
        parent_base = (
            workspace.parent
            if workspace is not None and workspace.parent.shape == (1,)
            else None
        )
        parent = (
            parent_base.at[:].set(-1)
            if parent_base is not None
            else jnp.full((1,), -1, dtype=INDEX_DTYPE)
        )
        node_base = (
            workspace.node_ranges
            if workspace is not None and workspace.node_ranges.shape == (1, 2)
            else None
        )
        node_ranges = (
            node_base.at[:].set(jnp.array([0, n - 1], dtype=INDEX_DTYPE))
            if node_base is not None
            else jnp.array([[0, n - 1]], dtype=INDEX_DTYPE)
        )
        left_child = jnp.empty((0,), dtype=INDEX_DTYPE)
        right_child = jnp.empty((0,), dtype=INDEX_DTYPE)
        left_is_leaf = jnp.empty((0,), dtype=jnp.bool_)
        right_is_leaf = jnp.empty((0,), dtype=jnp.bool_)
        level_offsets = jnp.zeros((MAX_TREE_LEVELS + 1,), dtype=INDEX_DTYPE)
        level_offsets = level_offsets.at[1].set(as_index(1))
        tree = RadixTree(
            parent=parent,
            left_child=left_child,
            right_child=right_child,
            left_is_leaf=left_is_leaf,
            right_is_leaf=right_is_leaf,
            particle_indices=sorted_indices,
            morton_codes=sorted_codes,
            node_ranges=node_ranges,
            num_particles=int(n),
            num_internal_nodes=0,
            node_level=jnp.array([0], dtype=INDEX_DTYPE),
            level_offsets=level_offsets,
            nodes_by_level=jnp.array([0], dtype=INDEX_DTYPE),
            num_levels=as_index(1),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            leaf_codes=leaf_codes,
            leaf_depths=leaf_depths,
            use_morton_geometry=morton_geometry_flag,
        )
        updated_workspace = RadixTreeWorkspace(
            parent=parent,
            left_child=left_child,
            right_child=right_child,
            left_is_leaf=left_is_leaf,
            right_is_leaf=right_is_leaf,
            node_ranges=node_ranges,
        )
        if return_reordered:
            outputs = [tree, pos_sorted, mass_sorted, inv]
        else:
            outputs = [tree]
        if return_workspace:
            outputs.append(updated_workspace)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    num_internal = int(num_leaves - 1)
    total_nodes = int(num_internal + num_leaves)
    indices = jnp.arange(num_internal, dtype=INDEX_DTYPE)

    def _delta_vec(i: jnp.ndarray, j: jnp.ndarray) -> jnp.ndarray:
        cond = (j >= 0) & (j < num_leaves)
        safe_j = jnp.where(cond, j, as_index(0))
        ci = leaf_codes[i]
        cj = leaf_codes[safe_j]
        same = ci == cj
        clz_code = _clz_u64(jnp.bitwise_xor(ci, cj))
        clz_idx = _clz_u64(
            jnp.bitwise_xor(
                jnp.asarray(i, dtype=jnp.uint64),
                jnp.asarray(safe_j, dtype=jnp.uint64),
            )
        )
        val = jnp.where(same, as_index(64) + clz_idx, clz_code)
        return jnp.where(cond, val, as_index(-1))

    delta_plus = _delta_vec(indices, indices + 1)
    delta_minus = _delta_vec(indices, indices - 1)

    d = jnp.sign(delta_plus - delta_minus)
    d = jnp.where(indices == 0, as_index(1), d)
    d = jnp.where(d == 0, as_index(1), d)

    delta_min = _delta_vec(indices, indices - d)

    span = jnp.zeros_like(indices)
    max_k = 31
    for k in range(max_k - 1, -1, -1):
        step = as_index(1 << k)
        candidate = indices + (span + step) * d
        delta_candidate = _delta_vec(indices, candidate)
        cond = (
            (candidate >= 0) & (candidate < num_leaves) & (delta_candidate > delta_min)
        )
        span = jnp.where(cond, span + step, span)

    end_idx = indices + span * d
    first = jnp.minimum(indices, end_idx)
    last = jnp.maximum(indices, end_idx)

    start_particle = leaf_starts[first]
    end_particle = leaf_ends_exclusive[last] - as_index(1)
    node_ranges_internal = jnp.stack([start_particle, end_particle], axis=1)

    first_code = leaf_codes[first]
    last_code = leaf_codes[last]
    common_prefix = _lcp_codes_only(first_code, last_code)

    split = first
    for k in range(max_k - 1, -1, -1):
        step = as_index(1 << k)
        mid = split + step
        safe_mid = jnp.where(mid < last, mid, last)
        cmp = _lcp_codes_only(first_code, leaf_codes[safe_mid])
        cond = (mid < last) & (cmp > common_prefix)
        split = jnp.where(cond, mid, split)

    left_is_leaf = split == first
    right_is_leaf = split + as_index(1) == last

    leaf_offset = as_index(num_internal)
    left_leaf_idx = leaf_offset + first
    right_leaf_idx = leaf_offset + last

    left_indices = jnp.where(left_is_leaf, left_leaf_idx, split)
    right_indices = jnp.where(
        right_is_leaf,
        right_leaf_idx,
        split + as_index(1),
    )

    left_child_vals = left_indices.astype(INDEX_DTYPE)
    right_child_vals = right_indices.astype(INDEX_DTYPE)

    left_child_base = (
        workspace.left_child
        if (workspace is not None and workspace.left_child.shape == (num_internal,))
        else None
    )
    left_child = (
        left_child_base.at[:].set(left_child_vals)
        if left_child_base is not None
        else left_child_vals
    )

    right_child_base = (
        workspace.right_child
        if (workspace is not None and workspace.right_child.shape == (num_internal,))
        else None
    )
    right_child = (
        right_child_base.at[:].set(right_child_vals)
        if right_child_base is not None
        else right_child_vals
    )

    left_flag_vals = left_is_leaf.astype(jnp.bool_)
    right_flag_vals = right_is_leaf.astype(jnp.bool_)

    left_flag_base = (
        workspace.left_is_leaf
        if (workspace is not None and workspace.left_is_leaf.shape == (num_internal,))
        else None
    )
    left_is_leaf_arr = (
        left_flag_base.at[:].set(left_flag_vals)
        if left_flag_base is not None
        else left_flag_vals
    )

    right_flag_base = (
        workspace.right_is_leaf
        if (workspace is not None and workspace.right_is_leaf.shape == (num_internal,))
        else None
    )
    right_is_leaf_arr = (
        right_flag_base.at[:].set(right_flag_vals)
        if right_flag_base is not None
        else right_flag_vals
    )

    parent_base = None
    if workspace is not None and workspace.parent.shape == (total_nodes,):
        parent_base = workspace.parent
    parent = (
        parent_base.at[:].set(-1)
        if parent_base is not None
        else jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    )
    parent = parent.at[left_child].set(indices)
    parent = parent.at[right_child].set(indices)

    node_base = None
    if workspace is not None and workspace.node_ranges.shape == (
        total_nodes,
        2,
    ):
        node_base = workspace.node_ranges
    node_ranges = (
        node_base
        if node_base is not None
        else jnp.zeros((total_nodes, 2), dtype=INDEX_DTYPE)
    )
    node_ranges = node_ranges.at[indices].set(node_ranges_internal)

    leaf_ranges = jnp.stack(
        [leaf_starts, leaf_ends_exclusive - as_index(1)],
        axis=1,
    )
    node_ranges = node_ranges.at[num_internal:].set(leaf_ranges)

    node_level = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    node_level = node_level.at[0].set(as_index(0))

    def propagate_level(idx, level_arr):
        current_level = level_arr[idx]

        def assign_child(child_idx, arr):
            def set_level(ci):
                return arr.at[ci].set(current_level + as_index(1))

            return lax.cond(
                child_idx >= 0,
                set_level,
                lambda _: arr,
                child_idx,
            )

        arr = level_arr
        arr = assign_child(left_child[idx], arr)
        arr = assign_child(right_child[idx], arr)
        return arr

    node_level = lax.fori_loop(0, num_internal, propagate_level, node_level)

    max_level = jnp.max(node_level)
    num_levels = max_level + as_index(1)

    level_counts = jnp.zeros((MAX_TREE_LEVELS,), dtype=INDEX_DTYPE)
    level_counts = level_counts.at[node_level].add(as_index(1))

    level_offsets = jnp.zeros((MAX_TREE_LEVELS + 1,), dtype=INDEX_DTYPE)
    level_offsets = level_offsets.at[1:].set(jnp.cumsum(level_counts))

    nodes_by_level = jnp.argsort(node_level, stable=True)

    num_levels = jnp.clip(
        num_levels,
        min=as_index(1),
        max=as_index(MAX_TREE_LEVELS),
    )

    tree = RadixTree(
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        left_is_leaf=left_is_leaf_arr,
        right_is_leaf=right_is_leaf_arr,
        particle_indices=sorted_indices,
        morton_codes=sorted_codes,
        node_ranges=node_ranges,
        num_particles=int(n),
        num_internal_nodes=num_internal,
        node_level=node_level,
        level_offsets=level_offsets,
        nodes_by_level=nodes_by_level,
        num_levels=num_levels,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        leaf_codes=leaf_codes,
        leaf_depths=leaf_depths,
        use_morton_geometry=morton_geometry_flag,
    )

    updated_workspace = RadixTreeWorkspace(
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        left_is_leaf=left_is_leaf_arr,
        right_is_leaf=right_is_leaf_arr,
        node_ranges=node_ranges,
    )

    outputs = [tree]
    if return_reordered:
        outputs.extend([pos_sorted, mass_sorted, inv])
    if return_workspace:
        outputs.append(updated_workspace)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def _resolve_fixed_depth_level(
    num_particles: int,
    target_leaf_particles: int,
    *,
    max_allowed_depth: int,
) -> int:
    target = max(1, int(target_leaf_particles))
    if num_particles <= target:
        return 0

    target_leaves = math.ceil(num_particles / target)
    depth = math.ceil(math.log(target_leaves, 8))
    return int(min(depth, max_allowed_depth))


def _fixed_depth_leaf_partitions(
    sorted_codes: Array,
    depth: int,
    num_particles: int,
) -> tuple[Array, Array, Array, Array]:
    depth = max(0, int(depth))
    if depth == 0:
        starts = jnp.asarray([0], dtype=INDEX_DTYPE)
        ends = jnp.asarray([num_particles], dtype=INDEX_DTYPE)
        codes = jnp.asarray([0], dtype=jnp.uint64)
        depths = jnp.asarray([0], dtype=INDEX_DTYPE)
        return starts, ends, codes, depths

    # Morton codes use 63 meaningful bits (21 per axis). Subtracting from 64
    # would drop an extra bit and halve the number of buckets, so subtract
    # from 63 instead.
    shift = max(0, 63 - depth * 3)
    leaf_ids = jnp.right_shift(sorted_codes, jnp.uint64(shift))
    leaf_ids = leaf_ids.astype(INDEX_DTYPE)
    num_leaves = int(1 << (3 * depth))

    counts = jnp.bincount(leaf_ids, length=num_leaves)
    counts = counts.astype(INDEX_DTYPE)
    leaf_starts = jnp.cumsum(counts, dtype=INDEX_DTYPE) - counts
    leaf_ends = leaf_starts + counts

    codes = jnp.arange(num_leaves, dtype=jnp.uint64) << jnp.uint64(shift)
    depths = jnp.full((num_leaves,), depth, dtype=INDEX_DTYPE)
    return leaf_starts, leaf_ends, codes, depths


__all__ = [
    "Bounds",
    "RadixTree",
    "RadixTreeWorkspace",
    "build_tree",
    "build_tree_jit",
    "build_fixed_depth_tree",
    "build_fixed_depth_tree_jit",
    "inverse_permutation",
    "reorder_particles_by_indices",
]
