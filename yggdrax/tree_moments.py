"""Mass moments for radix tree nodes.

This module accumulates per-node total masses and centres of mass from
particles that were reordered into Morton order by :mod:`yggdrax.tree`.
These summaries are useful when building multipole expansions during the
upward FMM sweep.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE, as_index
from .multipole_utils import (
    MAX_MULTIPOLE_ORDER,
    level_offset,
    multi_index_tuples,
    multi_power,
    total_coefficients,
)
from .tree import require_fmm_core_topology, resolve_tree_topology


class TreeMassMoments(NamedTuple):
    """Aggregate mass information for every node in a radix tree."""

    mass: Array
    center_of_mass: Array


_MULTI_INDEX_CACHE: Dict[int, Tuple[Tuple[int, int, int], ...]] = {
    level: multi_index_tuples(level) for level in range(MAX_MULTIPOLE_ORDER + 1)
}

_MULTI_INDEX_LOOKUP: Dict[int, Dict[Tuple[int, int, int], int]] = {
    level: {combo: idx for idx, combo in enumerate(combos)}
    for level, combos in _MULTI_INDEX_CACHE.items()
}


def _validate_order(max_order: int) -> int:
    order = int(max_order)
    if order < 0 or order > MAX_MULTIPOLE_ORDER:
        raise ValueError(
            f"max_order must be between 0 and {MAX_MULTIPOLE_ORDER} inclusive"
        )
    return order


def _binomial_multi(
    alpha: Tuple[int, int, int],
    beta: Tuple[int, int, int],
) -> int:
    ax, ay, az = alpha
    bx, by, bz = beta
    return math.comb(ax, bx) * math.comb(ay, by) * math.comb(az, bz)


_TOTAL_COMBO_COUNT = total_coefficients(MAX_MULTIPOLE_ORDER)
TranslationTables = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _build_translation_tables() -> TranslationTables:
    combos: List[Tuple[int, int, int]] = []
    for level in range(MAX_MULTIPOLE_ORDER + 1):
        combos.extend(_MULTI_INDEX_CACHE[level])

    combos_arr = np.asarray(combos, dtype=np.int64)
    total = combos_arr.shape[0]

    # Use 64-bit integer storage for index-like tables to match INDEX_DTYPE
    gamma = np.zeros((total, total, 3), dtype=np.int64)
    mask = np.zeros((total, total), dtype=np.bool_)
    binomial = np.zeros((total, total), dtype=np.float64)

    for i in range(total):
        alpha = combos_arr[i]
        for j in range(total):
            beta = combos_arr[j]
            valid = np.all(alpha >= beta)
            mask[i, j] = valid
            if valid:
                gamma[i, j] = alpha - beta
                binomial[i, j] = _binomial_multi(
                    tuple(alpha.tolist()),
                    tuple(beta.tolist()),
                )
            else:
                gamma[i, j] = 0

    return combos_arr, gamma, mask, binomial


(
    _COMBO_TABLE,
    _GAMMA_TABLE,
    _TRANSLATION_MASK,
    _BINOMIAL_TABLE,
) = _build_translation_tables()


class _M2MStencil(NamedTuple):
    gamma_indices: Array
    scales: Array


def _flatten_combos(order: int) -> Tuple[Tuple[int, int, int], ...]:
    combos: List[Tuple[int, int, int]] = []
    for level in range(order + 1):
        combos.extend(_MULTI_INDEX_CACHE[level])
    return tuple(combos)


_COMBOS_BY_ORDER: Tuple[Tuple[Tuple[int, int, int], ...], ...] = tuple(
    _flatten_combos(order) for order in range(MAX_MULTIPOLE_ORDER + 1)
)


def _build_m2m_stencils() -> Tuple[_M2MStencil, ...]:
    stencils: List[_M2MStencil] = []
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        total = total_coefficients(order)
        # gamma indices are index-like — store as 64-bit to match INDEX_DTYPE
        gamma_indices = np.zeros((total, total), dtype=np.int64)
        scales = np.zeros((total, total), dtype=np.float64)

        combo_lookup: Dict[Tuple[int, int, int], int] = {}
        for level in range(order + 1):
            combos_level = _MULTI_INDEX_CACHE[level]
            offset = level_offset(level)
            for idx, combo in enumerate(combos_level):
                combo_lookup[combo] = offset + idx

        for level_a in range(order + 1):
            combos_alpha = _MULTI_INDEX_CACHE[level_a]
            offset_a = level_offset(level_a)
            for idx_a, alpha in enumerate(combos_alpha):
                row = offset_a + idx_a
                for level_b in range(order + 1):
                    combos_beta = _MULTI_INDEX_CACHE[level_b]
                    offset_b = level_offset(level_b)
                    for idx_b, beta in enumerate(combos_beta):
                        col = offset_b + idx_b
                        if (
                            beta[0] <= alpha[0]
                            and beta[1] <= alpha[1]
                            and beta[2] <= alpha[2]
                        ):
                            gamma = (
                                alpha[0] - beta[0],
                                alpha[1] - beta[1],
                                alpha[2] - beta[2],
                            )
                            gamma_idx = combo_lookup[gamma]
                            gamma_indices[row, col] = gamma_idx
                            scales[row, col] = _binomial_multi(alpha, beta)
                        else:
                            gamma_indices[row, col] = 0
                            scales[row, col] = 0.0

        stencils.append(
            _M2MStencil(
                gamma_indices=jnp.asarray(gamma_indices, dtype=INDEX_DTYPE),
                scales=jnp.asarray(scales, dtype=jnp.float64),
            )
        )

    return tuple(stencils)


_M2M_STENCILS = _build_m2m_stencils()


class TreeMultipoleMoments(NamedTuple):
    """Multipole summaries (through hexadecapole) for tree nodes."""

    max_order: int
    mass: Array
    center: Array
    dipole: Array
    second_moment: Array
    quadrupole: Array
    third_moment: Array
    octupole: Array
    fourth_moment: Array
    hexadecapole: Array
    raw_packed: Array


def _level_bounds(level: int) -> Tuple[int, int]:
    return level_offset(level), level_offset(level + 1)


def _level_view(arr: Array, level: int) -> Array:
    start, end = _level_bounds(level)
    return arr[..., start:end]


def _assemble_tensor(
    values: Array,
    combos: Sequence[Tuple[int, int, int]],
    order: int,
) -> Array:
    if order == 0:
        return values
    num_nodes = values.shape[0]
    tensor_shape = (num_nodes,) + (3,) * order
    tensor = jnp.zeros(tensor_shape, dtype=values.dtype)
    for idx, combo in enumerate(combos):
        axes = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
        unique_perms = set(itertools.permutations(axes))
        value = values[:, idx]
        for perm in unique_perms:
            tensor = tensor.at[(slice(None),) + perm].set(value)
    return tensor


def _quadrupole_from_second(second: Array) -> Array:
    trace = jnp.trace(second, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=second.dtype)
    return 3.0 * second - trace[..., None, None] * identity


def _octupole_from_third(third: Array) -> Array:
    r2rk = jnp.stack(
        (
            third[..., 0, 0, 0] + third[..., 1, 1, 0] + third[..., 2, 2, 0],
            third[..., 0, 0, 1] + third[..., 1, 1, 1] + third[..., 2, 2, 1],
            third[..., 0, 0, 2] + third[..., 1, 1, 2] + third[..., 2, 2, 2],
        ),
        axis=-1,
    )

    octupole = jnp.zeros_like(third)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                value = 5.0 * third[..., i, j, k]
                if i == j:
                    value = value - r2rk[..., k]
                if i == k:
                    value = value - r2rk[..., j]
                if j == k:
                    value = value - r2rk[..., i]
                octupole = octupole.at[..., i, j, k].set(value)
    return octupole


def _hexadecapole_from_fourth(fourth: Array) -> Array:
    r2 = fourth[..., 0, 0, :, :] + fourth[..., 1, 1, :, :] + fourth[..., 2, 2, :, :]
    r4 = (
        fourth[..., 0, 0, 0, 0]
        + fourth[..., 1, 1, 1, 1]
        + fourth[..., 2, 2, 2, 2]
        + 2.0
        * (fourth[..., 0, 0, 1, 1] + fourth[..., 0, 0, 2, 2] + fourth[..., 1, 1, 2, 2])
    )

    hexadecapole = jnp.zeros_like(fourth)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    value = 35.0 * fourth[..., i, j, k, ell]
                    term = jnp.zeros_like(value)
                    if i == j:
                        term = term + r2[..., k, ell]
                    if i == k:
                        term = term + r2[..., j, ell]
                    if i == ell:
                        term = term + r2[..., j, k]
                    if j == k:
                        term = term + r2[..., i, ell]
                    if j == ell:
                        term = term + r2[..., i, k]
                    if k == ell:
                        term = term + r2[..., i, j]
                    value = value - 5.0 * term
                    delta_factor = 0
                    if i == j and k == ell:
                        delta_factor += 1
                    if i == k and j == ell:
                        delta_factor += 1
                    if i == ell and j == k:
                        delta_factor += 1
                    if delta_factor:
                        value = value + r4 * delta_factor
                    hexadecapole = hexadecapole.at[..., i, j, k, ell].set(value)
    return hexadecapole


def _pack_tensor_from_stf(
    tensor: Array,
    combos: Sequence[Tuple[int, int, int]],
) -> Array:
    values = []
    for combo in combos:
        index = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
        values.append(tensor[(slice(None),) + index])
    return jnp.stack(values, axis=1)


def _moments_from_raw_packed(
    packed: Array,
    max_order: int,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    dtype = packed.dtype
    num_nodes = packed.shape[0]

    mass = _level_view(packed, 0).reshape(num_nodes)

    dipole = jnp.zeros((num_nodes, 3), dtype=dtype)
    if max_order >= 1:
        level1 = _level_view(packed, 1)
        lookup1 = _MULTI_INDEX_LOOKUP[1]
        dipole = dipole.at[:, 0].set(level1[:, lookup1[(1, 0, 0)]])
        dipole = dipole.at[:, 1].set(level1[:, lookup1[(0, 1, 0)]])
        dipole = dipole.at[:, 2].set(level1[:, lookup1[(0, 0, 1)]])

    second_moment = jnp.zeros((num_nodes, 3, 3), dtype=dtype)
    if max_order >= 2:
        level2 = _level_view(packed, 2)
        lookup2 = _MULTI_INDEX_LOOKUP[2]
        second_moment = second_moment.at[:, 0, 0].set(level2[:, lookup2[(2, 0, 0)]])
        second_moment = second_moment.at[:, 1, 1].set(level2[:, lookup2[(0, 2, 0)]])
        second_moment = second_moment.at[:, 2, 2].set(level2[:, lookup2[(0, 0, 2)]])
        second_moment = second_moment.at[:, 0, 1].set(level2[:, lookup2[(1, 1, 0)]])
        second_moment = second_moment.at[:, 1, 0].set(second_moment[:, 0, 1])
        second_moment = second_moment.at[:, 0, 2].set(level2[:, lookup2[(1, 0, 1)]])
        second_moment = second_moment.at[:, 2, 0].set(second_moment[:, 0, 2])
        second_moment = second_moment.at[:, 1, 2].set(level2[:, lookup2[(0, 1, 1)]])
        second_moment = second_moment.at[:, 2, 1].set(second_moment[:, 1, 2])

    third_moment = jnp.zeros((num_nodes, 3, 3, 3), dtype=dtype)
    if max_order >= 3:
        level3 = _level_view(packed, 3)
        third_moment = _assemble_tensor(
            level3,
            _MULTI_INDEX_CACHE[3],
            3,
        )

    fourth_moment = jnp.zeros((num_nodes, 3, 3, 3, 3), dtype=dtype)
    if max_order >= 4:
        level4 = _level_view(packed, 4)
        fourth_moment = _assemble_tensor(
            level4,
            _MULTI_INDEX_CACHE[4],
            4,
        )

    quadrupole = _quadrupole_from_second(second_moment)

    if max_order >= 3:
        octupole = _octupole_from_third(third_moment)
    else:
        octupole = jnp.zeros_like(third_moment)

    if max_order >= 4:
        hexadecapole = _hexadecapole_from_fourth(fourth_moment)
    else:
        hexadecapole = jnp.zeros_like(fourth_moment)

    mass_mask_vec = mass[:, None] == 0
    mass_mask_mat = mass[:, None, None] == 0
    mass_mask_3 = mass[:, None, None, None] == 0
    mass_mask_4 = mass[:, None, None, None, None] == 0

    dipole = jnp.where(mass_mask_vec, 0.0, dipole)
    second_moment = jnp.where(mass_mask_mat, 0.0, second_moment)
    quadrupole = jnp.where(mass_mask_mat, 0.0, quadrupole)
    third_moment = jnp.where(mass_mask_3, 0.0, third_moment)
    octupole = jnp.where(mass_mask_3, 0.0, octupole)
    fourth_moment = jnp.where(mass_mask_4, 0.0, fourth_moment)
    hexadecapole = jnp.where(mass_mask_4, 0.0, hexadecapole)

    return (
        mass,
        dipole,
        second_moment,
        quadrupole,
        third_moment,
        octupole,
        fourth_moment,
        hexadecapole,
    )


def _moments_from_coefficients(
    packed: Array,
    max_order: int,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    dtype = packed.dtype
    num_nodes = packed.shape[0]

    mass = _level_view(packed, 0).reshape(num_nodes)

    dipole = jnp.zeros((num_nodes, 3), dtype=dtype)
    if max_order >= 1:
        level1 = _level_view(packed, 1)
        lookup1 = _MULTI_INDEX_LOOKUP[1]
        dipole = dipole.at[:, 0].set(level1[:, lookup1[(1, 0, 0)]])
        dipole = dipole.at[:, 1].set(level1[:, lookup1[(0, 1, 0)]])
        dipole = dipole.at[:, 2].set(level1[:, lookup1[(0, 0, 1)]])

    quadrupole = jnp.zeros((num_nodes, 3, 3), dtype=dtype)
    if max_order >= 2:
        level2 = _level_view(packed, 2)
        lookup2 = _MULTI_INDEX_LOOKUP[2]
        quadrupole = quadrupole.at[:, 0, 0].set(level2[:, lookup2[(2, 0, 0)]])
        quadrupole = quadrupole.at[:, 1, 1].set(level2[:, lookup2[(0, 2, 0)]])
        quadrupole = quadrupole.at[:, 2, 2].set(level2[:, lookup2[(0, 0, 2)]])
        quadrupole = quadrupole.at[:, 0, 1].set(level2[:, lookup2[(1, 1, 0)]])
        quadrupole = quadrupole.at[:, 1, 0].set(level2[:, lookup2[(1, 1, 0)]])
        quadrupole = quadrupole.at[:, 0, 2].set(level2[:, lookup2[(1, 0, 1)]])
        quadrupole = quadrupole.at[:, 2, 0].set(level2[:, lookup2[(1, 0, 1)]])
        quadrupole = quadrupole.at[:, 1, 2].set(level2[:, lookup2[(0, 1, 1)]])
        quadrupole = quadrupole.at[:, 2, 1].set(level2[:, lookup2[(0, 1, 1)]])

    octupole = jnp.zeros((num_nodes, 3, 3, 3), dtype=dtype)
    if max_order >= 3:
        level3 = _level_view(packed, 3)
        octupole = _assemble_tensor(level3, _MULTI_INDEX_CACHE[3], 3)

    hexadecapole = jnp.zeros((num_nodes, 3, 3, 3, 3), dtype=dtype)
    if max_order >= 4:
        level4 = _level_view(packed, 4)
        hexadecapole = _assemble_tensor(level4, _MULTI_INDEX_CACHE[4], 4)

    zero_second = jnp.zeros((num_nodes, 3, 3), dtype=dtype)
    zero_third = jnp.zeros((num_nodes, 3, 3, 3), dtype=dtype)
    zero_fourth = jnp.zeros((num_nodes, 3, 3, 3, 3), dtype=dtype)

    mass_mask_vec = mass[:, None] == 0
    mass_mask_mat = mass[:, None, None] == 0
    mass_mask_3 = mass[:, None, None, None] == 0
    mass_mask_4 = mass[:, None, None, None, None] == 0

    dipole = jnp.where(mass_mask_vec, 0.0, dipole)
    quadrupole = jnp.where(mass_mask_mat, 0.0, quadrupole)
    octupole = jnp.where(mass_mask_3, 0.0, octupole)
    hexadecapole = jnp.where(mass_mask_4, 0.0, hexadecapole)

    return (
        mass,
        dipole,
        zero_second,
        quadrupole,
        zero_third,
        octupole,
        zero_fourth,
        hexadecapole,
    )


def _tree_moments_from_raw(
    packed: Array,
    centers: Array,
    order: int,
) -> TreeMultipoleMoments:
    truncated = packed[:, : total_coefficients(order)]
    (
        mass,
        dipole,
        second_moment,
        quadrupole,
        third_moment,
        octupole,
        fourth_moment,
        hexadecapole,
    ) = _moments_from_raw_packed(truncated, order)

    return TreeMultipoleMoments(
        max_order=order,
        mass=mass,
        center=centers,
        dipole=dipole,
        second_moment=second_moment,
        quadrupole=quadrupole,
        third_moment=third_moment,
        octupole=octupole,
        fourth_moment=fourth_moment,
        hexadecapole=hexadecapole,
        raw_packed=truncated,
    )


@jaxtyped(typechecker=beartype)
def tree_moments_from_raw(
    packed: Array,
    centers: Array,
    max_order: int,
) -> TreeMultipoleMoments:
    """Build ``TreeMultipoleMoments`` from raw central coefficients."""

    order = _validate_order(max_order)
    if packed.ndim != 2:
        raise ValueError("packed multipole data must be rank-2")
    required = total_coefficients(order)
    if packed.shape[1] < required:
        raise ValueError(
            "packed data does not contain enough coefficients for max_order"
        )
    if centers.shape[0] != packed.shape[0]:
        raise ValueError("centers must align with packed coefficients")

    return _tree_moments_from_raw(packed, centers, order)


@jaxtyped(typechecker=beartype)
def multipole_from_packed(
    packed: Array,
    centers: Array,
    max_order: int,
) -> TreeMultipoleMoments:
    """Reconstruct multipole tensors from packed triangular coefficients."""

    order = _validate_order(max_order)
    if packed.ndim != 2:
        raise ValueError("packed multipole data must be rank-2")
    required = total_coefficients(order)
    if packed.shape[1] < required:
        raise ValueError(
            "packed data does not contain enough coefficients for max_order"
        )
    truncated = packed[:, :required]
    if centers.shape[0] != truncated.shape[0]:
        raise ValueError("centers must align with packed coefficients")
    (
        mass,
        dipole,
        second_moment,
        quadrupole,
        third_moment,
        octupole,
        fourth_moment,
        hexadecapole,
    ) = _moments_from_coefficients(truncated, order)

    return TreeMultipoleMoments(
        max_order=order,
        mass=mass,
        center=centers,
        dipole=dipole,
        second_moment=second_moment,
        quadrupole=quadrupole,
        third_moment=third_moment,
        octupole=octupole,
        fourth_moment=fourth_moment,
        hexadecapole=hexadecapole,
        raw_packed=truncated,
    )


@jaxtyped(typechecker=beartype)
def translate_packed_moments(
    packed_child: Array,
    delta: Array,
    max_order: int,
) -> Array:
    order = _validate_order(max_order)
    total = total_coefficients(order)
    dtype = packed_child.dtype
    packed = packed_child[:total]
    stencil = _M2M_STENCILS[order]

    delta_vec = jnp.asarray(delta, dtype=dtype)
    combos_flat = _COMBOS_BY_ORDER[order]
    delta_powers = jnp.asarray(
        [multi_power(delta_vec, combo) for combo in combos_flat],
        dtype=dtype,
    )

    scales = stencil.scales.astype(dtype)
    gathered = delta_powers[stencil.gamma_indices]
    matrix = scales * gathered

    return matrix @ packed


def _validate_inputs(
    tree: object,
    positions_sorted: Array,
    masses_sorted: Array,
) -> None:
    topology = resolve_tree_topology(tree)
    require_fmm_core_topology(topology)

    total_nodes = topology.parent.shape[0]
    if topology.node_ranges.shape[0] != total_nodes:
        raise ValueError("tree.node_ranges must align with tree.parent shape")
    try:
        num_particles = int(jnp.asarray(topology.num_particles))
    except jax.errors.ConcretizationTypeError:
        # Under outer jit, tree.num_particles can be traced. Fall back to
        # shape-based validation, which remains static and trace-safe.
        num_particles = int(positions_sorted.shape[0])
    if positions_sorted.shape[0] != num_particles:
        raise ValueError("positions_sorted must match tree.num_particles")
    if masses_sorted.shape[0] != num_particles:
        raise ValueError("masses_sorted must match tree.num_particles")
    if positions_sorted.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")


def compute_tree_mass_moments(
    tree: object,
    positions_sorted: Array,
    masses_sorted: Array,
) -> TreeMassMoments:
    """Compute total mass and centre of mass for every tree node."""

    topology = resolve_tree_topology(tree)
    require_fmm_core_topology(topology)
    _validate_inputs(topology, positions_sorted, masses_sorted)

    ranges = topology.node_ranges.astype(INDEX_DTYPE)
    starts = ranges[:, 0]
    ends = ranges[:, 1] + as_index(1)  # make end exclusive for prefix sums

    # Prefix sums over masses and mass-weighted positions enable O(1)
    # range queries per node.
    pad = jnp.zeros((1,), dtype=masses_sorted.dtype)
    mass_prefix = jnp.concatenate([pad, jnp.cumsum(masses_sorted)])

    weighted = masses_sorted[:, None] * positions_sorted
    weighted_pad = jnp.zeros(
        (1, positions_sorted.shape[1]), dtype=positions_sorted.dtype
    )
    weighted_prefix = jnp.concatenate(
        [weighted_pad, jnp.cumsum(weighted, axis=0)],
        axis=0,
    )

    total_mass = mass_prefix[ends] - mass_prefix[starts]
    moment_sum = weighted_prefix[ends] - weighted_prefix[starts]

    safe_mass = jnp.where(
        total_mass == 0,
        jnp.ones_like(total_mass),
        total_mass,
    )
    center = moment_sum / safe_mass[:, None]
    center = jnp.where(total_mass[:, None] == 0, 0.0, center)

    return TreeMassMoments(mass=total_mass, center_of_mass=center)


def compute_tree_multipole_moments(
    tree: object,
    positions_sorted: Array,
    masses_sorted: Array,
    expansion_centers: Optional[Array] = None,
    max_order: int = 2,
) -> TreeMultipoleMoments:
    """Compute multipole information up to ``max_order`` for each node.

    Parameters
    ----------
    tree : object
        Tree/topology exposing the FMM-core topology contract.
    positions_sorted : Array
        Particle positions in Morton order.
    masses_sorted : Array
        Particle masses reordered identically to ``positions_sorted``.
    expansion_centers : Optional[Array]
        Optional array ``(num_nodes, 3)`` specifying the expansion center for
        each node. When omitted, the node center-of-mass is used, which yields
        zero dipole moments.
    max_order : int
        Highest multipole order to accumulate (``0`` ≤ ``max_order`` ≤ ``4``).

    Returns
    -------
    TreeMultipoleMoments
        Packed multipole moments for each node through ``max_order``.

    Raises
    ------
    ValueError
        If ``expansion_centers`` has an incompatible shape or ``max_order`` is
        outside the supported range.
    """

    topology = resolve_tree_topology(tree)
    require_fmm_core_topology(topology)

    mass_moments = compute_tree_mass_moments(
        topology,
        positions_sorted,
        masses_sorted,
    )

    if expansion_centers is None:
        center = mass_moments.center_of_mass
    else:
        if expansion_centers.shape != mass_moments.center_of_mass.shape:
            raise ValueError("expansion_centers must have shape (num_nodes, 3)")
        center = expansion_centers

    order = _validate_order(max_order)

    dtype = jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    positions = positions_sorted.astype(dtype)
    masses = masses_sorted.astype(dtype)
    center = center.astype(dtype)

    ranges = topology.node_ranges.astype(INDEX_DTYPE)
    starts = ranges[:, 0]
    ends = ranges[:, 1] + as_index(1)

    combo_count = total_coefficients(order)
    combos = jnp.asarray(_COMBO_TABLE[:combo_count], dtype=INDEX_DTYPE)

    def axis_powers(values: Array) -> Array:
        base = jnp.asarray(values, dtype=dtype)[:, None]
        ones = jnp.ones_like(base)
        if order == 0:
            return ones
        repeated = jnp.repeat(base, repeats=order, axis=1)
        stacked = jnp.concatenate([ones, repeated], axis=1)
        return jnp.cumprod(stacked, axis=1)

    axis_powers_x = axis_powers(positions[:, 0])
    axis_powers_y = axis_powers(positions[:, 1])
    axis_powers_z = axis_powers(positions[:, 2])

    monomials = (
        axis_powers_x[:, combos[:, 0]]
        * axis_powers_y[:, combos[:, 1]]
        * axis_powers_z[:, combos[:, 2]]
    )

    weighted = masses[:, None] * monomials
    pad = jnp.zeros((1, combo_count), dtype=dtype)
    prefix = jnp.concatenate(
        [pad, jnp.cumsum(weighted, axis=0)],
        axis=0,
    )

    starts = jnp.asarray(starts, dtype=INDEX_DTYPE)
    ends = jnp.asarray(ends, dtype=INDEX_DTYPE)
    raw_sums = prefix[ends] - prefix[starts]

    neg_center = -center
    center_powers_x = axis_powers(neg_center[:, 0])
    center_powers_y = axis_powers(neg_center[:, 1])
    center_powers_z = axis_powers(neg_center[:, 2])

    gamma = jnp.asarray(
        _GAMMA_TABLE[:combo_count, :combo_count, :],
        dtype=INDEX_DTYPE,
    )
    mask_bool = jnp.asarray(
        _TRANSLATION_MASK[:combo_count, :combo_count],
        dtype=jnp.bool_,
    )
    binomial = jnp.asarray(
        _BINOMIAL_TABLE[:combo_count, :combo_count],
        dtype=dtype,
    )

    gamma_safe = jnp.where(mask_bool[..., None], gamma, 0)

    center_factor = (
        jnp.take(center_powers_x, gamma_safe[..., 0], axis=1)
        * jnp.take(center_powers_y, gamma_safe[..., 1], axis=1)
        * jnp.take(center_powers_z, gamma_safe[..., 2], axis=1)
    )

    mask = mask_bool.astype(dtype)
    transform = center_factor * mask[None, :, :] * binomial[None, :, :]

    packed = jnp.einsum("nij,nj->ni", transform, raw_sums)

    return _tree_moments_from_raw(packed, center, order)


@jaxtyped(typechecker=beartype)
def pack_multipole_expansions(
    moments: TreeMultipoleMoments,
    max_order: int,
) -> Array:
    """Pack multipole coefficients (triangular layout) up to ``max_order``."""

    order = _validate_order(max_order)
    if order > moments.max_order:
        raise ValueError("cannot pack coefficients beyond stored max_order")

    total = total_coefficients(order)
    num_nodes = moments.mass.shape[0]
    result = jnp.zeros((num_nodes, total), dtype=moments.mass.dtype)

    result = result.at[:, level_offset(0)].set(moments.mass)

    tensor_map = {
        1: moments.dipole,
        2: moments.quadrupole,
        3: moments.octupole,
        4: moments.hexadecapole,
    }

    for level in range(1, order + 1):
        combos = _MULTI_INDEX_CACHE[level]
        tensor = tensor_map[level]
        values = _pack_tensor_from_stf(tensor, combos)
        start = level_offset(level)
        end = start + len(combos)
        result = result.at[:, start:end].set(values)

    return result


__all__ = [
    "TreeMassMoments",
    "TreeMultipoleMoments",
    "compute_tree_mass_moments",
    "compute_tree_multipole_moments",
    "multipole_from_packed",
    "tree_moments_from_raw",
    "pack_multipole_expansions",
]
