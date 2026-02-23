"""Tests for tree mass and multipole moments."""

import jax.numpy as jnp
import pytest

from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import compute_tree_geometry
from yggdrax.multipole_utils import level_offset, total_coefficients
from yggdrax.tree import Tree, build_tree
from yggdrax.tree_moments import (
    compute_tree_mass_moments,
    compute_tree_multipole_moments,
    pack_multipole_expansions,
)

DEFAULT_TEST_LEAF_SIZE = 1


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.1, 0.0, 0.3],
            [0.6, 0.4, -0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    return tree, pos_sorted, mass_sorted


def test_mass_moment_shapes():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    total_nodes = tree.parent.shape[0]
    assert moments.mass.shape == (total_nodes,)
    assert moments.center_of_mass.shape == (total_nodes, 3)


def test_multipole_moment_shapes():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    multipole = compute_tree_multipole_moments(tree, pos_sorted, mass_sorted)

    total_nodes = tree.parent.shape[0]
    assert multipole.max_order == 2
    assert multipole.mass.shape == (total_nodes,)
    assert multipole.center.shape == (total_nodes, 3)
    assert multipole.dipole.shape == (total_nodes, 3)
    assert multipole.second_moment.shape == (total_nodes, 3, 3)
    assert multipole.quadrupole.shape == (total_nodes, 3, 3)
    assert multipole.third_moment.shape == (total_nodes, 3, 3, 3)
    assert multipole.octupole.shape == (total_nodes, 3, 3, 3)
    assert multipole.fourth_moment.shape == (total_nodes, 3, 3, 3, 3)
    assert multipole.hexadecapole.shape == (total_nodes, 3, 3, 3, 3)
    assert multipole.raw_packed.shape == (
        total_nodes,
        total_coefficients(2),
    )


def test_leaf_masses_and_centers_match_particles():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    num_internal = tree.num_internal_nodes
    assert jnp.allclose(moments.mass[num_internal:], mass_sorted)
    assert jnp.allclose(moments.center_of_mass[num_internal:], pos_sorted)

    multipole = compute_tree_multipole_moments(tree, pos_sorted, mass_sorted)
    assert jnp.allclose(multipole.mass[num_internal:], mass_sorted)
    assert jnp.allclose(multipole.center[num_internal:], pos_sorted)
    assert jnp.allclose(multipole.dipole[num_internal:], 0.0)
    assert jnp.allclose(multipole.quadrupole[num_internal:], 0.0)
    assert jnp.allclose(multipole.third_moment[num_internal:], 0.0)
    assert jnp.allclose(multipole.octupole[num_internal:], 0.0)
    assert jnp.allclose(multipole.fourth_moment[num_internal:], 0.0)
    assert jnp.allclose(multipole.hexadecapole[num_internal:], 0.0)


@pytest.mark.parametrize("tree_type", ["radix", "kdtree"])
def test_tree_wrapper_supports_mass_and_multipole_moments(tree_type: str):
    positions = jnp.array(
        [
            [-0.6, -0.3, -0.2],
            [-0.2, 0.4, -0.1],
            [0.1, -0.5, 0.3],
            [0.5, 0.2, -0.4],
            [0.8, 0.7, 0.6],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 2.0, 1.5, 0.5, 3.0], dtype=jnp.float64)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        return_reordered=True,
        leaf_size=2,
    )

    moments = compute_tree_mass_moments(
        tree,
        tree.positions_sorted,
        tree.masses_sorted,
    )
    multipole = compute_tree_multipole_moments(
        tree,
        tree.positions_sorted,
        tree.masses_sorted,
        max_order=3,
    )

    assert moments.mass.shape[0] == int(tree.num_nodes)
    assert multipole.raw_packed.shape == (
        int(tree.num_nodes),
        total_coefficients(3),
    )
    assert jnp.all(jnp.isfinite(moments.mass))
    assert jnp.all(jnp.isfinite(moments.center_of_mass))
    assert jnp.all(jnp.isfinite(multipole.raw_packed))
    assert jnp.isclose(jnp.max(moments.mass), jnp.sum(tree.masses_sorted))


def test_root_mass_and_com():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_mask = parents == -1
    assert int(root_mask.sum()) == 1
    root_index = int(jnp.argmax(root_mask.astype(INDEX_DTYPE)))

    expected_mass = jnp.sum(mass_sorted)
    expected_com = (
        jnp.sum(
            mass_sorted[:, None] * pos_sorted,
            axis=0,
        )
        / expected_mass
    )

    assert jnp.isclose(moments.mass[root_index], expected_mass)
    assert jnp.allclose(moments.center_of_mass[root_index], expected_com)

    multipole = compute_tree_multipole_moments(tree, pos_sorted, mass_sorted)

    rel = pos_sorted - multipole.center[root_index]
    dipole_direct = jnp.sum(mass_sorted[:, None] * rel, axis=0)
    second_direct = jnp.einsum("n,ni,nj->ij", mass_sorted, rel, rel)
    eye3 = jnp.eye(3, dtype=pos_sorted.dtype)
    quad_direct = 3.0 * second_direct - jnp.trace(second_direct) * eye3

    assert jnp.allclose(multipole.dipole[root_index], dipole_direct)
    assert jnp.allclose(multipole.second_moment[root_index], second_direct)
    assert jnp.allclose(multipole.quadrupole[root_index], quad_direct)


def test_multipole_about_geometry_center_matches_manual():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)
    multipole = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        expansion_centers=geom.center,
    )

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_idx = int(jnp.argmax((parents == -1).astype(INDEX_DTYPE)))

    center = geom.center[root_idx]
    rel = pos_sorted - center
    expected_dipole = jnp.sum(mass_sorted[:, None] * rel, axis=0)
    expected_second = jnp.einsum("n,ni,nj->ij", mass_sorted, rel, rel)
    eye3 = jnp.eye(3, dtype=pos_sorted.dtype)
    expected_quad = 3.0 * expected_second - jnp.trace(expected_second) * eye3

    assert jnp.allclose(multipole.center[root_idx], center)
    assert jnp.allclose(multipole.dipole[root_idx], expected_dipole)
    assert jnp.allclose(multipole.second_moment[root_idx], expected_second)
    assert jnp.allclose(multipole.quadrupole[root_idx], expected_quad)


def test_high_order_multipoles_match_manual():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    multipole = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=4,
    )

    total_nodes = tree.parent.shape[0]
    assert multipole.max_order == 4
    assert multipole.raw_packed.shape == (
        total_nodes,
        total_coefficients(4),
    )

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_idx = int(jnp.argmax((parents == -1).astype(INDEX_DTYPE)))

    center = multipole.center[root_idx]
    rel = pos_sorted - center

    third_direct = jnp.einsum(
        "n,ni,nj,nk->ijk",
        mass_sorted,
        rel,
        rel,
        rel,
    )
    fourth_direct = jnp.einsum(
        "n,ni,nj,nk,nl->ijkl",
        mass_sorted,
        rel,
        rel,
        rel,
        rel,
    )

    assert jnp.allclose(multipole.third_moment[root_idx], third_direct)
    assert jnp.allclose(multipole.fourth_moment[root_idx], fourth_direct)

    r2 = (
        fourth_direct[0, 0, :, :]
        + fourth_direct[1, 1, :, :]
        + fourth_direct[2, 2, :, :]
    )
    r4 = (
        fourth_direct[0, 0, 0, 0]
        + fourth_direct[1, 1, 1, 1]
        + fourth_direct[2, 2, 2, 2]
        + 2.0
        * (
            fourth_direct[0, 0, 1, 1]
            + fourth_direct[0, 0, 2, 2]
            + fourth_direct[1, 1, 2, 2]
        )
    )

    expected_octupole = jnp.zeros_like(third_direct)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                value = 5.0 * third_direct[i, j, k]
                if i == j:
                    value = value - (
                        third_direct[0, 0, k]
                        + third_direct[1, 1, k]
                        + third_direct[2, 2, k]
                    )
                if i == k:
                    value = value - (
                        third_direct[0, 0, j]
                        + third_direct[1, 1, j]
                        + third_direct[2, 2, j]
                    )
                if j == k:
                    value = value - (
                        third_direct[0, 0, i]
                        + third_direct[1, 1, i]
                        + third_direct[2, 2, i]
                    )
                expected_octupole = expected_octupole.at[i, j, k].set(value)

    assert jnp.allclose(multipole.octupole[root_idx], expected_octupole)

    expected_hexadecapole = jnp.zeros_like(fourth_direct)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    value = 35.0 * fourth_direct[i, j, k, ell]
                    term = 0.0
                    if i == j:
                        term = term + r2[k, ell]
                    if i == k:
                        term = term + r2[j, ell]
                    if i == ell:
                        term = term + r2[j, k]
                    if j == k:
                        term = term + r2[i, ell]
                    if j == ell:
                        term = term + r2[i, k]
                    if k == ell:
                        term = term + r2[i, j]
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
                    expected_hexadecapole = expected_hexadecapole.at[
                        i,
                        j,
                        k,
                        ell,
                    ].set(value)

    assert jnp.allclose(
        multipole.hexadecapole[root_idx],
        expected_hexadecapole,
    )

    packed = pack_multipole_expansions(multipole, max_order=4)
    assert packed.shape == (total_nodes, total_coefficients(4))

    def combos(level: int):
        result = []
        for ix in range(level + 1):
            for iy in range(level + 1 - ix):
                iz = level - ix - iy
                result.append((ix, iy, iz))
        return result

    def pack_tensor(tensor, level):
        entries = []
        for combo in combos(level):
            index = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
            entries.append(tensor[(slice(None),) + index])
        return jnp.stack(entries, axis=1)

    expected = jnp.concatenate(
        [
            multipole.mass[:, None],
            pack_tensor(multipole.dipole, 1),
            pack_tensor(multipole.quadrupole, 2),
            pack_tensor(multipole.octupole, 3),
            pack_tensor(multipole.hexadecapole, 4),
        ],
        axis=1,
    )

    assert jnp.allclose(packed, expected)

    with pytest.raises(ValueError):
        pack_multipole_expansions(multipole, max_order=5)


def test_pack_multipole_expansions_matches_layout():
    tree, pos_sorted, mass_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)
    multipole = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        expansion_centers=geom.center,
    )

    packed = pack_multipole_expansions(multipole, max_order=2)
    total_nodes = tree.parent.shape[0]
    assert packed.shape == (total_nodes, total_coefficients(2))

    num_internal = tree.num_internal_nodes
    parents = tree.parent[:num_internal]
    root_idx = int(jnp.argmax((parents == -1).astype(INDEX_DTYPE)))

    expected_vec = jnp.concatenate(
        [
            jnp.array([multipole.mass[root_idx]]),
            jnp.array(
                [
                    multipole.dipole[root_idx, 2],
                    multipole.dipole[root_idx, 1],
                    multipole.dipole[root_idx, 0],
                ]
            ),
            jnp.array(
                [
                    multipole.quadrupole[root_idx, 2, 2],
                    multipole.quadrupole[root_idx, 1, 2],
                    multipole.quadrupole[root_idx, 1, 1],
                    multipole.quadrupole[root_idx, 0, 2],
                    multipole.quadrupole[root_idx, 0, 1],
                    multipole.quadrupole[root_idx, 0, 0],
                ]
            ),
        ]
    )

    assert jnp.allclose(packed[root_idx], expected_vec)

    offset0 = level_offset(0)
    offset1 = level_offset(1)
    offset2 = level_offset(2)
    assert offset0 == 0
    assert offset1 == 1
    assert offset2 == 4

    with pytest.raises(ValueError):
        pack_multipole_expansions(multipole, max_order=3)


def test_zero_mass_nodes_return_zero_center():
    positions = jnp.array(
        [
            [-0.2, 0.3, 0.4],
            [0.1, -0.5, 0.2],
        ]
    )
    masses = jnp.zeros((positions.shape[0],))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    assert jnp.allclose(moments.mass, 0.0)
    assert jnp.allclose(moments.center_of_mass, 0.0)

    multipole = compute_tree_multipole_moments(tree, pos_sorted, mass_sorted)
    assert jnp.allclose(multipole.mass, 0.0)
    assert jnp.allclose(multipole.dipole, 0.0)
    assert jnp.allclose(multipole.second_moment, 0.0)
    assert jnp.allclose(multipole.quadrupole, 0.0)
    assert jnp.allclose(multipole.third_moment, 0.0)
    assert jnp.allclose(multipole.octupole, 0.0)
    assert jnp.allclose(multipole.fourth_moment, 0.0)
    assert jnp.allclose(multipole.hexadecapole, 0.0)
