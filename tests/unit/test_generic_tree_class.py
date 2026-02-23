"""Coverage for the generic public Tree wrapper."""

import jax
import jax.numpy as jnp
import pytest

from yggdrax import Tree


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(37)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def test_tree_from_particles_builds_adaptive_radix_tree():
    positions, masses = _sample_problem(n=64)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="radix",
        build_mode="adaptive",
        leaf_size=8,
        return_reordered=True,
    )

    assert tree.tree_type == "radix"
    assert tree.build_mode == "adaptive"
    assert tree.num_particles == 64
    assert tree.num_nodes > 0
    assert tree.positions_sorted is not None
    assert tree.masses_sorted is not None
    assert tree.inverse_permutation is not None


def test_tree_from_particles_builds_fixed_depth_tree():
    positions, masses = _sample_problem(n=64)
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type="radix",
        build_mode="fixed_depth",
        target_leaf_particles=16,
        return_reordered=True,
    )

    assert tree.tree_type == "radix"
    assert tree.build_mode == "fixed_depth"
    assert tree.num_particles == 64
    assert tree.positions_sorted is not None
    assert tree.masses_sorted is not None
    assert tree.inverse_permutation is not None


def test_tree_from_particles_rejects_unknown_tree_type():
    positions, masses = _sample_problem(n=16)
    with pytest.raises(ValueError, match="Unsupported tree_type"):
        Tree.from_particles(positions, masses, tree_type="octree")


def test_tree_from_particles_rejects_unknown_build_mode():
    positions, masses = _sample_problem(n=16)
    with pytest.raises(ValueError, match="Unsupported build_mode"):
        Tree.from_particles(positions, masses, build_mode="kdtree")
