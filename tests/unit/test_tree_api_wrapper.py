"""Coverage for yggdrax tree API wrappers."""

import jax
import jax.numpy as jnp

from yggdrax import (
    FixedDepthTreeBuildConfig,
    RadixTreeWorkspace,
    TreeBuildConfig,
    build_fixed_depth_octree,
    build_fixed_depth_tree,
    build_octree,
    build_tree,
)


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(23)
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


def test_build_tree_infers_bounds_when_omitted():
    positions, masses = _sample_problem(n=64)
    tree = build_tree(positions, masses, leaf_size=8)
    assert int(tree.num_particles) == 64
    assert tree.bounds_min.shape == (3,)
    assert tree.bounds_max.shape == (3,)


def test_build_tree_accepts_config_object():
    positions, masses = _sample_problem(n=64)
    cfg = TreeBuildConfig(leaf_size=8, return_reordered=True)
    tree, pos_sorted, mass_sorted, inv = build_tree(positions, masses, config=cfg)
    assert int(tree.num_particles) == 64
    assert pos_sorted.shape == positions.shape
    assert mass_sorted.shape == masses.shape
    assert inv.shape == (64,)


def test_build_octree_accepts_config_object():
    positions, masses = _sample_problem(n=64)
    cfg = TreeBuildConfig(leaf_size=8, return_reordered=True)
    tree, pos_sorted, mass_sorted, inv = build_octree(positions, masses, config=cfg)
    assert int(tree.num_particles) == 64
    assert tree.tree_type == "octree"
    assert tree.oct_num_nodes >= 1
    assert tree.oct_num_leaf_nodes >= 1
    assert tree.radix_leaf_to_oct.shape == (tree.num_leaves,)
    assert pos_sorted.shape == positions.shape
    assert mass_sorted.shape == masses.shape
    assert inv.shape == (64,)


def test_build_fixed_depth_tree_accepts_config_and_infers_bounds():
    positions, masses = _sample_problem(n=64)
    cfg = FixedDepthTreeBuildConfig(target_leaf_particles=16, return_reordered=True)
    tree, pos_sorted, mass_sorted, inv = build_fixed_depth_tree(
        positions,
        masses,
        config=cfg,
    )
    assert int(tree.num_particles) == 64
    assert pos_sorted.shape == positions.shape
    assert mass_sorted.shape == masses.shape
    assert inv.shape == (64,)


def test_build_fixed_depth_octree_accepts_config_and_infers_bounds():
    positions, masses = _sample_problem(n=64)
    cfg = FixedDepthTreeBuildConfig(target_leaf_particles=16, return_reordered=True)
    tree, pos_sorted, mass_sorted, inv = build_fixed_depth_octree(
        positions,
        masses,
        config=cfg,
    )
    assert int(tree.num_particles) == 64
    assert tree.tree_type == "octree"
    assert tree.oct_num_nodes >= 1
    assert tree.oct_num_leaf_nodes >= 1
    assert tree.radix_leaf_to_oct.shape == (tree.num_leaves,)
    assert pos_sorted.shape == positions.shape
    assert mass_sorted.shape == masses.shape
    assert inv.shape == (64,)


def test_build_tree_config_overrides_explicit_flags():
    positions, masses = _sample_problem(n=64)
    workspace = RadixTreeWorkspace(
        parent=jnp.zeros((63,), dtype=jnp.int32),
        left_child=jnp.zeros((63,), dtype=jnp.int32),
        right_child=jnp.zeros((63,), dtype=jnp.int32),
        left_is_leaf=jnp.zeros((63,), dtype=jnp.bool_),
        right_is_leaf=jnp.zeros((63,), dtype=jnp.bool_),
        node_ranges=jnp.zeros((63, 2), dtype=jnp.int32),
    )
    cfg = TreeBuildConfig(
        leaf_size=8,
        return_reordered=False,
        workspace=workspace,
        return_workspace=False,
    )
    tree = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
        return_workspace=True,
        config=cfg,
    )
    assert int(tree.num_particles) == 64
    assert tree.positions_sorted is None
    assert tree.workspace is None
