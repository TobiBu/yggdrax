"""Coverage for yggdrax tree API wrappers."""

import jax
import jax.numpy as jnp

from yggdrax import (
    FixedDepthTreeBuildConfig,
    TreeBuildConfig,
    build_fixed_depth_tree,
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
