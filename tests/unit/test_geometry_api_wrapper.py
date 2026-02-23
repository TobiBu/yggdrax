"""Coverage for yggdrax geometry API wrappers."""

import jax
import jax.numpy as jnp

from yggdrax import build_tree, compute_tree_geometry, geometry_to_level_major


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(29)
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


def test_compute_tree_geometry_wrapper_shapes():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)
    assert geometry.half_extent.shape == (total_nodes, 3)
    assert geometry.radius.shape == (total_nodes,)
    assert geometry.max_extent.shape == (total_nodes,)


def test_geometry_to_level_major_wrapper_shapes():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    level_major = geometry_to_level_major(tree, geometry)
    assert level_major.centers.ndim == 3
    assert level_major.half_extents.ndim == 3
    assert level_major.radii.ndim == 2
    assert level_major.max_extents.ndim == 2
    assert level_major.level_counts.ndim == 1
    assert level_major.node_indices.ndim == 2
