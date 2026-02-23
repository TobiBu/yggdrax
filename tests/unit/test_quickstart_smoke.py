"""Smoke test for the documented quick-start path."""

import jax
import jax.numpy as jnp

from yggdrax import (
    build_interactions_and_neighbors,
    build_tree,
    compute_tree_geometry,
)


def test_quickstart_pipeline_smoke():
    key = jax.random.PRNGKey(0)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (128, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (128,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )

    tree = build_tree(positions, masses, leaf_size=16)
    positions_sorted = positions[tree.particle_indices]
    geometry = compute_tree_geometry(tree, positions_sorted)
    interactions, neighbors = build_interactions_and_neighbors(tree, geometry)

    assert int(tree.num_particles) == 128
    assert int(tree.num_nodes) > 0
    assert int(tree.num_leaves) > 0
    assert int(jnp.sum(interactions.counts)) >= 0
    assert int(jnp.sum(neighbors.counts)) >= 0
