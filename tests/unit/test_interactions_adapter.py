"""Coverage for yggdrasil interaction config adapters."""

import jax
import jax.numpy as jnp

from yggdrasil.interactions import DualTreeTraversalConfig as ExpanseTraversalConfig
from yggdrasil import (
    DualTreeTraversalConfig,
    NodeInteractionList,
    NodeNeighborList,
    build_interactions_and_neighbors,
    build_tree,
    compute_tree_geometry,
    infer_bounds,
)


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(19)
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


def test_interactions_accepts_yggdrasil_traversal_config():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1


def test_interactions_accepts_expanse_traversal_config():
    positions, masses = _sample_problem(n=64)
    tree, pos_sorted, _, _ = build_tree(
        positions,
        masses,
        infer_bounds(positions),
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    traversal_cfg = ExpanseTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=traversal_cfg,
    )
    assert isinstance(interactions, NodeInteractionList)
    assert isinstance(neighbors, NodeNeighborList)
    assert interactions.sources.ndim == 1
    assert neighbors.neighbors.ndim == 1
