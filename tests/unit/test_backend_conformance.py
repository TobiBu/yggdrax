"""Backend conformance checks for tree/topology adapters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests.unit.backend_fixtures import BackendAdapter, conformance_adapters
from yggdrax import (
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)

_TEST_TRAVERSAL_CFG = DualTreeTraversalConfig(
    max_pair_queue=512,
    process_block=32,
    max_interactions_per_node=128,
    max_neighbors_per_leaf=128,
)


def _sample_problem(n: int = 128) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(101)
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


def _run_conformance(adapter: BackendAdapter) -> None:
    positions, masses = _sample_problem(n=32)

    tree, pos_sorted, mass_sorted, inverse = adapter.build_fn(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    assert pos_sorted.shape == positions.shape
    assert mass_sorted.shape == masses.shape
    assert inverse.shape == masses.shape

    # Topology/container compatibility contract.
    geometry = compute_tree_geometry(tree, pos_sorted)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)
    assert geometry.half_extent.shape == (total_nodes, 3)
    assert geometry.radius.shape == (total_nodes,)
    assert geometry.max_extent.shape == (total_nodes,)
    assert jnp.all(jnp.isfinite(geometry.center))

    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=0.6,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )

    assert interactions.sources.ndim == 1
    assert interactions.targets.ndim == 1
    assert neighbors.neighbors.ndim == 1
    assert int(jnp.sum(interactions.counts)) >= 0
    assert int(jnp.sum(neighbors.counts)) >= 0

    # Determinism for fixed inputs/config.
    tree2, _, _, _ = adapter.build_fn(
        positions,
        masses,
        leaf_size=16,
        return_reordered=True,
    )
    assert jnp.array_equal(tree.parent, tree2.parent)
    assert jnp.array_equal(tree.particle_indices, tree2.particle_indices)


@pytest.mark.parametrize(
    "adapter",
    conformance_adapters(),
    ids=lambda adapter: adapter.name,
)
def test_backend_conformance(adapter: BackendAdapter):
    _run_conformance(adapter)
