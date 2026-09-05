"""KD-tree backend parity with radix/octree (Phase 0, differentiable-apps paper).

The differentiable case studies (SVGD, correlation functions) drive the
KD-tree backend through the same ``pair_policy`` / dual-tree machinery as the
radix and octree backends. This module asserts that the KD-tree honours the
same *structural* interaction/neighbor contract on identical inputs.

Note on "same counts": the three backends build different topologies (a KD
median split is not a Morton radix split), so per-node interaction counts are
*not* expected to match numerically across backends -- only the CSR contract,
determinism, and geometry conformance are backend invariants. Exact physical
pair-coverage agreement is validated separately against Corrfunc/TreeCorr in
the Phase 2 correlation-function experiments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)

_BACKENDS = ("radix", "kdtree", "octree")

_TEST_TRAVERSAL_CFG = DualTreeTraversalConfig(
    max_pair_queue=2048,
    process_block=32,
    max_interactions_per_node=512,
    max_neighbors_per_leaf=1024,
)


def _sample_problem(n: int = 128):
    key = jax.random.PRNGKey(2026)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (n,), minval=0.5, maxval=1.5, dtype=jnp.float32
    )
    return positions, masses


def _build(tree_type: str, positions, masses, leaf_size: int = 8):
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        build_mode="adaptive",
        leaf_size=leaf_size,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    return tree, geometry


def _assert_interaction_contract(tree, interactions, neighbors) -> None:
    total_nodes = int(tree.num_nodes)

    offsets = jnp.asarray(interactions.offsets)
    counts = jnp.asarray(interactions.counts)
    level_offsets = jnp.asarray(interactions.level_offsets)
    sources = jnp.asarray(interactions.sources)
    targets = jnp.asarray(interactions.targets)

    # The far-interaction list is level-major: `offsets` holds per-node starts
    # *within* each level block and is therefore not a single global prefix
    # sum. The global CSR-style invariants live on `level_offsets` and the
    # total interaction count.
    total_far = int(sources.shape[0])
    assert offsets.shape == (total_nodes + 1,)
    assert counts.shape == (total_nodes,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == total_far == int(targets.shape[0])
    assert jnp.all(counts >= 0)
    assert int(jnp.sum(counts)) == total_far
    assert int(level_offsets[0]) == 0
    assert int(level_offsets[-1]) == total_far
    assert jnp.all(level_offsets[1:] >= level_offsets[:-1])

    if sources.shape[0] > 0:
        # Interaction endpoints must be valid node indices into the geometry.
        assert int(jnp.min(sources)) >= 0
        assert int(jnp.max(sources)) < total_nodes
        assert int(jnp.min(targets)) >= 0
        assert int(jnp.max(targets)) < total_nodes

    n_offsets = jnp.asarray(neighbors.offsets)
    n_counts = jnp.asarray(neighbors.counts)
    n_data = jnp.asarray(neighbors.neighbors)
    leaf_indices = jnp.asarray(neighbors.leaf_indices)

    assert leaf_indices.ndim == 1
    assert n_offsets.shape == (leaf_indices.shape[0] + 1,)
    assert n_counts.shape == (leaf_indices.shape[0],)
    assert int(n_offsets[0]) == 0
    assert int(n_offsets[-1]) == int(n_data.shape[0])
    assert jnp.all(n_offsets[1:] >= n_offsets[:-1])
    assert jnp.array_equal(n_offsets[1:] - n_offsets[:-1], n_counts)


@pytest.mark.parametrize("tree_type", _BACKENDS)
@pytest.mark.parametrize("theta", [0.3, 0.6, 1.0])
def test_interaction_contract_holds_for_all_backends(tree_type: str, theta: float):
    """The CSR interaction/neighbor contract holds identically per backend."""
    positions, masses = _sample_problem(n=128)
    tree, geometry = _build(tree_type, positions, masses)
    interactions, neighbors = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type="dehnen",
        traversal_config=_TEST_TRAVERSAL_CFG,
    )
    _assert_interaction_contract(tree, interactions, neighbors)


@pytest.mark.parametrize("tree_type", _BACKENDS)
def test_geometry_conformance(tree_type: str):
    """Continuous per-node geometry has the right shapes and is finite."""
    positions, masses = _sample_problem(n=128)
    tree, geometry = _build(tree_type, positions, masses)
    total_nodes = int(tree.parent.shape[0])
    assert geometry.center.shape == (total_nodes, 3)
    assert geometry.half_extent.shape == (total_nodes, 3)
    assert geometry.radius.shape == (total_nodes,)
    assert geometry.max_extent.shape == (total_nodes,)
    assert jnp.all(jnp.isfinite(geometry.center))
    assert jnp.all(geometry.radius >= 0.0)


@pytest.mark.parametrize("tree_type", _BACKENDS)
def test_build_is_deterministic(tree_type: str):
    """Rebuilding on identical inputs reproduces the topology bit-for-bit."""
    positions, masses = _sample_problem(n=128)
    tree_a, _ = _build(tree_type, positions, masses)
    tree_b, _ = _build(tree_type, positions, masses)
    assert jnp.array_equal(tree_a.parent, tree_b.parent)
    assert jnp.array_equal(tree_a.particle_indices, tree_b.particle_indices)


def test_all_backends_cover_the_same_particle_set():
    """Independent of topology, every backend indexes exactly the N particles."""
    positions, masses = _sample_problem(n=128)
    reference = None
    for tree_type in _BACKENDS:
        tree, _ = _build(tree_type, positions, masses)
        assert int(tree.num_particles) == 128
        covered = jnp.sort(jnp.asarray(tree.particle_indices))
        if reference is None:
            reference = covered
        assert jnp.array_equal(covered, jnp.arange(128))
