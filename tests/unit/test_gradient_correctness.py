"""Finite-difference vs. ``jax.grad`` for differentiable tree geometry outputs.

Phase 0 deliverable for the differentiable-applications paper. This validates
the central claim of the differentiability model (docs/differentiability_model.md,
paper section 2): *given a fixed tree topology and a fixed far/near
partition, the continuous per-node geometry -- and any smooth observable built
on top of it -- is correctly differentiable with respect to particle
positions.*

Method
------
1. Build the tree once at the base positions to freeze (a) the particle
   ordering / topology and (b) the ``theta``-dependent MAC far-pair set.
   Both are piecewise-constant in the positions and non-differentiable at
   their boundaries -- we deliberately stay away from those boundaries.
2. Define a smooth scalar observable over the frozen far node-pairs using the
   continuous node centres from ``compute_tree_geometry``.
3. Compare ``jax.grad`` of that observable against central finite differences.

The sweep over ``theta`` (MAC tightness) changes the frozen far-pair set;
gradient correctness must hold for every setting away from accept/reject
boundaries. Run in float64 so the finite-difference reference is meaningful.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import (
    DualTreeTraversalConfig,
    Tree,
    build_interactions_and_neighbors,
    compute_tree_geometry,
)

_BACKENDS = ("radix", "kdtree", "octree")
_THETAS = (0.6, 1.0)

_TRAVERSAL_CFG = DualTreeTraversalConfig(
    max_pair_queue=8192,
    process_block=64,
    max_interactions_per_node=2048,
    max_neighbors_per_leaf=2048,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Scope float64 to this module; restore the previous setting afterwards."""
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _make_positions(n: int = 1024):
    key = jax.random.PRNGKey(7)
    pos = jax.random.uniform(key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    mass = jnp.ones(n, dtype=jnp.float64)
    return pos, mass


def _freeze_topology(tree_type: str, positions, masses, theta: float):
    """Build once; return the frozen ordering and far node-pair endpoints."""
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        build_mode="adaptive",
        leaf_size=4,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    interactions, _ = build_interactions_and_neighbors(
        tree,
        geometry,
        theta=theta,
        mac_type="dehnen",
        traversal_config=_TRAVERSAL_CFG,
    )
    order = jnp.asarray(tree.particle_indices)  # sorted-slot -> original id
    src = jnp.asarray(interactions.sources)
    tgt = jnp.asarray(interactions.targets)
    return tree, order, src, tgt


def _make_observable(tree, order, src, tgt, n):
    """Smooth scalar over frozen far node-pairs, differentiable in positions."""

    def observable(pos_flat):
        pos = pos_flat.reshape(n, 3)
        # Reorder with the frozen permutation (differentiable gather), then
        # recompute continuous node geometry for the (fixed) topology.
        geometry = compute_tree_geometry(tree, pos[order])
        delta = geometry.center[src] - geometry.center[tgt]
        dist_sq = jnp.sum(delta * delta, axis=-1)
        return jnp.sum(jnp.exp(-dist_sq))

    return observable


@pytest.mark.parametrize("tree_type", _BACKENDS)
@pytest.mark.parametrize("theta", _THETAS)
def test_geometry_gradient_matches_finite_difference(tree_type: str, theta: float):
    n = 1024
    positions, masses = _make_positions(n)
    tree, order, src, tgt = _freeze_topology(tree_type, positions, masses, theta)

    assert src.shape[0] > 0, "expected far interactions at this N/leaf_size/theta"

    observable = _make_observable(tree, order, src, tgt, n)
    x = positions.reshape(-1)

    analytic = np.asarray(jax.grad(observable)(x))

    # Central finite differences on a random subset of coordinates.
    rng = np.random.default_rng(0)
    idxs = rng.choice(x.shape[0], size=16, replace=False)
    h = 1e-6
    fd = np.empty(idxs.shape[0])
    for m, i in enumerate(idxs):
        fp = float(observable(x.at[i].add(h)))
        fm = float(observable(x.at[i].add(-h)))
        fd[m] = (fp - fm) / (2.0 * h)

    analytic_sel = analytic[idxs]
    denom = np.maximum(np.abs(fd), 1e-6)
    max_rel_err = float(np.max(np.abs(analytic_sel - fd) / denom))

    # float64 central differences on a smooth observable: expect <~1e-4.
    assert max_rel_err < 1e-4, (
        f"{tree_type} theta={theta}: max relative gradient error "
        f"{max_rel_err:.2e} exceeds tolerance"
    )


@pytest.mark.parametrize("tree_type", _BACKENDS)
def test_gradient_through_rebuilt_tree_is_finite(tree_type: str):
    """Autodiff composes through the builder itself (topology recomputed).

    Mirrors the SVGD-style pattern where the tree is rebuilt inside the traced
    function every step; the gradient of a smooth observable must be finite and
    correctly shaped even though the topology is only piecewise-constant.
    """
    n = 256
    positions, masses = _make_positions(n)

    def loss(pos):
        tree = Tree.from_particles(
            pos,
            masses,
            tree_type=tree_type,
            build_mode="adaptive",
            leaf_size=8,
            return_reordered=True,
        )
        geometry = compute_tree_geometry(tree, tree.positions_sorted)
        # Mean squared node radius -- smooth in positions for fixed topology.
        return jnp.mean(geometry.radius**2)

    grad = jax.grad(loss)(positions)
    assert grad.shape == positions.shape
    assert jnp.all(jnp.isfinite(grad))
