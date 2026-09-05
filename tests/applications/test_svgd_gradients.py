"""Finite-difference vs. jax.grad for the tree SVGD update and bandwidth loop.

Checks that the tree-accelerated Stein update is correctly differentiable w.r.t.
the kernel bandwidth and the particle positions (fixed partition), and that a
short bandwidth-learning replay produces a finite, correctly-shaped gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd import targets as T
from yggdrax.applications.svgd.bandwidth_learning import learn_bandwidth
from yggdrax.applications.svgd.kernel import median_heuristic
from yggdrax.applications.svgd.sampler import (
    build_svgd_topology,
    svgd_phi_from_topology,
)

_CFG = DualTreeTraversalConfig(
    max_pair_queue=1 << 18,
    process_block=32,
    max_interactions_per_node=1 << 14,
    max_neighbors_per_leaf=1 << 14,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _far_active_setup(n=1000):
    p = jax.random.normal(jax.random.PRNGKey(0), (n, 3)) * 1.2
    sc = p * 0.5
    topo = build_svgd_topology(
        p, theta=0.6, leaf_size=8, backend="radix", traversal_config=_CFG
    )
    assert int(topo.far_tgt_slot.shape[0]) > 0
    return p, sc, topo


def test_gradient_wrt_bandwidth_matches_fd():
    p, sc, topo = _far_active_setup()
    h0 = 0.6

    def loss(h):
        return jnp.sum(svgd_phi_from_topology(p, sc, h, topo) ** 2)

    analytic = float(jax.grad(loss)(h0))
    hh = 1e-6
    fd = float((loss(h0 + hh) - loss(h0 - hh)) / (2 * hh))
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_gradient_wrt_positions_matches_fd():
    p, sc, topo = _far_active_setup(n=600)
    h0 = 0.6

    def loss(flat):
        return jnp.sum(svgd_phi_from_topology(flat.reshape(p.shape), sc, h0, topo))

    x = p.reshape(-1)
    analytic = np.asarray(jax.grad(loss)(x))
    rng = np.random.default_rng(0)
    idxs = rng.choice(x.shape[0], size=12, replace=False)
    h = 1e-6
    fd = np.array(
        [float((loss(x.at[i].add(h)) - loss(x.at[i].add(-h))) / (2 * h)) for i in idxs]
    )
    rel = float(np.max(np.abs(analytic[idxs] - fd) / np.maximum(np.abs(fd), 1e-6)))
    assert rel < 1e-4


def test_bandwidth_learning_runs_and_reduces_loss():
    # Tiny run: just assert the optimizer produces a finite trajectory and does
    # not increase the loss over a couple of outer steps.
    tgt = T.gaussian(jnp.array([0.0, 0.0]), jnp.array([1.0, 4.0]))
    p0 = jax.random.normal(jax.random.PRNGKey(0), (120, 2)) * 0.5
    samples = tgt.sample(jax.random.PRNGKey(3), 200)
    h0 = float(median_heuristic(p0))
    res = learn_bandwidth(
        p0,
        tgt.score,
        samples,
        h_init=h0,
        mmd_bandwidth=1.5,
        step_size=0.3,
        num_svgd_steps=8,
        theta=0.5,
        leaf_size=16,
        learning_rate=0.15,
        num_outer_steps=6,
        traversal_config=_CFG,
    )
    assert np.all(np.isfinite(res.h_history))
    assert res.h_final > 0.0
    assert res.loss_history[-1] <= res.loss_history[0] + 1e-9
