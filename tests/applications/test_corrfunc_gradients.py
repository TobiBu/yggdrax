"""Finite-difference vs. ``jax.grad`` for the soft pair-count estimator.

Differentiates the soft counts w.r.t. particle positions and w.r.t. bin edges
at a fixed pair topology (the intended-use differentiability model, section 2),
and checks agreement with central finite differences in double precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.corrfunc.binning import make_log_edges
from yggdrax.applications.corrfunc.estimator import (
    build_pair_topology,
    soft_pair_counts_from_topology,
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


def _catalog(n: int = 400, seed: int = 5):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, 3), dtype=jnp.float64)


def _central_diff(fn, x, idxs, h=1e-6):
    fd = np.empty(idxs.shape[0])
    for m, i in enumerate(idxs):
        fp = float(fn(x.at[i].add(h)))
        fm = float(fn(x.at[i].add(-h)))
        fd[m] = (fp - fm) / (2.0 * h)
    return fd


def test_gradient_wrt_positions_matches_finite_difference():
    pos = _catalog(n=500)
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 120.0
    # A loose-ish theta so both near and far contributions are exercised.
    topo = build_pair_topology(
        pos, theta=0.7, leaf_size=8, backend="radix", traversal_config=_CFG
    )

    def scalar(flat):
        counts = soft_pair_counts_from_topology(
            flat.reshape(pos.shape), topo, edges, sharpness
        )
        # Weighted sum over bins -> a smooth scalar sensitive to all bins.
        weights = jnp.arange(1, counts.shape[0] + 1, dtype=counts.dtype)
        return jnp.sum(counts * weights)

    x = pos.reshape(-1)
    analytic = np.asarray(jax.grad(scalar)(x))
    rng = np.random.default_rng(0)
    idxs = rng.choice(x.shape[0], size=16, replace=False)
    fd = _central_diff(scalar, x, idxs)
    rel = float(np.max(np.abs(analytic[idxs] - fd) / np.maximum(np.abs(fd), 1e-6)))
    assert rel < 1e-4, f"position gradient mismatch (rel err {rel:.2e})"


def test_gradient_wrt_edges_matches_finite_difference():
    pos = _catalog(n=400)
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 120.0
    topo = build_pair_topology(
        pos, theta=0.0, leaf_size=16, backend="radix", traversal_config=_CFG
    )

    def scalar(ed):
        counts = soft_pair_counts_from_topology(pos, topo, ed, sharpness)
        return jnp.sum(counts * jnp.arange(1, counts.shape[0] + 1, dtype=counts.dtype))

    analytic = np.asarray(jax.grad(scalar)(edges))
    idxs = np.arange(1, edges.shape[0] - 1)  # interior edges
    fd = _central_diff(scalar, edges, idxs, h=1e-6)
    rel = float(np.max(np.abs(analytic[idxs] - fd) / np.maximum(np.abs(fd), 1e-6)))
    assert rel < 1e-4, f"edge gradient mismatch (rel err {rel:.2e})"
