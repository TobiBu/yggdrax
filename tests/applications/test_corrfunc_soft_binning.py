"""Soft radial binning: convergence to hard bins and differentiability.

Validates the soft-window building block of the differentiable
correlation-function estimator (paper section 5), independent of the tree
traversal:

* the soft-binned pair-count histogram converges to the hard-binned histogram
  as the window sharpness increases (monotonically, per bin);
* the windows form an approximate partition of unity across the binned range;
* the soft counts are differentiable w.r.t. both separations and bin edges.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.baselines import (
    brute_force_pair_counts,
    brute_force_soft_pair_counts,
)
from yggdrax.applications.corrfunc.binning import (
    hard_bin_weights,
    make_log_edges,
    soft_bin_weights,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _separations(n: int = 20000):
    # Off-edge separations spanning the binned range, in log space.
    key = jax.random.PRNGKey(0)
    return jnp.exp(
        jax.random.uniform(key, (n,), minval=jnp.log(0.11), maxval=jnp.log(9.5))
    )


def test_soft_histogram_converges_to_hard():
    edges = make_log_edges(0.1, 10.0, 8)
    r = _separations()
    hard_hist = hard_bin_weights(r, edges).sum(0)

    prev = None
    errs = []
    for sharpness in (20.0, 100.0, 500.0):
        soft_hist = soft_bin_weights(r, edges, sharpness=sharpness).sum(0)
        rel = float(jnp.max(jnp.abs(soft_hist - hard_hist) / jnp.maximum(hard_hist, 1)))
        errs.append(rel)
        if prev is not None:
            assert rel < prev, "per-bin error should shrink as sharpness grows"
        prev = rel
    # At high sharpness the histograms agree closely.
    assert errs[-1] < 1e-2


def test_windows_are_approximate_partition_of_unity():
    edges = make_log_edges(0.1, 10.0, 8)
    r = _separations()
    soft = soft_bin_weights(r, edges, sharpness=300.0)
    rowsum = soft.sum(-1)
    # Within the binned range each separation's memberships sum to ~1.
    assert float(jnp.max(jnp.abs(rowsum - 1.0))) < 1e-2


def test_soft_counts_are_differentiable():
    edges = make_log_edges(0.1, 10.0, 8)
    r = _separations(n=2000)

    def loss_positions(rr):
        return jnp.sum(soft_bin_weights(rr, edges, 50.0))

    grad_r = jax.grad(loss_positions)(r)
    assert grad_r.shape == r.shape
    assert jnp.all(jnp.isfinite(grad_r))

    def loss_edges(ed):
        return jnp.sum(soft_bin_weights(r, ed, 50.0)[:, 3])

    grad_e = jax.grad(loss_edges)(edges)
    assert jnp.all(jnp.isfinite(grad_e))
    assert float(jnp.abs(grad_e).sum()) > 0.0


def test_brute_force_soft_converges_to_hard():
    key = jax.random.PRNGKey(1)
    pos = jax.random.uniform(key, (500, 3), dtype=jnp.float64)
    edges = make_log_edges(0.02, 1.0, 8)
    hard = brute_force_pair_counts(pos, edges)

    prev = None
    for sharpness in (50.0, 200.0, 800.0):
        soft = brute_force_soft_pair_counts(pos, edges, sharpness)
        rel = float(jnp.max(jnp.abs(soft - hard) / jnp.maximum(hard, 1)))
        if prev is not None:
            assert rel <= prev + 1e-9
        prev = rel
    assert prev < 5e-3
