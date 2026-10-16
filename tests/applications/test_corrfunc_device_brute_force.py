"""The device brute force computes the host reference's sum, as one program.

``brute_force_soft_pair_counts_device`` exists so the estimator can be timed
against a baseline that is jitted and warmed like it is; it is only worth that
if it is the *same* number as ``brute_force_soft_pair_counts``, the reference
the estimator is validated against. These tests pin that, for every way the
blocking can split the rows, and check it differentiates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.baselines import (
    brute_force_soft_pair_counts,
    brute_force_soft_pair_counts_device,
)
from yggdrax.applications.corrfunc.binning import make_linear_edges, make_log_edges


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _catalog(n: int = 500, dim: int = 3, seed: int = 3):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, dim), dtype=jnp.float64)


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0)))


@pytest.mark.parametrize("block_size", [None, 1, 7, 64, 500, 4096])
def test_matches_the_host_reference_for_any_blocking(block_size):
    """Every row split, padded or not, sums the same pairs."""
    pos = _catalog()
    edges = make_log_edges(0.01, 0.5, 10)
    want = brute_force_soft_pair_counts(pos, edges, 150.0)
    got = brute_force_soft_pair_counts_device(pos, edges, 150.0, block_size=block_size)
    assert _rel(got, want) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_bins_and_other_dimensions(dim):
    pos = _catalog(n=300, dim=dim)
    edges = make_linear_edges(0.02, 0.6, 6)
    want = brute_force_soft_pair_counts(pos, edges, 40.0, log=False)
    got = brute_force_soft_pair_counts_device(pos, edges, 40.0, log=False)
    assert _rel(got, want) < 1e-12


def test_jit_gives_the_same_counts():
    pos = _catalog()
    edges = make_log_edges(0.01, 0.5, 10)
    eager = brute_force_soft_pair_counts_device(pos, edges, 150.0, block_size=37)
    jitted = jax.jit(
        lambda p, e: brute_force_soft_pair_counts_device(p, e, 150.0, block_size=37)
    )(pos, edges)
    assert _rel(jitted, eager) < 1e-14


def test_float32_inputs_stay_float32():
    pos = _catalog().astype(jnp.float32)
    edges = make_log_edges(0.01, 0.5, 10)
    got = brute_force_soft_pair_counts_device(pos, edges.astype(jnp.float32), 150.0)
    assert got.dtype == jnp.float32
    want = brute_force_soft_pair_counts(pos.astype(jnp.float64), edges, 150.0)
    assert _rel(got, want) < 1e-5


def test_gradient_is_finite_and_matches_central_differences():
    """Differentiable through the blocks, the padding and the masked diagonal."""
    pos = _catalog(n=120)
    edges = make_log_edges(0.02, 0.6, 6)
    weights = jnp.linspace(1.0, 2.0, 6)

    def loss(p):
        counts = brute_force_soft_pair_counts_device(p, edges, 30.0, block_size=7)
        return jnp.sum(weights * counts)

    grad = jax.grad(loss)(pos)
    assert bool(jnp.all(jnp.isfinite(grad)))
    direction = jax.random.normal(jax.random.PRNGKey(9), pos.shape, dtype=pos.dtype)
    eps = 1e-6
    fd = (loss(pos + eps * direction) - loss(pos - eps * direction)) / (2 * eps)
    ad = jnp.sum(grad * direction)
    assert abs(float(fd - ad)) <= 1e-6 * max(abs(float(ad)), 1.0)
