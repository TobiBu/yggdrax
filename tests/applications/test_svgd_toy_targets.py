"""Convergence sanity checks for SVGD on toy targets.

Exact SVGD should recover the low-order structure of standard targets: a
Gaussian's mean, a bimodal mixture's two modes, and the banana's curvature.
The tree sampler is checked to track the exact sampler on the Gaussian.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd import targets as T
from yggdrax.applications.svgd.exact import run_svgd
from yggdrax.applications.svgd.kernel import median_heuristic
from yggdrax.applications.svgd.sampler import run_tree_svgd

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


def test_gaussian_mean_recovered():
    tgt = T.gaussian(jnp.array([2.0, -1.0]), jnp.array([1.0, 1.0]))
    p0 = jax.random.normal(jax.random.PRNGKey(0), (200, 2)) * 0.5
    h = float(median_heuristic(p0))
    pf = run_svgd(p0, tgt.score, h, 0.3, 400)
    assert jnp.max(jnp.abs(pf.mean(0) - jnp.array([2.0, -1.0]))) < 0.3


def test_gmm_covers_both_modes():
    gmm = T.gaussian_mixture(
        jnp.array([[-3.0, 0.0], [3.0, 0.0]]), jnp.array([0.5, 0.5])
    )
    p0 = jax.random.normal(jax.random.PRNGKey(1), (300, 2)) * 0.5
    h = float(median_heuristic(p0))
    pf = run_svgd(p0, gmm.score, h, 0.2, 600)
    frac_left = float((pf[:, 0] < 0).mean())
    assert 0.2 < frac_left < 0.8, "particles should populate both modes"
    assert pf[:, 0].min() < -1.5 and pf[:, 0].max() > 1.5


def test_banana_has_positive_curvature():
    tgt = T.banana(curvature=0.3, scale=2.0)
    p0 = jax.random.normal(jax.random.PRNGKey(2), (300, 2))
    h = float(median_heuristic(p0))
    pf = run_svgd(p0, tgt.score, h, 0.1, 400)
    # The banana bends upward in x1 away from x0=0: split by |x0| and compare.
    near = jnp.abs(pf[:, 0]) < 1.0
    far = jnp.abs(pf[:, 0]) > 2.0
    assert float(pf[far, 1].mean()) > float(pf[near, 1].mean())


def test_tree_tracks_exact_on_gaussian():
    tgt = T.gaussian(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))
    p0 = jax.random.normal(jax.random.PRNGKey(0), (200, 3)) * 0.6
    h = float(median_heuristic(p0))
    pe = run_svgd(p0, tgt.score, h, 0.3, 40)
    pt = run_tree_svgd(
        p0, tgt.score, h, 0.3, 40, theta=0.4, leaf_size=16, traversal_config=_CFG
    )
    assert jnp.max(jnp.abs(pe.std(0) - pt.std(0))) < 0.15
