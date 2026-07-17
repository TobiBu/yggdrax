"""Tree-accelerated SVGD vs. the exact O(N^2) reference.

* With no far pairs accepted (tight ``theta``) the near field is exact, so the
  tree Stein update equals the exact update to machine precision.
* With far pairs accepted, the monopole far approximation adds an error that
  shrinks as ``theta`` tightens.
* A short SVGD run matches the exact run in distribution (moments) at a
  moderate ``theta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd import targets as T
from yggdrax.applications.svgd.exact import exact_phi, run_svgd
from yggdrax.applications.svgd.kernel import median_heuristic
from yggdrax.applications.svgd.sampler import (
    build_svgd_topology,
    run_tree_svgd,
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


@pytest.mark.parametrize("backend", ["radix", "octree"])
def test_nearfield_phi_is_exact(backend):
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (300, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)
    topo = build_svgd_topology(
        p, theta=0.0, leaf_size=16, backend=backend, traversal_config=_CFG
    )
    assert int(topo.far_tgt_slot.shape[0]) == 0
    tree = svgd_phi_from_topology(p, sc, h, topo)
    rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
    assert rel < 1e-10, f"{backend}: near-field not exact (rel {rel:.2e})"


def test_far_monopole_error_shrinks_with_theta():
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (1500, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)

    prev = None
    saw_far = False
    for theta in (1.0, 0.6, 0.3):
        topo = build_svgd_topology(
            p, theta=theta, leaf_size=8, backend="radix", traversal_config=_CFG
        )
        saw_far = saw_far or int(topo.far_tgt_slot.shape[0]) > 0
        tree = svgd_phi_from_topology(p, sc, h, topo)
        rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
        if prev is not None:
            assert rel <= prev + 1e-9, "error should not grow as theta tightens"
        prev = rel
    assert saw_far, "expected far pairs at this configuration"
    assert prev < 1e-2


def test_distribution_matches_exact_short_run():
    tgt = T.gaussian(jnp.array([1.0, 0.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
    p0 = jax.random.normal(jax.random.PRNGKey(0), (200, 3)) * 0.6
    h = float(median_heuristic(p0))
    pe = run_svgd(p0, tgt.score, h, 0.3, 40)
    pt = run_tree_svgd(
        p0, tgt.score, h, 0.3, 40, theta=0.4, leaf_size=16, traversal_config=_CFG
    )
    # Moments agree within a loose tolerance (tree is an approximation).
    assert jnp.max(jnp.abs(pe.mean(0) - pt.mean(0))) < 0.1
    assert jnp.max(jnp.abs(pe.std(0) - pt.std(0))) < 0.15
