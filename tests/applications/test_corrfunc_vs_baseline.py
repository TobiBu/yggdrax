"""Validate the tree-accelerated estimator against the brute-force reference.

Two regimes:

* No far pairs accepted (tight ``theta``): the near field is exact, so the
  estimator must equal the brute-force soft counts to machine precision.
* Far pairs accepted (looser ``theta``): the monopole far approximation adds a
  bounded error that shrinks as ``theta`` tightens.

Also checks the hard-bin limit (soft counts at high sharpness match hard
brute-force counts). All three tree backends (radix, octree, and the leaf-only
KD-tree) give exact near-field coverage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.corrfunc.baselines import (
    brute_force_pair_counts,
    brute_force_soft_pair_counts,
)
from yggdrax.applications.corrfunc.binning import make_log_edges
from yggdrax.applications.corrfunc.estimator import (
    build_pair_topology,
    soft_pair_counts,
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


def _catalog(n: int = 400, seed: int = 2):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, 3), dtype=jnp.float64)


@pytest.mark.parametrize("backend", ["radix", "octree", "kdtree"])
def test_nearfield_exact_when_no_far_pairs(backend):
    pos = _catalog()
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 200.0
    exact = brute_force_soft_pair_counts(pos, edges, sharpness)

    topo = build_pair_topology(
        pos, theta=0.0, leaf_size=16, backend=backend, traversal_config=_CFG
    )
    assert int(topo.far_src_start.shape[0]) == 0, "expected no far pairs at theta=0"
    est = soft_pair_counts_from_topology(pos, topo, edges, sharpness)
    rel = float(jnp.max(jnp.abs(est - exact) / jnp.maximum(exact, 1.0)))
    assert rel < 1e-10, f"{backend}: near-field not exact (rel err {rel:.2e})"


@pytest.mark.parametrize("backend", ["radix", "octree", "kdtree"])
def test_farfield_error_shrinks_with_theta(backend):
    pos = _catalog(n=600)
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 200.0
    exact = brute_force_soft_pair_counts(pos, edges, sharpness)

    prev = None
    for theta in (0.9, 0.6, 0.3):
        topo = build_pair_topology(
            pos, theta=theta, leaf_size=8, backend=backend, traversal_config=_CFG
        )
        est = soft_pair_counts_from_topology(pos, topo, edges, sharpness)
        rel = float(jnp.abs(est.sum() - exact.sum()) / exact.sum())
        if prev is not None:
            assert rel <= prev + 1e-9, "error should not grow as theta tightens"
        prev = rel
    assert prev < 1e-2


def test_hard_limit_matches_hard_brute_force():
    pos = _catalog()
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 1500.0
    hard = brute_force_pair_counts(pos, edges)
    soft_bf = brute_force_soft_pair_counts(pos, edges, sharpness)
    # theta=0 -> near-field exact, so the estimator equals the soft brute-force
    # to machine precision (isolates the estimator from the soft-binning residual).
    est = soft_pair_counts(
        pos, edges, theta=0.0, sharpness=sharpness, leaf_size=16, traversal_config=_CFG
    )
    rel_est = float(jnp.max(jnp.abs(est - soft_bf) / jnp.maximum(soft_bf, 1.0)))
    assert rel_est < 1e-9, f"estimator != soft brute-force (rel err {rel_est:.2e})"
    # The soft window is a partition of unity, so the TOTAL soft count matches
    # the total hard count closely (per-bin near-edge redistribution aside,
    # which the binning test covers separately).
    rel_total = float(jnp.abs(est.sum() - hard.sum()) / hard.sum())
    assert rel_total < 1e-3, f"total soft vs hard mismatch (rel {rel_total:.2e})"


def test_kdtree_leaf_only_covers_all_pairs():
    """Leaf-only KD-tree tiles every pair: near-field is exact vs brute force.

    (The heap KD-tree used to be rejected here because its internal-node pivots
    were absent from the near-list; the leaf-only bucket build fixes that.)
    """
    pos = _catalog()
    edges = make_log_edges(0.02, 0.8, 8)
    sharpness = 200.0
    exact = brute_force_soft_pair_counts(pos, edges, sharpness)
    topo = build_pair_topology(
        pos, theta=0.0, leaf_size=16, backend="kdtree", traversal_config=_CFG
    )
    assert int(topo.far_src_start.shape[0]) == 0, "expected no far pairs at theta=0"
    est = soft_pair_counts_from_topology(pos, topo, edges, sharpness)
    rel = float(jnp.max(jnp.abs(est - exact) / jnp.maximum(exact, 1.0)))
    assert rel < 1e-10, f"kdtree near-field not exact (rel err {rel:.2e})"
