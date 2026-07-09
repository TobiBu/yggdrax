"""Enriched shared top tree (global coarse tree) tests (Phase 3 LET stage 1).

Invariants:
- the per-domain frontier is a mass-conserving partition (frontier mass ==
  domain root mass);
- the global coarse tree is identical on every device (built from identical
  gathered input) and its total mass == the global mass;
- coarse leaves carry valid origin tags.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_let.py -q
"""

import numpy as np
import pytest

import jax.numpy as jnp

from yggdrax.distributed import build_distributed_coarse_tree, device_count, make_mesh

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="coarse tree gather needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 24
_N = _PER_DEV * _NDEV
_CAP = 4 * _PER_DEV


@pytest.fixture(scope="module")
def metrics():
    mesh = make_mesh(_NDEV)
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1.0, 1.0, size=(_N, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(_N,)).astype(np.float32)
    m = build_distributed_coarse_tree(
        mesh,
        jnp.asarray(pts),
        jnp.asarray(mass),
        leaf_size=_LEAF,
        output_capacity=_CAP,
        equalize=True,
    )
    return pts, mass, m


def test_frontier_partitions_domain(metrics):
    _, _, m = metrics
    # frontier (all leaves) mass == domain root mass (exact partition)
    np.testing.assert_allclose(
        np.asarray(m.frontier_mass_sum), np.asarray(m.domain_mass), rtol=1e-4
    )


def test_coarse_tree_total_mass_is_global(metrics):
    _, mass, m = metrics
    coarse_root_mass = np.asarray(m.coarse_root_mass)  # [ndev]
    np.testing.assert_allclose(coarse_root_mass, mass.sum(), rtol=1e-4)


def test_coarse_tree_is_replicated_across_devices(metrics):
    _, _, m = metrics
    root_mass = np.asarray(m.coarse_root_mass)
    root_com = np.asarray(m.coarse_root_com)  # [ndev, 3]
    # identical on every device
    np.testing.assert_allclose(root_mass, root_mass[0], rtol=1e-5)
    for g in range(_NDEV):
        np.testing.assert_allclose(root_com[g], root_com[0], rtol=1e-5)


def test_coarse_leaves_have_valid_tags(metrics):
    _, _, m = metrics
    n_valid = np.asarray(m.n_coarse_valid)  # [ndev], replicated
    # every device sees the same global count of real (non-padding) coarse nodes
    assert (n_valid == n_valid[0]).all()
    assert int(n_valid[0]) > 0
