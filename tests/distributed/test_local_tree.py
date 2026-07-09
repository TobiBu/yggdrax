"""Per-GPU local tree + shared top tree tests (Phase 2).

Correctness anchor: the union of the per-domain coarse moments must reproduce
the *global* total mass and centre of mass that a single-device build over all
particles would give (the root of a single global tree).

The distributed build is expensive to compile, so it is run ONCE per (mesh)
session via a fixture and shared across the assertions.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_local_tree.py -q
"""

import numpy as np
import pytest

import jax.numpy as jnp

from yggdrax.distributed import device_count, distributed_tree_moments, make_mesh

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed tree build needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 16
_N = _PER_DEV * _NDEV
_CAP = 4 * _PER_DEV


@pytest.fixture(scope="module")
def built():
    """Run the distributed decompose+build+gather once; reuse across tests."""
    mesh = make_mesh(_NDEV)
    rng = np.random.default_rng(10)
    pts = rng.uniform(-1.0, 1.0, size=(_N, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(_N,)).astype(np.float32)
    res = distributed_tree_moments(
        mesh,
        jnp.asarray(pts),
        jnp.asarray(mass),
        leaf_size=_LEAF,
        output_capacity=_CAP,
        equalize=True,
    )
    return pts, mass, res


def test_domain_moments_reconstruct_global_mass_and_com(built):
    pts, mass, res = built
    domain_mass = np.asarray(res.domain_mass)      # [ndev]
    domain_com = np.asarray(res.domain_com)        # [ndev, 3]
    counts = np.asarray(res.counts)

    assert int(counts.sum()) == _N
    np.testing.assert_allclose(domain_mass.sum(), mass.sum(), rtol=1e-4)
    global_com = (mass[:, None] * pts).sum(0) / mass.sum()
    recon_com = (domain_mass[:, None] * domain_com).sum(0) / domain_mass.sum()
    np.testing.assert_allclose(recon_com, global_com, rtol=1e-3, atol=1e-4)


def test_shared_top_tree_is_replicated_and_matches_domains(built):
    _, _, res = built
    top_mass = np.asarray(res.top_mass).reshape(_NDEV, _NDEV)
    top_com = np.asarray(res.top_com).reshape(_NDEV, _NDEV, 3)
    domain_mass = np.asarray(res.domain_mass)
    domain_com = np.asarray(res.domain_com)

    # every device gathered the same coarse top tree
    for g in range(_NDEV):
        np.testing.assert_allclose(top_mass[g], top_mass[0], rtol=1e-5)
        np.testing.assert_allclose(top_com[g], top_com[0], rtol=1e-5)
    # and it equals the per-domain root moments
    np.testing.assert_allclose(top_mass[0], domain_mass, rtol=1e-5)
    np.testing.assert_allclose(top_com[0], domain_com, rtol=1e-5)


def test_local_root_node_equals_domain_moment(built):
    _, _, res = built
    total_nodes = np.asarray(res.node_mass).shape[0] // _NDEV
    node_mass = np.asarray(res.node_mass).reshape(_NDEV, total_nodes)
    domain_mass = np.asarray(res.domain_mass)
    # root node (index 0) mass == the domain aggregate mass
    np.testing.assert_allclose(node_mass[:, 0], domain_mass, rtol=1e-5)
