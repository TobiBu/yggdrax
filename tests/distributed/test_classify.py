"""Remote classification tests (Phase 3 LET stage 2).

Each GPU cross-walks its local tree against a coarse tree built from *other*
domains only. Invariants:
- the remote coarse tree excludes the own domain: its mass == global total minus
  own-domain mass (so combining with the local self-walk never double-counts);
- the walk produces non-trivial far (M2L) and near (import) work without
  capacity overflow.

The far/near admissibility partition itself is already proven by
test_cross_walk (the cross walk is correct for any two trees).

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_classify.py -q
"""

import numpy as np
import pytest

import jax.numpy as jnp

from yggdrax.distributed import classify_against_remote, device_count, make_mesh

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="remote classification needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 24
_N = _PER_DEV * _NDEV
_CAP = 4 * _PER_DEV


@pytest.fixture(scope="module")
def metrics():
    mesh = make_mesh(_NDEV)
    rng = np.random.default_rng(13)
    pts = rng.uniform(-1.0, 1.0, size=(_N, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(_N,)).astype(np.float32)
    m = classify_against_remote(
        mesh,
        jnp.asarray(pts),
        jnp.asarray(mass),
        leaf_size=_LEAF,
        output_capacity=_CAP,
        theta=0.5,
        equalize=True,
    )
    return pts, mass, m


def test_remote_tree_excludes_own_domain(metrics):
    _, _, m = metrics
    remote = np.asarray(m.remote_root_mass)   # [ndev]
    own = np.asarray(m.own_domain_mass)       # [ndev]
    total = np.asarray(m.total_mass)          # [ndev], replicated
    # remote mass == global total - own domain mass
    np.testing.assert_allclose(remote, total - own, rtol=1e-3, atol=1e-3)


def test_total_mass_is_global(metrics):
    _, mass, m = metrics
    np.testing.assert_allclose(np.asarray(m.total_mass), mass.sum(), rtol=1e-4)


def test_classification_is_nontrivial_and_bounded(metrics):
    _, _, m = metrics
    assert not bool(np.asarray(m.overflow).any())
    # every GPU has some remote far and some remote near work
    assert (np.asarray(m.far_count) > 0).all()
    assert (np.asarray(m.near_count) > 0).all()
