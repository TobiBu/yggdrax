"""Halo particle-import tests (Phase 3 LET stage 3).

The two-round request/fulfill import must fetch exactly the remote near-field
leaves each GPU classified, no more, no less. Invariants:
- **mass conservation / no over-import**: the imported halo mass equals the
  total mass of the requested remote leaves (fetching wrong, extra, or missing
  particles would break this);
- **provenance**: every imported particle is sourced from another domain
  (global id domain != self) -- never self-imported;
- no request-buffer overflow.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_import.py -q
"""

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed import device_count, distributed_let_import, make_mesh

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="halo import needs >= 2 devices"
)

_NDEV = min(4, device_count())
_LEAF = 8
_PER_DEV = 24
_N = _PER_DEV * _NDEV
_CAP = 4 * _PER_DEV


@pytest.fixture(scope="module")
def metrics():
    mesh = make_mesh(_NDEV)
    rng = np.random.default_rng(21)
    pts = rng.uniform(-1.0, 1.0, size=(_N, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(_N,)).astype(np.float32)
    m = distributed_let_import(
        mesh,
        jnp.asarray(pts),
        jnp.asarray(mass),
        leaf_size=_LEAF,
        output_capacity=_CAP,
        theta=0.5,
        equalize=True,
    )
    return pts, mass, m


def test_no_request_overflow(metrics):
    _, _, m = metrics
    assert not bool(np.asarray(m.request_overflow).any())


def test_import_mass_conserved(metrics):
    _, _, m = metrics
    # imported halo mass == mass of the requested remote leaves (exact fetch)
    np.testing.assert_allclose(
        np.asarray(m.imported_mass), np.asarray(m.needed_mass), rtol=1e-4, atol=1e-4
    )


def test_import_is_all_remote(metrics):
    _, _, m = metrics
    # no halo particle was sourced from the importing GPU itself
    assert not bool(np.asarray(m.wrong_domain).any())
    # and the import is non-trivial
    assert (np.asarray(m.n_halo_valid) > 0).all()


def test_near_to_halo_mapping_built(metrics):
    _, _, m = metrics
    # every device maps some near coarse leaves to halo blocks (the link the
    # combined near-field P2P uses); non-trivial and within request capacity
    n_mapped = np.asarray(m.n_mapped)
    assert (n_mapped > 0).all()
