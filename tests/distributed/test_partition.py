"""SFC domain-decomposition tests for ``yggdrax.distributed.partition``.

Run on forced host CPU devices (buffer-fallback exchange) or a multi-GPU node:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_partition.py -q
"""

import numpy as np
import pytest

import jax.numpy as jnp

from yggdrax.distributed import device_count, make_mesh, sfc_decompose
from yggdrax.morton import morton_encode

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="SFC decomposition needs >= 2 devices"
)


def _mesh(n=None):
    n = min(4, device_count()) if n is None else n
    return make_mesh(n)


def _random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)


def _valid_rows(domain, ndev, capacity):
    """Concatenate every device's valid rows in device order."""
    pos = np.asarray(domain.positions).reshape(ndev, capacity, 3)
    codes = np.asarray(domain.codes).reshape(ndev, capacity)
    counts = np.asarray(domain.counts)
    kept_pos, kept_codes, owner = [], [], []
    for g in range(ndev):
        c = int(counts[g])
        kept_pos.append(pos[g, :c])
        kept_codes.append(codes[g, :c])
        owner.append(np.full(c, g))
    return (
        np.concatenate(kept_pos),
        np.concatenate(kept_codes),
        np.concatenate(owner),
        counts,
    )


def test_decompose_preserves_all_particles_and_global_order():
    mesh = _mesh()
    ndev = mesh.size
    n = 16 * ndev
    cap = n  # generous padding
    pts = _random_points(n, seed=1)

    domain = sfc_decompose(
        mesh, jnp.asarray(pts), jnp.ones(n), output_capacity=cap, equalize=True
    )
    kept_pos, kept_codes, _, counts = _valid_rows(domain, ndev, cap)

    # No particles lost or duplicated.
    assert int(counts.sum()) == n
    # Reference: global Morton order of all inputs.
    from yggdrax.distributed.partition import global_bounds  # reuse encoder path

    # bounds must match what sfc used; recompute the same global box here.
    lo = pts.min(0)
    hi = pts.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    bounds = (jnp.asarray(lo - span * 1e-6), jnp.asarray(hi + span * 1e-6))
    ref_codes = np.asarray(morton_encode(jnp.asarray(pts), bounds))
    ref_sorted = np.sort(ref_codes)

    # Concatenated valid codes (device order) equal the global sorted codes.
    np.testing.assert_array_equal(np.sort(kept_codes), ref_sorted)
    # And they are globally non-decreasing across the device concatenation.
    assert np.all(np.diff(kept_codes) >= 0)


def test_equalize_balances_counts():
    mesh = _mesh()
    ndev = mesh.size
    n = 12 * ndev  # divisible -> perfectly equal
    cap = 4 * n
    pts = _random_points(n, seed=2)
    domain = sfc_decompose(
        mesh, jnp.asarray(pts), jnp.ones(n), output_capacity=cap, equalize=True
    )
    counts = np.asarray(domain.counts)
    assert int(counts.sum()) == n
    # exact balance: every device within 1 of N/ndev
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(counts, np.full(ndev, n // ndev))


def test_domains_are_contiguous_and_disjoint():
    mesh = _mesh()
    ndev = mesh.size
    n = 20 * ndev
    cap = n
    pts = _random_points(n, seed=3)
    domain = sfc_decompose(
        mesh, jnp.asarray(pts), jnp.ones(n), output_capacity=cap, equalize=True
    )
    codes = np.asarray(domain.codes).reshape(ndev, cap)
    counts = np.asarray(domain.counts)
    # each device's max valid code <= next device's min valid code
    for g in range(ndev - 1):
        cg, cn = int(counts[g]), int(counts[g + 1])
        if cg == 0 or cn == 0:
            continue
        assert codes[g, :cg].max() <= codes[g + 1, :cn].min()


def test_align_level_keeps_coarse_cells_on_one_device():
    mesh = _mesh()
    ndev = mesh.size
    n = 32 * ndev
    cap = 4 * n
    align_level = 2  # top 6 Morton bits define the cell
    pts = _random_points(n, seed=4)
    domain = sfc_decompose(
        mesh,
        jnp.asarray(pts),
        jnp.ones(n),
        output_capacity=cap,
        align_level=align_level,
        equalize=False,  # alignment and exact equalize are mutually exclusive
    )
    _, kept_codes, owner, counts = _valid_rows(domain, ndev, cap)
    assert int(counts.sum()) == n
    # No level-2 cell (top 6 bits) is owned by more than one device.
    shift = 63 - 3 * align_level
    cell = kept_codes.astype(np.uint64) >> np.uint64(shift)
    for c in np.unique(cell):
        owners = np.unique(owner[cell == c])
        assert owners.size == 1, f"cell {c} split across devices {owners}"
