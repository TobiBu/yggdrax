"""The neighbour-only summary exchange (C3) against the all-gather one (C1).

``DistributedStatsConfig(summary_exchange="neighbors")`` ships each summary cell
only to the devices it can touch, instead of gathering every device's cells on
every device. What is asserted, on 1, 2, 4 and 8 devices where they exist:

* both exchanges give the SAME halo (per-device halo and send counts are equal)
  and therefore bit-identical ξ counts, kNN distances and gradients; on points
  hugging faces and corners, clustered points, and a "line" of points whose
  Morton domains are slabs, so that a device's only neighbours are the two
  adjacent slabs, and the first and last slab are neighbours through the
  periodic face;
* on the line, every device sends its cells to exactly two devices with the
  periodic box, and the end slabs to one without it; it receives fewer cell rows
  than the all-gather path holds;
* too small a neighbour count or exchange capacity raises
  ``summary_exchange_overflow`` (and ``overflow``).

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_distributed_stats_exchange.py -q --no-cov
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.stats import (
    DistributedStatsConfig,
    distributed_knn,
    distributed_pair_counts,
)

_NDEVS = [d for d in (1, 2, 4, 8) if d <= device_count()]
_MULTI = [d for d in _NDEVS if d >= 4]
_L = 1.0
_EDGES = np.geomspace(0.01, 0.06, 7)  # r_hi = 0.082 < a slab of 1/8
_SHARP = 60.0
_K = 5
_RMAX = 0.08


def _mesh(ndev):
    if ndev > device_count():
        pytest.skip(f"needs {ndev} devices, have {device_count()}")
    return make_mesh(ndev)


def _faces_and_corners(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, _L, (n, 3))
    m = n // 5
    axis = rng.integers(0, 3, m)
    side = rng.integers(0, 2, m)
    offs = rng.uniform(0.0, 1e-3, m)
    pts[np.arange(m), axis] = np.where(side == 1, _L - offs, offs)
    return pts


def _clustered(n, seed=1):
    rng = np.random.default_rng(seed)
    centres = np.concatenate([np.zeros((1, 3)), rng.uniform(0, _L, (4, 3))])
    pts = centres[rng.integers(0, len(centres), n)] + rng.normal(0, 0.03, (n, 3))
    pts[: n // 5] = rng.uniform(0, _L, (n // 5, 3))
    return np.mod(pts, _L)


def _line(n, seed=2):
    """Points along x in a thin tube: the Morton domains are x-slabs."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(0.0, _L, n)
    pts[:, 1:] = 0.5 + rng.uniform(0.0, 0.015, (n, 2))
    return pts


_PTS = {
    "faces_corners": lambda: _faces_and_corners(1200),
    "clustered": lambda: _clustered(1200),
    "line": lambda: _line(1200),
}


def _cfg(n, ndev, mode, **kw):
    return DistributedStatsConfig.for_problem(
        max(1, n // ndev), halo_fraction=8.0, summary_exchange=mode, **kw
    )


def _np_stats(res):
    return {k: np.asarray(v) for k, v in res.stats.items()}


def _xi(pts, ndev, mode, *, box=_L, **kw):
    bounds = None if box is not None else (np.zeros(3), np.full(3, _L + 1e-6))
    return distributed_pair_counts(
        pts,
        _EDGES,
        box_size=box,
        bounds=bounds,
        config=_cfg(len(pts), ndev, mode, **kw),
        sharpness=_SHARP,
        mesh=_mesh(ndev),
    )


def _knn(pts, ndev, mode, **kw):
    return distributed_knn(
        pts,
        k=_K,
        r_max=_RMAX,
        box_size=_L,
        config=_cfg(len(pts), ndev, mode, **kw),
        mesh=_mesh(ndev),
    )


def _assert_same_halo(new, old):
    sn, so = _np_stats(new), _np_stats(old)
    for key in ("particles", "halo", "sent", "near_pairs"):
        np.testing.assert_array_equal(sn[key], so[key], err_msg=key)


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_pair_counts_are_identical(name, ndev):
    pts = _PTS[name]()
    new = _xi(pts, ndev, "neighbors")
    old = _xi(pts, ndev, "all_gather")
    assert not bool(new.overflow), {k: bool(v) for k, v in new.flags.items()}
    assert not bool(old.overflow)
    _assert_same_halo(new, old)
    np.testing.assert_array_equal(np.asarray(new.counts), np.asarray(old.counts))
    if ndev > 1:
        sn, so = _np_stats(new), _np_stats(old)
        # never more cells than the all-gather path receives
        assert np.all(sn["summary_recv"] <= so["summary_recv"])


@pytest.mark.parametrize("name", sorted(_PTS))
@pytest.mark.parametrize("ndev", _NDEVS)
def test_knn_is_identical(name, ndev):
    pts = _PTS[name]()
    new = _knn(pts, ndev, "neighbors")
    old = _knn(pts, ndev, "all_gather")
    assert not bool(new.overflow), {k: bool(v) for k, v in new.flags.items()}
    _assert_same_halo(new, old)
    np.testing.assert_array_equal(np.asarray(new.distances), np.asarray(old.distances))


@pytest.mark.parametrize("ndev", _MULTI)
def test_line_slabs_only_talk_to_adjacent_slabs(ndev):
    pts = _line(1200)
    res = _xi(pts, ndev, "neighbors")
    old = _xi(pts, ndev, "all_gather")
    assert not bool(res.overflow)
    st, so = _np_stats(res), _np_stats(old)
    # periodic: slab d touches d-1 and d+1, slab 0 and slab ndev-1 across x = 0
    np.testing.assert_array_equal(st["summary_dests"], np.full(ndev, 2))
    assert np.all(st["summary_recv"] < so["summary_recv"])
    np.testing.assert_array_equal(np.asarray(res.counts), np.asarray(old.counts))

    # non-periodic: the end slabs have one neighbour
    flat = _xi(pts, ndev, "neighbors", box=None)
    flat_old = _xi(pts, ndev, "all_gather", box=None)
    sf = _np_stats(flat)
    want = np.full(ndev, 2)
    want[0] = want[-1] = 1
    np.testing.assert_array_equal(sf["summary_dests"], want)
    _assert_same_halo(flat, flat_old)
    np.testing.assert_array_equal(np.asarray(flat.counts), np.asarray(flat_old.counts))


@pytest.mark.parametrize("ndev", _MULTI)
def test_line_wraps_through_the_periodic_face(ndev):
    """Slab 0 imports slab ndev-1's particles only through x = 0."""
    pts = _line(1200)
    per = _xi(pts, ndev, "neighbors")
    flat = _xi(pts, ndev, "neighbors", box=None)
    h_per, h_flat = _np_stats(per)["halo"], _np_stats(flat)["halo"]
    assert h_per[0] > h_flat[0] and h_per[-1] > h_flat[-1]
    np.testing.assert_array_equal(h_per[1:-1], h_flat[1:-1])


@pytest.mark.parametrize("ndev", _MULTI)
def test_gradients_are_identical(ndev):
    pts = jnp.asarray(_PTS["clustered"]())
    w = jnp.asarray(np.random.default_rng(7).uniform(0.5, 2.0, len(pts)))
    coef = jnp.linspace(1.0, 2.0, len(_EDGES) - 1)

    def grads(mode):
        cfg = _cfg(len(pts), ndev, mode)

        def loss(x, ww):
            c = distributed_pair_counts(
                x,
                _EDGES,
                box_size=_L,
                config=cfg,
                weights=ww,
                sharpness=_SHARP,
                mesh=_mesh(ndev),
            ).counts
            return jnp.sum(coef * jnp.log1p(c))

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(pts, w)

    (v_new, (gx_new, gw_new)), (v_old, (gx_old, gw_old)) = (
        grads("neighbors"),
        grads("all_gather"),
    )
    assert float(v_new) == float(v_old)
    np.testing.assert_array_equal(np.asarray(gx_new), np.asarray(gx_old))
    np.testing.assert_array_equal(np.asarray(gw_new), np.asarray(gw_old))


@pytest.mark.parametrize(
    "small",
    [dict(summary_neighbors=1), dict(summary_recv_capacity=2)],
    ids=["neighbors", "recv_capacity"],
)
def test_summary_exchange_overflow_fires(small):
    ndev = max(_NDEVS)
    if ndev < 4 and "summary_neighbors" in small:
        pytest.skip("one neighbour suffices below 4 devices on the line")
    if ndev < 2:
        pytest.skip("needs >= 2 devices")
    pts = _line(1200)
    res = _xi(pts, ndev, "neighbors", **small)
    assert bool(res.flags["summary_exchange_overflow"]) and bool(res.overflow)
    ok = _xi(pts, ndev, "neighbors")
    assert not bool(ok.flags["summary_exchange_overflow"])


def test_config_rejects_unknown_exchange():
    cfg = DistributedStatsConfig(capacity=64, halo_capacity=64)
    with pytest.raises(ValueError, match="summary_exchange"):
        dataclasses.replace(cfg, summary_exchange="ring").resolved()
