"""The smooth (blended) acceptance rule of the traced soft pair counts (A4b).

Covers ``TracedPairCountConfig(smooth_band > 1)``:

* accuracy against brute force: the window rule's error bound survives the
  blend (per-bin relative error below 1e-6), and the blended geometric rule is
  at least as accurate as the hard one at the same theta;
* gradients: ``jit(grad)`` matches brute-force autodiff (window rule) and
  central finite differences of the blended counts themselves (both rules), so
  the gradient carries the d(sigma)/dx terms of the band pairs;
* continuity: one particle scanned across acceptance switches at fixed leaf
  membership -- the hard rule jumps, the smooth rule does not (trapezoid-rule
  residual at the round-off floor);
* composition with ``vmap`` / ``lax.scan`` and the band-list overflow flag.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.binning import make_log_edges, soft_bin_weights
from yggdrax.applications.corrfunc.traced import (
    _build_topology,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)

_N = 400
_SHARP = 100.0
_EDGES = make_log_edges(0.02, 0.8, 8)
_WIDE = make_log_edges(0.02, 0.8, 3)
_BOUNDS = (jnp.full(3, -0.1), jnp.full(3, 1.1))
_BETA = 2.0


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _catalog(seed: int = 5, n: int = _N) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)


def _brute(positions, edges, sharpness=_SHARP):
    n = positions.shape[0]
    d = positions[:, None, :] - positions[None, :, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
    w = soft_bin_weights(r, edges, sharpness)
    upper = jnp.triu(jnp.ones((n, n), dtype=positions.dtype), k=1)
    return jnp.sum(w * upper[..., None], axis=(0, 1))


def _loss(counts):
    return jnp.sum(jnp.log1p(counts) * jnp.arange(1.0, counts.shape[0] + 1.0))


def _plan(positions, edges, **kw):
    kw.setdefault("bounds", _BOUNDS)
    kw.setdefault("leaf_size", 16)
    return plan_traced_pair_counts(positions, edges=edges, sharpness=_SHARP, **kw)


def _counts(positions, edges, cfg):
    return traced_soft_pair_counts(
        positions, edges, cfg, sharpness=_SHARP, bounds=_BOUNDS
    )


def _max_rel_err(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got) / np.asarray(ref) - 1.0)))


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("edges", [_EDGES, _WIDE], ids=["narrow", "wide"])
@pytest.mark.parametrize("leaf_size", [8, 16])
def test_smooth_window_rule_keeps_the_error_bound(edges, leaf_size):
    pos = _catalog(seed=11, n=500)
    cfg = _plan(pos, edges, smooth_band=_BETA, leaf_size=leaf_size)
    assert cfg.smooth and cfg.band_cap > 0
    res = jax.jit(lambda p: _counts(p, edges, cfg))(pos)
    assert not bool(res.overflow)
    assert _max_rel_err(res.counts, _brute(pos, edges)) < 1e-6


def test_smooth_window_rule_blends_some_pairs_at_a_loose_eps():
    """eps 1e-3 on wide bins: band pairs exist and are counted correctly."""
    pos = _catalog(seed=11, n=500)
    cfg = _plan(pos, _WIDE, smooth_band=_BETA, window_eps=1e-3)
    res = _counts(pos, _WIDE, cfg)
    assert int(res.band_count) > 0
    # the bound is eps per particle pair, far below these counts
    assert _max_rel_err(res.counts, _brute(pos, _WIDE)) < 1e-5


@pytest.mark.parametrize("theta", [0.3, 0.5])
def test_smooth_geometric_rule_is_no_worse_than_the_hard_rule(theta):
    pos = _catalog(seed=11, n=500)
    ref = _brute(pos, _WIDE)
    errs = {}
    for beta in (1.0, _BETA):
        cfg = _plan(pos, _WIDE, acceptance="geometric", theta=theta, smooth_band=beta)
        res = _counts(pos, _WIDE, cfg)
        assert not bool(res.overflow)
        errs[beta] = _max_rel_err(res.counts, ref)
    assert errs[_BETA] <= errs[1.0]


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("edges", [_EDGES, _WIDE], ids=["narrow", "wide"])
def test_smooth_window_gradient_matches_brute_force(edges):
    pos = _catalog(seed=3)
    cfg = _plan(pos, edges, smooth_band=_BETA)
    g = jax.jit(jax.grad(lambda p: _loss(_counts(p, edges, cfg).counts)))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p, edges)))(pos)
    rel = float(jnp.linalg.norm(g - g_ref) / jnp.linalg.norm(g_ref))
    assert rel < 1e-6


def _band_particles(pos, edges, cfg, k=6):
    """Slots of particles in the band pairs' nodes -- where d(sigma)/dx lives."""
    topo = _build_topology(pos, cfg, _BOUNDS, edges, _SHARP, True)
    nb = int(topo.walk.band_count)
    assert nb > 0
    ranges = np.asarray(topo.node_ranges)
    order = np.asarray(topo.order)
    slots = []
    accepted = np.asarray(topo.walk.band_accepted)[:nb]
    for a, b in zip(
        np.asarray(topo.walk.band_a)[:nb][accepted],
        np.asarray(topo.walk.band_b)[:nb][accepted],
    ):
        for node in (a, b):
            s, e = ranges[node]
            slots.extend(order[s : e + 1].tolist())
        if len(set(slots)) >= k:
            break
    return sorted(set(slots))[:k]


@pytest.mark.parametrize(
    ("acceptance", "kw"),
    [("window", dict(window_eps=1e-3)), ("geometric", dict(theta=0.5))],
)
def test_smooth_gradient_matches_finite_differences_of_the_blend(acceptance, kw):
    """FD of the blended counts, at particles of accepted band pairs."""
    pos = _catalog(seed=3)
    cfg = _plan(pos, _WIDE, acceptance=acceptance, smooth_band=_BETA, **kw)
    f = jax.jit(lambda p: _loss(_counts(p, _WIDE, cfg).counts))
    g = np.asarray(
        jax.jit(jax.grad(lambda p: _loss(_counts(p, _WIDE, cfg).counts)))(pos)
    )
    h = 1e-6
    for i in _band_particles(pos, _WIDE, cfg):
        for c in range(3):
            e = jnp.zeros_like(pos).at[i, c].set(h)
            fd = (float(f(pos + e)) - float(f(pos - e))) / (2 * h)
            assert abs(fd - g[i, c]) <= 1e-5 * max(1.0, abs(g[i, c])), (i, c)


# --------------------------------------------------------------------------
# continuity across acceptance switches
# --------------------------------------------------------------------------


def _scan(cfg, pos, i, direction, ts):
    def value(t):
        p = pos.at[i].add(t * direction)
        r = _counts(p, _WIDE, cfg)
        return _loss(r.counts), r.overflow

    vals, ov = jax.jit(jax.vmap(value))(ts)
    grads = jax.jit(jax.vmap(jax.grad(lambda t: value(t)[0])))(ts)
    ranges = jax.jit(
        jax.vmap(
            lambda t: _build_topology(
                pos.at[i].add(t * direction), cfg, _BOUNDS, _WIDE, _SHARP, True
            ).node_ranges
        )
    )(ts)
    assert not bool(jnp.any(ov))
    # fixed leaf membership along the whole scan: only acceptance switches
    assert bool(jnp.all(ranges == ranges[0]))
    v, g = np.asarray(vals), np.asarray(grads)
    dt = float(ts[1] - ts[0])
    # trapezoid residual: O(dt^3) for a C1 function, the jump size at a jump
    return np.abs(np.diff(v) - 0.5 * (g[1:] + g[:-1]) * dt), np.abs(np.diff(v))


def test_a_scan_across_acceptance_switches_jumps_only_under_the_hard_rule():
    pos = _catalog(seed=5, n=500)
    i = 7
    direction = jnp.array([1.0, 0.3, -0.2]) / jnp.linalg.norm(
        jnp.array([1.0, 0.3, -0.2])
    )
    ts = jnp.linspace(-0.004, 0.004, 161)
    probes = [pos.at[i].add(t * direction) for t in (-0.004, 0.0, 0.004)]
    out = {}
    for beta in (1.0, _BETA):
        cfg = _plan(probes, _WIDE, acceptance="geometric", theta=0.5, smooth_band=beta)
        out[beta] = _scan(cfg, pos, i, direction, ts)
    resid_hard, _ = out[1.0]
    resid_smooth, _ = out[_BETA]
    # the hard rule switches along this segment: a residual spike far above
    # its own O(dt^3) floor
    assert resid_hard.max() > 100.0 * np.median(resid_hard)
    # the smooth rule has none: its largest residual is the O(dt^3) floor, so
    # the value does not jump and the gradient integrates to the value change
    assert resid_smooth.max() < 1e-2 * resid_hard.max()
    assert resid_smooth.max() < 10.0 * np.max(
        resid_hard[resid_hard < 1e-3 * resid_hard.max()]
    )


# --------------------------------------------------------------------------
# composition and overflow
# --------------------------------------------------------------------------


def test_smooth_rule_runs_under_vmap_and_scan():
    batch = jnp.stack([_catalog(seed=s, n=200) for s in (1, 2, 3)])
    cfg = _plan(batch, _WIDE, smooth_band=_BETA, window_eps=1e-3)
    out = jax.jit(jax.vmap(lambda p: _counts(p, _WIDE, cfg)))(batch)
    assert not bool(jnp.any(out.overflow))
    for b in range(3):
        np.testing.assert_allclose(
            out.counts[b], _counts(batch[b], _WIDE, cfg).counts, rtol=1e-12
        )

    def step(p, _):
        g = jax.grad(lambda q: _loss(_counts(q, _WIDE, cfg).counts))(p)
        return jnp.clip(p - 1e-6 * g, 0.0, 1.0), None

    final, _ = jax.jit(lambda p: jax.lax.scan(step, p, None, length=2))(batch[0])
    assert bool(jnp.all(jnp.isfinite(final)))


def test_a_small_band_cap_raises_the_overflow_flag():
    pos = _catalog(seed=3)
    cfg = _plan(pos, _WIDE, acceptance="geometric", theta=0.5, smooth_band=_BETA)
    small = dataclasses.replace(cfg, band_cap=2)
    res = _counts(pos, _WIDE, small)
    assert bool(res.band_overflow) and bool(res.overflow)
    assert int(res.band_count) > 2


def test_smooth_band_below_one_is_refused():
    with pytest.raises(ValueError, match="smooth_band"):
        _plan(_catalog(n=50), _WIDE, smooth_band=0.5)


# --------------------------------------------------------------------------
# frozen leaf membership
# --------------------------------------------------------------------------


def test_frozen_membership_keeps_the_error_bound_at_moved_positions():
    pos0 = _catalog(seed=11, n=500)
    pos = jnp.clip(
        pos0 + 0.01 * jax.random.normal(jax.random.PRNGKey(4), pos0.shape), 0, 1
    )
    cfg = _plan([pos0, pos], _EDGES, smooth_band=_BETA)
    res = traced_soft_pair_counts(
        pos, _EDGES, cfg, sharpness=_SHARP, bounds=_BOUNDS, membership=pos0
    )
    assert not bool(res.overflow)
    assert _max_rel_err(res.counts, _brute(pos, _EDGES)) < 1e-6


def test_frozen_membership_makes_a_whole_trajectory_continuous():
    """Every particle moves (leaves change on the way): with the leaf partition
    frozen at the start, the smooth rule has no jump; rebuilding it has."""
    pos = _catalog(seed=5, n=400) * 0.8 + 0.1
    direction = 0.01 * jax.random.normal(jax.random.PRNGKey(1), pos.shape)
    ts = jnp.linspace(0.0, 1.0, 81)
    probes = [pos + t * direction for t in (0.0, 0.5, 1.0)]
    cfg = _plan(probes, _WIDE, acceptance="geometric", theta=0.5, smooth_band=_BETA)

    def resid(membership, exact=False):
        def value(t):
            if exact:
                return _loss(_brute(pos + t * direction, _WIDE)), jnp.asarray(False)
            r = traced_soft_pair_counts(
                pos + t * direction,
                _WIDE,
                cfg,
                sharpness=_SHARP,
                bounds=_BOUNDS,
                membership=membership,
            )
            return _loss(r.counts), r.overflow

        vals, ov = jax.jit(jax.vmap(value))(ts)
        grads = jax.jit(jax.vmap(jax.grad(lambda t: value(t)[0])))(ts)
        assert not bool(jnp.any(ov))
        v, g = np.asarray(vals), np.asarray(grads)
        dt = float(ts[1] - ts[0])
        return np.abs(np.diff(v) - 0.5 * (g[1:] + g[:-1]) * dt)

    leaves = jax.jit(
        jax.vmap(
            lambda t: _build_topology(
                pos + t * direction, cfg, _BOUNDS, _WIDE, _SHARP, True
            ).node_ranges
        )
    )(ts)
    assert not bool(jnp.all(leaves == leaves[0]))  # the partition does change
    rebuilt, frozen, exact = resid(None), resid(pos), resid(None, exact=True)
    # rebuilding: restructuring jumps, an order of magnitude above the exact
    # estimator's O(dt^3) residual; frozen: that residual and nothing more
    assert rebuilt.max() > 5.0 * exact.max()
    assert frozen.max() < 1.5 * exact.max()
    assert np.median(frozen) < 1.5 * np.median(exact)
