"""Tests of yggdrax.applications.positions (B4: matching ξ and kNN-CDFs)."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from yggdrax.applications.corrfunc.traced import traced_soft_pair_counts
from yggdrax.applications.knn_stats import knn_cdf
from yggdrax.applications.point_process import draw_thomas, thomas_positions
from yggdrax.applications.positions import (
    MatchSpec,
    PositionMatcher,
    optimise_positions,
)
from yggdrax.applications.positions.reference import dense_statistics

jax.config.update("jax_enable_x64", True)

N = 500
L = 1.0


def _spec(**kw) -> MatchSpec:
    base = MatchSpec(
        box_size=L,
        xi_edges=tuple(np.geomspace(0.02, 0.25, 9)),
        knn_ks=(1, 2, 4),
        knn_r_grid=tuple(np.geomspace(0.01, 0.4, 12)),
        xi_sharpness=40.0,
        knn_sharpness=8.0,
        candidate_margin=6,
        window_eps=1e-12,
    )
    return replace(base, **kw)


@pytest.fixture(scope="module")
def problem():
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    target = thomas_positions(
        jnp.asarray([0.03, 1.0]), draw_thomas(keys[0], N // 10, 10), L
    )
    init = jax.random.uniform(keys[1], (N, 3)) * L
    randoms = jax.random.uniform(keys[2], (N, 3)) * L
    mid = 0.5 * (init + target)
    matcher = PositionMatcher.build(
        _spec(), target, probes=[init, mid], randoms=randoms, leaf_size=16
    )
    return {"target": target, "init": init, "mid": mid, "m": matcher}


def test_a_fresh_state_reproduces_the_estimators(problem):
    """At a fresh state: exact kNN-CDFs, and the plain traced pair counts."""
    m, x = problem["m"], problem["mid"]
    spec = m.spec
    s = m.fresh_statistics(x)
    grid = jnp.asarray(spec.knn_r_grid)
    kw = dict(sharpness=spec.knn_sharpness, box_size=L)
    ref_r = knn_cdf(x, m.randoms, spec.knn_ks, grid, **kw)
    ref_d = knn_cdf(x, None, spec.knn_ks, grid, **kw)
    np.testing.assert_allclose(s.cdf[0], ref_r, atol=1e-14)
    np.testing.assert_allclose(s.cdf[1], ref_d, atol=1e-14)
    plain = traced_soft_pair_counts(
        x,
        jnp.asarray(spec.xi_edges),
        m.traced_config,
        sharpness=spec.xi_sharpness,
        box_size=L,
    )
    np.testing.assert_allclose(s.counts, plain.counts, rtol=1e-12)
    assert not bool(s.overflow)


def test_tree_statistics_and_gradient_match_the_dense_reference(problem):
    """Values and the full gradient against autodiff of the O(N^2) estimator."""
    m, x = problem["m"], problem["mid"]
    state = m.refresh(x)
    tree = m.statistics(x, state)
    dense = dense_statistics(m, x)
    np.testing.assert_allclose(tree.counts, dense.counts, rtol=1e-10)
    np.testing.assert_allclose(tree.cdf, dense.cdf, atol=1e-13)

    g_tree = jax.grad(lambda y: m.loss(y, state)[0])(x)
    g_dense = jax.grad(lambda y: m.loss_from_statistics(dense_statistics(m, y)).loss)(x)
    rel = float(jnp.linalg.norm(g_tree - g_dense) / jnp.linalg.norm(g_dense))
    assert rel < 1e-9, rel


def test_gradient_matches_central_differences(problem):
    """FD of the full chain (state refreshed at every evaluation)."""
    m, x = problem["m"], problem["mid"]
    state = m.refresh(x)
    g = jax.grad(lambda y: m.loss(y, state)[0])(x)
    chain = jax.jit(lambda y: m.loss_from_statistics(m.fresh_statistics(y)).loss)
    rng = np.random.default_rng(3)
    flat = rng.choice(x.size, size=8, replace=False)
    h = 1e-6
    scale = float(jnp.max(jnp.abs(g)))
    for f in flat:
        r, c = divmod(int(f), 3)
        e = jnp.zeros_like(x).at[r, c].set(h)
        fd = float(chain(x + e) - chain(x - e)) / (2 * h)
        assert abs(fd - float(g[r, c])) < 1e-6 * scale, (r, c, fd, float(g[r, c]))


def test_stale_candidates_are_detected(problem):
    """Zero staleness at the refresh point; positive after a large move."""
    m, x = problem["m"], problem["mid"]
    state = m.refresh(x)
    assert float(m.stale_fraction(x, state, state)) == 0.0
    moved = x + 0.08 * jax.random.normal(jax.random.PRNGKey(1), x.shape)
    fresh = m.refresh(moved)
    frac = float(m.stale_fraction(moved, state, fresh))
    assert 0.0 < frac < 1.0
    # A stale k-th distance can only be too large, never too small.
    stale_cdf = m.statistics(moved, state).cdf
    exact_cdf = m.statistics(moved, fresh).cdf
    assert bool(jnp.all(stale_cdf <= exact_cdf + 1e-14))


def test_adam_reduces_the_loss(problem):
    m = problem["m"]
    res = optimise_positions(
        m, problem["init"], optax.adam(0.004), num_steps=18, rebuild_every=6
    )
    assert len(res.history) == 18 and len(res.epochs) == 3
    final = m.loss_from_statistics(m.fresh_statistics(res.positions)).loss
    assert float(final) < 0.6 * res.history[0]["loss"]
    assert bool(jnp.all((res.positions >= 0) & (res.positions < L)))
    assert res.seconds_per_step > 0 and res.compile_seconds > 0


def test_lbfgs_reduces_the_loss(problem):
    m = problem["m"]
    res = optimise_positions(
        m, problem["init"], optax.lbfgs(), num_steps=6, rebuild_every=3
    )
    losses = [h["loss"] for h in res.history]
    assert losses[-1] < losses[0]


def test_an_overflow_replans_and_repeats_the_epoch(problem):
    m = problem["m"]
    tiny = m.with_traced_config(replace(m.traced_config, near_cap=8))
    res = optimise_positions(
        tiny, problem["init"], optax.adam(0.004), num_steps=4, rebuild_every=2
    )
    assert res.epochs[0]["replans"] >= 1
    assert res.matcher.traced_config.near_cap > 8
    assert len(res.history) == 4


def test_build_validates_its_inputs(problem):
    t = problem["target"]
    with pytest.raises(ValueError, match="randoms"):
        PositionMatcher.build(_spec(), t)
    with pytest.raises(ValueError, match="L / 2"):
        PositionMatcher.build(_spec(knn_r_grid=(0.1, 0.6), query_mode="data"), t)
    with pytest.raises(ValueError, match="size"):
        PositionMatcher.build(_spec(query_mode="data"), t, probes=[t[:-1]])


def test_bench_smoke(tmp_path, monkeypatch):
    """``bench/positions/optimise.py --smoke`` writes a complete JSON."""
    # select_device writes these; keep them from leaking into other tests.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    path = Path(__file__).resolve().parents[2] / "bench" / "positions" / "optimise.py"
    spec = importlib.util.spec_from_file_location("b4_optimise", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "smoke.json"
    mod.main(["--smoke", "--out", str(out), "--no-fig"])
    payload = json.loads(out.read_text())
    assert payload["steps"] == 20 and len(payload["history"]) == 20
    s = payload["summary"]
    assert s["final"]["loss"] < s["initial"]["loss"]
    assert {"initial", "final", "target", "independent_realisation"} <= set(
        payload["statistics"]
    )
    assert payload["seconds_per_step"] > 0
    assert (tmp_path / "smoke.npz").exists()
    if importlib.util.find_spec("matplotlib") is not None:
        png = tmp_path / "smoke.png"
        mod.main(["--replot", str(out), "--fig", str(png)])
        assert png.exists()
