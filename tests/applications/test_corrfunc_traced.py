"""The traced soft pair-count estimator: tree rebuilt inside the traced function.

Covers the A2 contract of :mod:`yggdrax.applications.corrfunc.traced`: values
and gradients against brute force, exact pair coverage at a finite opening
angle, and composition with ``jit(grad)``, ``lax.scan`` and ``vmap`` with the
rebuild inside, plus the overflow flags of every capacity.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.applications.corrfunc.binning import make_log_edges, soft_bin_weights
from yggdrax.applications.corrfunc.traced import (
    TracedPairCountConfig,
    _build_topology,
    plan_traced_pair_counts,
    traced_soft_pair_counts,
)

_N = 300
_LEAF = 16
_SHARP = 100.0
_EDGES = make_log_edges(0.02, 0.8, 8)
# Fixed Morton domain, so a finite-difference nudge cannot move the bounds.
_BOUNDS = (jnp.full(3, -0.1), jnp.full(3, 1.1))


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _catalog(seed: int = 0, n: int = _N) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)


def _brute(positions: jax.Array) -> jax.Array:
    """``sum_{i<j} w_k(r_ij)`` by the O(n^2) double loop."""
    n = positions.shape[0]
    d = positions[:, None, :] - positions[None, :, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
    w = soft_bin_weights(r, _EDGES, _SHARP)
    upper = jnp.triu(jnp.ones((n, n), dtype=positions.dtype), k=1)
    return jnp.sum(w * upper[..., None], axis=(0, 1))


def _loss(counts: jax.Array) -> jax.Array:
    # weights the bins unevenly so the gradient is not a symmetric sum
    return jnp.sum(jnp.log1p(counts) * jnp.arange(1.0, counts.shape[0] + 1.0))


def _traced(positions, config, bounds=_BOUNDS):
    return traced_soft_pair_counts(
        positions, _EDGES, config, sharpness=_SHARP, bounds=bounds
    )


@pytest.fixture(scope="module")
def near_only_config() -> TracedPairCountConfig:
    """theta so small that the walk accepts no multi-particle node as far.

    A single-particle leaf has radius 0, so the non-strict Dehnen test
    ``(r_a + r_b)^2 <= theta^2 d^2`` accepts a pair of them at any theta; their
    monopole is the exact pair term, so the sum is still exact.
    """
    jax.config.update("jax_enable_x64", True)
    probes = [_catalog(seed=0), _catalog(seed=3)]
    return plan_traced_pair_counts(probes, theta=0.02, leaf_size=_LEAF, bounds=_BOUNDS)


@pytest.fixture(scope="module")
def far_config() -> TracedPairCountConfig:
    """A working opening angle: far, near and within-leaf terms all present."""
    jax.config.update("jax_enable_x64", True)
    probes = [_catalog(s) for s in range(4)]
    return plan_traced_pair_counts(probes, theta=0.6, leaf_size=_LEAF, bounds=_BOUNDS)


def test_matches_brute_force_when_nothing_is_far(near_only_config):
    pos = _catalog()
    res = jax.jit(lambda p: _traced(p, near_only_config))(pos)
    assert not bool(res.overflow)
    assert int(res.near_count) > 0
    # every far pair (if any) is a single particle against a single particle
    topo = _build_topology(pos, near_only_config, _BOUNDS)
    ranges = np.asarray(topo.node_ranges)
    count = int(topo.walk.far_count)
    for nodes in (topo.walk.far_a, topo.walk.far_b):
        nodes = np.asarray(nodes)[:count]
        np.testing.assert_array_equal(ranges[nodes, 1] - ranges[nodes, 0], 0)
    np.testing.assert_allclose(res.counts, _brute(pos), rtol=1e-12)


def test_pair_lists_tile_every_pair_once_at_finite_theta(far_config):
    """Within-leaf + near + far cover the n(n-1)/2 pairs exactly once."""
    pos = _catalog()
    topo = _build_topology(pos, far_config, _BOUNDS)
    walk = topo.walk
    assert int(walk.far_count) > 0
    ranges = np.asarray(topo.node_ranges)

    def size(nodes):
        return np.maximum(ranges[nodes, 1] - ranges[nodes, 0] + 1, 0)

    far_a = np.asarray(walk.far_a)[: int(walk.far_count)]
    far_b = np.asarray(walk.far_b)[: int(walk.far_count)]
    near_a = np.asarray(walk.near_a)[: int(walk.near_count)]
    near_b = np.asarray(walk.near_b)[: int(walk.near_count)]
    leaves = size(np.arange(topo.num_internal, ranges.shape[0]))
    covered = (
        np.sum(size(far_a) * size(far_b))
        + np.sum(size(near_a) * size(near_b))
        + np.sum(leaves * (leaves - 1) // 2)
    )
    assert covered == _N * (_N - 1) // 2
    # the far monopole is an approximation, but a close one at theta = 0.6
    res = _traced(pos, far_config)
    total = float(jnp.sum(res.counts))
    assert abs(total - float(jnp.sum(_brute(pos)))) / total < 0.05


def test_gradient_matches_brute_force_autodiff(near_only_config):
    pos = _catalog(seed=3)
    assert not bool(_traced(pos, near_only_config).overflow)
    g = jax.jit(jax.grad(lambda p: _loss(_traced(p, near_only_config).counts)))(pos)
    g_ref = jax.grad(lambda p: _loss(_brute(p)))(pos)
    np.testing.assert_allclose(
        g, g_ref, rtol=1e-9, atol=1e-12 * float(jnp.abs(g_ref).max())
    )


def test_gradient_wrt_positions_and_edges_matches_finite_difference(far_config):
    """Fixed-topology gradient (far terms included) vs central differences."""
    pos = _catalog(seed=1)

    def f(p, edges):
        res = traced_soft_pair_counts(
            p, edges, far_config, sharpness=_SHARP, bounds=_BOUNDS
        )
        return _loss(res.counts)

    g_pos, g_edges = jax.jit(jax.grad(f, argnums=(0, 1)))(pos, _EDGES)
    f_jit = jax.jit(f)
    h = 1e-6
    rng = np.random.default_rng(0)
    for i, k in zip(rng.integers(0, _N, 6), rng.integers(0, 3, 6), strict=True):
        fd = (
            float(f_jit(pos.at[i, k].add(h), _EDGES))
            - float(f_jit(pos.at[i, k].add(-h), _EDGES))
        ) / (2 * h)
        np.testing.assert_allclose(g_pos[i, k], fd, rtol=1e-5, atol=1e-6)
    for k in (1, 4, 7):
        fd = (
            float(f_jit(pos, _EDGES.at[k].add(h)))
            - float(f_jit(pos, _EDGES.at[k].add(-h)))
        ) / (2 * h)
        np.testing.assert_allclose(g_edges[k], fd, rtol=1e-5, atol=1e-6)


def test_jit_grad_rebuilds_the_tree_inside(far_config):
    """One compiled grad serves different catalogues: the tree is not stale."""
    grad_fn = jax.jit(jax.grad(lambda p: _loss(_traced(p, far_config).counts)))
    jaxpr = str(jax.make_jaxpr(grad_fn)(_catalog()))
    assert "sort" in jaxpr and "while" in jaxpr  # Morton sort + mutual walk

    # Without a rebuild, a permuted catalogue would reuse the first tree's
    # leaf ranges on the wrong particles. With one, grad is equivariant.
    pos = _catalog(seed=2)
    perm = jax.random.permutation(jax.random.PRNGKey(9), _N)
    g = grad_fn(pos)
    g_perm = grad_fn(pos[perm])
    np.testing.assert_allclose(g_perm, g[perm], rtol=1e-10, atol=1e-12)


def test_scan_of_gradient_steps_with_in_loop_rebuilds(far_config):
    target = _traced(_catalog(seed=0), far_config).counts

    def loss(p):
        res = _traced(p, far_config)
        return jnp.mean((jnp.log1p(res.counts) - jnp.log1p(target)) ** 2), res.overflow

    def step(p, _):
        (value, overflow), g = jax.value_and_grad(loss, has_aux=True)(p)
        p = jnp.clip(p - 1e-4 * g, 0.0, 1.0)
        return p, (value, overflow)

    start = _catalog(seed=1)
    final, (losses, overflows) = jax.jit(
        lambda p: jax.lax.scan(step, p, None, length=3)
    )(start)
    assert losses.shape == (3,)
    assert not bool(jnp.any(overflows))
    assert bool(jnp.all(jnp.isfinite(final)))
    assert not np.allclose(final, start)
    np.testing.assert_allclose(losses[0], loss(start)[0], rtol=1e-12)


def test_vmap_over_catalogues_matches_a_loop(far_config):
    batch = jnp.stack([_catalog(seed=s) for s in range(3)])
    out = jax.jit(jax.vmap(lambda p: _traced(p, far_config)))(batch)
    assert out.counts.shape == (3, _EDGES.shape[0] - 1)
    assert not bool(jnp.any(out.overflow))
    grads = jax.jit(jax.vmap(jax.grad(lambda p: _loss(_traced(p, far_config).counts))))(
        batch
    )
    single = jax.jit(lambda p: _traced(p, far_config).counts)
    single_grad = jax.jit(jax.grad(lambda p: _loss(_traced(p, far_config).counts)))
    for b in range(3):
        np.testing.assert_allclose(out.counts[b], single(batch[b]), rtol=1e-12)
        np.testing.assert_allclose(
            grads[b], single_grad(batch[b]), rtol=1e-10, atol=1e-12
        )


def test_vmap_with_per_catalogue_bounds(far_config):
    """bounds=None infers each catalogue's own Morton box inside the vmap."""
    batch = jnp.stack([_catalog(seed=s) for s in range(2)])
    out = jax.vmap(lambda p: _traced(p, far_config, bounds=None))(batch)
    assert not bool(jnp.any(out.overflow))
    np.testing.assert_allclose(
        jnp.sum(out.counts, axis=1),
        jnp.stack([jnp.sum(_brute(p)) for p in batch]),
        rtol=0.05,
    )


@pytest.mark.parametrize(
    ("field", "flag"),
    [
        ("leaf_capacity", "leaf_overflow"),
        ("max_pair_queue", "queue_overflow"),
        ("far_cap", "far_overflow"),
        ("near_cap", "near_overflow"),
    ],
)
def test_too_small_capacity_raises_its_overflow_flag(far_config, field, flag):
    small = dataclasses.replace(far_config, **{field: 4})
    res = jax.jit(lambda p: _traced(p, small))(_catalog())
    assert bool(getattr(res, flag))
    assert bool(res.overflow)
    # the planned config itself is clean
    assert not bool(_traced(_catalog(), far_config).overflow)


def test_planner_covers_a_batch_and_reports_needs(far_config):
    for s in range(4):
        res = _traced(_catalog(seed=s), far_config)
        assert not bool(res.overflow)
        assert int(res.num_leaves) <= far_config.leaf_capacity
        assert int(res.far_count) <= far_config.far_cap
        assert int(res.near_count) <= far_config.near_cap
        assert int(res.peak_wavefront) <= far_config.max_pair_queue


def test_traced_inference_demo_moves_toward_truth():
    pytest.importorskip("optax")
    from yggdrax.applications.corrfunc.inference_demo import (
        make_cluster_model,
        recover_parameter_traced,
    )

    model = make_cluster_model(jax.random.PRNGKey(0), n=300, num_centers=6)
    res = recover_parameter_traced(
        model,
        make_log_edges(0.01, 0.5, 8),
        a_true=0.5,
        a_init=0.1,
        theta=0.4,
        leaf_size=_LEAF,
        num_steps=25,
        probe_a=(0.0, 0.5, 0.99),
    )
    assert len(res.a_history) == 26
    assert abs(res.a_final - 0.5) < abs(0.1 - 0.5)
    assert res.loss_history[-1] < res.loss_history[0]
