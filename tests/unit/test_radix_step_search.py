"""The radix build's binary searches: looped on CPU, unrolled elsewhere, same tree."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import _tree_impl
from yggdrax.morton import morton_encode_impl


def _lowered_build(n, leaf_size, seed, coincident):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(size=(n, 3))
    if coincident:
        # equal Morton codes exercise the index tie-break in the span search
        pos[: n // 3] = pos[0]
    positions = jnp.asarray(pos)
    masses = jnp.asarray(0.1 + rng.uniform(size=n))
    bounds = (jnp.zeros(3), jnp.ones(3))
    sorted_indices, sorted_codes = _tree_impl._morton_sort(
        morton_encode_impl(positions, bounds)
    )
    leaf_starts = jnp.arange(0, n, leaf_size)
    leaf_ends = jnp.minimum(leaf_starts + leaf_size, n)

    # a fresh function per call, so jit traces again and reads the patched switch
    def build(p, m, si, sc, ls, le):
        return _tree_impl._build_tree_from_leaf_partitions(
            p,
            m,
            si,
            sc,
            ls,
            le,
            bounds,
            leaf_size=leaf_size,
            return_reordered=True,
            workspace=None,
            return_workspace=False,
        )

    args = (positions, masses, sorted_indices, sorted_codes, leaf_starts, leaf_ends)
    return jax.jit(build).lower(*args), args


@pytest.mark.parametrize(
    "n, leaf_size, seed, coincident",
    [(512, 16, 0, False), (4096, 1, 2, False), (777, 5, 3, True)],
)
def test_looped_and_unrolled_step_searches_build_the_same_tree(
    monkeypatch, n, leaf_size, seed, coincident
):
    results = {}
    while_ops = {}
    for unroll in (False, True):
        monkeypatch.setattr(_tree_impl, "_unroll_step_searches", lambda u=unroll: u)
        lowered, args = _lowered_build(n, leaf_size, seed, coincident)
        while_ops[unroll] = lowered.as_text().count("stablehlo.while")
        results[unroll] = jax.tree_util.tree_leaves(lowered.compile()(*args))

    assert len(results[False]) == len(results[True])
    for looped, unrolled in zip(results[False], results[True]):
        assert looped.dtype == unrolled.dtype
        assert np.array_equal(np.asarray(looped), np.asarray(unrolled))
    # the CPU form keeps at least the two searches as loops (the unrolled chain
    # is what stalls XLA:CPU's LLVM loop vectorizer under JAX 0.11.2); other
    # stages may switch on _unroll_step_searches too, so this is a lower bound
    assert while_ops[False] >= while_ops[True] + 2


@pytest.mark.parametrize("unroll, expected_while_ops", [(False, 1), (True, 0)])
def test_descending_step_search_is_one_loop_or_unrolled(
    monkeypatch, unroll, expected_while_ops
):
    monkeypatch.setattr(_tree_impl, "_unroll_step_searches", lambda: unroll)
    target = jnp.arange(-3, 1000, 37, dtype=_tree_impl.INDEX_DTYPE)

    # the largest x <= target with x a sum of distinct powers 2**k, k < 10
    def search(t):
        def step_fn(step, x):
            return jnp.where(x + step <= t, x + step, x)

        return _tree_impl._descending_step_search(step_fn, jnp.zeros_like(t), 10)

    lowered = jax.jit(search).lower(target)
    assert lowered.as_text().count("stablehlo.while") == expected_while_ops
    expected = np.clip(np.asarray(target), 0, 2**10 - 1)
    assert np.array_equal(np.asarray(lowered.compile()(target)), expected)


def test_step_searches_loop_on_cpu_only(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert not _tree_impl._unroll_step_searches()
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _tree_impl._unroll_step_searches()
