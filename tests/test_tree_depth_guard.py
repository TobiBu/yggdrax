"""A radix tree deeper than ``MAX_TREE_LEVELS`` fails loudly (A2).

Its level tables hold 64 levels and ``num_levels`` is clipped to them, so a deeper
node is in no level and every level-ordered sweep skips it -- silently, before.
Nested shells down to 1e-7 and a clump of 512 particles inside one level-21 cell
(64 leaves of 8 sharing it) make a cells tree 70 levels deep on CPU: the eager
build raises :class:`~yggdrax.TreeDepthError` naming the box, the level-21 cell and
the farthest particle; a shallow tree passes :func:`~yggdrax.check_tree_depth`,
and a traced one is left to its caller (``None``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import (
    MAX_TREE_LEVELS,
    TreeDepthError,
    check_tree_depth,
    domain_report,
    format_domain_report,
)
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds


def _deep(n_shell=600, n_clump=512, seed=0):
    rng = np.random.default_rng(seed)
    r = np.exp(rng.uniform(np.log(1e-7), 0.0, n_shell))
    d = rng.normal(size=(n_shell, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    parts = [r[:, None] * d, 1e-9 * rng.normal(size=(n_clump, 3))]
    return jnp.asarray(np.concatenate(parts, axis=0))


def _cells(pos, leaf_size=8, leaf_capacity=512):
    out = build_static_cells_tree(
        pos,
        jnp.ones((pos.shape[0],)),
        infer_bounds(pos),
        leaf_size=leaf_size,
        leaf_capacity=leaf_capacity,
    )
    return out if hasattr(out, "node_level") else out[0]


def test_a_tree_past_the_level_tables_raises_with_the_box():
    pos = _deep()
    with pytest.raises(TreeDepthError) as err:
        _cells(pos)
    exc = err.value
    assert exc.depth > MAX_TREE_LEVELS and exc.bound == MAX_TREE_LEVELS
    message = str(exc)
    assert f"{exc.depth} levels deep" in message
    assert "Morton box" in message and "level-21 cell" in message
    far = exc.report["farthest"]["index"]
    assert f"farthest particle #{far}" in message
    radius = np.linalg.norm(
        np.asarray(pos) - np.median(np.asarray(pos), axis=0), axis=1
    )
    assert far == int(np.argmax(radius))
    assert exc.report["cell21"] == pytest.approx(
        exc.report["box_extent"] / 2.0**21, rel=1e-12
    )


def test_a_shallow_tree_passes_and_a_traced_one_is_the_callers():
    rng = np.random.default_rng(1)
    pos = jnp.asarray(rng.uniform(-1.0, 1.0, (512, 3)))
    tree = _cells(pos, leaf_size=16, leaf_capacity=128)
    depth = check_tree_depth(tree, pos)
    assert depth is not None and depth <= MAX_TREE_LEVELS
    assert depth == int(np.max(np.asarray(tree.node_level))) + 1
    assert jax.jit(lambda t: jnp.asarray(check_tree_depth(t) is None))(tree)
    # a bound below the tree's depth: the same error, from the check itself
    with pytest.raises(TreeDepthError, match="past the bound of 2"):
        check_tree_depth(tree, pos, bound=2)


def test_the_domain_report_names_the_outliers():
    rng = np.random.default_rng(2)
    bulk = rng.normal(size=(10_000, 3))
    far = np.array([[1e4, 0.0, 0.0], [0.0, -2e4, 0.0]])
    report = domain_report(jnp.asarray(np.concatenate([bulk, far])))
    assert report["farthest"]["index"] == 10_001
    assert report["r_max"] == pytest.approx(2e4, rel=1e-3)
    assert report["box_extent"] == pytest.approx(2e4, rel=1e-3)  # the y side
    assert report["r_quantile"] < 10.0 and report["dominance"] > 1e3
    assert report["n_beyond"] == 2
    text = format_domain_report(report)
    assert "farthest particle #10001" in text and "2 of 10002" in text
