"""Reverse-mode through tree geometry and the mutual walk UNDER ``jax.jit``.

``compute_tree_geometry``'s upward pass loops over the tree levels, and under
``jit`` the level count is traced. JAX cannot transpose a loop with a traced trip
count, so ``jax.jit(jax.grad(...))`` through the geometry used to raise on every
backend while the eager gradient worked (there the count is concrete). The pass
now chooses, without gradients, which leaf bound each internal node inherits and
gathers that bound from the live leaf bounds; the walks' outputs are integer, so
their float inputs carry no gradient. These tests pin all of it:

* the forward geometry equals a direct per-node reduction over its leaves, bit
  for bit, eager and under jit, so the change moves no value;
* ``jit(grad)`` equals the eager gradient and finite differences, and ``jvp``
  agrees with ``vjp``, on every backend and in float32 and float64;
* a loss whose far list comes from either mutual-walk backend differentiates
  under jit, and both backends give the same gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import Tree, compute_tree_geometry
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import dual_tree_walk_mutual, mutual_walk_inputs

_KINDS = ("radix", "octree", "kdtree", "cells")


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _cloud(n: int, dtype, seed: int = 1):
    kp, km = jax.random.split(jax.random.PRNGKey(seed))
    pos = jax.random.uniform(kp, (n, 3), minval=-1.0, maxval=1.0).astype(dtype)
    mass = jax.random.uniform(km, (n,), minval=0.5, maxval=1.5).astype(dtype)
    return pos, mass


def _tree(kind: str, n: int = 600, dtype=jnp.float64):
    pos, mass = _cloud(n, dtype)
    if kind == "cells":
        topo, ps, *_ = build_static_cells_tree(
            pos,
            mass,
            infer_bounds(pos),
            leaf_size=32,
            leaf_capacity=max(16, n // 4),
            return_reordered=True,
            return_overflow=True,
        )
        return topo, ps
    tree = Tree.from_particles(
        pos,
        mass,
        tree_type=kind,
        build_mode="adaptive",
        leaf_size=32,
        return_reordered=True,
    )
    return tree, tree.positions_sorted


def _loss(tree, positions):
    g = compute_tree_geometry(tree, positions)
    weights = (jnp.arange(g.center.size) % 7 + 1.0).reshape(g.center.shape)
    return (
        jnp.sum(weights.astype(positions.dtype) * g.center)
        + jnp.sum(g.half_extent**2)
        + jnp.sum(jnp.sin(g.radius))
    )


def _reference_geometry(tree, positions):
    """The upward pass as it was: children's boxes merged by min/max, in numpy.

    Leaf bounds come from the module's own leaf helpers (particles or Morton
    cells, as ``compute_tree_geometry`` picks them); the merge is a plain
    recursion over the children, independent of the level loop under test.
    """
    from yggdrax import _geometry_impl as gi

    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    ranges = np.asarray(tree.node_ranges)
    dtype = np.dtype(positions.dtype)
    if bool(getattr(tree, "use_morton_geometry", False)):
        leaf_min, leaf_max = gi._compute_leaf_bounds_from_morton(tree)
    else:
        leaf_ranges = jnp.asarray(tree.node_ranges)[num_internal:]
        leaf_min, leaf_max = gi._compute_leaf_bounds(
            positions,
            leaf_ranges[:, 0],
            leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1,
        )
    num_nodes = ranges.shape[0]
    mins = np.zeros((num_nodes, positions.shape[1]), dtype)
    maxs = np.zeros_like(mins)
    mins[num_internal:] = np.asarray(leaf_min).astype(dtype)
    maxs[num_internal:] = np.asarray(leaf_max).astype(dtype)
    left = np.asarray(tree.left_child)
    right = np.asarray(tree.right_child)
    done = np.zeros(num_nodes, bool)
    done[num_internal:] = True

    def visit(node):
        if done[node]:
            return
        visit(left[node])
        visit(right[node])
        mins[node] = np.minimum(mins[left[node]], mins[right[node]])
        maxs[node] = np.maximum(maxs[left[node]], maxs[right[node]])
        done[node] = True

    for node in range(num_internal):
        visit(node)
    # the cell tree's empty padding nodes keep +-finfo.max bounds and overflow
    # here exactly as they do in the library
    with np.errstate(over="ignore", invalid="ignore"):
        half = np.maximum(dtype.type(0.5) * (maxs - mins), dtype.type(1e-6))
        center = dtype.type(0.5) * (mins + maxs)
    return center, half


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_geometry_values_unchanged_bitwise(kind, dtype):
    """Centres and half-extents equal the plain min/max merge, bit for bit."""
    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    tree, positions = _tree(kind, dtype=dtype)
    ref_center, ref_half = _reference_geometry(tree, positions)
    for g in (
        compute_tree_geometry(tree, positions),
        jax.jit(lambda p: compute_tree_geometry(tree, p))(positions),
    ):
        assert np.asarray(g.center).dtype == np.dtype(dtype)
        assert np.array_equal(np.asarray(g.center), ref_center)
        assert np.array_equal(np.asarray(g.half_extent), ref_half)


@pytest.mark.parametrize("kind", _KINDS)
def test_jit_grad_matches_eager_grad(kind):
    tree, positions = _tree(kind)
    loss = lambda p: _loss(tree, p)  # noqa: E731
    eager = jax.grad(loss)(positions)
    compiled = jax.jit(jax.grad(loss))(positions)
    assert float(jnp.linalg.norm(compiled)) > 0.0
    np.testing.assert_allclose(compiled, eager, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", _KINDS)
def test_jit_grad_matches_finite_differences_and_jvp(kind):
    tree, positions = _tree(kind)
    loss = lambda p: _loss(tree, p)  # noqa: E731
    v = jax.random.normal(jax.random.PRNGKey(3), positions.shape)
    vjp_v = float(jnp.vdot(jax.jit(jax.grad(loss))(positions), v))
    jvp_v = float(jax.jit(lambda p: jax.jvp(loss, (p,), (v,))[1])(positions))
    eps = 1e-6
    fd = float((loss(positions + eps * v) - loss(positions - eps * v)) / (2 * eps))
    assert vjp_v == pytest.approx(jvp_v, rel=1e-12)
    assert vjp_v == pytest.approx(fd, rel=1e-7)


def test_jit_grad_float32():
    tree, positions = _tree("radix", dtype=jnp.float32)
    loss = lambda p: _loss(tree, p)  # noqa: E731
    compiled = jax.jit(jax.grad(loss))(positions)
    eager = jax.grad(loss)(positions)
    assert compiled.dtype == jnp.float32
    np.testing.assert_allclose(compiled, eager, rtol=0, atol=1e-5)


def _far_list_loss(tree, positions, backend):
    g = compute_tree_geometry(tree, positions)
    w = mutual_walk_inputs(tree, g, mac_type="dehnen")
    res = dual_tree_walk_mutual(
        w.left_child_full,
        w.right_child_full,
        w.centers,
        w.radii,
        0.6,
        w.root,
        max_pair_queue=1 << 13,
        far_cap=1 << 14,
        near_cap=1 << 14,
        mac_type="dehnen",
        backend=backend,
        pallas_interpret=True,
    )
    live = jnp.arange(res.far_a.shape[0]) < res.far_count
    a = jnp.where(live, res.far_a, 0)
    b = jnp.where(live, res.far_b, 0)
    d2 = jnp.sum((g.center[a] - g.center[b]) ** 2, axis=-1)
    return jnp.sum(jnp.where(live, d2, 0.0)), res.far_count


def test_mutual_walk_backends_differentiate_under_jit():
    """The walk feeds on live geometry; both backends give the same gradient."""
    tree, positions = _tree("radix", n=1200)
    v = jax.random.normal(jax.random.PRNGKey(5), positions.shape)
    grads, jvps = {}, {}
    for backend in ("flat", "pallas"):
        loss = lambda p, b=backend: _far_list_loss(tree, p, b)[0]  # noqa: E731
        grads[backend] = jax.jit(jax.grad(loss))(positions)
        jvps[backend] = float(
            jax.jit(lambda p, f=loss: jax.jvp(f, (p,), (v,))[1])(positions)
        )
    assert int(_far_list_loss(tree, positions, "flat")[1]) > 0
    assert float(jnp.linalg.norm(grads["flat"])) > 0.0
    np.testing.assert_allclose(grads["pallas"], grads["flat"], rtol=0, atol=1e-12)
    assert jvps["pallas"] == pytest.approx(jvps["flat"], rel=1e-12)
    assert float(jnp.vdot(grads["flat"], v)) == pytest.approx(jvps["flat"], rel=1e-10)
