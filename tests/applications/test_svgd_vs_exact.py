"""Tree-accelerated SVGD vs. the exact O(N^2) reference.

* With no far pairs accepted (tight ``theta``) the near field is exact, so the
  tree Stein update equals the exact update to machine precision.
* With far pairs accepted, the monopole far approximation adds an error that
  shrinks as ``theta`` tightens.
* A short SVGD run matches the exact run in distribution (moments) at a
  moderate ``theta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import DualTreeTraversalConfig
from yggdrax.applications.svgd import targets as T
from yggdrax.applications.svgd.exact import exact_phi, run_svgd
from yggdrax.applications.svgd.kernel import median_heuristic
from yggdrax.applications.svgd.sampler import (
    build_svgd_topology,
    run_tree_svgd,
    svgd_phi,
    svgd_phi_from_topology,
)

_CFG = DualTreeTraversalConfig(
    max_pair_queue=1 << 18,
    process_block=32,
    max_interactions_per_node=1 << 14,
    max_neighbors_per_leaf=1 << 14,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("backend", ["radix", "octree", "leaf_kdtree"])
def test_nearfield_phi_is_exact(backend):
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (300, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)
    topo = build_svgd_topology(
        p, theta=0.0, leaf_size=16, backend=backend, traversal_config=_CFG
    )
    assert int(topo.far_tgt_slot.shape[0]) == 0
    tree = svgd_phi_from_topology(p, sc, h, topo)
    rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
    assert rel < 1e-10, f"{backend}: near-field not exact (rel {rel:.2e})"


@pytest.mark.parametrize("dim", [2, 5, 8])
def test_dimension_general_nearfield_exact(dim):
    """The leaf-KD-tree Stein update is exact in arbitrary dimension.

    radix/octree are 3-D only; the default leaf-KD-tree tiles all pairs in any
    dimension, so at theta=0 the tree update equals the exact update to machine
    precision for d != 3 as well.
    """
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (300, dim)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)
    topo = build_svgd_topology(
        p, theta=0.0, leaf_size=16, backend="leaf_kdtree", traversal_config=_CFG
    )
    assert int(topo.far_tgt_slot.shape[0]) == 0
    tree = svgd_phi_from_topology(p, sc, h, topo)
    rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
    assert rel < 1e-10, f"d={dim}: near-field not exact (rel {rel:.2e})"


def test_far_monopole_error_shrinks_with_theta():
    key = jax.random.PRNGKey(0)
    p = jax.random.normal(key, (1500, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)

    prev = None
    saw_far = False
    for theta in (1.0, 0.6, 0.3):
        topo = build_svgd_topology(
            p, theta=theta, leaf_size=8, backend="radix", traversal_config=_CFG
        )
        saw_far = saw_far or int(topo.far_tgt_slot.shape[0]) > 0
        tree = svgd_phi_from_topology(p, sc, h, topo)
        rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
        if prev is not None:
            assert rel <= prev + 1e-9, "error should not grow as theta tightens"
        prev = rel
    assert saw_far, "expected far pairs at this configuration"
    assert prev < 1e-2


def test_distribution_matches_exact_short_run():
    tgt = T.gaussian(jnp.array([1.0, 0.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
    p0 = jax.random.normal(jax.random.PRNGKey(0), (200, 3)) * 0.6
    h = float(median_heuristic(p0))
    pe = run_svgd(p0, tgt.score, h, 0.3, 40)
    pt = run_tree_svgd(
        p0, tgt.score, h, 0.3, 40, theta=0.4, leaf_size=16, traversal_config=_CFG
    )
    # Moments agree within a loose tolerance (tree is an approximation).
    assert jnp.max(jnp.abs(pe.mean(0) - pt.mean(0))) < 0.1
    assert jnp.max(jnp.abs(pe.std(0) - pt.std(0))) < 0.15


# --- WP1: the kernel-aware far field ---------------------------------------
#
# The RBF kernel is effectively zero beyond a few bandwidths, so a node pair
# whose closest possible separation exceeds ``c * h`` contributes nothing and is
# dropped rather than refined or summarised. These tests pin the two properties
# that buys us: the update is still right, and the partition is much smaller.


def _toy_target(name):
    if name == "gaussian":
        return T.gaussian(jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0]))
    if name == "gmm":
        return T.gaussian_mixture(
            jnp.array([[-2.5, 0.0], [2.5, 0.0]]), jnp.array([0.5, 0.5])
        )
    return T.banana(curvature=0.3, scale=2.0)


@pytest.mark.parametrize("name", ["gaussian", "gmm", "banana"])
def test_kernel_cutoff_matches_exact_and_shrinks_the_partition(name):
    """c = 6 costs <= 1e-6 relative and removes most of the near list.

    ``theta = 0`` isolates the cutoff: nothing is accepted as a monopole, so the
    only difference from the exact update is the pairs the cutoff dropped. The
    dropped kernel values are bounded by ``exp(-c^2/2) = 1.5e-8``.
    """
    tgt = _toy_target(name)
    n = 2000
    p = tgt.sample(jax.random.PRNGKey(3), n)
    sc = tgt.score(p)
    h = float(median_heuristic(p))
    ref = exact_phi(p, sc, h)

    base = build_svgd_topology(
        p, theta=0.0, leaf_size=16, backend="radix", traversal_config=_CFG
    )
    cut = build_svgd_topology(
        p,
        theta=0.0,
        leaf_size=16,
        backend="radix",
        traversal_config=_CFG,
        kernel_cutoff=6.0 * h,
    )
    tree = svgd_phi_from_topology(p, sc, h, cut)
    rel = float(jnp.linalg.norm(tree - ref) / jnp.linalg.norm(ref))
    assert rel < 1e-6, f"{name}: cutoff update off by {rel:.2e}"

    # Something must actually have been dropped, or the tolerance above is
    # vacuous. How *much* is dropped is a property of the configuration, not of
    # the policy: a near pair satisfies d < (r_A + r_B) / theta, so it can only
    # exceed the cutoff when the leaves are themselves several bandwidths wide.
    assert (
        cut.near_target_row.shape[0] < base.near_target_row.shape[0]
    ), f"{name}: near pairs unchanged at {base.near_target_row.shape[0]}"
    assert int(cut.num_far_pairs) == 0, "theta=0 accepts no monopole pairs"


def test_kernel_cutoff_collapses_the_far_field():
    """At a working theta the cutoff removes most far entries, not just pairs."""
    # A cloud several cutoffs across. On a cloud only ~2 cutoffs across (the
    # scaling bench's sigma = 1.2 at h = 0.5) almost nothing is droppable at
    # c = 6, because a pair accepted by a size-relative MAC sits at
    # gap ~ (1 - theta) * d and the domain simply does not reach that far.
    p = jax.random.normal(jax.random.PRNGKey(0), (4000, 3)) * 3.0
    sc = p * 0.5
    h = 0.5
    kw = dict(theta=0.5, leaf_size=16, backend="radix", traversal_config=_CFG)

    base = build_svgd_topology(p, **kw)
    cut = build_svgd_topology(p, kernel_cutoff=6.0 * h, **kw)

    assert base.far_tgt_slot.shape[0] > 0, "expected a non-trivial far field"
    assert cut.far_tgt_slot.shape[0] < base.far_tgt_slot.shape[0]

    ref = exact_phi(p, sc, h)
    err_base = float(
        jnp.linalg.norm(svgd_phi_from_topology(p, sc, h, base) - ref)
        / jnp.linalg.norm(ref)
    )
    err_cut = float(
        jnp.linalg.norm(svgd_phi_from_topology(p, sc, h, cut) - ref)
        / jnp.linalg.norm(ref)
    )
    # Dropping pairs the kernel cannot reach must not cost accuracy: the
    # monopole error of the pairs that remain dominates either way.
    assert err_cut <= err_base * 1.5 + 1e-9, f"{err_base:.3e} -> {err_cut:.3e}"


def test_a_cutoff_wider_than_the_domain_changes_nothing():
    """A cutoff no pair can exceed must reproduce the plain MAC partition.

    This is the guard on the tagged-far-pair plumbing: with the policy installed
    but nothing droppable, the far list must be the same list the built-in MAC
    produces, in effect as well as in count.
    """
    p = jax.random.normal(jax.random.PRNGKey(1), (1500, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    kw = dict(theta=0.5, leaf_size=16, backend="radix", traversal_config=_CFG)

    base = build_svgd_topology(p, **kw)
    wide = build_svgd_topology(p, kernel_cutoff=1e6, **kw)

    assert int(wide.num_far_pairs) == int(base.num_far_pairs)
    assert wide.far_tgt_slot.shape[0] == base.far_tgt_slot.shape[0]
    assert wide.near_target_row.shape[0] == base.near_target_row.shape[0]

    a = svgd_phi_from_topology(p, sc, h, base)
    b = svgd_phi_from_topology(p, sc, h, wide)
    assert float(jnp.max(jnp.abs(a - b))) < 1e-12


def test_svgd_phi_cutoff_bandwidths_is_the_c_times_h_convention():
    """``cutoff_bandwidths=c`` on the one-shot helper means ``kernel_cutoff=c*h``."""
    p = jax.random.normal(jax.random.PRNGKey(2), (800, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    kw = dict(theta=0.4, leaf_size=16, backend="radix", traversal_config=_CFG)

    via_helper = svgd_phi(p, sc, h, cutoff_bandwidths=6.0, **kw)
    topo = build_svgd_topology(p, kernel_cutoff=6.0 * h, **kw)
    via_topo = svgd_phi_from_topology(p, sc, h, topo)
    assert float(jnp.max(jnp.abs(via_helper - via_topo))) < 1e-12


# --- the two near-field accumulations --------------------------------------
#
# The near field can be summed by scattering each unordered pair's two
# directions, or by a segmented reduction over the directed list. Which is
# faster is a property of the dtype (float32's `.at[].add()` is 43x float64's
# under index contention), so both exist. They must agree.


@pytest.mark.parametrize("backend", ["radix", "leaf_kdtree"])
def test_accumulations_agree(backend):
    p = jax.random.normal(jax.random.PRNGKey(5), (1200, 3)) * 1.2
    sc = p * 0.5
    h = 0.6
    topo = build_svgd_topology(
        p, theta=0.4, leaf_size=16, backend=backend, traversal_config=_CFG
    )
    by_scatter = svgd_phi_from_topology(p, sc, h, topo, accumulate="scatter")
    by_segment = svgd_phi_from_topology(p, sc, h, topo, accumulate="segment")
    assert float(jnp.max(jnp.abs(by_scatter - by_segment))) < 1e-12

    # "auto" is one of the two, never a third thing.
    by_auto = svgd_phi_from_topology(p, sc, h, topo, accumulate="auto")
    assert float(jnp.max(jnp.abs(by_auto - by_scatter))) < 1e-12


def test_segment_accumulation_is_exact_at_theta_zero():
    """The segmented path is a different summation order, not a different sum."""
    p = jax.random.normal(jax.random.PRNGKey(6), (900, 3)) * 1.2
    sc = p * 0.5
    h = float(median_heuristic(p))
    topo = build_svgd_topology(
        p, theta=0.0, leaf_size=16, backend="radix", traversal_config=_CFG
    )
    ref = exact_phi(p, sc, h)
    out = svgd_phi_from_topology(p, sc, h, topo, accumulate="segment")
    assert float(jnp.linalg.norm(out - ref) / jnp.linalg.norm(ref)) < 1e-10


def test_unknown_accumulation_is_rejected():
    p = jax.random.normal(jax.random.PRNGKey(7), (200, 3))
    topo = build_svgd_topology(
        p, theta=0.4, leaf_size=16, backend="radix", traversal_config=_CFG
    )
    with pytest.raises(ValueError, match="accumulate must be"):
        svgd_phi_from_topology(p, p * 0.5, 0.6, topo, accumulate="nonsense")


def test_directed_pair_list_is_sorted_and_twice_the_halved_one():
    """What makes the segmented reduction segmented."""
    p = jax.random.normal(jax.random.PRNGKey(8), (1500, 3)) * 1.2
    topo = build_svgd_topology(
        p, theta=0.4, leaf_size=16, backend="radix", traversal_config=_CFG
    )
    directed = np.asarray(topo.near_dir_target)
    assert directed.shape[0] == 2 * int(topo.near_target_row.shape[0])
    assert directed.shape[0] == int(topo.num_near_leaf_pairs)
    assert np.all(np.diff(directed) >= 0), "target rows must be non-decreasing"


# --- static capacities -----------------------------------------------------
#
# The partition's lengths are data dependent, so a per-step-rebuild sampler
# retraces every step: six rebuilds give six (near pairs, M) signatures, and 8
# steps at N=1e4 cost 35.5 s against 2.2 s padded. Padding must be inert.


@pytest.mark.parametrize("capacity", ["exact", "bucket", "pow2", 1 << 18])
@pytest.mark.parametrize("accumulate", ["scatter", "segment"])
def test_capacity_padding_is_inert(capacity, accumulate):
    p = jax.random.normal(jax.random.PRNGKey(11), (1500, 3)) * 1.2
    sc, h = p * 0.5, 0.6
    kw = dict(theta=0.5, leaf_size=16, backend="radix", traversal_config=_CFG)
    base = svgd_phi_from_topology(
        p, sc, h, build_svgd_topology(p, capacity="exact", **kw), accumulate="scatter"
    )
    topo = build_svgd_topology(p, capacity=capacity, **kw)
    got = svgd_phi_from_topology(p, sc, h, topo, accumulate=accumulate)
    assert float(jnp.max(jnp.abs(got - base))) < 1e-12


def test_padding_reduces_shape_churn_and_an_explicit_capacity_removes_it():
    """The property the padding exists for -- and the limit of the cheap policies.

    ``"pow2"`` and ``"bucket"`` pin a shape only while the count stays inside one
    bucket; a count sitting on a boundary still flips between rebuilds, as *M*
    does here (2048 vs 4096 at N = 4000). Only an explicit capacity is a
    guarantee, which is what a long run should pin after one probe build.
    """
    p0 = jax.random.normal(jax.random.PRNGKey(12), (4000, 3)) * 1.2
    kw = dict(theta=0.5, leaf_size=32, backend="radix", traversal_config=_CFG)

    def shapes_for(cap):
        seen = set()
        for step in range(4):
            p = p0 + 0.03 * step * jax.random.normal(
                jax.random.PRNGKey(200 + step), p0.shape
            )
            t = build_svgd_topology(p, capacity=cap, **kw)
            seen.add((int(t.near_target_row.shape[0]), int(t.far_tgt_slot.shape[0])))
        return seen

    exact = shapes_for("exact")
    assert len(exact) > 1, "the test problem must actually vary"
    assert len(shapes_for("pow2")) < len(exact)
    assert len(shapes_for(1 << 16)) == 1, "an explicit capacity is the guarantee"


def test_capacity_smaller_than_the_partition_is_rejected():
    p = jax.random.normal(jax.random.PRNGKey(13), (1500, 3)) * 1.2
    with pytest.raises(ValueError, match="smaller than"):
        build_svgd_topology(
            p,
            theta=0.5,
            leaf_size=16,
            backend="radix",
            traversal_config=_CFG,
            capacity=4,
        )


def test_unknown_capacity_is_rejected():
    p = jax.random.normal(jax.random.PRNGKey(14), (400, 3))
    with pytest.raises(ValueError, match="capacity must be"):
        build_svgd_topology(
            p,
            theta=0.5,
            leaf_size=16,
            backend="radix",
            traversal_config=_CFG,
            capacity="enormous",
        )
