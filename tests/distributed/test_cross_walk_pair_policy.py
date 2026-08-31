"""The cross walk's ``pair_policy`` hook, and the one thing it must not copy.

The self walk has taken a solver-owned ``pair_policy`` for a long time: a caller
replaces the geometric MAC's verdict with its own, evaluated per pair against state
the traversal knows nothing about. The cross walk had no such hook, so a distributed
solver could carry its criterion on the local walk and nowhere else -- and on a mesh
the cross-domain half is the larger half of the near-field work.

What is deliberately NOT copied from the self walk is the two-orientation agreement
rule. ``_resolve_pair_actions`` calls the policy twice, swapping target and source,
and accepts only when both agree; that is right for a self traversal, which emits
both directions of every pair. This walk is directed -- ordered ``(target, source)``,
never swapped -- and the two trees are disjoint index spaces, so the swapped call
would index a source-tree array with a target-tree node id. That reads the wrong node
rather than raising, which is why the forward-only rule is asserted here rather than
left as a comment.

The policies below are chosen so their verdict cannot be confused with the geometric
MAC's: one accepts nothing, one accepts every live pair, and one accepts on a node-id
predicate that has no geometric meaning at all. A hook that silently fell back to
``_default_pair_actions_only`` would reproduce the geometric counts and pass a test
written against anything subtler.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_cross_walk_pair_policy.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from yggdrax._interactions_impl import _ACTION_ACCEPT, _ACTION_NEAR, _ACTION_REFINE
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree import Tree

LEAF = 8
THETA = 0.5
CAPS = dict(
    max_interactions_per_node=2048,
    max_neighbors_per_leaf=2048,
    max_pair_queue=131072,
)


def _tree(points):
    tree = Tree.from_particles(
        jnp.asarray(points),
        jnp.ones((points.shape[0],)),
        tree_type="radix",
        return_reordered=True,
        leaf_size=LEAF,
    )
    order = np.asarray(tree.particle_indices)
    geom = compute_tree_geometry(tree, jnp.asarray(points)[order], max_leaf_size=LEAF)
    return tree, geom


@pytest.fixture(scope="module")
def two_trees():
    """A target cloud and a source cloud far enough apart to be separate domains."""
    rng = np.random.default_rng(20260831)
    target = rng.normal(size=(256, 3))
    source = rng.normal(size=(256, 3)) + np.array([6.0, 0.0, 0.0])
    return _tree(target), _tree(source)


def _walk(two_trees, **kwargs):
    (t_tree, t_geom), (s_tree, s_geom) = two_trees
    return dual_tree_walk_cross_impl(
        t_tree, t_geom, s_tree, s_geom, THETA, mac_type="dehnen", **CAPS, **kwargs
    )


def _counts(res):
    return int(np.asarray(res.far_pair_count)), int(np.asarray(res.near_pair_count))


def test_the_geometric_walk_is_unchanged_when_no_policy_is_given(two_trees):
    """``pair_policy=None`` must reproduce the walk byte for byte.

    The hook is additive; if the default path moved, every existing cross-walk
    result moved with it.
    """

    baseline = _walk(two_trees)
    explicit = _walk(two_trees, pair_policy=None, policy_state=None)

    assert _counts(baseline) == _counts(explicit)
    np.testing.assert_array_equal(
        np.asarray(baseline.interaction_sources),
        np.asarray(explicit.interaction_sources),
    )
    np.testing.assert_array_equal(
        np.asarray(baseline.neighbor_indices), np.asarray(explicit.neighbor_indices)
    )


def test_a_policy_that_refuses_everything_empties_the_far_list(two_trees):
    """The policy's verdict must override the geometric MAC's, not be ANDed with it.

    The geometric arm accepts a real far field on this IC, so a far count of zero
    can only come from the policy actually deciding.
    """

    def refuse_all(_state, **pair):
        shape = pair["valid_pairs"].shape
        return (
            jnp.full(shape, _ACTION_REFINE, dtype=jnp.int32),
            jnp.full(shape, -1, dtype=jnp.int32),
        )

    geometric_far, _ = _counts(_walk(two_trees))
    assert geometric_far > 0, "the geometric arm has no far field; nothing is measured"

    far, near = _counts(_walk(two_trees, pair_policy=refuse_all, policy_state=None))
    assert far == 0, f"a refuse-everything policy still accepted {far} far pairs"
    assert near == 0, "REFINE must not fall through to the near list either"


def test_a_policy_that_accepts_at_the_root_collapses_the_far_list_to_one_pair(
    two_trees,
):
    """Accepting the very first pair must stop the traversal descending.

    This is the sharpest statement that the hook runs BEFORE the refinement
    decision rather than filtering its output: the wavefront is seeded with
    ``(target_root, source_root)``, so accepting every live pair can only ever
    produce that single pair.
    """

    def accept_all(_state, **pair):
        shape = pair["valid_pairs"].shape
        return (
            jnp.where(
                pair["valid_pairs"],
                jnp.asarray(_ACTION_ACCEPT, dtype=jnp.int32),
                jnp.asarray(_ACTION_REFINE, dtype=jnp.int32),
            ),
            jnp.zeros(shape, dtype=jnp.int32),
        )

    far, near = _counts(_walk(two_trees, pair_policy=accept_all, policy_state=None))
    assert far == 1, f"accepting root-vs-root gave {far} far pairs, not 1"
    assert near == 0


def test_the_policy_is_evaluated_forward_only_and_never_with_the_trees_swapped(
    two_trees,
):
    """A policy keyed on the SOURCE tree's node ids must see only source ids.

    The self walk evaluates its policy in both orientations and requires them to
    agree. Doing that here would feed target-tree node ids into ``source_nodes``,
    which -- because the trees are disjoint index spaces of different sizes -- reads
    a different node instead of failing. So this policy accepts only when the source
    id is at least the source tree's internal-node count, i.e. only for source
    LEAVES, and separately records every id it was shown.

    If a reverse call happened, ``source_nodes`` would carry target ids and the
    recorded maximum would exceed the source tree's node count.
    """

    (_t_tree, _t_geom), (s_tree, s_geom) = two_trees
    s_total = int(np.asarray(s_tree.node_ranges).shape[0])
    s_internal = int(np.asarray(s_tree.left_child).shape[0])

    seen: list[np.ndarray] = []

    def source_leaves_only(_state, **pair):
        src = pair["source_nodes"]
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), src)
        accept = pair["valid_pairs"] & (src >= s_internal)
        actions = jnp.where(
            accept,
            jnp.asarray(_ACTION_ACCEPT, dtype=jnp.int32),
            jnp.asarray(_ACTION_REFINE, dtype=jnp.int32),
        )
        return actions, jnp.zeros(src.shape, dtype=jnp.int32)

    res = _walk(two_trees, pair_policy=source_leaves_only, policy_state=None)

    assert seen, "the policy was never called"
    observed = np.concatenate([s.ravel() for s in seen])
    assert observed.max() < s_total, (
        f"the policy saw source id {observed.max()}, beyond the source tree's "
        f"{s_total} nodes -- it was called with the trees swapped"
    )

    sources = np.asarray(res.interaction_sources)
    live = sources[sources >= 0]
    assert live.size > 0, "nothing was accepted; the predicate matched no pair"
    assert (
        live.min() >= s_internal
    ), "an accepted source is an internal node, but the policy accepted only leaves"


def test_a_leaf_leaf_pair_the_policy_marks_near_reaches_the_near_list(two_trees):
    """NEAR must be honoured, not just ACCEPT and REFINE.

    Without this, a policy could push its whole rejected set into the near list by
    accident and nothing here would notice.
    """

    (_t, _tg), (s_tree, _sg) = two_trees
    s_internal = int(np.asarray(s_tree.left_child).shape[0])

    def near_at_the_leaves(_state, **pair):
        both_leaves = pair["valid_pairs"] & pair["target_leaf"] & pair["source_leaf"]
        actions = jnp.where(
            both_leaves,
            jnp.asarray(_ACTION_NEAR, dtype=jnp.int32),
            jnp.asarray(_ACTION_REFINE, dtype=jnp.int32),
        )
        return actions, jnp.zeros(pair["valid_pairs"].shape, dtype=jnp.int32)

    far, near = _counts(
        _walk(two_trees, pair_policy=near_at_the_leaves, policy_state=None)
    )
    assert far == 0, "nothing was accepted, so the far list must be empty"
    assert near > 0, "leaf-leaf pairs marked NEAR never reached the near list"

    res = _walk(two_trees, pair_policy=near_at_the_leaves, policy_state=None)
    neighbours = np.asarray(res.neighbor_indices)
    live = neighbours[neighbours >= 0]
    assert live.min() >= s_internal, "a near source is not a source-tree leaf"
