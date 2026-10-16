"""The jitted cross walk the tests call standalone.

The library exported this wrapper as ``yggdrax.distributed.dual_tree_walk_cross``
until the 2026-10 cleanup (Y3) removed it: nothing outside the tests called it,
and jaccpot calls :func:`~yggdrax.distributed.cross_walk.dual_tree_walk_cross_impl`
inside its own transforms.
"""

from functools import partial

import jax

from yggdrax.distributed import dual_tree_walk_cross_impl

dual_tree_walk_cross = partial(
    jax.jit,
    static_argnames=(
        "max_interactions_per_node",
        "max_neighbors_per_leaf",
        "max_pair_queue",
        "mac_type",
        "separation_floor",
        "collect_far",
        "collect_near",
        # A Python callable, so it must be static: traced, it would be flattened to
        # nothing and the walk would silently fall back to the geometric MAC.
        "pair_policy",
    ),
)(dual_tree_walk_cross_impl)
