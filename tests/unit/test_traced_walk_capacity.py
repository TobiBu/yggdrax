"""The traced dual-tree walk must size its wavefront from ``max_pair_queue``.

``_run_dual_tree_walk_raw`` sizes the traversal wavefront by trying a ladder of
capacities and retrying whenever the walk reports an overflow. Reading that
report needs a concrete flag, so under an outer trace (``jax.jit``,
``shard_map``) the ladder cannot be climbed at all: it runs its first rung and
leaves. The first rung is ``max(1024, process_block * 16)``.

That made ``max_pair_queue`` inert on every traced path and ``process_block`` --
a vectorisation knob -- the thing that decided how much of the tree was
traversed. Measured on a 256-leaf disc before the fix: 204 near pairs against
the eager ladder's 43 854, unchanged across four decades of ``max_pair_queue``,
and identical at ``process_block`` 32 and 64 because both floor to 1024. At 2048
leaves it was 36 pairs against 1 117 432.

These tests pin the contract that replaced it:

* the traced walk agrees with the eager ladder when the caller's capacity is
  sufficient (it uses that capacity, not the first rung);
* ``process_block`` no longer changes what the traced walk finds;
* an insufficient capacity is still *reported*, so a caller that can retry --
  the distributed driver reduces these flags across devices -- has something
  true to retry on;
* the traced capacity does not depend on the process-global ladder cache, so a
  traced result cannot change depending on whether something ran eagerly first.

Eager behaviour is deliberately untested-for-change here: it must not move, and
the first test compares against it as ground truth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._interactions_impl import _DUAL_TREE_QUEUE_CACHE
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import (
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
)
from yggdrax.tree import Tree

#: Small enough to run in a unit test, dense enough that the 1024-pair first rung
#: truncates hard: 256 leaves, ~36k near pairs, ~250 of them reachable at the rung.
_N = 1024
_LEAF = 4

#: Well clear of the wavefront the tree actually needs, so a truncation here can
#: only come from the queue and not from the far/near list capacities.
_ROOMY = dict(max_interactions_per_node=4096, max_neighbors_per_leaf=4096)


def _disc(n: int, seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """A thin disc: flat enough that the walk's wavefront gets genuinely wide."""
    rng = np.random.default_rng(seed)
    r = 10.0 * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack(
        [r * np.cos(th), r * np.sin(th), rng.normal(scale=0.2, size=n)], axis=1
    )
    return pos.astype(np.float64), rng.uniform(0.8, 1.2, n).astype(np.float64)


@pytest.fixture(scope="module")
def disc_tree() -> tuple[object, object, int]:
    """A tree whose walk needs more than the 1024-pair first rung."""
    pos, mass = _disc(_N)
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.asarray(mass), leaf_size=_LEAF, tree_type="radix"
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=_LEAF)
    topology = tree.topology
    num_leaves = int(topology.parent.shape[0]) - int(topology.left_child.shape[0])
    return topology, geometry, num_leaves


def _walk(
    topology: object,
    geometry: object,
    *,
    jit: bool,
    process_block: int = 64,
    max_pair_queue: int = 1 << 15,
    cold_cache: bool = True,
    **caps: int,
) -> dict:
    """One walk, returning its pair counts and overflow flags as Python scalars."""
    config = DualTreeTraversalConfig(
        max_pair_queue=max_pair_queue,
        process_block=process_block,
        **{**_ROOMY, **caps},
    )

    def body(topology: object, geometry: object) -> tuple:
        _far, _near, result = build_interactions_and_neighbors(
            topology,
            geometry,
            theta=0.4,
            traversal_config=config,
            mac_type="dehnen",
            return_result=True,
        )
        return (
            result.near_pair_count,
            result.far_pair_count,
            result.queue_overflow,
            result.far_overflow,
            result.near_overflow,
        )

    if cold_cache:
        _DUAL_TREE_QUEUE_CACHE.clear()
    near, far, queue_ovf, far_ovf, near_ovf = (jax.jit(body) if jit else body)(
        topology, geometry
    )
    return {
        "near": int(near),
        "far": int(far),
        "queue_overflow": bool(queue_ovf),
        "far_overflow": bool(far_ovf),
        "near_overflow": bool(near_ovf),
    }


def test_traced_walk_matches_the_eager_ladder(disc_tree):
    """The headline: a sufficient ``max_pair_queue`` is enough under trace too.

    Eager is ground truth -- its flags are concrete, so its ladder converges on a
    capacity that fits. Before the fix the traced arm returned ~0.6 % of these
    pairs at the same capacity.
    """
    topology, geometry, num_leaves = disc_tree
    traced = _walk(topology, geometry, jit=True)
    eager = _walk(topology, geometry, jit=False)

    # Vacuity guard: a tree that needs only the first rung would let a broken
    # traced path pass this file, so assert the premise the file is built on.
    assert num_leaves >= 256, f"{num_leaves} leaves is too few to overflow the rung"
    assert eager["near"] > 10_000, (
        f"only {eager['near']} near pairs: this tree no longer stresses the "
        "wavefront, so the comparison below proves nothing"
    )

    assert traced == eager, (
        "the traced walk disagrees with the eager ladder at the same capacity: "
        f"traced {traced} vs eager {eager}"
    )


def test_process_block_does_not_change_what_the_traced_walk_finds(disc_tree):
    """``process_block`` is a vectorisation width, not a traversal capacity.

    The sharpest signature of the old behaviour was that 32 and 64 agreed exactly
    (both floored to a 1024-pair wavefront) while 128 and 256 each found strictly
    more. Nothing but a capacity derived from ``process_block`` produces that, so
    testing its absence is the cleanest guard against a regression.
    """
    topology, geometry, _num_leaves = disc_tree
    counts = {
        block: _walk(topology, geometry, jit=True, process_block=block)
        for block in (32, 512)
    }
    assert counts[32]["near"] > 0, "vacuous: the walk found no near pairs at all"
    assert counts[32] == counts[512], (
        "process_block still decides how much of the tree the traced walk sees: "
        f"{counts[32]} at 32 against {counts[512]} at 512"
    )


def test_a_traced_overflow_reaches_the_caller(disc_tree):
    """An insufficient capacity must still be reported, not declared a success.

    This is the other half of the contract: the fix makes ``max_pair_queue`` mean
    something, which is only useful if a caller can tell that the value it chose
    was too small. The distributed driver reduces exactly this flag across
    devices to decide whether to grow its caps and rebuild.
    """
    topology, geometry, _num_leaves = disc_tree
    starved = _walk(topology, geometry, jit=True, max_pair_queue=64)
    roomy = _walk(topology, geometry, jit=True)

    assert starved["queue_overflow"], (
        "a 64-pair wavefront on a 256-leaf tree reported no queue overflow, so a "
        "truncation on the traced path is invisible to its caller"
    )
    assert starved["near"] < roomy["near"], (
        "the starved walk found as much as the roomy one, so it did not actually "
        "truncate and the flag above says nothing"
    )


def test_the_traced_capacity_ignores_the_ladder_cache(disc_tree, monkeypatch):
    """A traced result must not depend on what ran eagerly earlier in the process.

    ``_run_dual_tree_walk_raw`` memoises the capacity its ladder settled on, keyed
    on tree shape and capacities. The traced path cannot verify a capacity, so it
    must not inherit one: otherwise the same jitted callable compiles to different
    wavefront shapes depending on process history, and a before/after measurement
    silently compares two different programs. (It did, once.)
    """
    topology, geometry, _num_leaves = disc_tree
    import yggdrax._interactions_impl as impl

    consulted: list[tuple] = []
    real_get = impl._get_cached_queue_capacity

    def spy(cache_key: tuple):
        consulted.append(cache_key)
        return real_get(cache_key)

    monkeypatch.setattr(impl, "_get_cached_queue_capacity", spy)

    cold = _walk(topology, geometry, jit=True)
    assert not consulted, "the traced walk read the ladder cache"

    # The eager arm both warms the cache and proves the spy is wired to a name
    # that is still called -- without it this test would pass if the cache lookup
    # were merely renamed.
    _walk(topology, geometry, jit=False, cold_cache=False)
    assert consulted, "the cache lookup was never called, so the guard is vacuous"

    warm = _walk(topology, geometry, jit=True, cold_cache=False)
    assert (
        warm == cold
    ), f"a warm ladder cache changed the traced result: {warm} against {cold}"
