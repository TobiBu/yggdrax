"""``dual_tree_walk_cross_impl(target_node_active=..., source_node_active=...)``.

The cross-walk form of ``tests/test_mutual_walk_node_active.py``, split in two
masks because this walk spans two index spaces.

Padding leaves of a capacity-padded cell partition sit at one degenerate centre
with a ~0 radius. That does something WORSE across two trees than within one: in
the self walk they only fail the MAC against each other, but here every padding
node is a distinct point that is near to some of the other tree and far from the
rest, so it emits spurious pairs of BOTH kinds and against the whole live tree.
That is the failure that made a cross-import cap sized for the real import
overflow at the entire live leaf count.

The mask must not buy that by pruning real work, so the last test is the control:
the admissibility partition of ``test_cross_walk.py`` -- far sources from a target
leaf and its ancestors plus its near sources must tile every source particle
exactly once -- has to still hold on the masked padded trees. It bites: killing one
live source leaf breaks 86 of 87 target leaves, one live internal source node 84,
and a target mask that is not ancestor-closed 2. It is also BLIND to the flood
(padding nodes carry no particles, so they cover nothing and an unmasked walk
partitions just as well), which is why the two tests are both here.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_cross_walk_node_active.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax._cell_partition import adaptive_cell_leaf_partition
from yggdrax._tree_impl import (
    RadixTree,
    _build_balanced_bucket_structure,
    reorder_particles_by_indices,
)
from yggdrax.bounds import infer_bounds
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.geometry import compute_tree_geometry
from yggdrax.morton import morton_encode

_LEAF = 8
_THETA = 0.5
_MAC = "bh"


def _cells_tree(points, bounds, leaf_size=_LEAF, capacity=None):
    """Cell leaves in a balanced bucket structure, padded to ``capacity``.

    ``capacity=None`` picks the next power of two above ``1.5 x`` the live leaf
    count, i.e. a shard with real padding; ``capacity=k`` gives the unpadded tree.
    """
    P = jnp.asarray(points)
    n = P.shape[0]
    M = jnp.ones((n,), P.dtype)
    codes = morton_encode(P, bounds)
    order = jnp.argsort(codes, stable=True)
    sorted_codes = codes[order]
    probe = adaptive_cell_leaf_partition(sorted_codes, leaf_size=leaf_size, capacity=n)
    k = int(probe.num_leaves)
    cap = int(capacity or (1 << int(np.ceil(np.log2(1.5 * k)))))
    part = adaptive_cell_leaf_partition(sorted_codes, leaf_size=leaf_size, capacity=cap)
    starts = np.asarray(part.leaf_starts).astype(np.int64)
    ends = np.asarray(part.leaf_ends).astype(np.int64)
    (
        parent,
        left,
        right,
        lil,
        ril,
        node_ranges,
        node_level,
        level_offsets,
        nodes_by_level,
        num_levels,
    ) = _build_balanced_bucket_structure(starts, ends)
    ps, _ms, _inv = reorder_particles_by_indices(P, M, order)
    I = jnp.int32
    topo = RadixTree(
        parent=jnp.asarray(parent, I),
        left_child=jnp.asarray(left, I),
        right_child=jnp.asarray(right, I),
        left_is_leaf=jnp.asarray(lil),
        right_is_leaf=jnp.asarray(ril),
        particle_indices=jnp.asarray(order, I),
        morton_codes=sorted_codes,
        node_ranges=jnp.asarray(node_ranges, I),
        num_particles=n,
        num_internal_nodes=cap - 1,
        node_level=jnp.asarray(node_level, I),
        level_offsets=jnp.asarray(level_offsets, I),
        nodes_by_level=jnp.asarray(nodes_by_level, I),
        num_levels=jnp.asarray(num_levels, I),
        bounds_min=jnp.asarray(bounds[0], P.dtype),
        bounds_max=jnp.asarray(bounds[1], P.dtype),
        leaf_codes=sorted_codes[jnp.asarray(np.minimum(starts, n - 1), I)],
        leaf_depths=jnp.asarray(part.leaf_depths, I),
        use_morton_geometry=jnp.asarray(False),
        leaf_size=leaf_size,
    )
    geom = compute_tree_geometry(topo, ps, max_leaf_size=leaf_size)
    return topo, geom, k, cap


def _active(tree):
    """A node is live exactly when its particle range is non-empty."""
    r = np.asarray(tree.node_ranges)
    return jnp.asarray(r[:, 1] >= r[:, 0])


def _two_domains(seed=0, n_t=300, n_s=300):
    rng = np.random.default_rng(seed)
    tgt = rng.uniform(-1.0, 0.2, size=(n_t, 3)).astype(np.float32)
    src = rng.uniform(-0.2, 1.0, size=(n_s, 3)).astype(np.float32)
    bounds = infer_bounds(jnp.asarray(np.concatenate([tgt, src], axis=0)))
    return tgt, src, bounds


def _padded_pair(seed=0, capacity=None):
    tgt, src, bounds = _two_domains(seed)
    t = _cells_tree(tgt, bounds, capacity=capacity)
    s = _cells_tree(src, bounds, capacity=capacity)
    return t, s


def _walk(t, s, t_active, s_active, *, kf=512, kn=512, queue=1 << 14):
    t_tree, t_geom, _tk, _tcap = t
    s_tree, s_geom, _sk, _scap = s
    return dual_tree_walk_cross_impl(
        t_tree,
        t_geom,
        s_tree,
        s_geom,
        _THETA,
        mac_type=_MAC,
        max_interactions_per_node=kf,
        max_neighbors_per_leaf=kn,
        max_pair_queue=queue,
        target_node_active=t_active,
        source_node_active=s_active,
    )


def _pairs(res, t_tree):
    """Far pairs as (target node, source node) and near pairs as (target leaf node, source node)."""
    tt = np.asarray(res.interaction_targets)
    ss = np.asarray(res.interaction_sources)
    m = tt >= 0
    far = set(zip(tt[m].tolist(), ss[m].tolist()))

    leaf_nodes = np.asarray(res.leaf_indices)
    off = np.asarray(res.neighbor_offsets)
    idx = np.asarray(res.neighbor_indices)
    cnt = np.asarray(res.neighbor_counts)
    near = set()
    for row, node in enumerate(leaf_nodes.tolist()):
        o, c = int(off[row]), int(cnt[row])
        near.update((node, int(v)) for v in idx[o : o + c])
    return far, near


def test_the_active_mask_is_ancestor_closed():
    # The walk never refines a dead node, so an empty-range internal node must not
    # sit above a live leaf. This is the premise the mask's contract rests on.
    (t_tree, _g, k, cap), _s = _padded_pair()
    parent = np.asarray(t_tree.parent)
    active = np.asarray(_active(t_tree))
    assert cap > k and not active.all()
    root = int(np.argmin(parent))
    for node in np.flatnonzero(active).tolist():
        while node != root:
            node = int(parent[node])
            assert active[node]


def test_padding_nodes_are_dead_with_the_mask_and_flood_without_it():
    t, s = _padded_pair()
    t_tree, s_tree = t[0], s[0]
    t_act, s_act = _active(t_tree), _active(s_tree)
    t_dead = set(np.flatnonzero(~np.asarray(t_act)).tolist())
    s_dead = set(np.flatnonzero(~np.asarray(s_act)).tolist())
    assert t_dead and s_dead

    res_m = _walk(t, s, t_act, s_act)
    assert not (
        bool(res_m.queue_overflow)
        or bool(res_m.far_overflow)
        or bool(res_m.near_overflow)
    )
    far_m, near_m = _pairs(res_m, t_tree)
    assert far_m and near_m
    # with the mask: nothing emitted touches a dead node on either side
    assert not any(a in t_dead or b in s_dead for a, b in far_m | near_m)

    # without it the padding nodes emit against the live tree, both kinds
    res_0 = _walk(t, s, None, None, kf=1 << 12, kn=1 << 12, queue=1 << 17)
    assert not (
        bool(res_0.queue_overflow)
        or bool(res_0.far_overflow)
        or bool(res_0.near_overflow)
    ), "widen the unmasked capacities: the comparison below needs full lists"
    far_0, near_0 = _pairs(res_0, t_tree)
    assert any(a in t_dead or b in s_dead for a, b in near_0)
    assert any(a in t_dead or b in s_dead for a, b in far_0)

    # and the live-only part of the unmasked lists is what the mask keeps
    def live_only(pairs):
        return {p for p in pairs if p[0] not in t_dead and p[1] not in s_dead}

    assert near_m == live_only(near_0)
    # a dead ancestor is never split, so the unmasked walk can reach live pairs
    # below one that the masked walk does not -- far is a subset, not an equality
    assert far_m <= live_only(far_0)


def test_unmasked_padding_overflows_the_caps_the_mask_fits():
    """The lane failure in miniature: caps sized for the real import do not hold.

    Fit ``max_interactions_per_node`` and ``max_neighbors_per_leaf`` to the masked
    walk's own worst row, keep a queue wide enough that neither run is cut short by
    it, and the unmasked walk overflows on the padding alone. Note the queue is
    the FIRST thing the padding blows in practice -- at the walk's default queue
    the unmasked run here never reaches a near decision at all -- so this fixes the
    queue to isolate the claim about the output caps.
    """
    t, s = _padded_pair()
    t_act, s_act = _active(t[0]), _active(s[0])
    wide_queue = 1 << 17

    ref = _walk(t, s, t_act, s_act, kf=1 << 12, kn=1 << 12, queue=wide_queue)
    kf = int(np.asarray(ref.interaction_counts).max())
    kn = int(np.asarray(ref.neighbor_counts).max())
    assert kf > 0 and kn > 0

    fitted = _walk(t, s, t_act, s_act, kf=kf, kn=kn, queue=wide_queue)
    assert not (
        bool(fitted.queue_overflow)
        or bool(fitted.far_overflow)
        or bool(fitted.near_overflow)
    )

    blown = _walk(t, s, None, None, kf=kf, kn=kn, queue=wide_queue)
    assert bool(blown.far_overflow) or bool(blown.near_overflow)

    # and where the blowup lives: the padding multiplies the per-leaf NEAR demand,
    # which is the cap the cross import is sized on
    loose = _walk(t, s, None, None, kf=1 << 12, kn=1 << 12, queue=wide_queue)
    assert int(np.asarray(loose.neighbor_counts).max()) > 3 * kn
    assert int(loose.near_pair_count) > 20 * int(ref.near_pair_count)


def test_all_active_mask_is_identical_to_no_mask():
    # An all-true mask must change nothing, field by field -- including on padded
    # trees, where "nothing" means it floods exactly as the unmasked walk does.
    t, s = _padded_pair()
    wide = dict(kf=1 << 12, kn=1 << 12, queue=1 << 17)
    ones_t = jnp.ones((int(t[0].parent.shape[0]),), bool)
    ones_s = jnp.ones((int(s[0].parent.shape[0]),), bool)
    a = _walk(t, s, ones_t, ones_s, **wide)
    b = _walk(t, s, None, None, **wide)
    for field in a._fields:
        assert np.array_equal(
            np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        ), field


def test_the_jitted_wrapper_takes_the_masks_too():
    """`dual_tree_walk_cross` is the entry point jaccpot imports, not the impl.

    Threading a parameter through the impl alone has left it unreachable from the
    public name before in this codebase, and a test that calls the impl directly
    does not notice.
    """
    from yggdrax.distributed import dual_tree_walk_cross

    t, s = _padded_pair()
    res = dual_tree_walk_cross(
        t[0],
        t[1],
        s[0],
        s[1],
        _THETA,
        mac_type=_MAC,
        max_interactions_per_node=512,
        max_neighbors_per_leaf=512,
        max_pair_queue=1 << 14,
        target_node_active=_active(t[0]),
        source_node_active=_active(s[0]),
    )
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )
    t_dead = set(np.flatnonzero(~np.asarray(_active(t[0]))).tolist())
    far, near = _pairs(res, t[0])
    assert far and near
    assert not any(a in t_dead for a, _ in far | near)


def test_mask_shape_is_checked():
    # Unlike ``policy_state``, these are shape-checked: a mask built over the wrong
    # tree raises instead of gathering the wrong node.
    t, s = _padded_pair()
    t_n = int(t[0].parent.shape[0])
    s_n = int(s[0].parent.shape[0])
    with pytest.raises(ValueError, match="target_node_active"):
        _walk(t, s, jnp.ones((t_n + 1,), bool), None)
    with pytest.raises(ValueError, match="source_node_active"):
        _walk(t, s, None, jnp.ones((s_n + 1,), bool))


def test_the_mask_does_not_prune_real_work():
    """The control: the admissibility partition still holds on the masked walk.

    For every LIVE target leaf, the source particles reached through far sources
    accepted for it or any ancestor, plus its near source leaves, must cover every
    source particle exactly once. An over-eager mask -- one that killed a live pair,
    or a dead ancestor above a live leaf -- shows up here as a coverage hole.
    """
    t, s = _padded_pair()
    t_tree, s_tree = t[0], s[0]
    t_act = np.asarray(_active(t_tree))
    res = _walk(t, s, jnp.asarray(t_act), _active(s_tree))
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )

    parent = np.asarray(t_tree.parent)
    s_ranges = np.asarray(s_tree.node_ranges)
    n_source = int(s_tree.num_particles)

    tt = np.asarray(res.interaction_targets)
    ss = np.asarray(res.interaction_sources)
    far_map: dict[int, list[int]] = {}
    for a, b in zip(tt[tt >= 0].tolist(), ss[tt >= 0].tolist()):
        far_map.setdefault(a, []).append(b)

    off = np.asarray(res.neighbor_offsets)
    idx = np.asarray(res.neighbor_indices)
    cnt = np.asarray(res.neighbor_counts)
    leaf_nodes = np.asarray(res.leaf_indices)
    root = int(np.argmin(parent))

    def src_particles(node):
        lo, hi = int(s_ranges[node, 0]), int(s_ranges[node, 1])
        return list(range(lo, hi + 1))

    checked = 0
    for row, leaf in enumerate(leaf_nodes.tolist()):
        if not t_act[leaf]:
            # a dead target leaf stands for nothing and must have emitted nothing
            assert int(cnt[row]) == 0
            continue
        sources = []
        node = leaf
        while True:
            sources.extend(far_map.get(node, []))
            if node == root:
                break
            node = int(parent[node])
        o, c = int(off[row]), int(cnt[row])
        sources.extend(int(v) for v in idx[o : o + c])
        covered = [p for sn in sources for p in src_particles(sn)]
        assert len(covered) == len(set(covered)), f"target leaf {leaf}: double cover"
        assert set(covered) == set(range(n_source)), f"target leaf {leaf}: hole"
        checked += 1
    assert checked > 0
