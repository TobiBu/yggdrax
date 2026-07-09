"""Cross-tree dual walk correctness (Phase 3 core).

Implementation-independent invariant: the **FMM admissibility partition**. For
every target leaf, the source particles it interacts with -- via far source
nodes accepted for the leaf or any of its ancestors, plus near source leaves --
must cover ALL source particles exactly once (a disjoint tiling). A correct
dual walk guarantees this; a buggy refine/emit breaks it.

Single-device (no shard_map needed for the kernel itself):

    JAX_PLATFORMS=cpu pytest tests/distributed/test_cross_walk.py -q
"""

import numpy as np

import jax.numpy as jnp

from yggdrax._tree_impl import build_tree
from yggdrax.bounds import infer_bounds
from yggdrax.geometry import compute_tree_geometry
from yggdrax.distributed import dual_tree_walk_cross

_LEAF = 8
_THETA = 0.5
_MAC = "bh"


def _build(points, bounds, leaf_size=_LEAF):
    pts = jnp.asarray(points)
    tree, pos_sorted, _mass_sorted, _inv = build_tree(
        pts, jnp.ones(pts.shape[0]), bounds, return_reordered=True, leaf_size=leaf_size
    )
    geom = compute_tree_geometry(tree, pos_sorted, max_leaf_size=leaf_size)
    return tree, geom


def _two_domains(seed, n_t=40, n_s=40):
    rng = np.random.default_rng(seed)
    # Two separated-but-overlapping-in-box clouds, shared coordinate frame.
    tgt = rng.uniform(-1.0, 0.2, size=(n_t, 3)).astype(np.float32)
    src = rng.uniform(-0.2, 1.0, size=(n_s, 3)).astype(np.float32)
    allpts = jnp.asarray(np.concatenate([tgt, src], axis=0))
    bounds = infer_bounds(allpts)
    return tgt, src, bounds


def _far_pairs(res):
    """Flat valid (target_node, source_node) far pairs."""
    tt = np.asarray(res.interaction_targets)
    ss = np.asarray(res.interaction_sources)
    m = tt >= 0
    return tt[m], ss[m]


def _run_partition_check(seed):
    tgt, src, bounds = _two_domains(seed)
    t_tree, t_geom = _build(tgt, bounds)
    s_tree, s_geom = _build(src, bounds)

    res = dual_tree_walk_cross(
        t_tree,
        t_geom,
        s_tree,
        s_geom,
        _THETA,
        mac_type=_MAC,
        max_interactions_per_node=256,
        max_neighbors_per_leaf=256,
        max_pair_queue=8192,
    )
    # capacities must be adequate
    assert not bool(res.queue_overflow)
    assert not bool(res.far_overflow)
    assert not bool(res.near_overflow)

    parent = np.asarray(t_tree.parent)
    t_ranges = np.asarray(t_tree.node_ranges)
    s_ranges = np.asarray(s_tree.node_ranges)
    n_source = int(s_ranges[np.asarray(res.leaf_indices)][:, 1].max()) + 1  # M-1 -> M

    # target-node -> list of far source nodes
    tt, ss = _far_pairs(res)
    far_map = {}
    for t_node, s_node in zip(tt.tolist(), ss.tolist()):
        far_map.setdefault(t_node, []).append(s_node)

    # target leaf -> near source leaf nodes
    near_off = np.asarray(res.neighbor_offsets)
    near_idx = np.asarray(res.neighbor_indices)
    near_cnt = np.asarray(res.neighbor_counts)
    leaf_indices = np.asarray(res.leaf_indices)  # target leaf node ids, leaf-row order

    def src_particles(node):
        lo, hi = int(s_ranges[node, 0]), int(s_ranges[node, 1])
        return set(range(lo, hi + 1))

    root = int(np.argmin(parent))
    for leaf_row, leaf_node in enumerate(leaf_indices.tolist()):
        # collect far sources from the leaf and all its ancestors
        far_sources = []
        node = leaf_node
        while True:
            far_sources.extend(far_map.get(node, []))
            if node == root:
                break
            node = int(parent[node])
        # near sources for this leaf row
        o = int(near_off[leaf_row])
        c = int(near_cnt[leaf_row])
        near_sources = near_idx[o : o + c].tolist()

        covered = []
        for sn in far_sources:
            covered.extend(sorted(src_particles(sn)))
        for sn in near_sources:
            covered.extend(sorted(src_particles(sn)))

        cov_set = set(covered)
        # complete coverage of all source particles
        assert cov_set == set(range(n_source)), (
            f"leaf_row {leaf_row}: covered {len(cov_set)} of {n_source} source parts"
        )
        # disjoint tiling (no source particle counted twice)
        assert len(covered) == n_source, (
            f"leaf_row {leaf_row}: overlap, {len(covered)} entries for {n_source} parts"
        )


def test_cross_walk_partitions_sources_seed0():
    _run_partition_check(0)


def test_cross_walk_partitions_sources_seed1():
    _run_partition_check(1)


def test_cross_walk_produces_far_and_near():
    tgt, src, bounds = _two_domains(2)
    t_tree, t_geom = _build(tgt, bounds)
    s_tree, s_geom = _build(src, bounds)
    res = dual_tree_walk_cross(
        t_tree, t_geom, s_tree, s_geom, _THETA,
        mac_type=_MAC,
        max_interactions_per_node=256,
        max_neighbors_per_leaf=256,
        max_pair_queue=8192,
    )
    # a non-trivial walk yields both far accepts and near leaf pairs
    assert int(res.far_pair_count) > 0
    assert int(res.near_pair_count) > 0
