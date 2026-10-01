"""End to end: export against a summary, assemble on the receiver, check the sum.

The property under test is the one a flat imported POOL fails: for every local target
leaf, the sender's particles reached through the imported set -- far sources accepted
for the leaf or any ancestor, plus its near sources -- must be covered EXACTLY ONCE.

A gap loses force. An overlap double-counts it. Neither shows up in a conservation
check: double counting leaves momentum exact. Measured on the pooled construction, all
583 target leaves double-counted; the per-cell construction is what fixes it, and this
is where that is demonstrated rather than argued.

    JAX_PLATFORMS=cpu pytest tests/distributed/test_receiver_lists.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax.distributed.export import build_send_buffers, export_walk
from yggdrax.distributed.import_cells import receiver_interaction_lists
from yggdrax.distributed.summary import occupancy_cut

from .test_export_walk import _children_full, _domains, _summaries

_THETA = 0.6
_MAC = "dehnen"


def _extents(geom):
    return np.asarray(geom.radius)


def _one_way(doms, sender, receiver, max_leaves):
    """Everything the receiver needs, without the collectives (tested separately).

    Returns the receiver's far/near lists plus the sender node behind each payload
    row, so coverage can be checked in the sender's particle space.
    """
    ndev = len(doms)
    cen, rad, act, cap, cuts = _summaries(doms, max_leaves=max_leaves)

    s_topo, s_geom = doms[sender]
    s_left, s_right = _children_full(s_topo)
    ex = export_walk(
        s_left,
        s_right,
        jnp.asarray(s_geom.center),
        jnp.asarray(s_geom.radius),
        jnp.argmin(s_topo.parent).astype(s_topo.parent.dtype),
        cen,
        rad,
        act,
        _THETA,
        jnp.asarray(sender),
        max_pair_queue=1 << 16,
        far_cap=1 << 18,
        near_cap=1 << 18,
        mac_type=_MAC,
    )
    assert not (
        bool(ex.far_overflow) or bool(ex.near_overflow) or bool(ex.queue_overflow)
    )

    n_s_nodes = int(s_topo.parent.shape[0])
    out = {}
    for which, (c, n, k) in (
        ("far", (ex.far_cell, ex.far_node, ex.far_count)),
        ("near", (ex.near_cell, ex.near_node, ex.near_count)),
    ):
        sb = build_send_buffers(
            c,
            n,
            k,
            ndev=ndev,
            max_cells=cap,
            num_nodes=n_s_nodes,
            node_capacity=1 << 14,
            csr_capacity=1 << 16,
        )
        assert not (bool(sb.node_overflow) or bool(sb.csr_overflow))
        ns = np.asarray(sb.node_sizes)
        cs = np.asarray(sb.csr_sizes)
        n_off = int(np.concatenate([[0], np.cumsum(ns)])[receiver])
        c_off = int(np.concatenate([[0], np.cumsum(cs)])[receiver])
        # single sender, so no rebase is needed: this is the identity case
        out[which] = dict(
            nodes=np.asarray(sb.node_rows)[n_off : n_off + ns[receiver]],
            cell=np.asarray(sb.csr_cell)[c_off : c_off + cs[receiver]],
            row=np.asarray(sb.csr_row)[c_off : c_off + cs[receiver]],
        )
    return out, cuts[receiver], cap


def _assemble(doms, receiver, imported_nodes, csr_cell, csr_row, cut, sender_geom):
    """Build the combined space and run the receiver's walk."""
    r_topo, r_geom = doms[receiver]
    n_local = int(r_topo.parent.shape[0])
    k = int(imported_nodes.size)
    idx = r_topo.parent.dtype
    left, right = _children_full(r_topo)
    left = jnp.concatenate([left, jnp.full((k,), -1, idx)])
    right = jnp.concatenate([right, jnp.full((k,), -1, idx)])
    centers = jnp.concatenate(
        [jnp.asarray(r_geom.center), jnp.asarray(sender_geom.center)[imported_nodes]]
    )
    extents = jnp.concatenate(
        [jnp.asarray(r_geom.radius), jnp.asarray(sender_geom.radius)[imported_nodes]]
    )
    return receiver_interaction_lists(
        left,
        right,
        centers,
        extents,
        n_local,
        cut.cells,
        jnp.asarray(csr_cell),
        jnp.asarray(csr_row),
        jnp.asarray(csr_cell.size),
        _THETA,
        max_pair_queue=1 << 18,
        far_cap=1 << 20,
        near_cap=1 << 20,
        mac_type=_MAC,
    )


@pytest.mark.parametrize("max_leaves", [1, 4, 16])
def test_every_target_leaf_sums_the_sender_exactly_once(max_leaves):
    doms = _domains(2)
    sender, receiver = 1, 0
    lists, cut, cap = _one_way(doms, sender, receiver, max_leaves)

    # one combined imported block: far payloads then near payloads
    imported = np.concatenate([lists["far"]["nodes"], lists["near"]["nodes"]])
    shift = lists["far"]["nodes"].size
    csr_cell = np.concatenate([lists["far"]["cell"], lists["near"]["cell"]])
    csr_row = np.concatenate([lists["far"]["row"], lists["near"]["row"] + shift])

    res = _assemble(doms, receiver, imported, csr_cell, csr_row, cut, doms[sender][1])
    assert not (
        bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow)
    )

    r_topo = doms[receiver][0]
    s_topo = doms[sender][0]
    s_ranges = np.asarray(s_topo.node_ranges)
    n_src = int(s_topo.num_particles)
    parent = np.asarray(r_topo.parent)
    root = int(np.argmin(parent))
    nint = int(r_topo.left_child.shape[0])

    by_target = {}
    for arr_t, arr_s, n in (
        (res.far_target, res.far_source, res.far_count),
        (res.near_target, res.near_source, res.near_count),
    ):
        t = np.asarray(arr_t)[: int(n)]
        s = np.asarray(arr_s)[: int(n)]
        assert np.all(s >= 0) and np.all(s < imported.size)
        for ti, si in zip(t.tolist(), s.tolist()):
            by_target.setdefault(ti, []).append(int(imported[si]))

    r_ranges = np.asarray(r_topo.node_ranges)
    n_loc = int(r_topo.num_particles)
    live = (r_ranges[:, 1] >= r_ranges[:, 0]) & (r_ranges[:, 0] < n_loc)
    leaves = np.flatnonzero(live)
    leaves = leaves[leaves >= nint]

    checked = 0
    for leaf in leaves.tolist():
        srcs, node = [], leaf
        while True:
            srcs.extend(by_target.get(node, []))
            if node == root:
                break
            node = int(parent[node])
        cov = np.zeros(n_src, np.int32)
        for sn in srcs:
            lo, hi = int(s_ranges[sn, 0]), int(s_ranges[sn, 1])
            if hi >= lo:
                cov[lo : hi + 1] += 1
        assert cov.max() <= 1, f"leaf {leaf}: a sender particle counted twice"
        assert cov.min() >= 1, f"leaf {leaf}: a sender particle never counted"
        checked += 1
    assert checked > 0


def test_pairs_are_local_target_then_imported_source():
    doms = _domains(2)
    lists, cut, cap = _one_way(doms, 1, 0, 4)
    imported = np.concatenate([lists["far"]["nodes"], lists["near"]["nodes"]])
    shift = lists["far"]["nodes"].size
    res = _assemble(
        doms,
        0,
        imported,
        np.concatenate([lists["far"]["cell"], lists["near"]["cell"]]),
        np.concatenate([lists["far"]["row"], lists["near"]["row"] + shift]),
        cut,
        doms[1][1],
    )
    n_local = int(doms[0][0].parent.shape[0])
    for t, s, n in (
        (res.far_target, res.far_source, res.far_count),
        (res.near_target, res.near_source, res.near_count),
    ):
        tt = np.asarray(t)[: int(n)]
        ss = np.asarray(s)[: int(n)]
        assert np.all((tt >= 0) & (tt < n_local)), "target must be a LOCAL node"
        assert np.all((ss >= 0) & (ss < imported.size)), "source must be a payload row"
    nint = int(doms[0][0].left_child.shape[0])
    nt = np.asarray(res.near_target)[: int(res.near_count)]
    assert np.all(nt >= nint), "near targets are leaves"
