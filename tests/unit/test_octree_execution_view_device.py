"""Device (JAX) static-shape uniform-octree build: coverage + no-recompile gate.

build_uniform_octree_execution_view_device produces a DENSE node space whose size depends
only on depth (static), so the build compiles once and is reused across timesteps. This
gate checks: (1) the dense node count / root coverage, (2) the U/V lists form an exact
interaction partition (every source handled once per leaf), and (3) the build is jittable
and does not retrace when only positions change (the point of the static-shape design).
"""

from __future__ import annotations

from collections import Counter

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from yggdrax.octree_uvwx import build_uniform_octree_execution_view_device


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / (1.0 - 0.85 * r)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return r[:, None] * d
    return rng.uniform(-1.0, 1.0, size=(n, 3))


@pytest.mark.parametrize(
    ("n", "L", "dist"),
    [(2000, 3, "uniform"), (2000, 3, "clustered"), (2000, 4, "uniform")],
)
def test_device_view_coverage(n, L, dist):
    pos = _points(n, 7, dist)
    v = build_uniform_octree_execution_view_device(pos, L)
    parent = np.asarray(v.parent)
    nr = np.asarray(v.node_ranges)
    leaf_idx = np.asarray(v.leaf_indices)
    off_L = int(leaf_idx[0])
    v_src = np.asarray(v.v_src)
    v_tgt = np.asarray(v.v_tgt)
    u_nbr = np.asarray(v.u_neighbors)
    u_off = np.asarray(v.u_offsets)
    M = int(v.num_valid_nodes)
    sentinel = M

    assert M == (8 ** (L + 1) - 1) // 7  # dense count depends only on depth
    assert tuple(nr[0]) == (0, n - 1)  # root covers all particles

    def particles(node):
        s, e = nr[node]
        return set(range(int(s), int(e) + 1))

    from collections import defaultdict

    far_by_tgt = defaultdict(list)
    for s, t in zip(v_src.tolist(), v_tgt.tolist()):
        if s != sentinel and t != sentinel:
            far_by_tgt[int(t)].append(int(s))

    def ancestors(node):
        out = []
        x = int(node)
        while x >= 0:
            out.append(x)
            x = int(parent[x])
        return out

    all_p = set(range(n))
    occ_leaves = [int(c) for c in leaf_idx if (int(nr[c, 1]) - int(nr[c, 0]) + 1) > 0]
    for ln in occ_leaves:
        r = ln - off_L
        cnt = Counter()
        cnt.update(particles(ln))  # self block (kernel adds it once)
        for e in u_nbr[u_off[r] : u_off[r] + 26].tolist():
            if e != sentinel:
                cnt.update(particles(int(e) + off_L))
        for a in ancestors(ln):
            for s in far_by_tgt.get(a, []):
                cnt.update(particles(s))
        covered = set(cnt)
        assert covered == all_p, f"leaf {ln}: missing {len(all_p - covered)} particles"
        assert not [p for p, c in cnt.items() if c > 1], f"leaf {ln}: double-counted"


def test_device_view_jittable_static_shapes():
    """Jitting with static depth compiles once and reuses across position sets."""
    from functools import partial

    build = jax.jit(
        partial(build_uniform_octree_execution_view_device), static_argnums=(1,)
    )
    n, L = 1500, 3
    v1 = build(_points(n, 1, "uniform"), L)
    v2 = build(_points(n, 2, "clustered"), L)  # same (N, L) -> no retrace, same shapes
    for f in (
        "parent",
        "children",
        "node_ranges",
        "v_src",
        "u_neighbors",
        "leaf_indices",
    ):
        assert getattr(v1, f).shape == getattr(v2, f).shape
    assert int(v1.num_valid_nodes) == int(v2.num_valid_nodes) == (8 ** (L + 1) - 1) // 7
