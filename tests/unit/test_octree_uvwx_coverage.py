"""Coverage-partition gate for the uniform-octree U/V interaction lists.

Structural correctness of the O(N)-FMM interaction structure, independent of the
multipole operators: for every ordered leaf-cell pair (A, B), A != B, the interaction
is handled EXACTLY ONCE -- either U (A, B adjacent at leaf level -> near P2P) or a V
interaction (M2L) at the unique COARSEST level where their ancestors are well separated
(parents still colleagues). This is the partition that guarantees a correct O(N) FMM:
U(leaf) union {V(ancestor) for all ancestors} = all sources, no gaps, no double counts.

Validated for uniform + clustered point sets across depths L=3..5 (see
benchmark_a100/RESUME.md, Phase 1a).
"""

from __future__ import annotations

import numpy as np
import pytest

from yggdrax.octree_uvwx import build_uniform_octree_uv


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
    [
        (3000, 3, "uniform"),
        (3000, 4, "uniform"),
        (5000, 4, "clustered"),
        (4000, 5, "clustered"),
    ],
)
def test_octree_uvwx_coverage(n, L, dist):
    oc = build_uniform_octree_uv(_points(n, 7, dist), L)
    parent = oc.parent
    is_leaf = oc.is_leaf
    level = oc.level

    # colleague test via Morton coords: two same-level cells are near iff |dcoord|<=1.
    # Recover per-node cell coord from centers is fragile; instead reuse the V/U lists +
    # the ancestor structure and check the partition invariant directly.
    # Build node particle sets (internal = union of descendant leaves).
    num_nodes = len(parent)
    from collections import defaultdict

    children = defaultdict(list)
    for c in range(num_nodes):
        if parent[c] >= 0:
            children[int(parent[c])].append(c)
    partset = {}
    for C in sorted(range(num_nodes), key=lambda x: -level[x]):
        if is_leaf[C]:
            s, e = oc.node_ranges[C]
            partset[C] = set(range(int(s), int(e) + 1))
        else:
            u = set()
            for ch in children[C]:
                u |= partset[ch]
            partset[C] = u

    V = defaultdict(list)
    for s, t in zip(oc.v_src.tolist(), oc.v_tgt.tolist()):
        V[t].append(s)

    def ancestors(C):
        out = []
        x = C
        while x >= 0:
            out.append(x)
            x = int(parent[x])
        return out

    n_part = oc.leaf_of.shape[0]
    all_particles = set(range(n_part))
    for r, L_node in enumerate(oc.leaf_indices.tolist()):
        from collections import Counter

        cnt = Counter()
        # far: union over ancestors of leaves-under-S for S in V(ancestor)
        for A in ancestors(L_node):
            for S in V.get(A, []):
                cnt.update(partset[S])
        # near: U-list (self included)
        for S in oc.u_neighbors[oc.u_offsets[r] : oc.u_offsets[r + 1]].tolist():
            cnt.update(partset[int(S)])
        covered = set(cnt)
        doubles = [p for p, c in cnt.items() if c > 1]
        assert covered == all_particles, (
            f"leaf {L_node}: covered {len(covered)}/{n_part} (missing "
            f"{len(all_particles - covered)})"
        )
        assert not doubles, f"leaf {L_node}: {len(doubles)} double-counted sources"
