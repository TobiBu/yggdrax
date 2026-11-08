"""The Pallas mutual walk: same pair SETS as the flat walk, and the backend switch.

``mutual_walk_pallas`` runs each wavefront round of ``dual_tree_walk_mutual`` as
one Pallas launch. Every test here runs the kernel in Pallas interpret mode, so it
runs on CPU; the native Triton lowering needs an Ampere-or-later GPU. Pinned:

* far and near lists equal the flat walk's as sets, with the same round count and
  peak wavefront, for both node layouts, both tree kinds and two opening angles;
* a full far list does not stop the walk and reports what the list needed; a full
  queue is flagged;
* the separation floor keeps the same sets in both walks and every far pair apart;
* Dehnen's eq (16a) from a walk table gives the flat walk's sets under
  ``dehnen_pair_accept``;
* each node record and eq (16a) row is ONE read per node of a pair (the walk is
  gather-bound);
* ``dual_tree_walk_mutual(backend=...)`` dispatches, refuses what the kernel
  cannot run, and ``"auto"`` is the flat walk where the kernel cannot run.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yggdrax import compute_tree_geometry
from yggdrax._tree_impl import build_static_cells_tree, build_static_radix_tree
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import (
    WALK_TABLE_GM,
    WALK_TABLE_POWER0,
    WALK_TABLE_THRESHOLD,
    dehnen_pair_accept,
    dual_tree_walk_mutual,
    mutual_walk_inputs,
    mutual_walk_pallas,
    pallas_mutual_walk_supported,
    walk_table_width,
)
from yggdrax.pallas import _compat
from yggdrax.tree_moments import compute_tree_mass_moments

_CAPS = {"max_pair_queue": 1 << 15, "far_cap": 1 << 17, "near_cap": 1 << 17}

# Every PAIR of (node layout, theta, tree kind) values in four cases rather than
# the eight-case grid: each factor drives a different part of the walk and none was
# seen to interact with another.
_PAIRWISE_CASES = [
    ("soa", 0.8, "cells"),
    ("record", 0.5, "cells"),
    ("record", 0.8, "buckets"),
    ("soa", 0.5, "buckets"),
]


def _plummer(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _walk_case(kind: str, *, n: int = 4000, leaf: int = 16, seed: int = 3, dtype=None):
    """Tree, sorted particles and walk inputs (masked empty nodes) on a Plummer sphere."""
    dtype = jnp.float32 if dtype is None else dtype
    pos = jnp.asarray(_plummer(n, seed), dtype)
    mass = jnp.full((n,), 1.0 / n, dtype)
    bounds = infer_bounds(pos)
    if kind == "buckets":
        topo, ps, ms, _ = build_static_radix_tree(
            pos, mass, bounds, leaf_size=leaf, return_reordered=True
        )
    else:
        topo, ps, ms, _ = build_static_cells_tree(
            pos, mass, bounds, leaf_size=leaf, leaf_capacity=1024, return_reordered=True
        )
    geometry = compute_tree_geometry(topo, ps)
    inputs = mutual_walk_inputs(
        topo, geometry, mac_type="dehnen", mask_empty_nodes=True
    )
    inputs = inputs._replace(
        centers=jnp.asarray(inputs.centers, dtype),
        radii=jnp.asarray(inputs.radii, dtype),
    )
    return topo, ps, ms, inputs


def _args(w) -> tuple[Any, ...]:
    return (w.left_child_full, w.right_child_full, w.centers, w.radii)


def _sets(res) -> tuple[set, set]:
    nf, nn = int(res.far_count), int(res.near_count)
    fa, fb = np.asarray(res.far_a)[:nf], np.asarray(res.far_b)[:nf]
    na, nb = np.asarray(res.near_a)[:nn], np.asarray(res.near_b)[:nn]
    return set(zip(fa.tolist(), fb.tolist())), set(zip(na.tolist(), nb.tolist()))


def _clean(res) -> bool:
    return not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )


@pytest.mark.parametrize("node_layout,theta,kind", _PAIRWISE_CASES)
def test_pallas_walk_lists_equal_the_flat_walk_as_sets(kind, theta, node_layout):
    # Interpret-mode cost tracks the PAIR count, not n.
    _, _, _, w = _walk_case(kind)
    ref = dual_tree_walk_mutual(
        *_args(w), theta, w.root, mac_type="dehnen", node_active=w.node_active, **_CAPS
    )
    got = mutual_walk_pallas(
        *_args(w),
        theta,
        w.root,
        node_active=w.node_active,
        block=64,
        interpret=True,
        node_layout=node_layout,
        **_CAPS,
    )
    assert _clean(ref) and _clean(got)
    assert int(ref.far_count) > 0, "vacuous: the MAC never fires"
    assert int(ref.near_count) > 0, "vacuous: no near pairs"
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert len(far_g) == int(got.far_count), "duplicate far emission"
    assert len(near_g) == int(got.near_count), "duplicate near emission"
    assert far_g == far_r
    assert near_g == near_r
    assert int(got.rounds) == int(ref.rounds)
    assert int(got.peak_wavefront) == int(ref.peak_wavefront)


def test_pallas_walk_flags_overflow():
    _, _, _, w = _walk_case("buckets", n=3000, leaf=8, seed=5)
    kw = dict(block=64, interpret=True)
    small = mutual_walk_pallas(
        *_args(w),
        0.6,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=256,
        near_cap=1 << 16,
        **kw,
    )
    assert bool(small.far_overflow) and not bool(small.near_overflow)
    # A full far list does not stop the walk: it reports what the list needed,
    # which is the count of a walk that fits, and the near list is complete.
    fits = mutual_walk_pallas(
        *_args(w),
        0.6,
        w.root,
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
        **kw,
    )
    assert not bool(fits.far_overflow) and int(fits.far_count) > 256
    assert int(small.far_needed) == int(fits.far_count)
    assert int(small.far_count) == 256
    assert int(small.near_needed) == int(small.near_count) == int(fits.near_count)
    assert _sets(small)[1] == _sets(fits)[1]
    tiny_q = mutual_walk_pallas(
        *_args(w),
        0.6,
        w.root,
        max_pair_queue=64,
        far_cap=1 << 16,
        near_cap=1 << 16,
        **kw,
    )
    assert bool(tiny_q.queue_overflow)


@pytest.mark.parametrize("floor", [0.05, 0.3])
def test_separation_floor_keeps_every_far_pair_apart_in_both_walks(floor):
    # An accepted far pair also needs |c_b - c_a| >= r_a + r_b + floor (radii that
    # bound every particle, so no two particles of the pair are closer than the
    # floor). Both walks keep the same SETS with it, every accepted pair honours it,
    # and it moves work from the far to the near list.
    theta = 0.8
    topo, ps, _, w = _walk_case("cells", seed=5)
    kw = dict(node_active=w.node_active, **_CAPS)
    base = dual_tree_walk_mutual(*_args(w), theta, w.root, mac_type="dehnen", **kw)
    ref = dual_tree_walk_mutual(
        *_args(w), theta, w.root, mac_type="dehnen", separation_floor=floor, **kw
    )
    got = mutual_walk_pallas(
        *_args(w),
        theta,
        w.root,
        block=64,
        interpret=True,
        separation_floor=floor,
        **kw,
    )
    assert _clean(base) and _clean(ref) and _clean(got)
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert far_g == far_r and near_g == near_r
    assert int(ref.far_count) < int(
        base.far_count
    ), "vacuous: the floor changed nothing"
    assert int(ref.near_count) > int(base.near_count)
    a = np.asarray([p[0] for p in far_r])
    b = np.asarray([p[1] for p in far_r])
    c = np.asarray(w.centers, np.float64)
    r = np.asarray(w.radii, np.float64)
    gap = np.linalg.norm(c[b] - c[a], axis=1) - r[a] - r[b]
    assert gap.min() >= floor * (1 - 1e-5)
    # and the closest particles of the tightest far pairs
    ranges = np.asarray(topo.node_ranges)
    pos = np.asarray(ps, np.float64)
    worst = np.inf
    for i in np.argsort(gap)[:50]:
        pa = pos[ranges[a[i], 0] : ranges[a[i], 1] + 1]
        pb = pos[ranges[b[i], 0] : ranges[b[i], 1] + 1]
        dmin = np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1))
        worst = min(worst, dmin)
    assert worst >= floor * (1 - 1e-5)


@dataclasses.dataclass(frozen=True)
class _TableAccept:
    """``pair_accept`` for the flat walk: eq (16a) from the same table rows."""

    order: int

    def __call__(self, data, a, b, dist_sq, radius_a, radius_b):
        table = jnp.asarray(data["table"], dist_sq.dtype)
        cols = WALK_TABLE_POWER0 + int(self.order)
        rows_a, rows_b = table[a], table[b]
        return dehnen_pair_accept(
            row_a=[rows_a[:, k] for k in range(cols)],
            row_b=[rows_b[:, k] for k in range(cols)],
            radius_a=radius_a,
            radius_b=radius_b,
            dist_sq=dist_sq,
            order=int(self.order),
            theta_max=jnp.asarray(data["theta_max"], dist_sq.dtype),
        )


def _error_table(topo, ps, ms, w, *, order: int, eps: float, dtype) -> jax.Array:
    """An eq (16a) table: G M, a smooth threshold, and seeded powers ``s_n`` in [0, 1].

    A real table holds the normalised multipole powers of each node; any values in
    their range exercise the same arithmetic in both walks.
    """
    mass = np.asarray(compute_tree_mass_moments(topo, ps, ms).mass, np.float64)
    c = np.asarray(w.centers, np.float64)
    nodes = int(c.shape[0])
    table = np.zeros((nodes, walk_table_width(order)))
    table[:, WALK_TABLE_GM] = mass
    # a stand-in for eps * min f_b: the Plummer field strength at the node
    table[:, WALK_TABLE_THRESHOLD] = eps / (np.sum(c * c, axis=1) + 1.0)
    rng = np.random.default_rng(11)
    table[:, WALK_TABLE_POWER0 : WALK_TABLE_POWER0 + order] = rng.uniform(
        0.0, 1.0, size=(nodes, order)
    )
    return jnp.asarray(table, dtype)


@pytest.mark.parametrize(
    "node_layout, dtype", [("record", "float32"), ("soa", "float64")]
)
@pytest.mark.parametrize("eps", [1e-2, 1e-4])
def test_pallas_walk_with_eq16a_equals_the_flat_walk_with_it(node_layout, dtype, eps):
    order = 4
    topo, ps, ms, w = _walk_case("buckets", seed=7, dtype=getattr(jnp, dtype))
    table = _error_table(
        topo, ps, ms, w, order=order, eps=eps, dtype=getattr(jnp, dtype)
    )
    kw = dict(node_active=w.node_active, **_CAPS)
    ref = dual_tree_walk_mutual(
        *_args(w),
        0.5,
        w.root,
        pair_accept=_TableAccept(order),
        pair_accept_data={"table": table, "theta_max": jnp.asarray(1.0, dtype)},
        **kw,
    )
    got = mutual_walk_pallas(
        *_args(w),
        0.5,
        w.root,
        block=64,
        interpret=True,
        node_layout=node_layout,
        error_table=table,
        error_order=order,
        theta_max=1.0,
        **kw,
    )
    assert _clean(ref) and _clean(got)
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert far_g == far_r
    assert near_g == near_r
    assert int(got.rounds) == int(ref.rounds)
    # the criterion is live: it differs from the geometric walk at the same theta
    geo = dual_tree_walk_mutual(*_args(w), 0.5, w.root, mac_type="dehnen", **kw)
    assert _sets(geo)[0] != far_r


def _round_kernel_reads(**walk_kwargs) -> collections.Counter:
    """Reads of each ref in ONE round kernel of a traced 7-node walk, by ref type.

    Counted in the kernel's jaxpr (``get`` per read), so no walk runs.
    """
    left = jnp.asarray([1, 3, 5, -1, -1, -1, -1], jnp.int32)
    right = jnp.asarray([2, 4, 6, -1, -1, -1, -1], jnp.int32)

    def _eqns(jaxpr):
        for eqn in jaxpr.eqns:
            yield eqn
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    inner = getattr(sub, "jaxpr", sub)  # a ClosedJaxpr's Jaxpr
                    if hasattr(inner, "eqns"):
                        yield from _eqns(inner)

    traced = jax.make_jaxpr(
        lambda: mutual_walk_pallas(
            left,
            right,
            jnp.zeros((7, 3), jnp.float32),
            jnp.ones((7,), jnp.float32),
            0.5,
            jnp.asarray(0, jnp.int32),
            max_pair_queue=64,
            far_cap=64,
            near_cap=64,
            interpret=True,
            rounds_per_check=1,
            **walk_kwargs,
        )
    )()
    (call,) = [e for e in _eqns(traced.jaxpr) if e.primitive.name == "pallas_call"]
    return collections.Counter(
        str(e.invars[0].aval.inner_aval)
        for e in _eqns(call.params["jaxpr"])
        if e.primitive.name == "get"
    )


@pytest.mark.parametrize(
    "order, width, layout, kernel_width",
    [(7, 16, "record", 16), (4, 6, "soa", 8)],
)
def test_pallas_walk_reads_each_eq16a_table_row_once(
    order, width, layout, kernel_width
):
    """One read of each node's eq (16a) row per pair, not one per column.

    The rows are read as vectors and split in registers; a table whose width is
    not a power of two (Triton splits only those) is padded.
    """
    reads = _round_kernel_reads(
        error_table=jnp.zeros((7, width), jnp.float32),
        error_order=order,
        node_layout=layout,
    )
    assert reads[f"float32[7,{kernel_width}]"] == 2, reads


@pytest.mark.parametrize("order", [0, 7])
def test_pallas_walk_reads_each_node_record_once(order):
    """The record ``(cx, cy, cz, r, left, right, active, 0)`` is one read per node."""
    extra = {}
    if order:
        extra = dict(error_table=jnp.zeros((7, 16), jnp.float32), error_order=order)
    reads = _round_kernel_reads(node_layout="record", **extra)
    assert reads["float32[7,8]"] == 2, reads


def _tiny_walk(**kw):
    return mutual_walk_pallas(
        jnp.full((3,), -1, jnp.int32),
        jnp.full((3,), -1, jnp.int32),
        jnp.zeros((3, 3), jnp.float32),
        jnp.zeros((3,), jnp.float32),
        0.5,
        jnp.asarray(0, jnp.int32),
        max_pair_queue=8,
        far_cap=8,
        near_cap=8,
        interpret=True,
        **kw,
    )


def test_pallas_walk_refuses_bad_arguments():
    with pytest.raises(ValueError, match="error_table"):
        _tiny_walk(error_order=6)
    with pytest.raises(ValueError, match="columns"):
        _tiny_walk(error_table=jnp.zeros((3, 4), jnp.float32), error_order=6)
    with pytest.raises(ValueError, match="node_layout"):
        _tiny_walk(node_layout="aos")
    with pytest.raises(ValueError, match="max_programs"):
        _tiny_walk(max_programs=0)
    with pytest.raises(ValueError, match="together"):
        _tiny_walk(seed_a=jnp.zeros((1,), jnp.int32))
    with pytest.raises(ValueError, match="exceeds"):
        _tiny_walk(seed_a=jnp.zeros((9,), jnp.int32), seed_b=jnp.zeros((9,), jnp.int32))


def test_max_programs_changes_slots_not_sets():
    """A capped grid strides over the queue's blocks: same sets, same rounds."""
    _, _, _, w = _walk_case("buckets", n=2000, seed=9)
    kw = dict(node_active=w.node_active, block=64, interpret=True, **_CAPS)
    wide = mutual_walk_pallas(*_args(w), 0.6, w.root, **kw)
    narrow = mutual_walk_pallas(*_args(w), 0.6, w.root, max_programs=2, **kw)
    assert _clean(wide) and _clean(narrow)
    assert _sets(narrow) == _sets(wide)
    assert int(narrow.rounds) == int(wide.rounds)


# --------------------------------------------------------------------------
# dual_tree_walk_mutual(backend=...)
# --------------------------------------------------------------------------


def test_backend_pallas_equals_flat_including_a_seeded_walk():
    _, _, _, w = _walk_case("cells", n=3000, seed=4)
    kw = dict(mac_type="dehnen", node_active=w.node_active, **_CAPS)
    ref = dual_tree_walk_mutual(*_args(w), 0.7, w.root, **kw)
    got = dual_tree_walk_mutual(
        *_args(w), 0.7, w.root, backend="pallas", pallas_interpret=True, **kw
    )
    assert _clean(ref) and _clean(got)
    assert got.far_a.dtype == ref.far_a.dtype
    assert got.rounds.dtype == ref.rounds.dtype
    assert _sets(got) == _sets(ref)
    assert int(got.rounds) == int(ref.rounds)
    assert int(got.peak_wavefront) == int(ref.peak_wavefront)
    # Seeds in the WRONG orientation: both backends canonicalise them, so the
    # radius tie-break splits the same node and the sets agree.
    root = int(w.root)
    lc, rc = int(w.left_child_full[root]), int(w.right_child_full[root])
    seed_a = jnp.asarray([rc, rc, -1], w.left_child_full.dtype)
    seed_b = jnp.asarray([lc, rc, -1], w.left_child_full.dtype)
    seeds = dict(seed_a=seed_a, seed_b=seed_b, seed_count=jnp.asarray(2))
    ref_s = dual_tree_walk_mutual(*_args(w), 0.7, w.root, **seeds, **kw)
    got_s = dual_tree_walk_mutual(
        *_args(w), 0.7, w.root, backend="pallas", pallas_interpret=True, **seeds, **kw
    )
    assert _clean(ref_s) and _clean(got_s)
    assert int(ref_s.far_count) > 0
    assert _sets(got_s) == _sets(ref_s)


def test_backend_auto_is_the_flat_walk_where_the_kernel_cannot_run():
    _, _, _, w = _walk_case("buckets", n=2000, seed=6)
    kw = dict(node_active=w.node_active, **_CAPS)
    flat = dual_tree_walk_mutual(*_args(w), 0.6, w.root, mac_type="dehnen", **kw)
    # mac_type=None is the strict rule the kernel does not implement
    strict = dual_tree_walk_mutual(*_args(w), 0.6, w.root, **kw)
    strict_auto = dual_tree_walk_mutual(*_args(w), 0.6, w.root, backend="auto", **kw)
    for name in flat._fields:
        assert np.array_equal(
            np.asarray(getattr(strict_auto, name)), np.asarray(getattr(strict, name))
        ), name
    auto = dual_tree_walk_mutual(
        *_args(w), 0.6, w.root, mac_type="dehnen", backend="auto", **kw
    )
    if pallas_mutual_walk_supported():
        assert _sets(auto) == _sets(flat)
    else:  # the flat walk itself: identical arrays, order included
        for name in flat._fields:
            assert np.array_equal(
                np.asarray(getattr(auto, name)), np.asarray(getattr(flat, name))
            ), name
    # pallas_interpret selects the kernel even here
    forced = dual_tree_walk_mutual(
        *_args(w),
        0.6,
        w.root,
        mac_type="dehnen",
        backend="auto",
        pallas_interpret=True,
        **kw,
    )
    assert _sets(forced) == _sets(flat)


def test_backend_pallas_refuses_what_the_kernel_cannot_run():
    _, _, _, w = _walk_case("buckets", n=500, seed=2)
    kw = dict(node_active=w.node_active, **_CAPS)
    with pytest.raises(ValueError, match="backend must be one of"):
        dual_tree_walk_mutual(*_args(w), 0.6, w.root, backend="cuda", **kw)
    with pytest.raises(ValueError, match="mac_type=None"):
        dual_tree_walk_mutual(
            *_args(w), 0.6, w.root, backend="pallas", pallas_interpret=True, **kw
        )
    with pytest.raises(ValueError, match="engblom"):
        dual_tree_walk_mutual(
            *_args(w),
            0.6,
            w.root,
            mac_type="engblom",
            backend="pallas",
            pallas_interpret=True,
            **kw,
        )
    with pytest.raises(ValueError, match="pair_accept"):
        dual_tree_walk_mutual(
            *_args(w),
            0.6,
            w.root,
            mac_type="dehnen",
            backend="pallas",
            pallas_interpret=True,
            pair_accept=_TableAccept(2),
            pair_accept_data={"table": jnp.zeros((1, 4)), "theta_max": 1.0},
            **kw,
        )
    if not pallas_mutual_walk_supported():
        with pytest.raises(ValueError, match="sm_80"):
            dual_tree_walk_mutual(
                *_args(w), 0.6, w.root, mac_type="dehnen", backend="pallas", **kw
            )
    with pytest.raises(ValueError, match="matching 1-D"):
        dual_tree_walk_mutual(
            *_args(w),
            0.6,
            w.root,
            mac_type="dehnen",
            backend="pallas",
            pallas_interpret=True,
            seed_a=jnp.zeros((2,), jnp.int32),
            seed_b=jnp.zeros((3,), jnp.int32),
            **kw,
        )
    with pytest.raises(ValueError, match="together"):
        dual_tree_walk_mutual(
            *_args(w),
            0.6,
            w.root,
            mac_type="dehnen",
            backend="pallas",
            pallas_interpret=True,
            seed_a=jnp.zeros((2,), jnp.int32),
            **kw,
        )

    @jax.jit
    def traced_theta(theta):
        return dual_tree_walk_mutual(
            *_args(w),
            theta,
            w.root,
            mac_type="dehnen",
            backend="pallas",
            pallas_interpret=True,
            **kw,
        )

    with pytest.raises(TypeError, match="concrete theta"):
        traced_theta(0.6)


# --------------------------------------------------------------------------
# the shared Pallas compat layer
# --------------------------------------------------------------------------


class _Device:
    def __init__(self, capability):
        self.compute_capability = capability


@pytest.mark.parametrize(
    "backend, capability, expected",
    [
        ("cpu", "8.0", False),
        ("gpu", "7.5", False),
        ("gpu", "8.0", True),
        ("gpu", "9.0", True),
        ("gpu", None, False),
    ],
)
def test_triton_gate_needs_an_ampere_gpu(monkeypatch, backend, capability, expected):
    monkeypatch.setattr(_compat.jax, "default_backend", lambda: backend)
    monkeypatch.setattr(_compat.jax, "devices", lambda: [_Device(capability)])
    assert _compat.pallas_triton_supported() is expected
    assert pallas_mutual_walk_supported() is expected


def test_backend_kwargs_name_triton_with_its_tuning():
    assert _compat.pallas_backend_kwargs("triton", interpret=True, num_warps=2) == {}
    kwargs = _compat.pallas_backend_kwargs("triton", num_warps=2, num_stages=3)
    params = kwargs["compiler_params"]
    assert params.num_warps == 2 and params.num_stages == 3
    assert ("backend" in kwargs) == _compat.PALLAS_CALL_TAKES_BACKEND
    with pytest.raises(NotImplementedError, match="mosaic"):
        _compat.pallas_backend_kwargs("mosaic")
