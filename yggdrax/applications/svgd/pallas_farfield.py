"""A fused, leaf-major Pallas kernel for the Stein far field (M2P).

The near field's story, told again one level up. The pure-JAX far field expands
every accepted ``(target node, source node)`` pair to the target node's
*particles* and scatters ``M`` monopole contributions back. At N = 1e5 that is
**M = 89 555 008** entries over 99 616 distinct targets, so the indices repeat
**899 times** on average -- against the near field's 62, which was already
enough to cost 258 ms there. Measured, far field only:

===================================  =========  ==========
part                                   float64     float32
===================================  =========  ==========
gather + arithmetic                    49.75 ms    25.57 ms
the scatter alone                       7.71 ms   271.71 ms
===================================  =========  ==========

:func:`~yggdrax.applications.svgd.sampler.assemble_svgd_topology` now pushes
each pair down to the *leaves* under its target node instead, which is ``ml``
times smaller (2.8 million entries) and lets the accumulation be a segmented
reduction plus a permutation. That fixes the forward -- 237.89 -> 4.82 ms in
float32 -- and **not** the reverse, because the transpose of the per-entry
target gather is the same scatter again.

This module is the reverse's answer, and the forward's best form. One program
owns a target leaf and streams that leaf's far source *monopoles* from a CSR,
accumulating in registers; the reverse streams the cotangent the same way. The
source-side cotangents (``d/dc_B``, ``d/dS_B``) are emitted per entry and summed
by source node through the partition's transpose CSR, which is a segmented
reduction over 2.8 million entries rather than a scatter over 89.5 million.

The chain from a node's monopole back to its particles is a **range update** --
every particle in the node's contiguous slot range receives the same vector --
so it is a difference array and a ``cumsum`` over ``F`` nodes, with no atomics
anywhere in either pass.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from yggdrax.applications.svgd._pallas_compat import KernelRef, pallas_backend_kwargs
from yggdrax.applications.svgd.pallas_nearfield import (
    _integer_cotangent,
    _pow2_ceil,
    _resolve_tile,
    _target_rows_from_offsets,
)

if TYPE_CHECKING:
    from jax.experimental import pallas as pl
else:
    try:  # pragma: no cover - import is environment-dependent
        from jax.experimental import pallas as pl
    except Exception:  # pragma: no cover - Pallas is optional
        pl = None


def _prepare(
    leaf_x: Array,
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    h: float | Float[Array, ""],
) -> tuple[Any, ...]:
    """Validate and pad the far-field operands for a kernel launch.

    Args:
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``.
        h: Kernel bandwidth.

    Returns:
        ``(tables, shapes)`` -- the padded arrays and the static tile sizes.

    Raises:
        RuntimeError: If ``jax.experimental.pallas`` could not be imported.
        ValueError: If the input shapes are mutually inconsistent.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    leaf_x = jnp.asarray(leaf_x)
    dtype = leaf_x.dtype
    if leaf_x.ndim != 3:
        raise ValueError("leaf_x must have shape (L, W, d)")
    num_leaves, leaf_width, num_dims = (int(v) for v in leaf_x.shape)
    if jnp.shape(src_offsets) != (num_leaves + 1,):
        raise ValueError("src_offsets must have shape (L + 1,)")

    lane_width = _pow2_ceil(num_dims)
    width = _pow2_ceil(leaf_width)
    pad = ((0, 0), (0, width - leaf_width), (0, lane_width - num_dims))
    tbl_x = jnp.pad(leaf_x, pad)
    tbl_mask = jnp.pad(jnp.asarray(leaf_mask).astype(bool), pad[:2])
    lane_pad = ((0, 0), (0, lane_width - num_dims))
    com = jnp.pad(jnp.asarray(mono_com, dtype), lane_pad)
    sums = jnp.pad(jnp.asarray(mono_sums, dtype), lane_pad)
    count = jnp.asarray(mono_count, dtype).reshape(-1)
    nodes = jnp.asarray(src_node, jnp.int32).reshape(-1)
    if int(nodes.shape[0]) == 0:
        nodes = jnp.zeros((1,), jnp.int32)
    return (
        tbl_x,
        tbl_mask,
        count,
        com,
        sums,
        jnp.asarray(src_offsets, jnp.int32),
        nodes,
        jnp.asarray([h], dtype),
    ), (num_leaves, leaf_width, num_dims, width, lane_width, int(nodes.shape[0]))


def farfield_stein_jax(
    leaf_x: Float[Array, "L W d"],
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    h: float | Float[Array, ""],
    *,
    chunk_entries: int | None = None,
) -> Float[Array, "L W d"]:
    """Leaf-major Stein far field in pure JAX -- the kernel's twin.

    For every target lane ``i`` of leaf ``l`` and every far source node ``B`` in
    that leaf's CSR row::

        r = x_i - com_B
        out_i += exp(-|r|^2 / 2h^2) * (sum_s_B + count_B * r / h^2)

    Args:
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``. Entries past
            ``src_offsets[-1]`` are capacity padding and contribute nothing.
        h: Kernel bandwidth.
        chunk_entries: Entries per rematerialised chunk. ``None`` takes them all.

    Returns:
        The far-field contribution, leaf-major, shape ``(L, W, d)``.
    """
    leaf_x = jnp.asarray(leaf_x)
    dtype = leaf_x.dtype
    valid = jnp.asarray(leaf_mask).astype(bool)
    count = jnp.asarray(mono_count, dtype).reshape(-1, 1)
    com = jnp.asarray(mono_com, dtype)
    sums = jnp.asarray(mono_sums, dtype)
    offsets = jnp.asarray(src_offsets)
    nodes = jnp.asarray(src_node)
    num_leaves, width, num_dims = leaf_x.shape
    num_entries = int(nodes.shape[0])
    acc = jnp.zeros((num_leaves, width, num_dims), dtype)
    if num_entries == 0:
        return acc
    inv_h2 = 1.0 / (h * h)

    rows = _target_rows_from_offsets(offsets, num_entries)
    live = jnp.arange(num_entries, dtype=offsets.dtype) < offsets[-1]
    chunk = (
        num_entries if chunk_entries is None else min(int(chunk_entries), num_entries)
    )
    num_chunks = -(-num_entries // chunk)
    pad = num_chunks * chunk - num_entries
    if pad:
        rows = jnp.concatenate([rows, jnp.full((pad,), num_leaves - 1, rows.dtype)])
        nodes = jnp.concatenate([nodes, jnp.zeros((pad,), nodes.dtype)])
        live = jnp.concatenate([live, jnp.zeros((pad,), bool)])

    def _step(carry: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, None]:
        row, node, alive = xs
        sep = leaf_x[row] - com[node][:, None, :]
        kern = jnp.exp(-jnp.sum(sep * sep, axis=-1, keepdims=True) * (0.5 * inv_h2))
        kern = jnp.where(alive[:, None, None], kern, 0.0)
        contrib = kern * (
            sums[node][:, None, :] + count[node][:, None, :] * sep * inv_h2
        )
        return (
            carry
            + jax.ops.segment_sum(
                contrib, row, num_segments=num_leaves, indices_are_sorted=True
            ),
            None,
        )

    acc, _ = lax.scan(
        _step,
        acc,
        (
            rows.reshape(num_chunks, chunk),
            nodes.reshape(num_chunks, chunk),
            live.reshape(num_chunks, chunk),
        ),
    )
    return acc * valid[..., None].astype(dtype)


def _farfield_kernel(
    tgt_x_ref: KernelRef,
    tgt_mask_ref: KernelRef,
    count_ref: KernelRef,
    com_ref: KernelRef,
    sums_ref: KernelRef,
    offsets_ref: KernelRef,
    node_ref: KernelRef,
    h_ref: KernelRef,
    out_ref: KernelRef,
    *,
    num_dims: int,
    lane_width: int,
) -> None:
    """Accumulate one target subtile's far field entirely in registers.

    A monopole is a handful of scalars, so unlike the near field there is no
    inner lane loop to tile: each iteration broadcasts one source node against
    the whole target vector.

    Args:
        tgt_x_ref: Target positions for this subtile, shape ``(1, Bt, dp)``.
        tgt_mask_ref: Which target lanes are real, shape ``(1, Bt)``.
        count_ref: Particles per far source node, shape ``(F,)``.
        com_ref: Centre of mass per far source node, shape ``(F, dp)``.
        sums_ref: Summed score per far source node, shape ``(F, dp)``.
        offsets_ref: CSR offsets by target leaf, shape ``(L + 1,)``.
        node_ref: Far source node of each entry, shape ``(P,)``.
        h_ref: Kernel bandwidth, shape ``(1,)``.
        out_ref: **Output**, shape ``(1, Bt, dp)``.
        num_dims: The true dimension ``d``. Static.
        lane_width: The padded dimension ``dp >= d``. Static.

    Returns:
        None. The result is the write to ``out_ref``.
    """
    leaf = pl.program_id(0)
    tvalid = tgt_mask_ref[0, :]
    tgt_x = [tgt_x_ref[0, :, k] for k in range(num_dims)]
    bandwidth = h_ref[0]
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    half_inv_h2 = 0.5 * inv_h2
    zero = jnp.zeros_like(tgt_x[0])

    def _entry(p: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
        node = node_ref[p]
        sep = [tgt_x[k] - com_ref[node, k] for k in range(num_dims)]
        dist_sq = sep[0] * sep[0]
        for k in range(1, num_dims):
            dist_sq = dist_sq + sep[k] * sep[k]
        kern = jnp.exp(-dist_sq * half_inv_h2)
        weight = kern * count_ref[node] * inv_h2
        return tuple(
            acc[k] + kern * sums_ref[node, k] + weight * sep[k] for k in range(num_dims)
        )

    acc = lax.fori_loop(
        offsets_ref[leaf],
        offsets_ref[leaf + 1],
        _entry,
        tuple(zero for _ in range(num_dims)),
    )
    for k in range(num_dims):
        out_ref[0, :, k] = jnp.where(tvalid, acc[k], zero)
    for k in range(num_dims, lane_width):
        out_ref[0, :, k] = zero


def farfield_stein_pallas(
    leaf_x: Float[Array, "L W d"],
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    h: float | Float[Array, ""],
    *,
    target_subtile: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
) -> Float[Array, "L W d"]:
    """Fused leaf-major Stein far field: one Pallas program per target leaf.

    Numerically equivalent to :func:`farfield_stein_jax`.

    Args:
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``.
        h: Kernel bandwidth.
        target_subtile: Target lanes per program.
        num_warps: Triton warps per program.
        num_stages: Triton pipeline stages.
        interpret: Run Pallas in interpret mode.

    Returns:
        The far-field contribution, leaf-major, shape ``(L, W, d)``.
    """
    tables, shapes = _prepare(
        leaf_x, leaf_mask, mono_count, mono_com, mono_sums, src_offsets, src_node, h
    )
    num_leaves, leaf_width, num_dims, width, lane_width, num_entries = shapes
    if num_leaves == 0 or leaf_width == 0:
        return jnp.zeros_like(jnp.asarray(leaf_x))
    subtile = _resolve_tile(target_subtile, width)
    if num_warps is None:
        num_warps = max(1, min(4, subtile // 32))

    def _kernel(*refs: KernelRef) -> None:
        return _farfield_kernel(*refs, num_dims=num_dims, lane_width=lane_width)

    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (num_leaves, width, lane_width), tables[0].dtype
        ),
        in_specs=[
            pl.BlockSpec((1, subtile, lane_width), lambda leaf, sub: (leaf, sub, 0)),
            pl.BlockSpec((1, subtile), lambda leaf, sub: (leaf, sub)),
            pl.BlockSpec((tables[2].shape[0],), lambda leaf, sub: (0,)),
            pl.BlockSpec(tables[3].shape, lambda leaf, sub: (0, 0)),
            pl.BlockSpec(tables[4].shape, lambda leaf, sub: (0, 0)),
            pl.BlockSpec((num_leaves + 1,), lambda leaf, sub: (0,)),
            pl.BlockSpec((num_entries,), lambda leaf, sub: (0,)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
        ],
        out_specs=pl.BlockSpec(
            (1, subtile, lane_width), lambda leaf, sub: (leaf, sub, 0)
        ),
        grid=(num_leaves, width // subtile),
        interpret=bool(interpret),
        name=f"stein_farfield_t{subtile}_w{width}_d{num_dims}",
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        ),
    )(*tables)
    return out[:, :leaf_width, :num_dims]


def _farfield_bwd_kernel(
    tgt_x_ref: KernelRef,
    tgt_g_ref: KernelRef,
    tgt_mask_ref: KernelRef,
    count_ref: KernelRef,
    com_ref: KernelRef,
    sums_ref: KernelRef,
    offsets_ref: KernelRef,
    node_ref: KernelRef,
    h_ref: KernelRef,
    dx_ref: KernelRef,
    dcom_ref: KernelRef,
    dsums_ref: KernelRef,
    dh_ref: KernelRef,
    *,
    num_dims: int,
    lane_width: int,
) -> None:
    """Reverse of :func:`_farfield_kernel`, target side in registers.

    The target-side cotangent ``d/dx_i`` is a per-leaf reduction and stays in
    registers. The source side cannot: a far source node is seen by many target
    leaves, so ``d/dcom_B`` and ``d/dsum_s_B`` are emitted **per entry** and
    summed by node outside, through the partition's transpose CSR. That is a
    segmented reduction over ``P`` (2.8 million) rather than a scatter over
    ``M`` (89.5 million), and the source side is exactly the target side negated
    -- the same antisymmetry the near field's reverse uses.

    The grid must not split the leaf into subtiles: the per-entry writes are
    indexed by entry, so two programs sharing a leaf would race on them.

    Args:
        tgt_x_ref: Target positions for this leaf, shape ``(1, W, dp)``.
        tgt_g_ref: Output cotangent for this leaf, shape ``(1, W, dp)``.
        tgt_mask_ref: Lane validity, shape ``(1, W)``.
        count_ref: Particles per far source node, shape ``(F,)``.
        com_ref: Centre of mass per far source node, shape ``(F, dp)``.
        sums_ref: Summed score per far source node, shape ``(F, dp)``.
        offsets_ref: CSR offsets by target leaf, shape ``(L + 1,)``.
        node_ref: Far source node of each entry, shape ``(P,)``.
        h_ref: Kernel bandwidth, shape ``(1,)``.
        dx_ref: **Output**, cotangent of ``leaf_x``, shape ``(1, W, dp)``.
        dcom_ref: **Output**, per-entry cotangent of ``com``, shape ``(P, dp)``.
        dsums_ref: **Output**, per-entry cotangent of ``sums``, shape ``(P, dp)``.
        dh_ref: **Output**, per-entry ``d/dh``, shape ``(P,)``.
        num_dims: The true dimension ``d``. Static.
        lane_width: The padded dimension ``dp >= d``. Static.

    Returns:
        None. The results are the writes to the four output refs.
    """
    leaf = pl.program_id(0)
    tvalid = tgt_mask_ref[0, :]
    tgt_x = [tgt_x_ref[0, :, k] for k in range(num_dims)]
    tgt_g = [tgt_g_ref[0, :, k] for k in range(num_dims)]
    bandwidth = h_ref[0]
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    half_inv_h2 = 0.5 * inv_h2
    inv_h3 = inv_h2 / bandwidth
    zero = jnp.zeros_like(tgt_x[0])

    def _entry(p: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
        node = node_ref[p]
        cnt = count_ref[node]
        sep = [tgt_x[k] - com_ref[node, k] for k in range(num_dims)]
        dist_sq = sep[0] * sep[0]
        g_dot_s = tgt_g[0] * sums_ref[node, 0]
        g_dot_r = tgt_g[0] * sep[0]
        for k in range(1, num_dims):
            dist_sq = dist_sq + sep[k] * sep[k]
            g_dot_s = g_dot_s + tgt_g[k] * sums_ref[node, k]
            g_dot_r = g_dot_r + tgt_g[k] * sep[k]
        kern = jnp.where(tvalid, jnp.exp(-dist_sq * half_inv_h2), 0.0)
        coef = kern * inv_h2 * (g_dot_s + cnt * g_dot_r * inv_h2)
        weight = kern * cnt * inv_h2
        out = list(acc)
        for k in range(num_dims):
            side = weight * tgt_g[k] - coef * sep[k]
            out[k] = out[k] + side
            # d/dcom is the target side negated; d/dsum_s is just k * g.
            dcom_ref[p, k] = -jnp.sum(side)
            dsums_ref[p, k] = jnp.sum(kern * tgt_g[k])
        for k in range(num_dims, lane_width):
            dcom_ref[p, k] = 0.0
            dsums_ref[p, k] = 0.0
        dh_ref[p] = jnp.sum(
            kern
            * inv_h3
            * (dist_sq * (g_dot_s + cnt * g_dot_r * inv_h2) - 2.0 * cnt * g_dot_r)
        )
        return tuple(out)

    acc = lax.fori_loop(
        offsets_ref[leaf],
        offsets_ref[leaf + 1],
        _entry,
        tuple(zero for _ in range(num_dims)),
    )
    for k in range(num_dims):
        dx_ref[0, :, k] = jnp.where(tvalid, acc[k], zero)
    for k in range(num_dims, lane_width):
        dx_ref[0, :, k] = zero


def farfield_stein_pallas_bwd(
    cotangent: Float[Array, "L W d"],
    leaf_x: Float[Array, "L W d"],
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    node_offsets: Array,
    entry_perm: Array,
    h: float | Float[Array, ""],
    *,
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Fused reverse of :func:`farfield_stein_pallas`.

    Args:
        cotangent: Cotangent of the forward output, shape ``(L, W, d)``.
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``.
        node_offsets: Transpose CSR offsets by source node, shape ``(F + 1,)``.
        entry_perm: Entry indices ordered by source node, shape ``(P,)``.
        h: Kernel bandwidth.
        num_warps: Triton warps per program.
        num_stages: Triton pipeline stages.
        interpret: Run Pallas in interpret mode.

    Returns:
        Cotangents of ``leaf_x``, ``mono_com``, ``mono_sums`` and ``h``.
    """
    tables, shapes = _prepare(
        leaf_x, leaf_mask, mono_count, mono_com, mono_sums, src_offsets, src_node, h
    )
    num_leaves, leaf_width, num_dims, width, lane_width, num_entries = shapes
    dtype = tables[0].dtype
    num_nodes = int(tables[3].shape[0])
    if num_leaves == 0 or leaf_width == 0:
        zeros = jnp.zeros_like(jnp.asarray(leaf_x))
        return (
            zeros,
            jnp.zeros_like(jnp.asarray(mono_com)),
            jnp.zeros_like(jnp.asarray(mono_sums)),
            jnp.zeros((), dtype),
        )
    tbl_g = jnp.pad(
        jnp.asarray(cotangent, dtype),
        ((0, 0), (0, width - leaf_width), (0, lane_width - num_dims)),
    )
    if num_warps is None:
        num_warps = max(1, min(4, width // 32))

    def _kernel(*refs: KernelRef) -> None:
        return _farfield_bwd_kernel(*refs, num_dims=num_dims, lane_width=lane_width)

    d_x, d_com_entry, d_sums_entry, d_h_entry = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_leaves, width, lane_width), dtype),
            jax.ShapeDtypeStruct((num_entries, lane_width), dtype),
            jax.ShapeDtypeStruct((num_entries, lane_width), dtype),
            jax.ShapeDtypeStruct((num_entries,), dtype),
        ),
        in_specs=[
            pl.BlockSpec((1, width, lane_width), lambda leaf: (leaf, 0, 0)),
            pl.BlockSpec((1, width, lane_width), lambda leaf: (leaf, 0, 0)),
            pl.BlockSpec((1, width), lambda leaf: (leaf, 0)),
            pl.BlockSpec((num_nodes,), lambda leaf: (0,)),
            pl.BlockSpec((num_nodes, lane_width), lambda leaf: (0, 0)),
            pl.BlockSpec((num_nodes, lane_width), lambda leaf: (0, 0)),
            pl.BlockSpec((num_leaves + 1,), lambda leaf: (0,)),
            pl.BlockSpec((num_entries,), lambda leaf: (0,)),
            pl.BlockSpec((1,), lambda leaf: (0,)),
        ],
        out_specs=(
            pl.BlockSpec((1, width, lane_width), lambda leaf: (leaf, 0, 0)),
            pl.BlockSpec((num_entries, lane_width), lambda leaf: (0, 0)),
            pl.BlockSpec((num_entries, lane_width), lambda leaf: (0, 0)),
            pl.BlockSpec((num_entries,), lambda leaf: (0,)),
        ),
        grid=(num_leaves,),
        interpret=bool(interpret),
        name=f"stein_farfield_bwd_w{width}_d{num_dims}",
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        ),
    )(
        tables[0],
        tbl_g,
        tables[1],
        tables[2],
        tables[3],
        tables[4],
        tables[5],
        tables[6],
        tables[7],
    )

    # Per-entry source cotangents -> per-node, through the transpose CSR. The
    # permutation makes the segment ids sorted, so this is a segmented reduction
    # over P entries and not a scatter over M.
    perm = jnp.asarray(entry_perm)[:num_entries]
    node_rows = _target_rows_from_offsets(jnp.asarray(node_offsets), num_entries)
    live = (jnp.arange(num_entries) < jnp.asarray(node_offsets)[-1])[:, None]
    d_com = jax.ops.segment_sum(
        d_com_entry[perm] * live,
        node_rows,
        num_segments=num_nodes,
        indices_are_sorted=True,
    )
    d_sums = jax.ops.segment_sum(
        d_sums_entry[perm] * live,
        node_rows,
        num_segments=num_nodes,
        indices_are_sorted=True,
    )
    d_h = jnp.sum(d_h_entry[perm] * live[:, 0])
    return (
        d_x[:, :leaf_width, :num_dims],
        d_com[:, :num_dims],
        d_sums[:, :num_dims],
        d_h,
    )


@partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12))
def _farfield_stein_fused(
    leaf_x: Array,
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    node_offsets: Array,
    entry_perm: Array,
    h: Array,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
) -> Array:
    """The fused far field with :func:`farfield_stein_pallas_bwd` as its reverse.

    Args:
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``.
        node_offsets: Transpose CSR offsets by source node, shape ``(F + 1,)``.
        entry_perm: Entry indices ordered by source node, shape ``(P,)``.
        h: Kernel bandwidth.
        num_warps: Triton warps per program. ``nondiff``.
        num_stages: Triton pipeline stages. ``nondiff``.
        interpret: Run Pallas in interpret mode. ``nondiff``.

    Returns:
        The far-field contribution, leaf-major, shape ``(L, W, d)``.
    """
    del node_offsets, entry_perm
    return farfield_stein_pallas(
        leaf_x,
        leaf_mask,
        mono_count,
        mono_com,
        mono_sums,
        src_offsets,
        src_node,
        h,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )


def _farfield_fwd(
    leaf_x,
    leaf_mask,
    mono_count,
    mono_com,
    mono_sums,
    src_offsets,
    src_node,
    node_offsets,
    entry_perm,
    h,
    num_warps,
    num_stages,
    interpret,
):
    out = farfield_stein_pallas(
        leaf_x,
        leaf_mask,
        mono_count,
        mono_com,
        mono_sums,
        src_offsets,
        src_node,
        h,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )
    return out, (
        leaf_x,
        leaf_mask,
        mono_count,
        mono_com,
        mono_sums,
        src_offsets,
        src_node,
        node_offsets,
        entry_perm,
        h,
    )


def _farfield_bwd(num_warps, num_stages, interpret, residual, cotangent):
    (
        leaf_x,
        leaf_mask,
        mono_count,
        mono_com,
        mono_sums,
        src_offsets,
        src_node,
        node_offsets,
        entry_perm,
        h,
    ) = residual
    d_x, d_com, d_sums, d_h = farfield_stein_pallas_bwd(
        cotangent,
        leaf_x,
        leaf_mask,
        mono_count,
        mono_com,
        mono_sums,
        src_offsets,
        src_node,
        node_offsets,
        entry_perm,
        h,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )
    return (
        d_x,
        _integer_cotangent(leaf_mask),
        jnp.zeros_like(jnp.asarray(mono_count)),
        d_com,
        d_sums,
        _integer_cotangent(src_offsets),
        _integer_cotangent(src_node),
        _integer_cotangent(node_offsets),
        _integer_cotangent(entry_perm),
        jnp.asarray(d_h, jnp.result_type(h)),
    )


_farfield_stein_fused.defvjp(_farfield_fwd, _farfield_bwd)


def farfield_stein(
    leaf_x: Float[Array, "L W d"],
    leaf_mask: Array,
    mono_count: Array,
    mono_com: Array,
    mono_sums: Array,
    src_offsets: Array,
    src_node: Array,
    node_offsets: Array,
    entry_perm: Array,
    h: float | Float[Array, ""],
    *,
    backend: str = "pallas",
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
    chunk_entries: int | None = None,
) -> Float[Array, "L W d"]:
    """Differentiable leaf-major Stein far field, fused kernel or twin.

    Args:
        leaf_x: Target positions, leaf-major, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        mono_count: Particles per far source node, shape ``(F,)``.
        mono_com: Centre of mass per far source node, shape ``(F, d)``.
        mono_sums: Summed score per far source node, shape ``(F, d)``.
        src_offsets: CSR offsets by target leaf, shape ``(L + 1,)``.
        src_node: Far source node of each entry, shape ``(P,)``.
        node_offsets: Transpose CSR offsets by source node, shape ``(F + 1,)``.
        entry_perm: Entry indices ordered by source node, shape ``(P,)``.
        h: Kernel bandwidth.
        backend: ``"pallas"`` for the kernel, ``"jax"`` for the twin.
        num_warps: Triton warps per program (kernel only).
        num_stages: Triton pipeline stages (kernel only).
        interpret: Run Pallas in interpret mode (kernel only); implies the kernel.
        chunk_entries: Entries per rematerialised chunk (twin only).

    Returns:
        The far-field contribution, leaf-major, shape ``(L, W, d)``.

    Raises:
        ValueError: If ``backend`` is not ``"pallas"`` or ``"jax"``.
    """
    if backend not in ("pallas", "jax"):
        raise ValueError(f"backend must be 'pallas' or 'jax'; got {backend!r}")
    if backend == "jax" and not interpret:
        return farfield_stein_jax(
            leaf_x,
            leaf_mask,
            mono_count,
            mono_com,
            mono_sums,
            src_offsets,
            src_node,
            h,
            chunk_entries=chunk_entries,
        )
    dtype = jnp.asarray(leaf_x).dtype
    return _farfield_stein_fused(
        jnp.asarray(leaf_x),
        jnp.asarray(leaf_mask),
        jnp.asarray(mono_count, dtype),
        jnp.asarray(mono_com, dtype),
        jnp.asarray(mono_sums, dtype),
        jnp.asarray(src_offsets),
        jnp.asarray(src_node),
        jnp.asarray(node_offsets),
        jnp.asarray(entry_perm),
        jnp.asarray(h, dtype),
        num_warps,
        num_stages,
        interpret,
    )
