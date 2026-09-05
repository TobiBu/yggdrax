"""A fused, leaf-major Pallas kernel for the Stein near field.

The pure-JAX near field evaluates cross-leaf pairs by materialising, per chunk
of leaf pairs, a dense ``(chunk, W, W, d)`` difference tensor that XLA writes to
HBM, and then reduces it -- either by scattering both directions of each
unordered pair (``accumulate="scatter"``) or by a segmented reduction over a
directed pair list (``accumulate="segment"``). Measured at N = 1e5 on an A100,
near field only:

==========================================  =========  =========
strategy                                      float64    float32
==========================================  =========  =========
gather + arithmetic only (the floor)          22.60 ms   20.47 ms
halved pairs, two scatters                    28.60 ms  278.90 ms
directed pairs, segment_sum, chunked          56.20 ms   32.50 ms
==========================================  =========  =========

Neither is good enough under *differentiation*, and the reason is structural:
the transpose of a gather is a scatter, so reverse mode reintroduces exactly the
contended ``atomicAdd`` the segmented forward removed. No rearrangement of JAX
primitives escapes that.

This module is the escape. One Pallas program owns a single **target leaf** (a
vector of ``W`` target lanes) and streams that leaf's source leaves in a
``fori_loop`` driven by a CSR offset table; the ``W x W`` products live in
registers and never reach HBM, and the output is written leaf-major so placing
it is a *permutation*, not a scatter. The design is jaccpot's
``pallas/nearfield_fused_leaf.py``, which documents the same pathology for the
Plummer-softened gravity kernel; the arithmetic and the signature are this
application's own.

Two things are deliberate:

* :func:`nearfield_stein_jax` is a pure-JAX twin with the **same signature**. It
  is the correctness reference the kernel is tested against, and the fallback on
  any machine without an Ampere-or-later GPU. It is not scaffolding.
* The kernel is opt-in behind :func:`pallas_stein_nearfield_supported`. The two
  existing accumulations stay exactly as they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from yggdrax.applications.svgd._pallas_compat import KernelRef, pallas_backend_kwargs

if TYPE_CHECKING:
    # Under a type checker Pallas is present, so ``pl`` is the module and the
    # kernel body's ``pl.program_id`` / ``pl.dslice`` type-check. At runtime it
    # is optional, and every entry point that touches it guards on ``pl is None``.
    from jax.experimental import pallas as pl
else:
    try:  # pragma: no cover - import is environment-dependent
        from jax.experimental import pallas as pl
    except Exception:  # pragma: no cover - Pallas is optional
        pl = None


#: Default width of both the target subtile (target lanes per program) and the
#: source tile (source lanes per inner iteration). Clamped to a power of two not
#: exceeding the padded leaf width, because Triton only admits power-of-two tile
#: shapes. 32 x 32 is the measured optimum at ``leaf_size = 32``.
_DEFAULT_TILE = 32


def _pow2_floor(value: int) -> int:
    """Return the largest power of two not exceeding ``value`` (at least 1).

    Args:
        value: Requested width.

    Returns:
        A power of two in ``[1, value]``.
    """
    if value <= 1:
        return 1
    return 1 << (int(value).bit_length() - 1)


def _pow2_ceil(value: int) -> int:
    """Return the smallest power of two at least ``value`` (at least 1).

    Args:
        value: Requested width.

    Returns:
        A power of two ``>= value``.
    """
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _resolve_tile(requested: int | None, leaf_width: int) -> int:
    """Return a power-of-two tile width dividing ``leaf_width``.

    Args:
        requested: Requested width, or ``None`` for the default.
        leaf_width: Padded lanes per leaf block, itself a power of two.

    Returns:
        A power of two in ``[1, max(1, leaf_width)]``.
    """
    want = _DEFAULT_TILE if requested is None else int(requested)
    return max(1, min(_pow2_floor(want), _pow2_floor(max(1, leaf_width))))


def pallas_stein_nearfield_supported() -> bool:
    """Return whether this machine can run the fused Stein near-field kernel.

    Returns:
        ``True`` only when Pallas and its Triton backend import and the default
        backend is an Ampere-or-later GPU. Device-discovery failures return
        ``False`` rather than raising, because this is called to pick a lane.
    """
    if pl is None:
        return False
    if jax.default_backend() != "gpu":
        return False
    try:
        device = jax.devices()[0]
    except Exception:  # pragma: no cover - backend discovery is environmental
        return False
    capability = getattr(device, "compute_capability", None)
    if capability is None:
        return False
    try:
        return float(capability) >= 8.0
    except (TypeError, ValueError):  # pragma: no cover - vendor-specific strings
        return False


def _target_rows_from_offsets(src_offsets: Array, num_entries: int) -> Array:
    """Return the target leaf row of each CSR entry.

    Args:
        src_offsets: CSR offsets, shape ``(L + 1,)``, non-decreasing.
        num_entries: Length of the source list, ``P``.

    Returns:
        Target leaf row per entry, shape ``(P,)``. Entries past
        ``src_offsets[-1]`` (capacity padding) map to the last leaf and are
        masked separately.
    """
    positions = jnp.arange(num_entries, dtype=src_offsets.dtype)
    rows = jnp.searchsorted(src_offsets, positions, side="right") - 1
    return jnp.clip(rows, 0, src_offsets.shape[0] - 2)


def nearfield_stein_jax(
    leaf_x: Float[Array, "L W d"],
    leaf_s: Float[Array, "L W d"],
    leaf_mask: Array,
    src_offsets: Array,
    src_leaf: Array,
    h: float | Float[Array, ""],
    *,
    include_self: bool = True,
    chunk_pairs: int | None = None,
) -> Float[Array, "L W d"]:
    """Leaf-major Stein near field in pure JAX -- the kernel's twin.

    Computes, for every target lane ``i`` of leaf ``l``::

        sum over source leaves s of l, over lanes j of s:
            k * s_j + k * (x_i - x_j) / h^2,   k = exp(-|x_i - x_j|^2 / 2h^2)

    plus, when ``include_self``, leaf ``l`` paired with itself (which is where
    the ``i == j`` diagonal term ``s_i`` comes from).

    This is the parity reference for :func:`nearfield_stein_pallas` and the
    fallback where the kernel cannot run. It is deliberately written against the
    *same* CSR arguments so the two are interchangeable at the call site.

    Args:
        leaf_x: Positions, leaf-major and padded, shape ``(L, W, d)``.
        leaf_s: Target scores in the same layout, shape ``(L, W, d)``.
        leaf_mask: Which lanes hold a real particle, shape ``(L, W)``.
        src_offsets: CSR offsets into ``src_leaf`` per target leaf, shape
            ``(L + 1,)``.
        src_leaf: Source leaf row of each directed near pair, shape ``(P,)``,
            ascending in target leaf. Entries beyond ``src_offsets[-1]`` are
            capacity padding and are ignored.
        h: Kernel bandwidth.
        include_self: Whether each leaf is also its own source.
        chunk_pairs: Directed pairs per rematerialised chunk. ``None`` picks the
            whole list, which is what the parity tests want; large problems
            should pass a chunk that keeps ``(chunk, W, W, d)`` in memory.

    Returns:
        The near-field contribution, leaf-major, shape ``(L, W, d)``.
    """
    leaf_x = jnp.asarray(leaf_x)
    leaf_s = jnp.asarray(leaf_s, dtype=leaf_x.dtype)
    valid = jnp.asarray(leaf_mask).astype(bool)
    src_offsets = jnp.asarray(src_offsets)
    src_leaf = jnp.asarray(src_leaf)

    num_leaves, leaf_width, num_dims = leaf_x.shape
    num_entries = int(src_leaf.shape[0])
    inv_h2 = 1.0 / (h * h)

    def _pair_block(rows_t: Array, rows_s: Array, live: Array) -> Array:
        """Return one chunk of directed pairs' contribution to their targets."""
        x_t, x_s = leaf_x[rows_t], leaf_x[rows_s]
        s_s = leaf_s[rows_s]
        mask_s = valid[rows_s] & live[:, None]
        diff = x_t[:, :, None, :] - x_s[:, None, :, :]
        kern = jnp.exp(-jnp.sum(diff * diff, axis=-1) * (0.5 * inv_h2))
        kern = jnp.where(mask_s[:, None, :], kern, 0.0)[..., None]
        return jnp.sum(kern * s_s[:, None, :, :] + kern * diff * inv_h2, axis=2)

    acc = jnp.zeros((num_leaves, leaf_width, num_dims), dtype=leaf_x.dtype)

    if num_entries:
        rows_t_all = _target_rows_from_offsets(src_offsets, num_entries)
        live_all = jnp.arange(num_entries, dtype=src_offsets.dtype) < src_offsets[-1]
        chunk = (
            num_entries if chunk_pairs is None else min(int(chunk_pairs), num_entries)
        )
        num_chunks = -(-num_entries // chunk)
        pad = num_chunks * chunk - num_entries
        rows_s_all = src_leaf
        if pad:
            tail_t = jnp.full((pad,), num_leaves - 1, dtype=rows_t_all.dtype)
            tail_s = jnp.zeros((pad,), dtype=rows_s_all.dtype)
            rows_t_all = jnp.concatenate([rows_t_all, tail_t])
            rows_s_all = jnp.concatenate([rows_s_all, tail_s])
            live_all = jnp.concatenate([live_all, jnp.zeros((pad,), dtype=bool)])

        def _step(carry: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, None]:
            contrib = _pair_block(xs[0], xs[1], xs[2])
            return (
                carry
                + jax.ops.segment_sum(
                    contrib, xs[0], num_segments=num_leaves, indices_are_sorted=True
                ),
                None,
            )

        acc, _ = lax.scan(
            _step,
            acc,
            (
                rows_t_all.reshape(num_chunks, chunk),
                rows_s_all.reshape(num_chunks, chunk),
                live_all.reshape(num_chunks, chunk),
            ),
        )

    if include_self:
        rows = jnp.arange(num_leaves, dtype=jnp.int32)
        acc = acc + _pair_block(rows, rows, jnp.ones((num_leaves,), dtype=bool))

    return acc * valid[..., None].astype(leaf_x.dtype)


def _stein_nearfield_kernel(
    tgt_x_ref: KernelRef,
    tgt_mask_ref: KernelRef,
    tbl_x_ref: KernelRef,
    tbl_s_ref: KernelRef,
    tbl_mask_ref: KernelRef,
    offsets_ref: KernelRef,
    src_leaf_ref: KernelRef,
    h_ref: KernelRef,
    out_ref: KernelRef,
    *,
    num_dims: int,
    lane_width: int,
    leaf_width: int,
    source_tile: int,
    include_self: bool,
) -> None:
    """Accumulate one target subtile's Stein near field entirely in registers.

    Sources arrive as leaf *rows* through a CSR offset table and are gathered
    inside the kernel from the full leaf-major tables, so the dense
    ``(P, W, W, d)`` tensor the pure-JAX path materialises never exists, and no
    padding to a worst-case neighbour count is needed: the trip count is this
    leaf's own.

    The inner loop walks each source leaf in tiles of ``source_tile`` lanes and
    forms a ``(Bt, source_tile)`` outer product per dimension. Broadcasting one
    source lane at a time against the target vector also works and is simpler,
    but is 1.6x slower in float64 (51.7 vs 32.9 ms at N = 1e5): one scalar load
    per lane leaves too little work in flight to hide its latency.

    Args:
        tgt_x_ref: Target positions for this subtile, shape ``(1, Bt, dp)``.
        tgt_mask_ref: Which target lanes are real, shape ``(1, Bt)``.
        tbl_x_ref: FULL leaf-major position table, shape ``(L, W, dp)`` -- not
            narrowed to this program, because the gather indexes it by leaf row.
        tbl_s_ref: Full leaf-major score table, shape ``(L, W, dp)``.
        tbl_mask_ref: Full leaf-major validity table, shape ``(L, W)``.
        offsets_ref: CSR offsets, shape ``(L + 1,)``.
        src_leaf_ref: Source leaf rows, shape ``(P,)``.
        h_ref: Kernel bandwidth, shape ``(1,)``.
        out_ref: **Output**, shape ``(1, Bt, dp)``.
        num_dims: The true dimension ``d``. Static.
        lane_width: The padded dimension ``dp >= d``. Static.
        leaf_width: Lanes per leaf block ``W``, a power of two. Static.
        source_tile: Source lanes per inner iteration, a power of two dividing
            ``leaf_width``. Static.
        include_self: Whether to add the target leaf's own block.

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
    acc0 = tuple(zero for _ in range(num_dims))
    num_tiles = leaf_width // source_tile

    def _source_block(sid: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
        """Add every lane of source leaf ``sid`` to the running accumulator."""

        def _tile(t: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
            lanes = pl.dslice(t * source_tile, source_tile)
            src_x = [tbl_x_ref[sid, lanes, k] for k in range(num_dims)]
            src_s = [tbl_s_ref[sid, lanes, k] for k in range(num_dims)]
            active = tvalid[:, None] & tbl_mask_ref[sid, lanes][None, :]
            diff = [tgt_x[k][:, None] - src_x[k][None, :] for k in range(num_dims)]
            dist_sq = diff[0] * diff[0]
            for k in range(1, num_dims):
                dist_sq = dist_sq + diff[k] * diff[k]
            kern = jnp.where(active, jnp.exp(-dist_sq * half_inv_h2), 0.0)
            return tuple(
                acc[k]
                + jnp.sum(kern * src_s[k][None, :] + kern * diff[k] * inv_h2, axis=1)
                for k in range(num_dims)
            )

        return lax.fori_loop(0, num_tiles, _tile, acc)

    acc = lax.fori_loop(
        offsets_ref[leaf],
        offsets_ref[leaf + 1],
        lambda p, acc: _source_block(src_leaf_ref[p], acc),
        acc0,
    )
    if include_self:
        acc = _source_block(leaf, acc)

    for k in range(num_dims):
        out_ref[0, :, k] = jnp.where(tvalid, acc[k], zero)
    for k in range(num_dims, lane_width):
        out_ref[0, :, k] = zero


def nearfield_stein_pallas(
    leaf_x: Float[Array, "L W d"],
    leaf_s: Float[Array, "L W d"],
    leaf_mask: Array,
    src_offsets: Array,
    src_leaf: Array,
    h: float | Float[Array, ""],
    *,
    include_self: bool = True,
    target_subtile: int | None = None,
    source_tile: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
) -> Float[Array, "L W d"]:
    """Fused leaf-major Stein near field: one Pallas program per target leaf.

    Numerically equivalent to :func:`nearfield_stein_jax`, which is the twin the
    parity tests compare against.

    Args:
        leaf_x: Positions, leaf-major and padded, shape ``(L, W, d)``.
        leaf_s: Scores in the same layout, shape ``(L, W, d)``.
        leaf_mask: Which lanes hold a real particle, shape ``(L, W)``.
        src_offsets: CSR offsets into ``src_leaf``, shape ``(L + 1,)``.
        src_leaf: Source leaf row of each directed near pair, shape ``(P,)``,
            ascending in target leaf. Entries beyond ``src_offsets[-1]`` are
            capacity padding and are never read.
        h: Kernel bandwidth.
        include_self: Whether each leaf is also its own source, which is where
            the ``i == j`` diagonal term comes from.
        target_subtile: Target lanes per program. ``None`` uses 32, clamped to a
            power of two not exceeding the padded leaf width.
        source_tile: Source lanes per inner iteration. ``None`` uses 32, clamped
            the same way. Both measured flat within ~5 % over 8/16/32 at
            ``W = 32``; 32 is marginally best in float32.
        num_warps: Triton warps per program. ``None`` derives one per 64 tile
            elements, capped at 4 -- 8 warps measured 10-25 % slower.
        num_stages: Triton pipeline stages.
        interpret: Run Pallas in interpret mode (CPU semantics, no lowering).

    Returns:
        The near-field contribution, leaf-major, shape ``(L, W, d)``.

    Raises:
        RuntimeError: If ``jax.experimental.pallas`` could not be imported.
        ValueError: If the input shapes are mutually inconsistent.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    leaf_x = jnp.asarray(leaf_x)
    dtype = leaf_x.dtype
    leaf_s = jnp.asarray(leaf_s, dtype=dtype)
    mask = jnp.asarray(leaf_mask).astype(bool)
    offsets = jnp.asarray(src_offsets, dtype=jnp.int32)
    sources = jnp.asarray(src_leaf, dtype=jnp.int32)
    bandwidth = jnp.asarray([h], dtype=dtype)

    if leaf_x.ndim != 3:
        raise ValueError("leaf_x must have shape (L, W, d)")
    if leaf_s.shape != leaf_x.shape:
        raise ValueError("leaf_s must have the same shape as leaf_x")
    num_leaves, leaf_width, num_dims = (int(v) for v in leaf_x.shape)
    if mask.shape != (num_leaves, leaf_width):
        raise ValueError("leaf_mask must have shape (L, W)")
    if offsets.shape != (num_leaves + 1,):
        raise ValueError("src_offsets must have shape (L + 1,)")

    if num_leaves == 0 or leaf_width == 0:
        return jnp.zeros_like(leaf_x)

    # Triton admits only power-of-two tile shapes, so both the lane axis and the
    # leaf-width axis are padded up. The kernel reads dimensions [0, d) and
    # lanes the mask marks live; the padding is sliced off before returning.
    lane_width = _pow2_ceil(num_dims)
    width = _pow2_ceil(leaf_width)
    pad_dims = lane_width - num_dims
    pad_lanes = width - leaf_width
    tbl_x = jnp.pad(leaf_x, ((0, 0), (0, pad_lanes), (0, pad_dims)))
    tbl_s = jnp.pad(leaf_s, ((0, 0), (0, pad_lanes), (0, pad_dims)))
    tbl_mask = jnp.pad(mask, ((0, 0), (0, pad_lanes)))

    subtile = _resolve_tile(target_subtile, width)
    tile = _resolve_tile(source_tile, width)
    num_subtiles = width // subtile
    if num_warps is None:
        num_warps = max(1, min(4, (subtile * tile) // 64))

    # A zero-length source list still has to compile; give Pallas a shape it can
    # block and let the CSR bounds (all equal) make the loop empty.
    if int(sources.shape[0]) == 0:
        sources = jnp.zeros((1,), dtype=jnp.int32)
    num_entries = int(sources.shape[0])

    def _kernel(*refs: KernelRef) -> None:
        return _stein_nearfield_kernel(
            *refs,
            num_dims=num_dims,
            lane_width=lane_width,
            leaf_width=width,
            source_tile=tile,
            include_self=include_self,
        )

    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((num_leaves, width, lane_width), dtype),
        in_specs=[
            pl.BlockSpec((1, subtile, lane_width), lambda leaf, sub: (leaf, sub, 0)),
            pl.BlockSpec((1, subtile), lambda leaf, sub: (leaf, sub)),
            # Full gather tables: indexed by a data-dependent source leaf row.
            pl.BlockSpec((num_leaves, width, lane_width), lambda leaf, sub: (0, 0, 0)),
            pl.BlockSpec((num_leaves, width, lane_width), lambda leaf, sub: (0, 0, 0)),
            pl.BlockSpec((num_leaves, width), lambda leaf, sub: (0, 0)),
            pl.BlockSpec((num_leaves + 1,), lambda leaf, sub: (0,)),
            pl.BlockSpec((num_entries,), lambda leaf, sub: (0,)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
        ],
        out_specs=pl.BlockSpec(
            (1, subtile, lane_width), lambda leaf, sub: (leaf, sub, 0)
        ),
        grid=(num_leaves, num_subtiles),
        interpret=bool(interpret),
        name=f"stein_nearfield_t{subtile}s{tile}_w{width}_d{num_dims}",
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        ),
    )
    out = kernel(tbl_x, tbl_mask, tbl_x, tbl_s, tbl_mask, offsets, sources, bandwidth)
    return out[:, :leaf_width, :num_dims]
