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

from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
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

#: Fewest leaves at which the fused kernel is worth launching, for
#: ``backend="auto"``. One program owns one target leaf, so below a few hundred
#: leaves there is not enough work to fill the device and the launch is
#: overhead. Measured on an A100 (d = 3, leaf 32, theta = 0.5), speed-up of the
#: fused path over ``accumulate="scatter"`` on the *full* update:
#:
#: ======  =====  ==========  ==========  ==========
#:      N      L  f64 fwd     f64 fwd+dx  f32 fwd+dx
#: ======  =====  ==========  ==========  ==========
#:    500     16        0.63        0.74        1.09
#:   1000     32        0.66        0.74        0.94
#:   2000     63        1.02        1.10        1.74
#:   4000    125        0.98        0.90        1.66
#:   8000    250        1.73        1.24        2.05
#:  16000    500        2.85        1.23        1.50
#:  32000   1000        2.93        1.88        1.48
#: ======  =====  ==========  ==========  ==========
#:
#: The only clear losses are the two rows with fewer leaves than the device has
#: SMs (108); from L = 63 on it is a wash at worst and 2-3x at the sizes that
#: matter. 64 is where that boundary sits, so that is the threshold -- not a
#: round number chosen for looking like one.
_MIN_LEAVES_FOR_KERNEL = 64


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


def prefer_fused_nearfield(num_leaves: int) -> bool:
    """Whether the fused kernel is the right lane for a partition this size.

    Combines the two conditions: the machine can run it at all, and there is
    enough of it to run. Both the ``backend="auto"`` selector here and
    ``accumulate="auto"`` in :mod:`~yggdrax.applications.svgd.sampler` go
    through this, so they cannot disagree.

    It gates the **far**-field kernel too, though the threshold was measured on
    the near field. That is sound rather than convenient: both kernels put one
    program on one target leaf, so both are limited by the same thing below a
    few hundred leaves -- there is not enough work to fill the device and the
    launch is overhead. A far-field-specific threshold was not measured.

    Args:
        num_leaves: Leaves in the partition, i.e. programs the kernel launches.

    Returns:
        ``True`` when the kernel should be preferred over the pure-JAX paths.
    """
    return (
        int(num_leaves) >= _MIN_LEAVES_FOR_KERNEL and pallas_stein_nearfield_supported()
    )


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


class _Layout(NamedTuple):
    """Padded operands and static tile parameters shared by both kernels.

    Triton admits only power-of-two tile shapes, so the leaf-width axis and the
    dimension axis are both padded up and the padding is masked out. Deriving it
    once means the forward and the reverse cannot disagree about a shape, which
    would show up as a silently wrong gradient rather than an error.
    """

    tbl_x: Array
    tbl_s: Array
    tbl_mask: Array
    offsets: Array
    sources: Array
    bandwidth: Array
    num_leaves: int
    leaf_width: int
    num_dims: int
    width: int
    lane_width: int
    subtile: int
    tile: int
    num_warps: int
    num_entries: int

    @property
    def grid(self) -> tuple[int, int]:
        """The ``(leaf, target subtile)`` launch grid."""
        return (self.num_leaves, self.width // self.subtile)

    def name(self, stem: str) -> str:
        """Return a kernel name carrying the shapes it was specialised for.

        Args:
            stem: A prefix naming the kernel.

        Returns:
            The name.
        """
        return f"{stem}_t{self.subtile}s{self.tile}_w{self.width}_d{self.num_dims}"

    def spec_target(self) -> Any:
        """Block spec for a ``(1, Bt, dp)`` per-program slice of a table."""
        return pl.BlockSpec(
            (1, self.subtile, self.lane_width), lambda leaf, sub: (leaf, sub, 0)
        )

    def spec_target_mask(self) -> Any:
        """Block spec for a ``(1, Bt)`` per-program slice of the mask."""
        return pl.BlockSpec((1, self.subtile), lambda leaf, sub: (leaf, sub))

    def spec_table(self) -> Any:
        """Block spec for a full ``(L, W, dp)`` gather table."""
        return pl.BlockSpec(
            (self.num_leaves, self.width, self.lane_width),
            lambda leaf, sub: (0, 0, 0),
        )

    def spec_table_mask(self) -> Any:
        """Block spec for the full ``(L, W)`` validity table."""
        return pl.BlockSpec((self.num_leaves, self.width), lambda leaf, sub: (0, 0))

    def spec_offsets(self) -> Any:
        """Block spec for the ``(L + 1,)`` CSR offsets."""
        return pl.BlockSpec((self.num_leaves + 1,), lambda leaf, sub: (0,))

    def spec_sources(self) -> Any:
        """Block spec for the ``(P,)`` source-leaf list."""
        return pl.BlockSpec((self.num_entries,), lambda leaf, sub: (0,))

    def spec_scalar(self) -> Any:
        """Block spec for a ``(1,)`` scalar operand."""
        return pl.BlockSpec((1,), lambda leaf, sub: (0,))

    def out_struct(self) -> Any:
        """Shape/dtype of a ``(L, W, dp)`` leaf-major output."""
        return jax.ShapeDtypeStruct(
            (self.num_leaves, self.width, self.lane_width), self.tbl_x.dtype
        )

    def out_struct_lane(self) -> Any:
        """Shape/dtype of a ``(L, W)`` per-lane output."""
        return jax.ShapeDtypeStruct((self.num_leaves, self.width), self.tbl_x.dtype)

    def pad_like(self, values: Array) -> Array:
        """Pad a ``(L, W, d)`` array into the kernel's padded table layout.

        Args:
            values: An array in the caller's unpadded layout.

        Returns:
            The ``(L, W_pad, dp)`` array the block specs expect.
        """
        return jnp.pad(
            jnp.asarray(values, self.tbl_x.dtype),
            (
                (0, 0),
                (0, self.width - self.leaf_width),
                (0, self.lane_width - self.num_dims),
            ),
        )

    def unpad(self, out: Array) -> Array:
        """Strip the leaf-width and dimension padding from a kernel output.

        Args:
            out: A ``(L, W_pad, dp)`` kernel output.

        Returns:
            The ``(L, W, d)`` slice the caller asked for.
        """
        return out[:, : self.leaf_width, : self.num_dims]


def _prepare_layout(
    leaf_x: Array,
    leaf_s: Array,
    leaf_mask: Array,
    src_offsets: Array,
    src_leaf: Array,
    h: float | Float[Array, ""],
    *,
    target_subtile: int | None,
    source_tile: int | None,
    num_warps: int | None,
) -> _Layout | None:
    """Validate, pad and tile the operands for one kernel launch.

    Args:
        leaf_x: Positions, shape ``(L, W, d)``.
        leaf_s: Scores, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        src_offsets: CSR offsets, shape ``(L + 1,)``.
        src_leaf: Source leaf rows, shape ``(P,)``.
        h: Kernel bandwidth.
        target_subtile: Requested target lanes per program, or ``None``.
        source_tile: Requested source lanes per inner iteration, or ``None``.
        num_warps: Requested Triton warps, or ``None`` to derive one.

    Returns:
        The layout, or ``None`` when the partition is empty and there is nothing
        to launch.

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
        return None

    lane_width = _pow2_ceil(num_dims)
    width = _pow2_ceil(leaf_width)
    pad = ((0, 0), (0, width - leaf_width), (0, lane_width - num_dims))
    tbl_x = jnp.pad(leaf_x, pad)
    tbl_s = jnp.pad(leaf_s, pad)
    tbl_mask = jnp.pad(mask, pad[:2])

    subtile = _resolve_tile(target_subtile, width)
    tile = _resolve_tile(source_tile, width)
    if num_warps is None:
        num_warps = max(1, min(4, (subtile * tile) // 64))

    # A zero-length source list still has to compile; give Pallas a shape it can
    # block and let the CSR bounds (all equal) make the loop empty.
    if int(sources.shape[0]) == 0:
        sources = jnp.zeros((1,), dtype=jnp.int32)

    return _Layout(
        tbl_x=tbl_x,
        tbl_s=tbl_s,
        tbl_mask=tbl_mask,
        offsets=offsets,
        sources=sources,
        bandwidth=bandwidth,
        num_leaves=num_leaves,
        leaf_width=leaf_width,
        num_dims=num_dims,
        width=width,
        lane_width=lane_width,
        subtile=subtile,
        tile=tile,
        num_warps=int(num_warps),
        num_entries=int(sources.shape[0]),
    )


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
    layout = _prepare_layout(
        leaf_x,
        leaf_s,
        leaf_mask,
        src_offsets,
        src_leaf,
        h,
        target_subtile=target_subtile,
        source_tile=source_tile,
        num_warps=num_warps,
    )
    if layout is None:
        return jnp.zeros_like(leaf_x)

    def _kernel(*refs: KernelRef) -> None:
        return _stein_nearfield_kernel(
            *refs,
            num_dims=layout.num_dims,
            lane_width=layout.lane_width,
            leaf_width=layout.width,
            source_tile=layout.tile,
            include_self=include_self,
        )

    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (layout.num_leaves, layout.width, layout.lane_width), layout.tbl_x.dtype
        ),
        in_specs=[
            layout.spec_target(),
            layout.spec_target_mask(),
            layout.spec_table(),
            layout.spec_table(),
            layout.spec_table_mask(),
            layout.spec_offsets(),
            layout.spec_sources(),
            layout.spec_scalar(),
        ],
        out_specs=layout.spec_target(),
        grid=layout.grid,
        interpret=bool(interpret),
        name=layout.name("stein_nearfield"),
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=layout.num_warps,
            num_stages=int(num_stages),
        ),
    )
    out = kernel(
        layout.tbl_x,
        layout.tbl_mask,
        layout.tbl_x,
        layout.tbl_s,
        layout.tbl_mask,
        layout.offsets,
        layout.sources,
        layout.bandwidth,
    )
    return layout.unpad(out)


def _stein_nearfield_bwd_kernel(
    tgt_x_ref: KernelRef,
    tgt_s_ref: KernelRef,
    tgt_g_ref: KernelRef,
    tgt_mask_ref: KernelRef,
    tbl_x_ref: KernelRef,
    tbl_s_ref: KernelRef,
    tbl_g_ref: KernelRef,
    tbl_mask_ref: KernelRef,
    offsets_ref: KernelRef,
    src_leaf_ref: KernelRef,
    h_ref: KernelRef,
    dx_ref: KernelRef,
    ds_ref: KernelRef,
    dh_ref: KernelRef,
    *,
    num_dims: int,
    lane_width: int,
    leaf_width: int,
    source_tile: int,
    include_self: bool,
) -> None:
    """Reverse of :func:`_stein_nearfield_kernel`, also entirely in registers.

    **This is the whole point of the exercise.** Reverse-mode autodiff of the
    forward would transpose its gathers into scatters and put the contended
    ``atomicAdd`` straight back; here the ``j``-side accumulation is a reduction
    over the tile's source axis, so it stays in the kernel.

    That works because the near pair list is *symmetric*. A target leaf's
    sources are exactly the leaves for which it is itself a source, so one loop
    over ``sources(l)`` reaches every pair leaf ``l`` takes part in, in either
    role, and both roles reduce over the same tile axis:

    * ``l`` as **target** (lanes ``p``, sources ``q``) gives the ``x_i`` side;
    * ``l`` as **source** (targets ``q``, lanes ``p``) gives the ``x_j`` and
      ``s_j`` sides, whose separation vector is the same tile's negated.

    Args:
        tgt_x_ref: This subtile's positions, shape ``(1, Bt, dp)``.
        tgt_s_ref: This subtile's scores, shape ``(1, Bt, dp)``.
        tgt_g_ref: This subtile's output cotangent, shape ``(1, Bt, dp)``.
        tgt_mask_ref: Which lanes are real, shape ``(1, Bt)``.
        tbl_x_ref: Full leaf-major position table, shape ``(L, W, dp)``.
        tbl_s_ref: Full leaf-major score table, shape ``(L, W, dp)``.
        tbl_g_ref: Full leaf-major cotangent table, shape ``(L, W, dp)``.
        tbl_mask_ref: Full leaf-major validity table, shape ``(L, W)``.
        offsets_ref: CSR offsets, shape ``(L + 1,)``.
        src_leaf_ref: Source leaf rows, shape ``(P,)``.
        h_ref: Kernel bandwidth, shape ``(1,)``.
        dx_ref: **Output**, cotangent of ``leaf_x``, shape ``(1, Bt, dp)``.
        ds_ref: **Output**, cotangent of ``leaf_s``, shape ``(1, Bt, dp)``.
        dh_ref: **Output**, this subtile's partial ``d/dh``, shape ``(1, Bt)``.
        num_dims: The true dimension ``d``. Static.
        lane_width: The padded dimension ``dp >= d``. Static.
        leaf_width: Lanes per leaf block ``W``, a power of two. Static.
        source_tile: Source lanes per inner iteration. Static.
        include_self: Whether each leaf is also its own source.

    Returns:
        None. The results are the writes to the three output refs.
    """
    leaf = pl.program_id(0)
    tvalid = tgt_mask_ref[0, :]
    x_p = [tgt_x_ref[0, :, k] for k in range(num_dims)]
    s_p = [tgt_s_ref[0, :, k] for k in range(num_dims)]
    g_p = [tgt_g_ref[0, :, k] for k in range(num_dims)]
    bandwidth = h_ref[0]
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    half_inv_h2 = 0.5 * inv_h2
    inv_h3 = inv_h2 / bandwidth
    zero = jnp.zeros_like(x_p[0])
    acc0 = tuple(zero for _ in range(2 * num_dims + 1))
    num_tiles = leaf_width // source_tile

    def _source_block(sid: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
        """Add leaf ``sid``'s contribution to this leaf's three cotangents."""

        def _tile(t: Any, acc: tuple[Any, ...]) -> tuple[Any, ...]:
            lanes = pl.dslice(t * source_tile, source_tile)
            x_q = [tbl_x_ref[sid, lanes, k] for k in range(num_dims)]
            s_q = [tbl_s_ref[sid, lanes, k] for k in range(num_dims)]
            g_q = [tbl_g_ref[sid, lanes, k] for k in range(num_dims)]
            active = tvalid[:, None] & tbl_mask_ref[sid, lanes][None, :]
            sep = [x_p[k][:, None] - x_q[k][None, :] for k in range(num_dims)]
            dist_sq = sep[0] * sep[0]
            for k in range(1, num_dims):
                dist_sq = dist_sq + sep[k] * sep[k]
            kern = jnp.where(active, jnp.exp(-dist_sq * half_inv_h2), 0.0)
            scaled = kern * inv_h2

            # p as target: contract this lane's cotangent with the source's
            # score and with the separation.
            g_dot_s = g_p[0][:, None] * s_q[0][None, :]
            g_dot_r = g_p[0][:, None] * sep[0]
            # p as source: contract the *source lane's* cotangent instead.
            gq_dot_s = g_q[0][None, :] * s_p[0][:, None]
            gq_dot_r = g_q[0][None, :] * sep[0]
            for k in range(1, num_dims):
                g_dot_s = g_dot_s + g_p[k][:, None] * s_q[k][None, :]
                g_dot_r = g_dot_r + g_p[k][:, None] * sep[k]
                gq_dot_s = gq_dot_s + g_q[k][None, :] * s_p[k][:, None]
                gq_dot_r = gq_dot_r + g_q[k][None, :] * sep[k]

            coef = scaled * (g_dot_s + g_dot_r * inv_h2) + scaled * (
                gq_dot_s - gq_dot_r * inv_h2
            )
            out = list(acc)
            for k in range(num_dims):
                out[k] = out[k] + jnp.sum(
                    scaled * (g_p[k][:, None] - g_q[k][None, :]) - coef * sep[k],
                    axis=1,
                )
                out[num_dims + k] = out[num_dims + k] + jnp.sum(
                    kern * g_q[k][None, :], axis=1
                )
            # d/dh, taken on the target side only, which visits every directed
            # pair exactly once across the whole grid.
            out[2 * num_dims] = out[2 * num_dims] + jnp.sum(
                kern
                * inv_h3
                * (dist_sq * (g_dot_s + g_dot_r * inv_h2) - 2.0 * g_dot_r),
                axis=1,
            )
            return tuple(out)

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
        dx_ref[0, :, k] = jnp.where(tvalid, acc[k], zero)
        ds_ref[0, :, k] = jnp.where(tvalid, acc[num_dims + k], zero)
    for k in range(num_dims, lane_width):
        dx_ref[0, :, k] = zero
        ds_ref[0, :, k] = zero
    dh_ref[0, :] = jnp.where(tvalid, acc[2 * num_dims], zero)


def nearfield_stein_pallas_bwd(
    cotangent: Float[Array, "L W d"],
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
) -> tuple[Float[Array, "L W d"], Float[Array, "L W d"], Float[Array, ""]]:
    """Fused reverse of :func:`nearfield_stein_pallas`.

    **Precondition: the near pair list must be symmetric** -- every directed
    pair ``(a, b)`` present with ``(b, a)``. The reverse pass relies on it to
    reach a leaf's ``x_j`` and ``s_j`` cotangents from the leaf's *own* source
    list, which is what keeps the accumulation inside the kernel. The SVGD
    partition guarantees it and :func:`~yggdrax.applications.svgd.sampler.assemble_svgd_topology`
    raises if the traversal ever emits otherwise, so this is a checked property
    rather than an assumption; a caller assembling a CSR by hand must honour it.

    Args:
        cotangent: Cotangent of the forward output, shape ``(L, W, d)``.
        leaf_x: Positions, shape ``(L, W, d)``.
        leaf_s: Scores, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        src_offsets: CSR offsets, shape ``(L + 1,)``.
        src_leaf: Source leaf rows, shape ``(P,)``.
        h: Kernel bandwidth.
        include_self: Whether each leaf is also its own source.
        target_subtile: Target lanes per program.
        source_tile: Source lanes per inner iteration.
        num_warps: Triton warps per program.
        num_stages: Triton pipeline stages.
        interpret: Run Pallas in interpret mode.

    Returns:
        The cotangents of ``leaf_x``, ``leaf_s`` and ``h``.
    """
    layout = _prepare_layout(
        leaf_x,
        leaf_s,
        leaf_mask,
        src_offsets,
        src_leaf,
        h,
        target_subtile=target_subtile,
        source_tile=source_tile,
        num_warps=num_warps,
    )
    if layout is None:
        zeros = jnp.zeros_like(jnp.asarray(leaf_x))
        return zeros, zeros, jnp.zeros((), jnp.asarray(leaf_x).dtype)
    tbl_g = layout.pad_like(cotangent)

    def _kernel(*refs: KernelRef) -> None:
        return _stein_nearfield_bwd_kernel(
            *refs,
            num_dims=layout.num_dims,
            lane_width=layout.lane_width,
            leaf_width=layout.width,
            source_tile=layout.tile,
            include_self=include_self,
        )

    kernel = pl.pallas_call(
        _kernel,
        out_shape=(
            layout.out_struct(),
            layout.out_struct(),
            layout.out_struct_lane(),
        ),
        in_specs=[
            layout.spec_target(),
            layout.spec_target(),
            layout.spec_target(),
            layout.spec_target_mask(),
            layout.spec_table(),
            layout.spec_table(),
            layout.spec_table(),
            layout.spec_table_mask(),
            layout.spec_offsets(),
            layout.spec_sources(),
            layout.spec_scalar(),
        ],
        out_specs=(
            layout.spec_target(),
            layout.spec_target(),
            layout.spec_target_mask(),
        ),
        grid=layout.grid,
        interpret=bool(interpret),
        name=layout.name("stein_nearfield_bwd"),
        **pallas_backend_kwargs(
            "triton",
            interpret=bool(interpret),
            num_warps=layout.num_warps,
            num_stages=int(num_stages),
        ),
    )
    d_x, d_s, d_h = kernel(
        layout.tbl_x,
        layout.tbl_s,
        tbl_g,
        layout.tbl_mask,
        layout.tbl_x,
        layout.tbl_s,
        tbl_g,
        layout.tbl_mask,
        layout.offsets,
        layout.sources,
        layout.bandwidth,
    )
    return layout.unpad(d_x), layout.unpad(d_s), jnp.sum(d_h)


def _integer_cotangent(values: Array) -> np.ndarray:
    """Return the empty cotangent JAX expects for a non-differentiable input.

    ``leaf_mask``, ``src_offsets`` and ``src_leaf`` describe the *partition*, a
    discrete object the update is not differentiated with respect to. JAX's
    tangent type for such an input is ``float0``, a zero-sized dtype, and a
    ``custom_vjp`` reverse rule has to return one per primal.

    Args:
        values: The primal input.

    Returns:
        A ``float0`` array of the primal's shape.
    """
    return np.zeros(jnp.shape(values), dtype=jax.dtypes.float0)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11))
def _nearfield_stein_fused(
    leaf_x: Array,
    leaf_s: Array,
    leaf_mask: Array,
    src_offsets: Array,
    src_leaf: Array,
    h: Array,
    include_self: bool,
    target_subtile: int | None,
    source_tile: int | None,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
) -> Array:
    """The fused kernel with :func:`nearfield_stein_pallas_bwd` as its reverse.

    Args:
        leaf_x: Positions, shape ``(L, W, d)``.
        leaf_s: Scores, shape ``(L, W, d)``.
        leaf_mask: Lane validity, shape ``(L, W)``.
        src_offsets: CSR offsets, shape ``(L + 1,)``.
        src_leaf: Source leaf rows, shape ``(P,)``.
        h: Kernel bandwidth.
        include_self: Whether each leaf is also its own source. ``nondiff``.
        target_subtile: Target lanes per program. ``nondiff``.
        source_tile: Source lanes per inner iteration. ``nondiff``.
        num_warps: Triton warps per program. ``nondiff``.
        num_stages: Triton pipeline stages. ``nondiff``.
        interpret: Run Pallas in interpret mode. ``nondiff``.

    Returns:
        The near-field contribution, shape ``(L, W, d)``.
    """
    return nearfield_stein_pallas(
        leaf_x,
        leaf_s,
        leaf_mask,
        src_offsets,
        src_leaf,
        h,
        include_self=include_self,
        target_subtile=target_subtile,
        source_tile=source_tile,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )


def _nearfield_stein_fused_fwd(
    leaf_x,
    leaf_s,
    leaf_mask,
    src_offsets,
    src_leaf,
    h,
    include_self,
    target_subtile,
    source_tile,
    num_warps,
    num_stages,
    interpret,
):
    out = nearfield_stein_pallas(
        leaf_x,
        leaf_s,
        leaf_mask,
        src_offsets,
        src_leaf,
        h,
        include_self=include_self,
        target_subtile=target_subtile,
        source_tile=source_tile,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )
    # The residual is the inputs, not the (L, W, d) x (L, W, d) products: the
    # reverse kernel recomputes them in registers, which is the entire reason
    # this rule exists.
    return out, (leaf_x, leaf_s, leaf_mask, src_offsets, src_leaf, h)


def _nearfield_stein_fused_bwd(
    include_self,
    target_subtile,
    source_tile,
    num_warps,
    num_stages,
    interpret,
    residual,
    cotangent,
):
    leaf_x, leaf_s, leaf_mask, src_offsets, src_leaf, h = residual
    d_x, d_s, d_h = nearfield_stein_pallas_bwd(
        cotangent,
        leaf_x,
        leaf_s,
        leaf_mask,
        src_offsets,
        src_leaf,
        h,
        include_self=include_self,
        target_subtile=target_subtile,
        source_tile=source_tile,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )
    return (
        d_x,
        d_s,
        _integer_cotangent(leaf_mask),
        _integer_cotangent(src_offsets),
        _integer_cotangent(src_leaf),
        jnp.asarray(d_h, jnp.result_type(h)),
    )


_nearfield_stein_fused.defvjp(_nearfield_stein_fused_fwd, _nearfield_stein_fused_bwd)


def nearfield_stein(
    leaf_x: Float[Array, "L W d"],
    leaf_s: Float[Array, "L W d"],
    leaf_mask: Array,
    src_offsets: Array,
    src_leaf: Array,
    h: float | Float[Array, ""],
    *,
    include_self: bool = True,
    backend: str = "auto",
    target_subtile: int | None = None,
    source_tile: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
    chunk_pairs: int | None = None,
) -> Float[Array, "L W d"]:
    """Differentiable leaf-major Stein near field, fused kernel or twin.

    The one entry point callers should use. Both lanes compute the same thing to
    round-off; they differ in *how reverse mode gets there*. The twin's reverse
    is autodiff, which transposes its gathers into scatters; the kernel's is a
    hand-written rule that accumulates the ``j`` side in registers.

    Args:
        leaf_x: Positions, leaf-major and padded, shape ``(L, W, d)``.
        leaf_s: Scores in the same layout, shape ``(L, W, d)``.
        leaf_mask: Which lanes hold a real particle, shape ``(L, W)``.
        src_offsets: CSR offsets into ``src_leaf``, shape ``(L + 1,)``.
        src_leaf: Source leaf row of each directed near pair, shape ``(P,)``.
            Must be a *symmetric* pair list -- see
            :func:`nearfield_stein_pallas_bwd`.
        h: Kernel bandwidth.
        include_self: Whether each leaf is also its own source.
        backend: ``"auto"`` (default) uses the kernel where
            :func:`pallas_stein_nearfield_supported` says it can run *and* there
            are at least :data:`_MIN_LEAVES_FOR_KERNEL` leaves to keep the
            device busy, and the twin otherwise; ``"pallas"`` forces the kernel;
            ``"jax"`` forces the twin. Forcing the kernel on an unsupported
            machine raises rather than silently falling back.
        target_subtile: Target lanes per program (kernel only).
        source_tile: Source lanes per inner iteration (kernel only).
        num_warps: Triton warps per program (kernel only).
        num_stages: Triton pipeline stages (kernel only).
        interpret: Run Pallas in interpret mode (kernel only). Implies the
            kernel, since interpret mode needs no GPU.
        chunk_pairs: Directed pairs per rematerialised chunk (twin only).

    Returns:
        The near-field contribution, leaf-major, shape ``(L, W, d)``.

    Raises:
        ValueError: If ``backend`` is not one of the three names.
    """
    if backend not in ("auto", "pallas", "jax"):
        raise ValueError(f"backend must be 'auto', 'pallas' or 'jax'; got {backend!r}")
    use_kernel = (
        interpret
        or backend == "pallas"
        or (backend == "auto" and prefer_fused_nearfield(jnp.shape(leaf_x)[0]))
    )
    if not use_kernel:
        return nearfield_stein_jax(
            leaf_x,
            leaf_s,
            leaf_mask,
            src_offsets,
            src_leaf,
            h,
            include_self=include_self,
            chunk_pairs=chunk_pairs,
        )
    return _nearfield_stein_fused(
        jnp.asarray(leaf_x),
        jnp.asarray(leaf_s, dtype=jnp.asarray(leaf_x).dtype),
        jnp.asarray(leaf_mask),
        jnp.asarray(src_offsets),
        jnp.asarray(src_leaf),
        jnp.asarray(h, dtype=jnp.asarray(leaf_x).dtype),
        include_self,
        target_subtile,
        source_tile,
        num_warps,
        num_stages,
        interpret,
    )
