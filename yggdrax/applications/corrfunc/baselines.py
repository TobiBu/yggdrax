"""Validation baselines for the differentiable pair-count estimator.

The always-available reference is a brute-force :math:`O(N^2)` pair count
(exact, chunked to bound memory) in both hard-binned and soft-binned forms.
The soft-binned brute force is the key control: comparing it against the
tree-accelerated soft estimator isolates the *tree/MAC approximation* error
from the *soft-binning* error.

Thin wrappers around ``Corrfunc`` and ``TreeCorr`` are also provided for
external cross-checks. They import lazily and raise an informative error if the
package is absent, so they are exercised only where those libraries are
installed (typically the GPU servers), while the brute-force reference keeps
the local test suite self-contained.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from yggdrax.applications.corrfunc.binning import (
    hard_bin_weights,
    soft_bin_weights,
)


def brute_force_pair_counts(
    positions: Float[Array, "n 3"],
    edges: Float[Array, " nbins+1"],
    *,
    log: bool = True,
    chunk: int = 1024,
) -> Float[Array, " nbins"]:
    """Exact hard-binned counts of unordered pairs, :math:`O(N^2)`.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        edges: Radial bin edges, shape ``(nbins + 1,)``.
        log: Bin in ``log`` separation (must match the estimator).
        chunk: Row-block size bounding the pairwise memory to ``chunk * n``.

    Returns:
        Per-bin count of unordered pairs (each ``i < j`` counted once).
    """
    pos = np.asarray(positions)
    n = pos.shape[0]
    nbins = int(edges.shape[0]) - 1
    counts = np.zeros(nbins, dtype=np.float64)
    for i in range(0, n, chunk):
        block = pos[i : i + chunk]
        d = block[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1))
        rows = np.arange(block.shape[0])
        # Keep only ordered pairs (global row index < global col index) so each
        # unordered pair is counted exactly once and self-pairs are excluded.
        global_rows = (i + rows)[:, None]
        cols = np.arange(n)[None, :]
        keep = global_rows < cols
        r_keep = r[keep]
        w = np.asarray(hard_bin_weights(jnp.asarray(r_keep), edges, log=log))
        counts += w.sum(axis=0)
    return jnp.asarray(counts)


def brute_force_soft_pair_counts(
    positions: Float[Array, "n 3"],
    edges: Float[Array, " nbins+1"],
    sharpness: float,
    *,
    log: bool = True,
    chunk: int = 1024,
) -> Float[Array, " nbins"]:
    """Exact soft-binned counts of unordered pairs, :math:`O(N^2)`.

    Same accounting as :func:`brute_force_pair_counts` but with the smooth
    :func:`soft_bin_weights` window. This is the control the tree-accelerated
    soft estimator is validated against.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        edges: Radial bin edges, shape ``(nbins + 1,)``.
        sharpness: Soft-window sharpness (must match the estimator).
        log: Bin in ``log`` separation (must match the estimator).
        chunk: Row-block size bounding the pairwise memory to ``chunk * n``.

    Returns:
        Per-bin soft count of unordered pairs.
    """
    pos = np.asarray(positions)
    n = pos.shape[0]
    nbins = int(edges.shape[0]) - 1
    counts = np.zeros(nbins, dtype=np.float64)
    for i in range(0, n, chunk):
        block = pos[i : i + chunk]
        d = block[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1))
        rows = np.arange(block.shape[0])
        global_rows = (i + rows)[:, None]
        cols = np.arange(n)[None, :]
        keep = global_rows < cols
        r_keep = jnp.asarray(r[keep])
        w = np.asarray(soft_bin_weights(r_keep, edges, sharpness, log=log))
        counts += w.sum(axis=0)
    return jnp.asarray(counts)


def corrfunc_pair_counts(
    positions: Float[Array, "n 3"],
    edges: Float[Array, " nbins+1"],
    *,
    boxsize: float | None = None,
    nthreads: int = 1,
) -> np.ndarray:
    """Hard-binned pair counts via ``Corrfunc`` (``theory.DD``).

    Lazy import; raises a helpful error if Corrfunc is not installed. Corrfunc
    bins linearly in separation, so pass linear ``edges``.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        edges: Linear radial bin edges, shape ``(nbins + 1,)``.
        boxsize: Periodic box size, or None for a non-periodic count.
        nthreads: OpenMP threads for Corrfunc.

    Returns:
        Per-bin unordered-pair counts as a numpy array of length ``nbins``.

    Raises:
        ImportError: If Corrfunc is not installed.
    """
    try:
        from Corrfunc.theory.DD import DD  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Corrfunc is not installed; install it to use this baseline "
            "(the brute-force reference is always available)."
        ) from exc

    pos = np.asarray(positions, dtype=np.float64)
    bins = np.asarray(edges, dtype=np.float64)
    x, y, z = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    res = DD(
        autocorr=1,
        nthreads=nthreads,
        binfile=bins,
        X1=x,
        Y1=y,
        Z1=z,
        periodic=boxsize is not None,
        boxsize=boxsize if boxsize is not None else 0.0,
    )
    # Corrfunc returns ordered self-pair counts including both (i,j) and (j,i);
    # halve to match the unordered convention of the brute-force reference.
    return np.asarray(res["npairs"], dtype=np.float64) / 2.0


def treecorr_pair_counts(
    positions: Float[Array, "n 3"],
    edges: Float[Array, " nbins+1"],
) -> np.ndarray:
    """Hard-binned pair counts via ``TreeCorr`` (``NNCorrelation``, 3D).

    Lazy import; raises a helpful error if TreeCorr is not installed.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        edges: Radial bin edges, shape ``(nbins + 1,)``.

    Returns:
        Per-bin unordered-pair counts as a numpy array of length ``nbins``.

    Raises:
        ImportError: If TreeCorr is not installed.
    """
    try:
        import treecorr  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TreeCorr is not installed; install it to use this baseline "
            "(the brute-force reference is always available)."
        ) from exc

    pos = np.asarray(positions, dtype=np.float64)
    bins = np.asarray(edges, dtype=np.float64)
    cat = treecorr.Catalog(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2])
    nn = treecorr.NNCorrelation(
        min_sep=float(bins[0]),
        max_sep=float(bins[-1]),
        nbins=len(bins) - 1,
        bin_type="Log",
    )
    nn.process(cat)
    return np.asarray(nn.npairs, dtype=np.float64)
