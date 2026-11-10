"""Shared setup of the periodic ξ benches (B2): catalogues, references, arms.

Every bench in this directory parses its command line and calls
:func:`select_device` (from ``bench/differentiability/_a5_common.py``) before it
imports JAX, because GPU selection (autocvd, the org rule) must happen before
JAX initialises CUDA.

The problem is ξ in a periodic unit box: ``N`` points, soft log bins, the
natural estimator ``ξ = DD / RR - 1`` with the analytic ``RR`` of the box
(:func:`yggdrax.applications.point_process.natural_estimator_rr`). Catalogues
are uniform or a Thomas process from :mod:`yggdrax.applications.point_process`
(no external simulator, decision D2).

References. The exact soft ``DD = Σ_{i<j} W_b(r_ij)`` (minimum image) and the
exact hard counts come from one list of every pair closer than ``r_ref``, built
by ``scipy.spatial.cKDTree(boxsize=L)`` in row chunks. ``r_ref`` is where every
window has dropped below ``1e-15``, so the pairs beyond it change no count
measurably. The window is evaluated in float64 numpy.

Baselines. TreeCorr (``NNCorrelation``, ``metric="Periodic"``) and Corrfunc
(``theory.DD``, ``periodic=True``) count HARD log bins. Their counts are
compared with the exact hard counts (their own error), and their ξ with ours
(the soft-against-hard window difference, not an error). Corrfunc has no
macOS-arm64 wheel; its arm is skipped with the reason recorded wherever it
cannot be imported.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, os.path.join(_REPO, "bench", "differentiability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _a5_common import (  # noqa: E402  (path set up above)
    add_common_args,
    environment,
    select_device,
    write_json,
)

__all__ = [
    "add_common_args",
    "add_problem_args",
    "catalogue",
    "corrfunc_counts",
    "default_path",
    "environment",
    "exact_counts",
    "hard_rr",
    "select_device",
    "time_call",
    "treecorr_counts",
    "write_json",
]


def add_problem_args(p) -> None:
    """Box, binning and catalogue options shared by the ξ benches."""
    p.add_argument("--box", type=float, default=1.0, help="periodic box length L")
    p.add_argument("--r-min", type=float, default=0.004)
    p.add_argument("--r-max", type=float, default=0.04)
    p.add_argument("--nbins", type=int, default=10)
    p.add_argument("--sharpness", type=float, default=100.0, help="in ln r")
    p.add_argument("--eps", type=float, default=1e-6, help="window_eps and RR cut")
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument(
        "--catalogues",
        nargs="+",
        default=["uniform", "thomas"],
        choices=["uniform", "thomas"],
    )
    p.add_argument("--thomas-sigma", type=float, default=0.01, help="cluster size")
    p.add_argument("--thomas-beta", type=float, default=1.0, help="profile shape")
    p.add_argument("--thomas-children", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5, help="timed calls per arm")
    p.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="threads of TreeCorr and Corrfunc",
    )


def default_path(name: str, ext: str, args) -> str:
    out = os.path.join(_HERE, "results")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"{name}_{args.platform}.{ext}")


def edges_of(args):
    import numpy as np

    return np.geomspace(args.r_min, args.r_max, args.nbins + 1)


# --------------------------------------------------------------------------
# catalogues
# --------------------------------------------------------------------------


def catalogue(kind: str, n: int, args, seed: int | None = None):
    """``(n, 3)`` positions in ``[0, L)``: uniform, or a Thomas process.

    The Thomas catalogue has ``n // children`` parents with ``children`` points
    each (``n`` is rounded down to a multiple), Gaussian offsets of rms ``sigma``
    per axis for ``beta = 1`` (:func:`~yggdrax.applications.point_process.thomas_positions`).
    """
    import jax
    import jax.numpy as jnp

    seed = args.seed if seed is None else seed
    key = jax.random.PRNGKey(seed)
    if kind == "uniform":
        return jax.random.uniform(key, (n, 3), dtype=jnp.float64) * args.box
    if kind == "thomas":
        from yggdrax.applications.point_process import draw_thomas, thomas_positions

        m = int(args.thomas_children)
        draws = draw_thomas(key, max(1, n // m), m)
        theta = jnp.asarray([args.thomas_sigma / args.box, args.thomas_beta])
        return thomas_positions(theta, draws, 1.0) * args.box
    raise ValueError(kind)


# --------------------------------------------------------------------------
# exact references
# --------------------------------------------------------------------------


def _windows(r, edges, sharpness):
    """Soft log windows ``(len(r), nbins)`` in float64 numpy."""
    import numpy as np

    u = np.log(np.maximum(r, np.finfo(np.float64).tiny))[:, None]
    gate = 0.5 * (1.0 + np.tanh(0.5 * sharpness * (u - np.log(edges)[None, :])))
    return gate[:, :-1] - gate[:, 1:]


def exact_counts(positions, edges, sharpness: float, box: float, chunks: int = 0):
    """Exact soft and hard pair counts (minimum image), with the pair count.

    Returns ``dict(soft=(nbins,), hard=(nbins,), pairs_within_ref=int,
    r_ref=float, seconds=float)``. Hard bins are ``[e_b, e_{b+1})``.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    x = np.asarray(positions, dtype=np.float64) % box
    x[x >= box] = 0.0
    n = x.shape[0]
    e = np.asarray(edges, dtype=np.float64)
    r_ref = float(e[-1] * math.exp(math.log(1e15) / sharpness))
    if r_ref >= 0.5 * box:
        raise ValueError(f"r_ref = {r_ref:.4g} reaches L/2")
    full = cKDTree(x, boxsize=box)
    if chunks <= 0:
        chunks = max(1, n // 50_000)
    soft = np.zeros(len(e) - 1)
    hard = np.zeros(len(e) - 1)
    pairs = 0
    for rows in np.array_split(np.arange(n), chunks):
        part = cKDTree(x[rows], boxsize=box)
        coo = part.sparse_distance_matrix(full, r_ref, output_type="ndarray")
        i = rows[coo["i"]]
        keep = i < coo["j"]
        r = coo["v"][keep]
        pairs += int(r.size)
        soft += _windows(r, e, sharpness).sum(axis=0)
        hard += np.histogram(r, bins=e)[0]
    return {
        "soft": soft,
        "hard": hard,
        "pairs_within_ref": pairs,
        "r_ref": r_ref,
        "seconds": time.perf_counter() - t0,
    }


def hard_rr(edges, n: int, box: float):
    """Analytic hard-bin ``RR`` of ``n`` uniform points in a periodic cube."""
    import numpy as np

    e = np.asarray(edges, dtype=np.float64)
    shell = 4.0 / 3.0 * math.pi * (e[1:] ** 3 - e[:-1] ** 3)
    return 0.5 * n * (n - 1) * shell / box**3


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def time_call(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    """Run ``fn`` once untimed-warm (its own compile), then ``repeats`` times.

    ``fn`` must block until its result is ready. Returns the first call's time
    (compile + run for a jitted function) and the median / min of the rest.
    """
    t0 = time.perf_counter()
    out = fn()
    first = time.perf_counter() - t0
    runs = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        runs.append(time.perf_counter() - t0)
    return {
        "first_s": first,
        "median_s": statistics.median(runs),
        "min_s": min(runs),
        "runs_s": runs,
        "result": out,
    }


# --------------------------------------------------------------------------
# external baselines (hard bins)
# --------------------------------------------------------------------------


def treecorr_counts(
    positions, edges, box: float, threads: int, bin_slop: float | None = None
):
    """Unordered-pair hard counts from TreeCorr, periodic. Raises ImportError."""
    import numpy as np
    import treecorr  # pyright: ignore[reportMissingImports]

    x = np.asarray(positions, dtype=np.float64) % box
    e = np.asarray(edges, dtype=np.float64)
    kw: dict[str, Any] = {}
    if bin_slop is not None:
        kw["bin_slop"] = float(bin_slop)
    cat = treecorr.Catalog(x=x[:, 0], y=x[:, 1], z=x[:, 2])
    nn = treecorr.NNCorrelation(
        min_sep=float(e[0]),
        max_sep=float(e[-1]),
        nbins=len(e) - 1,
        bin_type="Log",
        metric="Periodic",
        xperiod=box,
        yperiod=box,
        zperiod=box,
        num_threads=int(threads),
        verbose=0,
        **kw,
    )
    nn.process(cat, num_threads=int(threads))
    # TreeCorr's auto-correlation counts each unordered pair once
    return np.asarray(nn.npairs, dtype=np.float64)


def corrfunc_counts(positions, edges, box: float, threads: int):
    """Unordered-pair hard counts from Corrfunc ``theory.DD``, periodic.

    Corrfunc bins in ``r`` with arbitrary edges (log edges are fine) and counts
    ordered pairs for an auto-correlation, so the result is halved. Raises
    ImportError where Corrfunc is not installed (no macOS-arm64 wheel).
    """
    import numpy as np
    from Corrfunc.theory.DD import DD  # pyright: ignore[reportMissingImports]

    x = np.asarray(positions, dtype=np.float64) % box
    res = DD(
        autocorr=1,
        nthreads=int(threads),
        binfile=np.asarray(edges, dtype=np.float64),
        X1=x[:, 0].copy(),
        Y1=x[:, 1].copy(),
        Z1=x[:, 2].copy(),
        periodic=True,
        boxsize=float(box),
    )
    return np.asarray(res["npairs"], dtype=np.float64) / 2.0


def baseline_available(name: str) -> tuple[bool, str]:
    """Whether a baseline imports here, and the reason if not."""
    try:
        if name == "treecorr":
            import treecorr  # noqa: F401  # pyright: ignore[reportMissingImports]
        elif name == "corrfunc":
            from Corrfunc.theory.DD import (  # noqa: F401  # pyright: ignore[reportMissingImports]
                DD,
            )
        else:
            raise ValueError(name)
    except ImportError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def max_rel_err(got, ref, floor: float = 0.0) -> float:
    """``max_b |got_b / ref_b - 1|`` over bins with ``ref_b > floor``."""
    import numpy as np

    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    live = ref > floor
    if not np.any(live):
        return float("nan")
    return float(np.max(np.abs(got[live] / ref[live] - 1.0)))
