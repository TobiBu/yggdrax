"""kNN forward against other codes: the cost of pure JAX (paper v2, B1 baseline).

Times one forward k-nearest-neighbour search, tree build included, at matched
``N``, ``Q``, ``k`` and dtype, for every code that is installed:

* ``yggdrax``: ``build_kdtree`` + ``query_neighbors(backend="tree")``, the
  search under ``yggdrax.applications.knn_stats`` (jitted, pure JAX);
* ``jaxkd``: ``jaxkd.build_and_query`` (pure JAX; Dodge, github.com/dodgebc/jaxkd);
* ``jaxkd-cuda``: the same with ``cuda=True`` (its optional CUDA extension; GPU);
* ``jztree``: ``jztree.knn.knn`` (JZ-Tree, Stücker et al., arXiv:2604.05885;
  CUDA FFI, so GPU only);
* ``ckdtree``: ``scipy.spatial.cKDTree`` on the host, all cores (context only).

Forward only: none of the baselines differentiates through the search, and the
paper states the gap as the price of pure JAX, not as a race.

Exactness is checked **before** timing, at every size, against a float64
``cKDTree`` reference on ``--check-queries`` of the queries: an arm's sorted
neighbour distances must match to ``--check-rtol`` and its neighbour index sets
must agree on at least ``--check-min-index-agreement`` of the checked queries
(float32 near-ties may swap a pair). An arm that fails the check is not timed
at that size (``--time-anyway`` overrides, for diagnosis).

Points and queries are independent uniform draws in ``[0, L)^3``, so there is no
self-match convention to reconcile between codes. ``--periodic`` searches with
the minimum image; codes without periodic support (jaxkd) are skipped there.

Each arm that runs out of memory, or whose median call exceeds
``--time-budget-s``, stops at that size (``sweep.stopped_early[arm]``); the
others continue. The JSON is rewritten after every size.

Examples::

    # local CPU check (jztree / jaxkd-cuda skip themselves)
    python bench/baselines/knn_forward.py --smoke --gpu-select none

    # A100 server (autocvd picks a free card, the org rule)
    python bench/baselines/knn_forward.py --gpu-select free \\
        --sizes 10000 100000 1000000 10000000 --ks 8 32 \\
        --output results/baselines/knn_forward_open.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (  # noqa: E402
    dump_json,
    is_out_of_memory,
    run_metadata,
    select_free_gpu,
    time_callable,
)

ARMS = ("yggdrax", "jaxkd", "jaxkd-cuda", "jztree", "ckdtree")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    p.add_argument("--ks", type=int, nargs="+", default=[8])
    p.add_argument("--queries-per-point", type=float, default=1.0)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--periodic", action="store_true", help="minimum-image search")
    p.add_argument("--box", type=float, default=1.0, help="side L of the cube")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--leaf-size", type=int, default=32, help="yggdrax KD leaf size")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-queries", type=int, default=2048)
    p.add_argument(
        "--check-rtol",
        type=float,
        default=None,
        help="relative tolerance on sorted distances (default 1e-4 f32, 1e-10 f64)",
    )
    p.add_argument("--check-min-index-agreement", type=float, default=None)
    p.add_argument("--time-anyway", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=120.0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument("--output", default="results/baselines/knn_forward.json")
    p.add_argument("--smoke", action="store_true", help="tiny sizes, 1 timed run")
    args = p.parse_args(argv)
    if args.smoke:
        args.sizes, args.ks = [2000, 8000], [8]
        args.runs, args.warmup, args.check_queries = 1, 1, 256
    if args.check_rtol is None:
        args.check_rtol = 1e-4 if args.dtype == "float32" else 1e-10
    if args.check_min_index_agreement is None:
        args.check_min_index_agreement = 0.99 if args.dtype == "float32" else 1.0
    return args


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _make_arms(
    args: argparse.Namespace, platform: str
) -> tuple[dict[str, Callable], dict[str, Any]]:
    """Build each requested arm's search ``fn(data, queries, k) -> (dist, idx)``.

    Parameters
    ----------
    args
        The parsed command line.
    platform
        ``jax.default_backend()``.

    Returns
    -------
    tuple
        The runnable arms, and per arm ``{available, reason, version}``.
    """
    import jax
    import numpy as np

    box = args.box if args.periodic else None
    fns: dict[str, Callable] = {}
    info: dict[str, Any] = {}

    def skip(name: str, reason: str) -> None:
        info[name] = {"available": False, "reason": reason}
        print(f"[knn-forward] skip {name}: {reason}")

    for name in args.arms:
        if name == "yggdrax":
            from yggdrax import build_kdtree, query_neighbors

            def ygg(data, queries, k, _box=box):
                tree = build_kdtree(data, leaf_size=args.leaf_size)
                idx, dist = query_neighbors(
                    tree, queries, k=k, backend="tree", box_size=_box
                )
                return dist, idx

            fns[name] = jax.jit(ygg, static_argnums=2)
            info[name] = {"available": True, "version": _version("yggdrax")}
        elif name in ("jaxkd", "jaxkd-cuda"):
            if args.periodic:
                skip(name, "jaxkd has no periodic boundaries")
                continue
            try:
                import jaxkd
            except Exception as exc:  # pragma: no cover - env-dependent
                skip(name, f"not installed ({type(exc).__name__}: {exc})")
                continue
            cuda = name == "jaxkd-cuda"
            if cuda:
                if platform != "gpu":
                    skip(name, "the jaxkd-cuda extension needs a GPU")
                    continue
                try:
                    import jaxkd_cuda  # noqa: F401
                except Exception as exc:  # pragma: no cover - env-dependent
                    skip(name, f"jaxkd-cuda not installed ({exc})")
                    continue

            def jkd(data, queries, k, _cuda=cuda):
                # jaxkd 0.1.2 mixes float32 and float64 in a lax.cond once x64
                # is on (as yggdrax needs), so it is traced and run without it;
                # float32 inputs are unaffected, float64 ones would be cast.
                with jax.enable_x64(False):
                    idx, dist = jaxkd.build_and_query(data, queries, k=k, cuda=_cuda)
                return dist, idx

            fns[name] = jkd  # build_and_query is jitted (k static) by jaxkd
            info[name] = {
                "available": True,
                "version": _version("jaxkd"),
                "note": "traced with jax_enable_x64 off (jaxkd 0.1.2 breaks under x64)",
            }
            if args.dtype == "float64":
                skip(name, "jaxkd runs without x64, so it cannot match float64")
                fns.pop(name)
        elif name == "jztree":
            if platform != "gpu":
                skip(name, "JZ-Tree is CUDA only (GPU)")
                continue
            try:
                import jztree.knn
            except Exception as exc:  # pragma: no cover - env-dependent
                skip(name, f"not installed ({type(exc).__name__}: {exc})")
                continue

            def jzt(data, queries, k, _box=box):
                rnn, inn = jztree.knn.knn(data, k, boxsize=_box, part_query=queries)
                return rnn, inn

            fns[name] = jax.jit(jzt, static_argnums=2)
            info[name] = {
                "available": True,
                "version": _version("jztree"),
                "backend_version": _version("jztree-cu12"),
            }
        elif name == "ckdtree":
            from scipy.spatial import cKDTree

            def ckd(data, queries, k, _box=box):
                tree = cKDTree(np.asarray(data), boxsize=_box)
                dist, idx = tree.query(np.asarray(queries), k=k, workers=-1)
                return dist.reshape(len(queries), k), idx.reshape(len(queries), k)

            fns[name] = ckd
            info[name] = {
                "available": True,
                "version": _version("scipy"),
                "note": "host, all cores; context only",
            }
    return fns, info


def _reference(data64, queries64, k: int, box):
    """Exact float64 neighbours of the check queries (cKDTree)."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(data64, boxsize=box).query(queries64, k=k, workers=-1)
    q = len(queries64)
    return dist.reshape(q, k), idx.reshape(q, k)


def _agreement(dist, idx, ref_dist, ref_idx, args) -> dict[str, Any]:
    """Compare an arm's neighbours with the reference on the check queries."""
    import numpy as np

    dist = np.sort(np.asarray(dist, dtype=np.float64), axis=1)
    idx = np.asarray(idx)
    rel = np.abs(dist - ref_dist) / np.maximum(ref_dist, 1e-300)
    same_set = np.all(np.sort(idx, axis=1) == np.sort(ref_idx, axis=1), axis=1)
    frac = float(np.mean(same_set))
    max_rel = float(np.max(rel))
    ok = bool(max_rel <= args.check_rtol and frac >= args.check_min_index_agreement)
    return {
        "ok": ok,
        "checked_queries": int(len(ref_dist)),
        "max_rel_distance_error": max_rel,
        "index_set_agreement": frac,
        "rtol": args.check_rtol,
        "min_index_agreement": args.check_min_index_agreement,
    }


def _block(out):
    import jax

    return jax.block_until_ready(out)


def main(argv=None) -> int:
    args = _parse_args(argv)
    select_free_gpu(args.gpu_select, tag="knn-forward")

    import jax

    # yggdrax needs x64 (not switched on at import since #97); the positions
    # keep --dtype.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    platform = jax.default_backend()
    fns, arm_info = _make_arms(args, platform)
    box = args.box if args.periodic else None
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    payload: dict[str, Any] = {
        "metadata": run_metadata(
            {
                "bench": "baselines/knn_forward",
                "walk": "yggdrax: build_kdtree + query_neighbors(backend='tree')",
            }
        ),
        "params": vars(args),
        "arms": arm_info,
        "results": [],
        "sweep": {"stopped_early": {}},
    }
    stopped: dict[str, Any] = payload["sweep"]["stopped_early"]
    rng = np.random.default_rng(args.seed)

    for n in args.sizes:
        q = max(1, int(round(args.queries_per_point * n)))
        # positions as the arms see them (rounded to --dtype, kept inside
        # [0, L) after rounding), and the reference computed on exactly those
        # rounded values in float64
        np_dtype = np.dtype(args.dtype)
        top = np.nextafter(np_dtype.type(args.box), np_dtype.type(0))
        data_np = np.minimum(rng.uniform(0.0, args.box, (n, 3)).astype(np_dtype), top)
        queries_np = np.minimum(
            rng.uniform(0.0, args.box, (q, 3)).astype(np_dtype), top
        )
        data64 = data_np.astype(np.float64)
        queries64 = queries_np.astype(np.float64)
        data = jnp.asarray(data_np, dtype=dtype)
        queries = jnp.asarray(queries_np, dtype=dtype)
        nc = min(args.check_queries, q)
        for k in args.ks:
            if k > n:
                continue
            ref_dist, ref_idx = _reference(data64, queries64[:nc], k, box)
            for name, fn in fns.items():
                if name in stopped:
                    continue
                row: dict[str, Any] = {"n": n, "q": q, "k": k, "arm": name}
                try:
                    t0 = time.perf_counter()
                    dist, idx = _block(fn(data, queries, k))
                    row["first_call_s"] = time.perf_counter() - t0
                    agree = _agreement(
                        np.asarray(dist)[:nc],
                        np.asarray(idx)[:nc],
                        ref_dist,
                        ref_idx,
                        args,
                    )
                    row["agreement"] = agree
                    print(
                        f"[knn-forward] n={n} k={k} {name}: agreement ok={agree['ok']} "
                        f"max_rel={agree['max_rel_distance_error']:.2e} "
                        f"sets={agree['index_set_agreement']:.4f}"
                    )
                    if not agree["ok"] and not args.time_anyway:
                        row["timing"] = None
                        row["not_timed"] = "disagrees with the exact reference"
                    elif row["first_call_s"] > 2 * args.time_budget_s:
                        # one call (compile included) is already far over budget:
                        # keep it as an upper bound, do not repeat it
                        row["timing"] = None
                        row["not_timed"] = "first call over 2x --time-budget-s"
                        stopped[name] = {
                            "reason": "time_budget",
                            "at_n": n,
                            "first_call_s": row["first_call_s"],
                        }
                    else:
                        timing = time_callable(
                            lambda: fn(data, queries, k),
                            warmup=args.warmup,
                            runs=args.runs,
                        )
                        row["timing"] = timing.as_dict()
                        print(
                            f"[knn-forward] n={n} k={k} {name}: "
                            f"median {timing.median_s * 1e3:.2f} ms"
                        )
                        if timing.median_s > args.time_budget_s:
                            stopped[name] = {
                                "reason": "time_budget",
                                "after_n": n,
                                "median_s": timing.median_s,
                            }
                except Exception as exc:
                    if not is_out_of_memory(exc):
                        raise
                    stopped[name] = {
                        "reason": "out_of_memory",
                        "at_n": n,
                        "k": k,
                        "detail": f"{type(exc).__name__}: {str(exc)[:400]}",
                    }
                    row["error"] = stopped[name]["detail"]
                    print(f"[knn-forward] {name} stopped at n={n}: out of memory")
                payload["results"].append(row)
            dump_json(payload, args.output)
    dump_json(payload, args.output)
    bad = [
        r
        for r in payload["results"]
        if r.get("agreement") is not None and not r["agreement"]["ok"]
    ]
    if bad:
        print(f"[knn-forward] {len(bad)} row(s) disagree with the exact reference")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
