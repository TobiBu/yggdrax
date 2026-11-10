"""Periodic ξ: accuracy and speed of yggdrax against TreeCorr and Corrfunc (B2).

For each catalogue (uniform, Thomas process) and size ``N``, in a periodic box:

* **exact reference**: soft and hard pair counts from every pair within reach
  (``scipy.spatial.cKDTree(boxsize=L)``, see ``_xi_common.exact_counts``);
* **yggdrax traced** (``--arms traced``): the natural estimator
  ``make_xi_statistic(backend="traced")`` on
  ``traced_soft_pair_counts(box_size=L)`` with the bin-aware window rule;
  forward time and value+grad time of ``L = Σ_b ξ_b²``;
* **yggdrax distributed** (``--arms distributed``): the same ξ from
  ``distributed_pair_counts`` on one device (exact acceptance, hard cut);
* **TreeCorr** (``--arms treecorr``): ``NNCorrelation(metric="Periodic")`` at its
  default ``bin_slop`` and at ``bin_slop=0``;
* **Corrfunc** (``--arms corrfunc``): ``theory.DD(periodic=True, boxsize=L)``;
  skipped, with the reason recorded, where it does not import (macOS arm64).

Accuracy: each code against its own exact definition (soft counts for yggdrax,
hard counts for the baselines), and the ξ of each baseline against ours. The
latter differs by the soft-against-hard window, which is not an error: the
exact soft-against-hard difference is reported next to it.

One JSON row per (catalogue, N, arm), rewritten after every row, so a long run
can be read while it goes and resumed with ``--resume``.

Local (CPU; Corrfunc skipped on macOS):

    PYTHONPATH=$PWD python bench/corrfunc/xi_box.py --sizes 10000 100000

A100 server (autocvd picks a free GPU, the org rule):

    PYTHONPATH=$PWD python bench/corrfunc/xi_box.py --platform gpu \\
        --gpu-select free --sizes 10000 100000 1000000

Results: ``bench/corrfunc/results/xi_box_<platform>.json`` (kept off main; the
paper branch holds results).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _xi_common as common  # noqa: E402

_ARMS = ("traced", "distributed", "treecorr", "corrfunc")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_args(p)
    common.add_problem_args(p)
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    p.add_argument("--arms", nargs="+", default=list(_ARMS), choices=_ARMS)
    p.add_argument(
        "--reference-max-n",
        type=int,
        default=1_000_000,
        help="skip the exact reference above this N",
    )
    p.add_argument(
        "--grad-max-n",
        type=int,
        default=10_000_000,
        help="skip value+grad timing above this N",
    )
    p.add_argument("--resume", action="store_true", help="keep rows already in --out")
    p.add_argument("--smoke", action="store_true", help="tiny sizes, 2 repeats")
    args = p.parse_args(argv)
    if args.smoke:
        args.sizes = [2000]
        args.repeats = 2
    return args


def _row_key(row) -> tuple:
    return (row["catalogue"], row["n"], row["arm"])


def _traced_arm(pos, args, edges, ref, rr) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.corrfunc.traced import (
        plan_traced_pair_counts,
        traced_soft_pair_counts,
    )
    from yggdrax.applications.point_process import make_xi_statistic

    t0 = time.perf_counter()
    cfg = plan_traced_pair_counts(
        pos,
        edges=jnp.asarray(edges),
        sharpness=args.sharpness,
        window_eps=args.eps,
        leaf_size=args.leaf_size,
        box_size=args.box,
    )
    plan_s = time.perf_counter() - t0
    stat = make_xi_statistic(
        edges,
        box_size=args.box,
        n_points=int(pos.shape[0]),
        sharpness=args.sharpness,
        eps=args.eps,
        backend="traced",
        traced_config=cfg,
    )
    fwd = jax.jit(stat.fn)

    def loss(p):
        xi, _ = stat.fn(p)
        return jnp.sum(xi * xi)

    vg = jax.jit(jax.value_and_grad(loss))
    t_fwd = common.time_call(lambda: jax.block_until_ready(fwd(pos)), args.repeats)
    xi, overflow = t_fwd.pop("result")
    diag = jax.jit(
        lambda p: traced_soft_pair_counts(
            p, jnp.asarray(edges), cfg, sharpness=args.sharpness, box_size=args.box
        )
    )(pos)
    row = {
        "plan_s": plan_s,
        "forward": t_fwd,
        "overflow": bool(overflow) or bool(diag.overflow),
        "xi": np.asarray(xi).tolist(),
        "dd": ((np.asarray(xi) + 1.0) * rr).tolist(),
        "config": {
            "leaf_size": cfg.leaf_size,
            "leaf_capacity": cfg.leaf_capacity,
            "max_pair_queue": cfg.max_pair_queue,
            "far_cap": cfg.far_cap,
            "near_cap": cfg.near_cap,
            "window_eps": cfg.window_eps,
        },
        "walk": {
            "num_leaves": int(diag.num_leaves),
            "far_count": int(diag.far_count),
            "near_count": int(diag.near_count),
            "peak_wavefront": int(diag.peak_wavefront),
        },
    }
    if pos.shape[0] <= args.grad_max_n:
        t_vg = common.time_call(lambda: jax.block_until_ready(vg(pos)), args.repeats)
        value, grad = t_vg.pop("result")
        row["value_and_grad"] = t_vg
        row["grad_finite"] = bool(jnp.all(jnp.isfinite(grad)))
        row["grad_over_forward"] = t_vg["median_s"] / t_fwd["median_s"]
    if ref is not None:
        dd = np.asarray(row["dd"])
        row["dd_rel_err_vs_exact_soft"] = common.max_rel_err(dd, ref["soft"], 1.0)
        row["dd_abs_err_vs_exact_soft"] = float(np.max(np.abs(dd - ref["soft"])))
    return row


def _distributed_arm(pos, args, edges, ref, rr) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.point_process import make_xi_statistic
    from yggdrax.distributed import make_mesh
    from yggdrax.distributed.stats import (
        DistributedStatsConfig,
        soft_window_cutoffs,
    )

    n = int(pos.shape[0])
    r_hi = soft_window_cutoffs(edges, args.sharpness, args.eps)[1]
    mesh = make_mesh(1)
    clustering = 1.0 if args._kind == "uniform" else 4.0
    for _attempt in range(5):
        cfg = DistributedStatsConfig.for_box(
            n,
            ndev=1,
            box_size=args.box,
            reach=r_hi,
            leaf_size=args.leaf_size,
            clustering=clustering,
        )
        stat = make_xi_statistic(
            edges,
            box_size=args.box,
            n_points=n,
            sharpness=args.sharpness,
            eps=args.eps,
            config=cfg,
            mesh=mesh,
        )
        fwd = jax.jit(stat.fn)
        xi, overflow = fwd(pos)
        if not bool(overflow):
            break
        clustering *= 2.0
    else:
        return {"skipped": f"overflow at clustering={clustering / 2}"}

    def loss(p):
        xi, _ = stat.fn(p)
        return jnp.sum(xi * xi)

    vg = jax.jit(jax.value_and_grad(loss))
    t_fwd = common.time_call(lambda: jax.block_until_ready(fwd(pos)), args.repeats)
    xi, overflow = t_fwd.pop("result")
    row = {
        "forward": t_fwd,
        "overflow": bool(overflow),
        "clustering_factor": clustering,
        "xi": np.asarray(xi).tolist(),
        "dd": ((np.asarray(xi) + 1.0) * rr).tolist(),
    }
    if n <= args.grad_max_n:
        t_vg = common.time_call(lambda: jax.block_until_ready(vg(pos)), args.repeats)
        t_vg.pop("result")
        row["value_and_grad"] = t_vg
        row["grad_over_forward"] = t_vg["median_s"] / t_fwd["median_s"]
    if ref is not None:
        dd = np.asarray(row["dd"])
        row["dd_rel_err_vs_exact_soft"] = common.max_rel_err(dd, ref["soft"], 1.0)
    return row


def _hard_arm(name, pos, args, edges, ref, rr_hard, bin_slop=None) -> dict:
    import numpy as np

    ok, why = common.baseline_available(name)
    if not ok:
        return {"skipped": why}
    if name == "treecorr":

        def run():
            return common.treecorr_counts(pos, edges, args.box, args.threads, bin_slop)

    else:

        def run():
            return common.corrfunc_counts(pos, edges, args.box, args.threads)

    t = common.time_call(run, args.repeats)
    counts = t.pop("result")
    row = {
        "forward": t,
        "threads": args.threads,
        "dd_hard": counts.tolist(),
        "xi_hard": (counts / rr_hard - 1.0).tolist(),
    }
    if bin_slop is not None:
        row["bin_slop"] = bin_slop
    if ref is not None:
        row["dd_rel_err_vs_exact_hard"] = common.max_rel_err(counts, ref["hard"], 1.0)
        row["dd_abs_err_vs_exact_hard"] = float(np.max(np.abs(counts - ref["hard"])))
    return row


def main(argv=None) -> None:
    args = _parse_args(argv)
    common.select_device(args)

    import jax
    import numpy as np

    from yggdrax.applications.point_process import natural_estimator_rr

    out = args.out or common.default_path("xi_box", "json", args)
    rows: list[dict] = []
    if args.resume and os.path.exists(out):
        with open(out) as f:
            rows = json.load(f).get("rows", [])
    done = {_row_key(r) for r in rows}
    edges = common.edges_of(args)
    meta = {
        "environment": common.environment(),
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "edges": edges.tolist(),
    }

    def flush():
        common.write_json(out, {**meta, "rows": rows})

    for kind in args.catalogues:
        args._kind = kind
        for n_req in args.sizes:
            pos = common.catalogue(kind, n_req, args)
            n = int(pos.shape[0])
            rr = natural_estimator_rr(
                edges,
                box_size=args.box,
                n_points=n,
                sharpness=args.sharpness,
                eps=args.eps,
            )
            rr_hard = common.hard_rr(edges, n, args.box)
            ref = None
            if n <= args.reference_max_n:
                key = (kind, n, "exact")
                if key in done:
                    prev = next(r for r in rows if _row_key(r) == key)
                    ref = {"soft": np.asarray(prev["dd_soft"]), "hard": prev["dd_hard"]}
                    ref["hard"] = np.asarray(ref["hard"])
                else:
                    r = common.exact_counts(pos, edges, args.sharpness, args.box)
                    ref = r
                    rows.append(
                        {
                            "catalogue": kind,
                            "n": n,
                            "arm": "exact",
                            "seconds": r["seconds"],
                            "pairs_within_ref": r["pairs_within_ref"],
                            "r_ref": r["r_ref"],
                            "dd_soft": r["soft"].tolist(),
                            "dd_hard": r["hard"].tolist(),
                            "xi_soft": (r["soft"] / rr - 1.0).tolist(),
                            "xi_hard": (r["hard"] / rr_hard - 1.0).tolist(),
                            # what a soft-vs-hard comparison of exact counts gives
                            "soft_vs_hard_ratio_err": _ratio_err(
                                r["hard"] / rr_hard, r["soft"] / rr, r["hard"]
                            ),
                        }
                    )
                    flush()
                    print(
                        f"{kind:8s} n={n:>8d} exact      {r['seconds']:8.2f} s "
                        f"({r['pairs_within_ref']} pairs)",
                        flush=True,
                    )
            arms: list[tuple[str, dict]] = []
            for arm in args.arms:
                if arm == "treecorr":
                    arms += [("treecorr", {}), ("treecorr_slop0", {"bin_slop": 0.0})]
                else:
                    arms.append((arm, {}))
            for label, kw in arms:
                if (kind, n, label) in done:
                    continue
                if label == "traced":
                    row = _traced_arm(pos, args, edges, ref, rr)
                elif label == "distributed":
                    row = _distributed_arm(pos, args, edges, ref, rr)
                else:
                    name = "treecorr" if label.startswith("treecorr") else "corrfunc"
                    row = _hard_arm(name, pos, args, edges, ref, rr_hard, **kw)
                row.update({"catalogue": kind, "n": n, "arm": label})
                traced = _find(rows, kind, n)
                if "xi_hard" in row and "xi" in traced:
                    row["ratio_err_vs_yggdrax"] = _ratio_err(
                        np.asarray(row["xi_hard"]) + 1.0,
                        np.asarray(traced["xi"]) + 1.0,
                        np.asarray(row["dd_hard"]),
                    )
                rows.append(row)
                flush()
                _print_row(row)
            jax.clear_caches()


def _ratio_err(a, b, counts, min_count: float = 100.0) -> float:
    """``max |a / b - 1|`` over bins holding at least ``min_count`` hard pairs.

    Compares ``1 + xi`` of a hard-bin count with the soft one. Bins with few
    pairs are dominated by the discreteness of the hard count and are left out.
    """
    import numpy as np

    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, counts))
    live = c >= min_count
    if not np.any(live):
        return float("nan")
    return float(np.max(np.abs(a[live] / b[live] - 1.0)))


def _find(rows, kind, n) -> dict:
    for r in rows:
        if r["catalogue"] == kind and r["n"] == n and r["arm"] == "traced":
            return r
    return {}


def _print_row(row) -> None:
    head = f"{row['catalogue']:8s} n={row['n']:>8d} {row['arm']:14s}"
    if "skipped" in row:
        print(f"{head} skipped: {row['skipped']}", flush=True)
        return
    fwd = row["forward"]["median_s"]
    msg = f"{head} fwd {fwd:9.4f} s"
    if "value_and_grad" in row:
        msg += f"  v+g {row['value_and_grad']['median_s']:9.4f} s"
    for k in ("dd_rel_err_vs_exact_soft", "dd_rel_err_vs_exact_hard"):
        if k in row:
            msg += f"  err {row[k]:.2e}"
    if "ratio_err_vs_yggdrax" in row:
        msg += f"  vs ygg {row['ratio_err_vs_yggdrax']:.2e}"
    if row.get("overflow"):
        msg += "  OVERFLOW"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
