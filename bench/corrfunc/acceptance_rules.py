"""Periodic ξ: the bin-aware window rule against the geometric rule (B2).

Same problem as ``xi_box.py`` (periodic box, soft log bins, uniform and Thomas
catalogues). For one ``N`` per catalogue, sweep

* the window rule over ``--eps-list`` (``window_eps``: the bound per particle pair and
  bin), and
* the geometric (opening-angle) rule over ``--thetas``,

optionally both again with the smooth rule (``--smooth-band beta``). Each
point records the largest per-bin relative error of ``DD`` against the exact
soft counts, the forward time, the value+grad time and the walk's list sizes.
Then, for each target error, the fastest point of each rule that meets it
(**matched accuracy**) and the ratio of their times.

Local (CPU):

    PYTHONPATH=$PWD python bench/corrfunc/acceptance_rules.py --n 30000

A100 server (autocvd picks a free GPU, the org rule):

    PYTHONPATH=$PWD python bench/corrfunc/acceptance_rules.py --platform gpu \\
        --gpu-select free --n 1000000

Results: ``bench/corrfunc/results/acceptance_rules_<platform>.{json,png}`` (kept
off main). ``--replot JSON`` redraws the figure only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _xi_common as common  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_args(p)
    common.add_problem_args(p)
    p.add_argument("--n", type=int, default=30_000)
    p.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
        help="window_eps values of the sweep (--eps stays the RR cut)",
    )
    p.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.5, 0.4, 0.3],
        help="opening angles (theta 0.3 already costs ~12x the window rule at N=3e4)",
    )
    p.add_argument(
        "--smooth-band",
        type=float,
        default=0.0,
        help="also run every point with the smooth rule at this beta (> 1)",
    )
    p.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    )
    p.add_argument("--no-grad", action="store_true", help="skip value+grad timing")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n, args.repeats = 3000, 2
        args.eps_list, args.thetas, args.targets = [1e-3, 1e-6], [0.6, 0.2], [1e-2]
    return args


def _points(args):
    """``(family, label, plan kwargs)`` for every point of the sweep."""
    out = []
    bands = [1.0] + ([args.smooth_band] if args.smooth_band > 1.0 else [])
    for beta in bands:
        tag = "" if beta == 1.0 else f", smooth beta={beta:g}"
        fam = "" if beta == 1.0 else " smooth"
        for eps in args.eps_list:
            out.append(
                (
                    "window" + fam,
                    f"window eps={eps:g}{tag}",
                    {"acceptance": "window", "window_eps": eps, "smooth_band": beta},
                )
            )
        for theta in args.thetas:
            out.append(
                (
                    "geometric" + fam,
                    f"geometric theta={theta:g}{tag}",
                    {"acceptance": "geometric", "theta": theta, "smooth_band": beta},
                )
            )
    return out


def _run_point(pos, args, edges, ref, kw) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from yggdrax.applications.corrfunc.traced import (
        plan_traced_pair_counts,
        traced_soft_pair_counts,
    )

    e = jnp.asarray(edges)
    t0 = time.perf_counter()
    try:
        cfg = plan_traced_pair_counts(
            pos,
            edges=e,
            sharpness=args.sharpness,
            leaf_size=args.leaf_size,
            box_size=args.box,
            sub_block=args.sub_block,
            **kw,
        )
    except RuntimeError as exc:
        return {"skipped": f"planner: {exc}"}
    plan_s = time.perf_counter() - t0

    def counts(p):
        return traced_soft_pair_counts(
            p, e, cfg, sharpness=args.sharpness, box_size=args.box
        )

    fwd = jax.jit(lambda p: counts(p).counts)
    t_fwd = common.time_call(lambda: jax.block_until_ready(fwd(pos)), args.repeats)
    dd = np.asarray(t_fwd.pop("result"))
    diag = jax.jit(counts)(pos)
    row = {
        "plan_s": plan_s,
        "forward": t_fwd,
        "overflow": bool(diag.overflow),
        "dd": dd.tolist(),
        "rel_err": common.max_rel_err(dd, ref["soft"], 1.0),
        "abs_err": float(np.max(np.abs(dd - ref["soft"]))),
        "walk": {
            "num_leaves": int(diag.num_leaves),
            "far_count": int(diag.far_count),
            "near_count": int(diag.near_count),
            "band_count": int(diag.band_count),
            "peak_wavefront": int(diag.peak_wavefront),
            "sub_count": int(diag.sub_count),
        },
        "sub_block": cfg.sub_block,
    }
    if not args.no_grad:
        rr = jax.lax.stop_gradient(jnp.asarray(ref["soft"]) + 1.0)

        def loss(p):
            return jnp.sum((counts(p).counts / rr - 1.0) ** 2)

        vg = jax.jit(jax.value_and_grad(loss))
        t_vg = common.time_call(lambda: jax.block_until_ready(vg(pos)), args.repeats)
        t_vg.pop("result")
        row["value_and_grad"] = t_vg
    return row


def _matched(rows, targets, families, time_key) -> list[dict]:
    """Fastest point per family meeting each target error."""
    out = []
    for kind in sorted({r["catalogue"] for r in rows}):
        for tgt in targets:
            entry = {"catalogue": kind, "target": tgt, "time": time_key}
            for fam in families:
                ok = [
                    r
                    for r in rows
                    if r["catalogue"] == kind
                    and r["family"] == fam
                    and "skipped" not in r
                    and not r["overflow"]
                    and time_key in r
                    and r["rel_err"] <= tgt
                ]
                if ok:
                    best = min(ok, key=lambda r: r[time_key]["median_s"])
                    entry[fam] = {
                        "label": best["label"],
                        "seconds": best[time_key]["median_s"],
                        "rel_err": best["rel_err"],
                    }
                else:
                    entry[fam] = None
            w, g = entry.get("window"), entry.get("geometric")
            if w and g:
                entry["geometric_over_window"] = g["seconds"] / w["seconds"]
            out.append(entry)
    return out


def _plot(payload, path) -> None:
    from _a5_common import import_pyplot

    plt = import_pyplot()
    rows = [r for r in payload["rows"] if "skipped" not in r]
    kinds = sorted({r["catalogue"] for r in rows})
    keys = ["forward"] + (
        ["value_and_grad"] if any("value_and_grad" in r for r in rows) else []
    )
    fig, axes = plt.subplots(
        len(keys),
        len(kinds),
        figsize=(4.2 * len(kinds), 3.4 * len(keys)),
        squeeze=False,
    )
    styles = {
        "window": ("C0", "o-"),
        "geometric": ("C3", "s-"),
        "window smooth": ("C0", "o--"),
        "geometric smooth": ("C3", "s--"),
    }
    for col, kind in enumerate(kinds):
        for row_i, key in enumerate(keys):
            ax = axes[row_i][col]
            for fam, (color, style) in styles.items():
                pts = [
                    r
                    for r in rows
                    if r["catalogue"] == kind and r["family"] == fam and key in r
                ]
                if not pts:
                    continue
                pts.sort(key=lambda r: -r["rel_err"])
                ax.plot(
                    [r[key]["median_s"] for r in pts],
                    [max(r["rel_err"], 1e-16) for r in pts],
                    style,
                    color=color,
                    label=fam,
                    ms=4,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(
                ("forward" if key == "forward" else "value+grad") + " time [s]"
            )
            ax.set_ylabel("max per-bin |DD / DD_exact - 1|")
            ax.set_title(f"{kind}, N = {payload['rows'][0]['n']}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    print(f"# wrote {path}")


def _print_matched(payload) -> None:
    for m in payload["matched_forward"] + payload["matched_value_and_grad"]:
        cells = []
        for fam in ("window", "geometric"):
            best = m.get(fam)
            cells.append(
                f"{best['label']} {best['seconds']:.4f} s" if best else f"{fam} -"
            )
        ratio = m.get("geometric_over_window")
        print(
            f"{m['catalogue']:8s} {m['time']:15s} target {m['target']:.0e}: "
            + "  |  ".join(cells)
            + (f"  ratio {ratio:.2f}" if ratio else ""),
            flush=True,
        )


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.replot:
        with open(args.replot) as f:
            payload = json.load(f)
        # recompute the matched tables at this run's --targets
        rows = payload["rows"]
        families = sorted({r["family"] for r in rows})
        for key in ("forward", "value_and_grad"):
            payload[f"matched_{key}"] = _matched(rows, args.targets, families, key)
        common.write_json(args.replot, payload)
        _print_matched(payload)
        _plot(payload, args.fig or args.replot.replace(".json", ".png"))
        return
    common.select_device(args)

    import jax

    out = args.out or common.default_path("acceptance_rules", "json", args)
    fig = args.fig or out.replace(".json", ".png")
    edges = common.edges_of(args)
    points = _points(args)
    families = sorted({fam for fam, _, _ in points})
    meta = {
        "environment": common.environment(),
        "args": vars(args),
        "edges": edges.tolist(),
    }
    rows: list[dict] = []
    for kind in args.catalogues:
        pos = common.catalogue(kind, args.n, args)
        n = int(pos.shape[0])
        ref = common.exact_counts(pos, edges, args.sharpness, args.box)
        print(f"{kind:8s} n={n} exact {ref['seconds']:.2f} s", flush=True)
        for fam, label, kw in points:
            row = _run_point(pos, args, edges, ref, kw)
            row.update({"catalogue": kind, "n": n, "family": fam, "label": label})
            rows.append(row)
            if "skipped" in row:
                print(f"  {label:34s} skipped: {row['skipped']}", flush=True)
            else:
                vg = row.get("value_and_grad", {}).get("median_s", float("nan"))
                print(
                    f"  {label:34s} err {row['rel_err']:.2e}  fwd "
                    f"{row['forward']['median_s']:8.4f} s  v+g {vg:8.4f} s  "
                    f"far {row['walk']['far_count']:>9d}  near "
                    f"{row['walk']['near_count']:>9d}"
                    + ("  OVERFLOW" if row["overflow"] else ""),
                    flush=True,
                )
            jax.clear_caches()
        payload = {
            **meta,
            "exact_seconds": ref["seconds"],
            "rows": rows,
            "matched_forward": _matched(rows, args.targets, families, "forward"),
            "matched_value_and_grad": _matched(
                rows, args.targets, families, "value_and_grad"
            ),
        }
        common.write_json(out, payload)
    _print_matched(payload)
    try:
        _plot(payload, fig)
    except ImportError as exc:  # matplotlib missing
        print(f"# no figure: {exc}")


if __name__ == "__main__":
    main()
