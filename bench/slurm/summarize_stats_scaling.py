"""Tabulate the JSONs of bench/distributed/scaling.py: time, memory, efficiency.

    python bench/slurm/summarize_stats_scaling.py $OUT/cases          # every row
    python bench/slurm/summarize_stats_scaling.py $OUT/cases --scaling weak --csv w.csv

One row per (tag, scaling, ics, payload, mode, devices, N). Parallel efficiency
is relative to the smallest device count of the same series: weak ``t_1 / t_p``,
strong ``t_1 p_1 / (t_p p)``. A run with an error or an overflow is listed but
not used as a reference.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys


def _load(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            with open(path) as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        schema = r.get("schema", "")
        # rows/<tag>.json (the resumable-row markers) repeat the cases: skip them
        if schema.startswith("yggdrax.scaling/") and not schema.endswith("+run"):
            r["_path"] = path
            rows.append(r)
    return rows


def _flat(r):
    t = r.get("timings", {})
    st = t.get("stages", {})
    mem = r.get("memory", {})
    peak = mem.get("runtime", {}).get("peak_bytes", []) or [-1]
    cs = r.get("counters_summary", {})
    return dict(
        tag=r.get("tag", ""),
        scaling=r["scaling"],
        ics=r["ics"],
        payload=r["payload"],
        mode=r["mode"],
        ndev=r["topology"]["ndev"],
        nodes=r["topology"].get("nodes"),
        n_total=r["problem"]["n_total"],
        n_per_device=r["problem"]["n_per_device"],
        time_s=t.get("time_s"),
        compile_s=t.get("compile_s"),
        partition_s=st.get("partition_s"),
        halo_s=st.get("halo_s"),
        temp_gb=(mem.get("compiled", {}).get("temp", 0) or 0) / 1e9,
        peak_gb=max(peak) / 1e9 if max(peak) > 0 else None,
        halo_max=cs.get("halo", {}).get("max"),
        near_max=cs.get("near_pairs", {}).get("max"),
        summary_recv_max=cs.get("summary_recv", {}).get("max"),
        overflow=r.get("overflow"),
        error=(r.get("error") or "")[:60],
        commit=(r.get("software", {}).get("yggdrax", {}).get("commit") or "")[:9],
        jax=r.get("software", {}).get("jax"),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root")
    p.add_argument("--tag", default=None, help="regex over the row tags")
    p.add_argument("--scaling", choices=["weak", "strong"], default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)
    rows = [_flat(r) for r in _load(args.root)]
    if args.tag is not None:
        rows = [r for r in rows if re.search(args.tag, r["tag"])]
    if args.scaling is not None:
        rows = [r for r in rows if r["scaling"] == args.scaling]
    if not rows:
        print("no scaling JSONs found", file=sys.stderr)
        return 1
    key = lambda r: (r["tag"], r["scaling"], r["ics"], r["payload"], r["mode"])
    rows.sort(key=lambda r: key(r) + (r["ndev"], r["n_total"]))
    ref = {}
    for r in rows:
        ok = r["time_s"] and not r["overflow"] and not r["error"]
        k = key(r)
        if ok and k not in ref:
            ref[k] = r
        r0 = ref.get(k)
        eff = None
        if ok and r0 is not None:
            if r["scaling"] == "weak":
                eff = r0["time_s"] / r["time_s"]
            else:
                eff = (r0["time_s"] * r0["ndev"]) / (r["time_s"] * r["ndev"])
        r["efficiency"] = eff

    def f(x, fmt):
        return "-" if x is None else format(x, fmt)

    head = (
        f"{'tag':8} {'scal':6} {'ics':9} {'pl':4} {'mode':14} {'ndev':>5} "
        f"{'N/dev':>10} {'time s':>9} {'eff':>5} {'part s':>7} {'halo s':>7} "
        f"{'temp GB':>8} {'peak GB':>8} {'halo':>10} {'over':>5} error"
    )
    print(head)
    for r in rows:
        print(
            f"{r['tag'][:8]:8} {r['scaling']:6} {r['ics']:9} {r['payload']:4} "
            f"{r['mode']:14} {r['ndev']:5d} {r['n_per_device']:10.3g} "
            f"{f(r['time_s'], '9.3f')} {f(r['efficiency'], '5.2f')} "
            f"{f(r['partition_s'], '7.3f')} {f(r['halo_s'], '7.3f')} "
            f"{f(r['temp_gb'], '8.2f')} {f(r['peak_gb'], '8.2f')} "
            f"{f(r['halo_max'], '10.0f')} {str(r['overflow'])[:5]:>5} {r['error']}"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
