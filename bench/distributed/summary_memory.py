"""Per-device memory of the summary exchange: all-gather (C1) vs neighbours (C3).

Two tables:

1. **Measured** on fake CPU devices (weak scaling, ``--n-per-device`` uniform
   points, ξ reach): the summary cell rows each device holds for its export walk
   (static, what memory is allocated for) and the live rows it received, for
   ``summary_exchange="all_gather"`` and ``"neighbors"``, at 1, 2, 4, 8 devices.
   Both paths give the same halo; the script checks that.
2. **Model** (:func:`~yggdrax.distributed.stats.estimate_capacities`) for the
   flagship's weak series, 4 to 1024 GPUs at ``--model-n-per-device``.

A cell row costs about 50 bytes in float32 (82 in float64): centre, radius and
flag in the gathered or received arrays, plus the export walk's per-node arrays
(centre, extent, two children, active) and its seed pair. The neighbours path
adds a send buffer (``summary_send_capacity`` rows of 4 floats) and the gathered
coarse boxes (``ndev x 2**depth x 7`` floats).

    python bench/distributed/summary_memory.py            # 8 fake CPU devices
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

ROW_BYTES = {"float32": 50, "float64": 82}


def _mb(rows, dtype="float32"):
    return rows * ROW_BYTES[dtype] / 2**20


def measured(n_per, ndevs, reach_spacings=3.4):
    import jax
    import numpy as np

    from yggdrax.distributed import make_mesh
    from yggdrax.distributed.stats import (
        DistributedStatsConfig,
        distributed_pair_counts,
        soft_window_cutoffs,
    )

    print(
        f"\nmeasured: uniform, {n_per} per device, xi reach {reach_spacings} "
        "spacings, float64\n"
    )
    print(
        f"{'ndev':>4} | {'all_gather rows':>15} {'MB':>7} {'recv live':>9} | "
        f"{'neighbors rows':>14} {'MB':>7} {'recv live':>9} {'dests':>5} | same halo"
    )
    for nd in ndevs:
        n = n_per * nd
        pts = np.random.default_rng(0).uniform(0, 1, (n, 3))
        d = 1.0 / n ** (1 / 3)
        edges = np.geomspace(0.5 * d, 2.5 * d, 9)
        lo, hi = soft_window_cutoffs(edges, 60.0, 1e-8)
        out = {}
        for mode in ("all_gather", "neighbors"):
            cfg = DistributedStatsConfig.for_box(
                n_per,
                ndev=nd,
                box_size=1.0,
                reach=hi,
                clustering=1.5,
                summary_exchange=mode,
            )
            r = distributed_pair_counts(
                pts, edges, box_size=1.0, config=cfg, sharpness=60.0, mesh=make_mesh(nd)
            )
            jax.block_until_ready(r.counts)
            assert not bool(r.overflow), {k: bool(v) for k, v in r.flags.items()}
            st = {k: np.asarray(v) for k, v in r.stats.items()}
            if mode == "all_gather":
                rows = nd * cfg.summary_capacity
            else:
                rows = cfg.summary_recv_capacity if nd > 1 else cfg.summary_capacity
            out[mode] = (rows, int(st["summary_recv"].max()), st)
        same = np.array_equal(out["all_gather"][2]["halo"], out["neighbors"][2]["halo"])
        a, b = out["all_gather"], out["neighbors"]
        print(
            f"{nd:4d} | {a[0]:15d} {_mb(a[0], 'float64'):7.2f} {a[1]:9d} | "
            f"{b[0]:14d} {_mb(b[0], 'float64'):7.2f} {b[1]:9d} "
            f"{int(b[2]['summary_dests'].max()):5d} | {same}"
        )


def model(n_per, ndevs, reach_spacings, clustering, dtype):
    from yggdrax.distributed.stats import estimate_capacities

    print(
        f"\nmodel: {n_per:.3g} per device, reach {reach_spacings} spacings, "
        f"clustering {clustering}, {dtype}, box 1, weak scaling\n"
    )
    print(
        f"{'ndev':>5} {'cells/dev':>10} | {'all_gather rows':>15} {'GB':>8} | "
        f"{'neighbors rows':>14} {'GB':>8} {'K':>4} | {'ratio':>6}"
    )
    for nd in ndevs:
        d = 1.0 / (n_per * nd) ** (1 / 3)
        est = estimate_capacities(
            n_per,
            ndev=nd,
            box_size=1.0,
            reach=reach_spacings * d,
            clustering=clustering,
        )
        cf = est["config"]
        old = nd * cf["summary_capacity"]
        new = cf["summary_recv_capacity"]
        extra = cf["summary_send_capacity"] * 16 + nd * 16 * 28  # send rows + boxes
        print(
            f"{nd:5d} {cf['summary_capacity']:10d} | {old:15d} "
            f"{_mb(old, dtype) / 1024:8.2f} | {new:14d} "
            f"{(_mb(new, dtype) + extra / 2**20) / 1024:8.2f} "
            f"{cf['summary_neighbors']:4d} | {old / max(new, 1):6.1f}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-per-device", type=int, default=16384)
    p.add_argument("--ndev", default="1,2,4,8")
    p.add_argument("--model-n-per-device", type=float, default=2**26)
    p.add_argument("--model-ndev", default="4,16,64,128,256,1024")
    p.add_argument("--reach-spacings", type=float, default=3.4)
    p.add_argument("--clustering", type=float, default=2.0)
    p.add_argument("--no-measure", action="store_true")
    args = p.parse_args(argv)
    ndevs = [int(x) for x in args.ndev.split(",")]
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1"
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            flags + f" --xla_force_host_platform_device_count={max(ndevs)}"
        ).strip()
    if not args.no_measure:
        measured(args.n_per_device, ndevs, args.reach_spacings)
    model(
        args.model_n_per_device,
        [int(x) for x in args.model_ndev.split(",")],
        args.reach_spacings,
        args.clustering,
        "float32",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
