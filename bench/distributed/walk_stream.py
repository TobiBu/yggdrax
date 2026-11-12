"""Memory and time of the streamed walk (``walk_chunks``) against the single walk.

For one problem (particles per device, devices, leaf size) and each
``walk_chunks`` in ``--chunks``, a payload's per-device body inside a
``shard_map`` (ξ: :func:`~yggdrax.distributed.stats.pair_counts_shard`; kNN:
:func:`~yggdrax.distributed.stats.knn_shard` with a soft CDF), sized by
:meth:`DistributedStatsConfig.for_box` with that chunk count:

* ``plan``: XLA's compiled temp per device (``memory_analysis()``), forward and
  value+grad (ξ: positions and weights; kNN: positions, through the CDF);
* without ``--plan-only``: run time (median and min of ``--repeats``) on
  uniform points, the overflow flags, and the largest relative deviation of the
  counts and gradients from the first entry of ``--chunks`` (put 1 first);
* the per-walk capacities and the counters (near pairs per walk and in total,
  the frontier, sub-block pairs).

``--plan-only`` compiles from shapes alone (no data), so it reaches the C4 sizes
(6.7e7 per device) on a laptop. The CPU plan uses the ``buf`` exchange, whose
all-gathers of the send buffers the GPU's native exchange does not have, so it
is an upper bound for a GPU.

Examples::

    # memory plan at the C4 size (2048^3 on 128 GPUs), compile only
    python bench/distributed/walk_stream.py --plan-only --n-per-device 6.7e7 \\
        --ndev-model 128 --chunks 1,16,64,256,1024

    # CPU timing, 2 fake devices
    python bench/distributed/walk_stream.py --n-per-device 131072 --ndev 2 \\
        --chunks 1,4,16,64 --out bench/results/walk_stream_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-per-device", type=float, default=131072)
    p.add_argument("--ndev", type=int, default=2, help="devices of the compile/run")
    p.add_argument(
        "--ndev-model",
        type=int,
        default=None,
        help="devices the capacities are sized for (default --ndev)",
    )
    p.add_argument("--payload", choices=["xi", "knn"], default="xi")
    p.add_argument("--chunks", default="1,4,16,64")
    p.add_argument("--leaf-size", type=int, default=64)
    p.add_argument("--sub-block", type=int, default=8)
    p.add_argument("--clustering", type=float, default=1.5)
    p.add_argument("--edges", default="0.5,2.5", help="xi bin range, in spacings")
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--sharpness", type=float, default=60.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r-max", type=float, default=3.0, help="kNN cap, in spacings")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--modes", default="forward,value_and_grad")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    flags = os.environ.get("XLA_FLAGS", "")
    if "device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={args.ndev}".strip()
        )
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    jax.config.update("jax_enable_x64", True)
    from yggdrax.distributed import stats as S

    ndev = int(args.ndev)
    ndev_model = int(args.ndev_model or ndev)
    npd = float(args.n_per_device)
    dtype = jnp.dtype(args.dtype)
    box = 1.0
    spacing = box / (npd * ndev_model) ** (1.0 / 3.0)
    e0, e1 = (float(v) for v in args.edges.split(","))
    edges = np.geomspace(e0 * spacing, e1 * spacing, args.nbins + 1)
    r_lo, r_hi = S.soft_window_cutoffs(edges, args.sharpness, args.eps, True)
    mesh = Mesh(np.array(jax.devices()[:ndev]), (S.AXIS_NAME,))
    sh = NamedSharding(mesh, P(S.AXIS_NAME))
    rows = int(npd)
    n_in = rows * ndev
    edges_j = jnp.asarray(edges, dtype)

    r_max = float(args.r_max) * spacing
    r_grid = jnp.asarray(np.geomspace(0.3 * spacing, 0.9 * r_max, 16), dtype)
    knn = args.payload == "knn"

    def program(cfg, mode):
        def body(x, w, c):
            if knn:
                r = S.knn_shard(
                    x,
                    c[0],
                    k=args.k,
                    r_max=r_max,
                    ndev=ndev,
                    config=cfg,
                    box_size=box,
                    r_grid=r_grid,
                    sharpness=20.0,
                )
                del w
                assert r.cdf is not None
                return r.cdf, r.overflow, r.stats
            r = S.pair_counts_shard(
                x,
                w,
                c[0],
                edges_j,
                ndev=ndev,
                config=cfg,
                box_size=box,
                r_lo=float(r_lo),
                r_hi=float(r_hi),
                sharpness=args.sharpness,
            )
            return r.counts, r.overflow, r.stats

        fn = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(S.AXIS_NAME),) * 3,
            out_specs=(P(), P(), P(S.AXIS_NAME)),
            check_vma=False,
        )
        if mode == "forward":
            return jax.jit(fn)
        coef = jnp.linspace(1.0, 2.0, 16 if knn else args.nbins, dtype=dtype)

        def loss(x, w, c):
            counts, over, st = fn(x, w, c)
            return jnp.sum(coef * jnp.log1p(counts)), (counts, over, st)

        argnums = (0,) if knn else (0, 1)
        return jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))

    abstract = (
        jax.ShapeDtypeStruct((n_in, 3), dtype, sharding=sh),
        jax.ShapeDtypeStruct((n_in,), dtype, sharding=sh),
        jax.ShapeDtypeStruct((ndev,), jnp.int64, sharding=sh),
    )
    data = None
    if not args.plan_only:
        rng = np.random.default_rng(args.seed)
        x = jax.device_put(jnp.asarray(rng.uniform(0, box, (n_in, 3)), dtype), sh)
        w = jax.device_put(jnp.asarray(rng.uniform(0.5, 1.5, n_in), dtype), sh)
        c = jax.device_put(jnp.full((ndev,), rows, jnp.int64), sh)
        data = (x, w, c)

    rows_out = []
    ref: dict = {}
    for chunks in [int(v) for v in args.chunks.split(",")]:
        cfg = S.DistributedStatsConfig.for_box(
            npd,
            ndev=ndev_model,
            box_size=box,
            reach=r_max if knn else r_hi,
            leaf_size=args.leaf_size,
            clustering=args.clustering,
            sub_block=args.sub_block,
            walk_chunks=chunks,
        )
        row = dict(
            walk_chunks=chunks,
            config={
                k: getattr(cfg, k)
                for k in (
                    "capacity",
                    "halo_capacity",
                    "max_pair_queue",
                    "near_cap",
                    "sub_pair_cap",
                    "walk_frontier",
                )
            },
        )
        for mode in args.modes.split(","):
            fn = program(cfg, mode)
            t0 = time.perf_counter()
            comp = fn.lower(*abstract).compile()
            ma = comp.memory_analysis()
            entry = dict(
                compile_s=time.perf_counter() - t0,
                temp_bytes=int(ma.temp_size_in_bytes),
                temp_bytes_per_particle=ma.temp_size_in_bytes / npd,
            )
            if data is not None:
                out = jax.block_until_ready(comp(*data))
                times = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    jax.block_until_ready(comp(*data))
                    times.append(time.perf_counter() - t0)
                entry.update(time_s=statistics.median(times), time_min_s=min(times))
                if mode == "forward":
                    counts, over, st = out
                    grads = ()
                else:
                    (_v, (counts, over, st)), grads = out
                vals = [np.asarray(counts)] + [np.asarray(g) for g in grads]
                entry["overflow"] = bool(over)
                if mode not in ref:
                    ref[mode] = vals
                else:
                    entry["max_rel_dev"] = max(
                        float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))
                        for a, b in zip(vals, ref[mode])
                    )
                entry["stats"] = {
                    k: np.asarray(st[k]).tolist()
                    for k in (
                        "near_pairs",
                        "near_pairs_total",
                        "walk_frontier",
                        "peak_wavefront",
                        "sub_pairs",
                    )
                    if k in st
                }
            row[mode] = entry
            print(
                f"K={chunks:5d} {mode:15s} temp {entry['temp_bytes'] / 1e9:7.3f} GB "
                f"({entry['temp_bytes_per_particle']:6.0f} B/particle)"
                + (
                    f"  time {entry['time_s']:7.3f} s  overflow {entry['overflow']}"
                    + (
                        f"  dev {entry['max_rel_dev']:.1e}"
                        if "max_rel_dev" in entry
                        else ""
                    )
                    if "time_s" in entry
                    else ""
                ),
                flush=True,
            )
        rows_out.append(row)

    record = dict(
        schema="yggdrax.walk_stream/1",
        args=vars(args),
        jax=jax.__version__,
        platform=jax.devices()[0].platform,
        rows=rows_out,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
