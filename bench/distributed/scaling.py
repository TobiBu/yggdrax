"""Weak and strong scaling of the distributed clustering statistics (C3/C4).

One invocation measures ONE point of a scaling curve: one device count (all the
devices this job sees), one problem size, for each requested payload (ξ pair
counts, kNN) and mode (``forward``, ``value_and_grad``), on a periodic box. It
writes one JSON file per (payload, mode) to ``--out-dir``. A scaling curve is a
series of such runs at 1, 4, 16, 64, 128 ... devices (``bench/slurm/stats_scaling_*.sbatch``).

* ``--scaling weak``: ``--n-per-device`` particles on every device; the total
  grows with the device count.
* ``--scaling strong``: ``--n-total`` particles split over the devices.

Lengths are in units of the mean interparticle spacing ``d = L / N^(1/3)`` by
default (``--length-unit spacing``), so a weak-scaling series keeps the work per
particle fixed: the ξ bin edges and the kNN ``r_max`` are multiples of ``d``.
``--length-unit box`` takes them in units of the box side instead.

Initial conditions are generated on the devices (``--ics uniform`` or
``clustered``: a fraction of the points in Gaussian blobs around random centres,
the rest uniform, wrapped into the box). Nothing of size N goes through a host.

Per (payload, mode) the JSON holds:

* topology: devices, processes, nodes, device kind, SLURM job, NCCL/XLA env;
* problem and the full ``DistributedStatsConfig``. It is sized by
  :meth:`DistributedStatsConfig.for_box`. On an overflow it is grown and rerun up
  to ``--retry-overflow`` times, and the retries are logged. With ``--tighten F``,
  every capacity is refit to ``F`` x the counters of a clean probe and checked
  by a second probe; memory follows the capacities, so this is what a memory
  ceiling should be measured with;
* timings: compile time, median/min of ``--repeats`` runs, and for ``forward``
  cheap stage splits (``--stages``): the partition alone, and partition + local
  tree + summary exchange + halo import with the payload's maximum reach; the
  walk and the statistic are the remainder;
* memory: XLA's compiled plan per device (temp/argument/output bytes) and, on
  GPU, every device's allocator high-water mark (``peak_bytes_in_use``, a
  running maximum over the process: forward runs first, so ``value_and_grad``'s
  peak is ``max(forward, value_and_grad)``; run ``--modes value_and_grad`` alone
  for its own peak) and ``bytes_limit``;
* the per-device counters and overflow flags of the statistic;
* the yggdrax commit (and whether the tree was dirty), JAX and jaxlib versions.

Where the devices come from:

* CPU (testing): ``--platform cpu --ndev 8`` forces 8 host devices.
* One GPU node, one process (the group's shared A100 server): ``--platform gpu
  --ndev 4`` picks 4 free GPUs with ``autocvd`` first (``--gpu-select free``, the
  org rule; ``least-used`` takes the idlest now, ``none`` keeps
  ``CUDA_VISIBLE_DEVICES``).
* SLURM, one process per GPU (HoreKa, JUWELS Booster): ``srun python
  scaling.py --platform gpu ...``; ``jax.distributed.initialize()`` reads the
  SLURM environment (``--distributed auto`` turns it on when ``SLURM_NTASKS >
  1``). GPUs are exclusive there, so ``autocvd`` is not used.

Examples::

    # CPU smoke, 4 fake devices, both payloads and modes (a minute)
    python bench/distributed/scaling.py --smoke

    # CPU, weak scaling point at 8 fake devices
    python bench/distributed/scaling.py --platform cpu --ndev 8 \\
        --scaling weak --n-per-device 16384 --out-dir bench/results/scaling_cpu

    # shared A100 server, 4 free GPUs
    python bench/distributed/scaling.py --platform gpu --ndev 4 --gpu-select free \\
        --scaling weak --n-per-device 4194304 --out-dir bench/results/scaling_a100
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SCHEMA = "yggdrax.scaling/1"


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scaling", choices=["weak", "strong"], default="weak")
    p.add_argument("--n-per-device", type=float, default=2**16)
    p.add_argument("--n-total", type=float, default=None, help="strong scaling")
    p.add_argument("--payload", default="xi,knn", help="comma list of xi, knn")
    p.add_argument("--modes", default="forward,value_and_grad")
    p.add_argument("--ics", choices=["uniform", "clustered"], default="uniform")
    p.add_argument("--cluster-frac", type=float, default=0.6)
    p.add_argument("--clusters-per-device", type=int, default=32)
    p.add_argument(
        "--cluster-sigma", type=float, default=4.0, help="blob width (length unit)"
    )
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--length-unit", choices=["spacing", "box"], default="spacing")
    p.add_argument("--edges", default="0.5,2.5", help="xi bin range (length unit)")
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--sharpness", type=float, default=60.0, help="in ln r")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r-max", type=float, default=3.0, help="kNN cap (length unit)")
    p.add_argument("--cdf-sharpness", type=float, default=20.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--clustering", type=float, default=None, help="sizing factor")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument(
        "--sub-block",
        type=int,
        default=None,
        help="ξ culled near-field sub-block (0 = dense leaf blocks; default: config's)",
    )
    p.add_argument("--exchange", choices=["auto", "buf", "native"], default="auto")
    p.add_argument(
        "--summary-exchange", choices=["neighbors", "all_gather"], default="neighbors"
    )
    p.add_argument(
        "--config-json",
        default=None,
        help="JSON dict of DistributedStatsConfig fields overriding the sizing",
    )
    p.add_argument("--retry-overflow", type=int, default=3)
    p.add_argument(
        "--tighten",
        type=float,
        default=0.0,
        help="after a clean probe, refit every capacity to this factor x the "
        "measured counters (e.g. 1.3) and re-probe; memory follows capacities",
    )
    p.add_argument(
        "--verify-exchange",
        action="store_true",
        help="also run the forward with exchange='buf' and compare bit for bit "
        "(small N only: buf all-gathers every send buffer)",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--stages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    p.add_argument(
        "--ndev", type=int, default=None, help="cpu: fake devices; gpu: GPUs to use"
    )
    p.add_argument(
        "--gpu-select", choices=["free", "least-used", "none"], default="free"
    )
    p.add_argument("--gpu-wait-timeout", type=int, default=None)
    p.add_argument("--distributed", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--out-dir", default=os.path.join(_REPO, "bench/results/scaling"))
    p.add_argument(
        "--output",
        default=None,
        help="also write every case of this run into ONE JSON here, only when all "
        "of them succeeded (the resumable-row marker of bench/slurm/)",
    )
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run, 4 devices")
    args = p.parse_args(argv)
    if args.smoke:
        args.platform, args.ndev = "cpu", args.ndev or 4
        args.n_per_device, args.repeats, args.warmup = 2048, 1, 0
        args.tag = args.tag or "smoke"
    if args.scaling == "strong" and args.n_total is None:
        p.error("--scaling strong needs --n-total")
    return args


# ---------------------------------------------------------------------------
# environment, before JAX starts
# ---------------------------------------------------------------------------


def _distributed(args) -> bool:
    if args.distributed != "auto":
        return args.distributed == "on"
    return int(os.environ.get("SLURM_NTASKS", "1")) > 1


def _prepare_env(args) -> None:
    os.environ["JAX_ENABLE_X64"] = "1"  # the statistics require it (ids, codes)
    if args.platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        n = int(args.ndev or 4)
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            flags += f" --xla_force_host_platform_device_count={n}"
        os.environ["XLA_FLAGS"] = flags.strip()
        return
    # GPU: grow the pool on demand, so peak_bytes_in_use is the working set
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if _distributed(args):
        return  # SLURM: exclusive GPUs, no selection
    if args.gpu_select == "none":
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("# CUDA_VISIBLE_DEVICES already set; not selecting", file=sys.stderr)
        return
    from autocvd import autocvd  # site tool: sets CUDA_VISIBLE_DEVICES (org rule)

    picked = autocvd(
        num_gpus=int(args.ndev or 1),
        least_used=args.gpu_select == "least-used",
        timeout=args.gpu_wait_timeout,
    )
    print(f"# autocvd selected GPU(s): {picked}", file=sys.stderr, flush=True)


def _init_distributed(args, jax) -> None:
    if not _distributed(args):
        return
    kw = {}
    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        # e.g. JUWELS: the first node's InfiniBand hostname (``<host>i:<port>``)
        kw["coordinator_address"] = os.environ["JAX_COORDINATOR_ADDRESS"]
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    if len(visible) == 1:
        kw["local_device_ids"] = [0]  # SLURM bound one GPU to this task
    elif "SLURM_LOCALID" in os.environ and args.platform == "gpu":
        kw["local_device_ids"] = [int(os.environ["SLURM_LOCALID"])]
    jax.distributed.initialize(**kw)


def _git() -> dict:
    out = {"commit": os.environ.get("YGGDRAX_COMMIT"), "dirty": None}
    try:
        out["commit"] = (
            subprocess.check_output(
                ["git", "-C", _REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = subprocess.run(
            ["git", "-C", _REPO, "diff", "--quiet", "HEAD", "--", "yggdrax"],
            stderr=subprocess.DEVNULL,
        ).returncode
        out["dirty"] = bool(dirty)
    except Exception:  # pragma: no cover - no git on the node
        pass
    return out


def _topology(jax, ndev) -> dict:
    devs = jax.devices()
    env = os.environ
    return dict(
        ndev=ndev,
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        local_device_count=jax.local_device_count(),
        nodes=int(env.get("SLURM_JOB_NUM_NODES", env.get("SLURM_NNODES", "1"))),
        host=socket.gethostname(),
        platform=jax.default_backend(),
        device_kind=str(devs[0].device_kind),
        slurm={
            k: env[k]
            for k in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_NTASKS",
                "SLURM_NTASKS_PER_NODE",
                "SLURM_JOB_PARTITION",
                "SLURM_CLUSTER_NAME",
            )
            if k in env
        },
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        nccl_env={k: v for k, v in env.items() if k.startswith(("NCCL_", "UCX_"))},
        xla_flags=env.get("XLA_FLAGS", ""),
    )


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

_GROW = {
    # flag -> (fields, factor)
    "partition_overflow": (("capacity", "leaf_capacity", "halo_leaf_capacity"), 1.25),
    "leaf_overflow": (("leaf_capacity", "halo_leaf_capacity"), 1.5),
    "summary_overflow": (("summary_capacity",), 2.0),
    "summary_exchange_overflow": (
        ("summary_neighbors", "summary_send_capacity", "summary_recv_capacity"),
        2.0,
    ),
    "export_overflow": (("export_near_cap", "export_queue"), 2.0),
    "send_overflow": (("send_capacity", "send_leaf_capacity"), 2.0),
    "halo_overflow": (
        ("halo_capacity", "halo_leaf_capacity", "send_capacity", "send_leaf_capacity"),
        1.5,
    ),
    "walk_overflow": (("near_cap", "max_pair_queue", "far_cap"), 2.0),
    "sub_overflow": (("sub_pair_cap",), 2.0),
}


def _grow(cfg, bad, ndev):
    fields = dataclasses.asdict(cfg)
    for flag in bad:
        names, f = _GROW.get(flag, ((), 1.0))
        for name in names:
            v = fields.get(name)
            if v is not None:
                fields[name] = int(math.ceil(v * f))
    if fields.get("summary_neighbors") is not None:
        fields["summary_neighbors"] = min(int(fields["summary_neighbors"]), ndev - 1)
        fields["summary_neighbors"] = max(1, fields["summary_neighbors"])
    return type(cfg)(**fields).resolved()


def _tighten(cfg, st, f, payload):
    """Capacities refit to ``f`` x the largest per-device counter of a probe."""
    import math as _m

    def cap(x, floor=16):
        return max(floor, int(_m.ceil(f * float(max(x)))))

    fields = dataclasses.asdict(cfg)
    walk = cap(list(st["near_pairs"]) + list(st["peak_wavefront"]), 1 << 10)
    fields.update(
        capacity=cap(st["particles"]),
        halo_capacity=cap(st["halo"]),
        send_capacity=cap(st["sent"]),
        leaf_capacity=cap(st["leaves_local"]),
        halo_leaf_capacity=cap(st["leaves"]),
        summary_capacity=cap(st["summary_cells"]),
        export_near_cap=cap(st["export_pairs"], 1 << 10),
        send_leaf_capacity=cap(st["export_pairs"], 1 << 10),
        near_cap=walk,
        max_pair_queue=walk,
    )
    if payload == "xi" and fields.get("sub_block") and "sub_pairs" in st:
        fields["sub_pair_cap"] = cap(st["sub_pairs"], 1 << 10)
    if cfg.summary_exchange == "neighbors":
        rows = cap(list(st["summary_sent"]) + list(st["summary_recv"]))
        fields.update(
            summary_send_capacity=rows,
            summary_recv_capacity=rows,
            summary_neighbors=max(1, cap(st["summary_dests"], 1)),
        )
        # the export walk is seeded with every received row
        fields["export_queue"] = max(
            rows + fields["export_near_cap"], cap(st["export_wavefront"], 1 << 10)
        )
    if payload == "knn":
        # the local first pass walks the local tree with the same capacities
        fields["near_cap"] = fields["max_pair_queue"] = max(
            walk, cap(st["leaves_local"]) * 8
        )
    ndev = len(st["particles"])
    if fields.get("summary_neighbors") is not None:
        fields["summary_neighbors"] = max(1, min(fields["summary_neighbors"], ndev - 1))
    return type(cfg)(**fields).resolved()


def _run(args) -> list:
    import jax

    _init_distributed(args, jax)
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from yggdrax.distributed import AXIS_NAME, make_mesh
    from yggdrax.distributed import stats as S
    from yggdrax.distributed.comm import resolve_ragged_method

    ndev = len(jax.devices()) if args.platform == "gpu" else int(args.ndev or 4)
    if args.platform == "gpu" and args.ndev and not _distributed(args):
        ndev = min(ndev, int(args.ndev))
    mesh = make_mesh(ndev)
    dtype = jnp.dtype(args.dtype)
    L = float(args.box)
    if args.scaling == "weak":
        per = int(round(args.n_per_device))
    else:
        per = -(-int(round(args.n_total)) // ndev)
    n_total = per * ndev
    spacing = L / n_total ** (1.0 / 3.0)
    unit = spacing if args.length_unit == "spacing" else L
    e0, e1 = (float(x) * unit for x in args.edges.split(","))
    edges = np.geomspace(e0, e1, args.nbins + 1)
    r_lo, r_hi = S.soft_window_cutoffs(edges, args.sharpness, args.eps)
    r_max = float(args.r_max) * unit
    r_grid = np.geomspace(0.1 * r_max, r_max, 16)
    coef = np.linspace(1.0, 2.0, args.nbins)
    clustering = args.clustering
    if clustering is None:
        clustering = 1.5 if args.ics == "uniform" else 4.0
    leader = jax.process_index() == 0
    sh = NamedSharding(mesh, P(AXIS_NAME))
    rep = NamedSharding(mesh, P())

    # -- initial conditions, generated on the devices
    def gen(dev_ids):
        d = dev_ids[0]
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), d)
        k_u, k_c, k_w, k_n = jax.random.split(key, 4)
        x = jax.random.uniform(k_u, (per, 3), dtype=jnp.float64) * L
        if args.ics == "clustered":
            n_c = max(1, args.clusters_per_device * ndev)
            centres = jax.random.uniform(
                jax.random.PRNGKey(args.seed + 7919), (n_c, 3), dtype=jnp.float64
            )
            which = jax.random.randint(k_c, (per,), 0, n_c)
            blob = centres[which] * L + float(args.cluster_sigma) * unit * (
                jax.random.normal(k_n, (per, 3), dtype=jnp.float64)
            )
            in_blob = jax.random.uniform(k_w, (per,)) < args.cluster_frac
            x = jnp.where(in_blob[:, None], jnp.mod(blob, L), x)
        return x.astype(dtype)

    dev_ids = jax.make_array_from_callback(
        (ndev,), sh, lambda idx: np.arange(ndev, dtype=np.int64)[idx]
    )
    make_pos = jax.jit(
        shard_map(gen, mesh=mesh, in_specs=P(AXIS_NAME), out_specs=P(AXIS_NAME))
    )
    if args.compile_only:
        # lower from shapes: nothing of size N is allocated, so a CPU host can
        # read XLA's memory plan for a problem that only fits on many GPUs
        pos = jax.ShapeDtypeStruct((n_total, 3), dtype, sharding=sh)
    else:
        pos = jax.block_until_ready(make_pos(dev_ids))
    counts = jax.make_array_from_callback(
        (ndev,), sh, lambda idx: np.full((ndev,), per, np.int64)[idx]
    )

    def gather(x):
        """A sharded per-device array as a host array (all processes)."""
        if jax.process_count() == 1:
            return np.asarray(x)
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))

    def host(x):
        """A replicated array as a host value."""
        return np.asarray(x.addressable_data(0))

    base_info = dict(
        schema=SCHEMA,
        tag=args.tag,
        scaling=args.scaling,
        ics=args.ics,
        dtype=args.dtype,
        topology=_topology(jax, ndev),
        problem=dict(
            n_total=n_total,
            n_per_device=per,
            box=L,
            spacing=spacing,
            length_unit=args.length_unit,
            edges=edges.tolist(),
            r_cut=[r_lo, r_hi],
            sharpness=args.sharpness,
            eps=args.eps,
            k=args.k,
            r_max=r_max,
            r_grid=r_grid.tolist(),
            cdf_sharpness=args.cdf_sharpness,
            clustered=(
                dict(
                    frac=args.cluster_frac,
                    clusters_per_device=args.clusters_per_device,
                    sigma=args.cluster_sigma * unit,
                )
                if args.ics == "clustered"
                else None
            ),
        ),
        software=dict(
            yggdrax=_git(),
            jax=jax.__version__,
            jaxlib=getattr(jax.lib, "__version__", None)
            or __import__("jaxlib").__version__,
            python=sys.version.split()[0],
            exchange=resolve_ragged_method(args.exchange),
        ),
        argv=sys.argv[1:],
    )

    def make_stat(payload, cfg):
        if payload == "xi":

            def body(x, c):
                res = S.pair_counts_shard(
                    x,
                    jnp.ones((x.shape[0],), x.dtype),
                    c[0],
                    jnp.asarray(edges, x.dtype),
                    ndev=ndev,
                    config=cfg,
                    box_size=L,
                    r_lo=r_lo,
                    r_hi=r_hi,
                    sharpness=args.sharpness,
                )
                return res.counts, res.flags, res.stats

            fn = shard_map(
                body,
                mesh=mesh,
                in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
                out_specs=(P(), P(), P(AXIS_NAME)),
                check_vma=False,
            )

            def loss(x):
                c = fn(x, counts)[0]
                return jnp.sum(jnp.asarray(coef, c.dtype) * jnp.log1p(c))

            return fn, loss

        def body(x, c):
            res = S.knn_shard(
                x,
                c[0],
                k=args.k,
                r_max=r_max,
                ndev=ndev,
                config=cfg,
                box_size=L,
                r_grid=jnp.asarray(r_grid, x.dtype),
                sharpness=args.cdf_sharpness,
            )
            d = res.distances
            d_sum = jax.lax.psum(jnp.sum(jnp.where(jnp.isfinite(d), d, 0.0)), AXIS_NAME)
            return (res.cdf, d_sum), res.flags, res.stats

        fn = shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
            out_specs=(P(), P(), P(AXIS_NAME)),
            check_vma=False,
        )

        def loss(x):
            cdf, d_sum = fn(x, counts)[0]
            return jnp.sum(cdf**2) + 1e-3 * d_sum / n_total

        return fn, loss

    def stage_fns(payload, cfg):
        """Partition alone; partition + tree + summary exchange + halo import."""
        box = (L, L, L)
        reach = r_hi if payload == "xi" else r_max

        def part(x, c):
            w = jnp.ones((x.shape[0],), x.dtype)
            fl = jnp.ones((x.shape[0], 2), jnp.int32)
            gid = jnp.arange(x.shape[0], dtype=jnp.int64)
            out = S._partition(x, w, fl, gid, c[0], ndev, box, None, cfg)
            return out

        def p_only(x, c):
            p_pos, *_, p_count, over = part(x, c)
            return jax.lax.psum(jnp.sum(p_pos[:1]) + p_count, AXIS_NAME)

        def halo(x, c):
            p_pos, p_w, p_flags, p_gid, p_count, _ = part(x, c)
            lo, hi = S._morton_box(box, None, x.dtype)
            t = S._cell_tree(
                p_pos, p_count, box, (lo, hi), cfg.leaf_size, cfg.leaf_capacity
            )
            h = S._import_halo(
                t,
                p_pos,
                p_w,
                p_flags,
                p_gid,
                p_count,
                jnp.asarray(reach, x.dtype),
                ndev,
                box,
                cfg,
            )
            return jax.lax.psum(jnp.sum(h.pos[:1]) + h.num, AXIS_NAME)

        def wrap(f):
            return jax.jit(
                shard_map(
                    f,
                    mesh=mesh,
                    in_specs=(P(AXIS_NAME), P(AXIS_NAME)),
                    out_specs=P(),
                    check_vma=False,
                )
            )

        return dict(partition=wrap(p_only), halo=wrap(halo))

    def timed(compiled, *a):
        for _ in range(max(0, args.warmup)):
            jax.block_until_ready(compiled(*a))
        ts = []
        out = None
        for _ in range(max(1, args.repeats)):
            t0 = time.perf_counter()
            out = jax.block_until_ready(compiled(*a))
            ts.append(time.perf_counter() - t0)
        return out, ts

    def compile_(fn, *a):
        t0 = time.perf_counter()
        c = fn.lower(*a).compile()
        dt = time.perf_counter() - t0
        mem = {}
        try:
            ma = c.memory_analysis()
            for key in (
                "temp_size_in_bytes",
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "generated_code_size_in_bytes",
            ):
                mem[key.replace("_size_in_bytes", "")] = int(getattr(ma, key, 0) or 0)
        except Exception as exc:  # pragma: no cover - backend dependent
            mem["error"] = f"{type(exc).__name__}: {exc}"
        return c, dt, mem

    def memory_now():
        rows = []
        for dvc in jax.local_devices():
            try:
                st = dvc.memory_stats() or {}
            except Exception:  # pragma: no cover - CPU has no allocator stats
                st = {}
            rows.append(
                [
                    int(st.get("peak_bytes_in_use", -1)),
                    int(st.get("bytes_in_use", -1)),
                    int(st.get("bytes_limit", -1)),
                ]
            )
        arr = np.asarray(rows, np.int64)
        if jax.process_count() > 1:
            arr = np.asarray(multihost_utils.process_allgather(arr)).reshape(-1, 3)
        return dict(
            peak_bytes=arr[:, 0].tolist(),
            in_use_bytes=arr[:, 1].tolist(),
            limit_bytes=arr[:, 2].tolist(),
        )

    results = []
    for payload in [s.strip() for s in args.payload.split(",") if s.strip()]:
        reach = r_hi if payload == "xi" else r_max
        est = S.estimate_capacities(
            per,
            ndev=ndev,
            box_size=L,
            reach=reach,
            leaf_size=args.leaf_size,
            clustering=clustering,
        )
        over = json.loads(args.config_json) if args.config_json else {}
        cfg = S.DistributedStatsConfig.for_box(
            per,
            ndev=ndev,
            box_size=L,
            reach=reach,
            leaf_size=args.leaf_size,
            clustering=clustering,
            chunk=args.chunk,
            exchange=args.exchange,
            summary_exchange=args.summary_exchange,
            **({} if args.sub_block is None else {"sub_block": args.sub_block}),
        )
        if over:  # explicit fields win over the sizing
            cfg = dataclasses.replace(cfg, **over).resolved()

        # forward probe: settle the capacities before anything is timed
        retries = []
        stat = None
        res = None
        for attempt in range(args.retry_overflow + 1):
            fn, _ = make_stat(payload, cfg)
            stat = jax.jit(fn)
            if args.compile_only:
                break
            t0 = time.perf_counter()
            res = jax.block_until_ready(stat(pos, counts))
            flags = {k: bool(host(v)) for k, v in res[1].items()}
            bad = [k for k, v in flags.items() if v and k != "local_pass_overflow"]
            retries.append(
                dict(attempt=attempt, overflow=bad, probe_s=time.perf_counter() - t0)
            )
            if not bad or attempt == args.retry_overflow:
                break
            cfg = _grow(cfg, bad, ndev)

        tightened = None
        if args.tighten and res is not None and not retries[-1]["overflow"]:
            # memory follows the capacities, not the live counts: refit them to
            # the counters this probe measured, then confirm with a second probe
            st = {k: gather(v) for k, v in res[2].items()}
            t_cfg = _tighten(cfg, st, float(args.tighten), payload)
            t_fn, _ = make_stat(payload, t_cfg)
            t_res = jax.block_until_ready(jax.jit(t_fn)(pos, counts))
            t_flags = {k: bool(host(v)) for k, v in t_res[1].items()}
            t_bad = [k for k, v in t_flags.items() if v and k != "local_pass_overflow"]
            tightened = dict(factor=float(args.tighten), overflow=t_bad)
            if not t_bad:
                cfg, res = t_cfg, t_res

        exchange_check = None
        if args.verify_exchange and res is not None:
            ref_cfg = dataclasses.replace(cfg, exchange="buf")
            ref_fn, _ = make_stat(payload, ref_cfg)
            ref = jax.block_until_ready(jax.jit(ref_fn)(pos, counts))
            got = jax.tree_util.tree_leaves(res[0])
            want = jax.tree_util.tree_leaves(ref[0])
            same = all(np.array_equal(host(a), host(b)) for a, b in zip(got, want))
            diff = max(
                float(np.max(np.abs(host(a) - host(b)))) for a, b in zip(got, want)
            )
            halo_same = bool(
                np.array_equal(gather(res[2]["halo"]), gather(ref[2]["halo"]))
            )
            exchange_check = dict(
                method=resolve_ragged_method(cfg.exchange),
                identical=bool(same),
                halo_identical=halo_same,
                max_abs_diff=diff,
            )
            if leader:
                print(f"# exchange check {payload}: {exchange_check}", flush=True)

        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            row = dict(base_info, payload=payload, mode=mode)
            row["config"] = dataclasses.asdict(cfg)
            row["sizing"] = est["model"]
            row["overflow_retries"] = retries
            row["exchange_check"] = exchange_check
            row["tightened"] = tightened
            try:
                fn, loss = make_stat(payload, cfg)
                if mode == "forward":
                    target = jax.jit(loss)
                elif mode == "value_and_grad":
                    target = jax.jit(jax.value_and_grad(loss))
                else:
                    raise SystemExit(f"unknown mode {mode!r}")
                compiled, c_s, mem = compile_(target, pos)
                row["timings"] = dict(compile_s=c_s)
                row["memory"] = dict(compiled=mem)
                if not args.compile_only:
                    out, ts = timed(compiled, pos)
                    row["timings"].update(
                        time_s=statistics.median(ts), time_min_s=min(ts), times_s=ts
                    )
                    value = out[0] if mode == "value_and_grad" else out
                    row["loss"] = float(host(value))
                    if mode == "forward" and args.stages:
                        stages = {}
                        for name, f in stage_fns(payload, cfg).items():
                            c2, cs2, _ = compile_(f, pos, counts)
                            _, ts2 = timed(c2, pos, counts)
                            stages[f"{name}_s"] = statistics.median(ts2)
                            stages[f"{name}_compile_s"] = cs2
                        stages["walk_and_statistic_s"] = (
                            row["timings"]["time_s"] - stages["halo_s"]
                        )
                        row["timings"]["stages"] = stages
                    row["memory"]["runtime"] = memory_now()
                    res = jax.block_until_ready(jax.jit(fn)(pos, counts))
                    flags = {k: bool(host(v)) for k, v in res[1].items()}
                    row["flags"] = flags
                    row["overflow"] = any(
                        v for k, v in flags.items() if k != "local_pass_overflow"
                    )
                    st = {k: gather(v).tolist() for k, v in res[2].items()}
                    row["counters"] = st
                    row["counters_summary"] = {
                        k: dict(min=min(v), mean=float(np.mean(v)), max=max(v))
                        for k, v in st.items()
                    }
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()[-4000:]
            row["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            results.append(row)
            if leader:
                _write(args, row)
                _print_row(row)
    return results


def _write(args, row) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.tag}_" if args.tag else ""
    ndev = row["topology"]["ndev"]
    n = row["problem"]["n_total"]
    name = (
        f"{tag}{row['scaling']}_{row['ics']}_{row['payload']}_{row['mode']}"
        f"_nd{ndev:04d}_N{n:.3g}.json"
    ).replace("+", "")
    path = os.path.join(args.out_dir, name)
    with open(path, "w") as fh:
        json.dump(row, fh, indent=1)
    print(f"# wrote {path}", file=sys.stderr, flush=True)


def _print_row(row) -> None:
    t = row.get("timings", {})
    mem = row.get("memory", {})
    peak = max(mem.get("runtime", {}).get("peak_bytes", [-1]) or [-1])
    temp = mem.get("compiled", {}).get("temp", -1)
    st = t.get("stages", {})
    print(
        f"{row['payload']:4} {row['mode']:14} ndev={row['topology']['ndev']:5d} "
        f"N/dev={row['problem']['n_per_device']:.3g} "
        f"compile={t.get('compile_s', float('nan')):7.2f}s "
        f"time={t.get('time_s', float('nan')):9.4f}s "
        + (
            f"[part {st['partition_s']:.4f} halo {st['halo_s']:.4f}] "
            if "halo_s" in st
            else ""
        )
        + f"temp={temp / 2**20:9.1f}MB peak={peak / 2**20:9.1f}MB "
        f"overflow={row.get('overflow')} {row.get('error', '')[:200]}",
        flush=True,
    )


def main(argv=None) -> int:
    args = _parse(argv)
    _prepare_env(args)
    rows = _run(args)
    ok = all("error" not in r and not r.get("overflow") for r in rows)
    if ok and args.output and (not rows or rows[0]["topology"]["process_index"] == 0):
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        tmp = args.output + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(dict(schema=SCHEMA + "+run", cases=rows), fh, indent=1)
        os.replace(tmp, args.output)  # atomic: a killed job leaves no marker
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
