"""Peak memory and time of forward vs value+grad for the distributed statistics (C2 → C3).

The groundwork for C3's memory ceiling: how much a gradient costs on top of the
forward, per device, at a given N and device count, in float32 and float64. Both
payloads run the way a large run will: the per-device bodies
(:func:`~yggdrax.distributed.stats.pair_counts_shard`,
:func:`~yggdrax.distributed.stats.knn_shard`) inside the caller's own
``shard_map``, with the input already sharded over the mesh.

Per case (payload × dtype × mode) it reports:

* ``compiled``: XLA's static plan for ONE device (``memory_analysis()`` of the
  compiled SPMD program): temp, argument and output bytes. ``temp`` is the
  number to read: the peak working set the program allocates beyond its inputs
  and outputs. Available on CPU too.
* ``runtime_peak``: ``peak_bytes_in_use`` per device after the case (GPU only;
  the allocator's high-water mark). Each case runs in its own subprocess
  (default ``--isolate``), so this is the case's own peak, not a running max.
* ``compile_s``, ``time_s`` (median of ``--repeats``) and ``time_min_s``.
* the overflow flags (a run that overflowed measured a truncated problem).

Modes: ``forward`` is the statistic alone; ``value_and_grad`` is
``jax.value_and_grad`` of a scalar loss of it with respect to the positions
(and weights for ξ).

Examples::

    # CPU smoke, 4 fake devices (seconds)
    python bench/distributed/grad_memory.py --smoke

    # CPU, small N, both dtypes, 8 fake devices
    python bench/distributed/grad_memory.py --platform cpu --ndev 8 --n 16384

    # A100 server: picks free GPUs with autocvd (org rule), 4 GPUs
    python bench/distributed/grad_memory.py --platform gpu --ndev 4 --n 1048576 \\
        --gpu-select free --out bench/results/grad_memory_4xA100.json

GPU selection happens before JAX starts: ``--gpu-select free`` waits for idle
cards via ``autocvd`` (``least-used`` takes the idlest now, ``none`` keeps
``CUDA_VISIBLE_DEVICES``). Timings from a contended card are meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--ndev", type=int, default=4, help="devices in the mesh")
    p.add_argument("--n", type=int, default=8192, help="particles (global)")
    p.add_argument("--payload", choices=["xi", "knn", "both"], default="both")
    p.add_argument("--dtypes", default="float32,float64")
    p.add_argument("--modes", default="forward,value_and_grad")
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument(
        "--edges",
        default="0.01,0.05",
        help="first,last log bin edge (box units) for ξ",
    )
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--sharpness", type=float, default=60.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r-max", type=float, default=0.08)
    p.add_argument("--cdf-sharpness", type=float, default=20.0)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument(
        "--halo-fraction",
        type=float,
        default=None,
        help="halo capacity / particles per device (default: from the reach)",
    )
    p.add_argument("--slack", type=float, default=1.5)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--exchange", choices=["auto", "buf", "native"], default="auto")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="one subprocess per case, so the runtime peak is the case's own",
    )
    p.add_argument(
        "--gpu-select", choices=["free", "least-used", "none"], default="free"
    )
    p.add_argument("--gpu-wait-timeout", type=int, default=None)
    p.add_argument("--out", default=None, help="write the JSON results here")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run")
    p.add_argument("--case", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.smoke:
        args.platform, args.ndev, args.n, args.repeats = "cpu", 4, 1024, 1
    return args


def _select_gpus(args) -> None:
    """Pin free GPUs before JAX initialises CUDA (org rule: autocvd)."""
    if args.platform != "gpu" or args.gpu_select == "none":
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("# CUDA_VISIBLE_DEVICES already set; not selecting", file=sys.stderr)
        return
    from autocvd import autocvd  # site tool: sets CUDA_VISIBLE_DEVICES

    picked = autocvd(
        num_gpus=args.ndev,
        least_used=args.gpu_select == "least-used",
        timeout=args.gpu_wait_timeout,
    )
    print(f"# autocvd selected GPU(s): {picked}", file=sys.stderr, flush=True)


def _env_for(args) -> dict:
    env = dict(os.environ)
    if args.platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        flags = env.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            flags += f" --xla_force_host_platform_device_count={args.ndev}"
        env["XLA_FLAGS"] = flags.strip()
    else:
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cases(args):
    payloads = ["xi", "knn"] if args.payload == "both" else [args.payload]
    return [
        (pl, dt, md)
        for pl in payloads
        for dt in args.dtypes.split(",")
        for md in args.modes.split(",")
    ]


# ---------------------------------------------------------------------------
# one case, in this process
# ---------------------------------------------------------------------------


def _run_case(args, payload: str, dtype_name: str, mode: str) -> dict:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from yggdrax.distributed import AXIS_NAME, make_mesh
    from yggdrax.distributed.comm import resolve_ragged_method
    from yggdrax.distributed.stats import (
        DistributedStatsConfig,
        knn_shard,
        pair_counts_shard,
        soft_window_cutoffs,
    )

    ndev = int(args.ndev)
    if len(jax.devices()) < ndev:
        raise SystemExit(f"need {ndev} devices, have {len(jax.devices())}")
    mesh = make_mesh(ndev)
    dtype = jnp.dtype(dtype_name)
    per = -(-int(args.n) // ndev)
    n = per * ndev
    L = float(args.box)
    rng = np.random.default_rng(args.seed)
    pos = rng.uniform(0.0, L, (n, 3)).astype(dtype_name)
    w = rng.uniform(0.5, 2.0, n).astype(dtype_name)

    e0, e1 = (float(x) for x in args.edges.split(","))
    edges = np.geomspace(e0, e1, args.nbins + 1)
    r_lo, r_hi = soft_window_cutoffs(edges, args.sharpness, args.eps)
    reach = r_hi if payload == "xi" else float(args.r_max)
    counts = np.full((ndev,), per, np.int64)
    sh = NamedSharding(mesh, P(AXIS_NAME))
    pos_d = jax.device_put(jnp.asarray(pos), sh)
    w_d = jax.device_put(jnp.asarray(w), sh)
    c_d = jax.device_put(jnp.asarray(counts), sh)
    coef = np.linspace(1.0, 2.0, args.nbins)
    r_grid = np.geomspace(0.1 * args.r_max, args.r_max, 8)

    def make_cfg(halo_fraction):
        return DistributedStatsConfig.for_problem(
            per,
            halo_fraction=halo_fraction,
            slack=args.slack,
            leaf_size=args.leaf_size,
            chunk=args.chunk,
            exchange=args.exchange,
        )

    def counters(res):
        return {
            k: jax.lax.pmax(v[0], AXIS_NAME)
            for k, v in res.stats.items()
            if k in ("halo", "sent", "near_pairs", "export_pairs", "particles")
        }

    def build(cfg):
        """``(stat, loss, args_in, argnums)`` for one configuration."""
        if payload == "xi":

            def body(x, ww, c):
                res = pair_counts_shard(
                    x,
                    ww,
                    c[0],
                    jnp.asarray(edges, dtype),
                    ndev=ndev,
                    config=cfg,
                    box_size=L,
                    r_lo=r_lo,
                    r_hi=r_hi,
                    sharpness=args.sharpness,
                )
                return res.counts, res.flags, counters(res)

            stat = jax.jit(
                shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(P(AXIS_NAME),) * 3,
                    out_specs=(P(), P(), P()),
                    check_vma=False,
                )
            )

            def loss(x, ww):
                c = stat(x, ww, c_d)[0]
                return jnp.sum(jnp.asarray(coef, c.dtype) * jnp.log1p(c))

            return stat, loss, (pos_d, w_d), (0, 1)

        def body(x, c):
            res = knn_shard(
                x,
                c[0],
                k=args.k,
                r_max=args.r_max,
                ndev=ndev,
                config=cfg,
                box_size=L,
                r_grid=jnp.asarray(r_grid, dtype),
                sharpness=args.cdf_sharpness,
            )
            d = res.distances
            d_sum = jax.lax.psum(jnp.sum(jnp.where(jnp.isfinite(d), d, 0.0)), AXIS_NAME)
            return (res.cdf, d_sum), res.flags, counters(res)

        stat = jax.jit(
            shard_map(
                body,
                mesh=mesh,
                in_specs=(P(AXIS_NAME),) * 2,
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        )

        def loss(x):
            cdf, d_sum = stat(x, c_d)[0]
            return jnp.sum(cdf**2) + 1e-3 * d_sum

        return stat, loss, (pos_d,), (0,)

    def probe(cfg):
        stat, _, args_in, _ = build(cfg)
        _, flags, ctr = stat(*args_in, c_d)
        bad = [k for k, v in flags.items() if bool(v) and k != "local_pass_overflow"]
        return bad, {k: int(v) for k, v in ctr.items()}

    cap_hf = max(float(ndev - 1), 0.05)
    if args.halo_fraction is None:
        # size the halo from a forward probe, as a production run would: grow
        # until nothing overflows, then tighten to the measured halo + 15 %
        hf, probe_log = min(1.0, cap_hf), []
        for _ in range(8):
            bad, ctr = probe(make_cfg(hf))
            probe_log.append(dict(halo_fraction=hf, overflow=bad, **ctr))
            if not bad or hf >= cap_hf:
                break
            hf = min(2.0 * hf, cap_hf)
        if not bad:
            tight = min(1.15 * ctr["halo"] / per + 16.0 / per, cap_hf)
            bad_t, ctr_t = probe(make_cfg(tight))
            probe_log.append(dict(halo_fraction=tight, overflow=bad_t, **ctr_t))
            if not bad_t:
                hf = tight
        halo_fraction = hf
    else:
        halo_fraction, probe_log = float(args.halo_fraction), []
    cfg = make_cfg(halo_fraction)
    stat, loss, args_in, argnums = build(cfg)

    if mode == "forward":
        fn = jax.jit(loss)
    elif mode == "value_and_grad":
        fn = jax.jit(jax.value_and_grad(loss, argnums=argnums))
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    t0 = time.perf_counter()
    compiled = fn.lower(*args_in).compile()
    compile_s = time.perf_counter() - t0
    mem: dict = {}
    try:
        ma = compiled.memory_analysis()
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

    out = jax.block_until_ready(compiled(*args_in))  # warm-up
    times = []
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        out = jax.block_until_ready(compiled(*args_in))
        times.append(time.perf_counter() - t0)

    over_flags, ctr = probe(cfg)
    peaks = []
    for dev in mesh.devices.flat:
        try:
            st = dev.memory_stats() or {}
            peaks.append(int(st.get("peak_bytes_in_use", -1)))
        except Exception:  # pragma: no cover - CPU has no allocator stats
            peaks.append(-1)
    value = out[0] if mode == "value_and_grad" else out
    return dict(
        payload=payload,
        dtype=dtype_name,
        mode=mode,
        n=n,
        ndev=ndev,
        n_per_device=per,
        platform=jax.default_backend(),
        devices=[str(d.device_kind) for d in mesh.devices.flat][:1],
        jax=jax.__version__,
        exchange=resolve_ragged_method(args.exchange),
        halo_fraction=halo_fraction,
        capacity=int(cfg.capacity),
        halo_capacity=int(cfg.halo_capacity),
        reach=reach,
        compile_s=compile_s,
        time_s=statistics.median(times),
        time_min_s=min(times),
        compiled=mem,
        runtime_peak_bytes=peaks,
        overflow=bool(over_flags),
        overflow_flags=over_flags,
        counters_max=ctr,
        halo_probe=probe_log,
        loss=float(value),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _mb(x) -> str:
    return "-" if x is None or x < 0 else f"{x / 2**20:9.1f}"


def _summary(rows: list) -> str:
    lines = [
        f"{'payload':7} {'dtype':7} {'mode':14} {'temp MB':>9} {'arg MB':>9} "
        f"{'peak MB':>9} {'compile s':>9} {'time s':>9} overflow"
    ]
    for r in rows:
        if "error" in r:
            lines.append(
                f"{r['payload']:7} {r['dtype']:7} {r['mode']:14} ERROR "
                f"{r['error'][:60]}"
            )
            continue
        c = r["compiled"]
        peak = max(r["runtime_peak_bytes"]) if r["runtime_peak_bytes"] else -1
        lines.append(
            f"{r['payload']:7} {r['dtype']:7} {r['mode']:14} {_mb(c.get('temp'))} "
            f"{_mb(c.get('argument'))} {_mb(peak)} {r['compile_s']:9.2f} "
            f"{r['time_s']:9.4f} {','.join(r['overflow_flags']) or '-'}"
        )
    # gradient / forward ratios
    by = {(r["payload"], r["dtype"], r["mode"]): r for r in rows if "error" not in r}
    for (pl, dt, md), r in sorted(by.items()):
        if md != "value_and_grad" or (pl, dt, "forward") not in by:
            continue
        f = by[(pl, dt, "forward")]
        tf, tg = f["compiled"].get("temp", 0), r["compiled"].get("temp", 0)
        lines.append(
            f"  {pl} {dt}: value+grad / forward = {tg / max(tf, 1):.2f}x temp, "
            f"{r['time_s'] / max(f['time_s'], 1e-12):.2f}x time"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.case is not None:
        payload, dtype_name, mode = args.case.split(":")
        row = _run_case(args, payload, dtype_name, mode)
        print("RESULT " + json.dumps(row), flush=True)
        return 0

    _select_gpus(args)
    rows = []
    if not args.isolate:
        for env_key, val in _env_for(args).items():
            os.environ[env_key] = val
        for case in _cases(args):
            rows.append(_run_case(args, *case))
    else:
        base = [a for a in (argv if argv is not None else sys.argv[1:])]
        for pl, dt, md in _cases(args):
            cmd = [sys.executable, os.path.abspath(__file__), *base]
            cmd += ["--case", f"{pl}:{dt}:{md}"]
            proc = subprocess.run(
                cmd, env=_env_for(args), capture_output=True, text=True
            )
            got = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
            if proc.returncode != 0 or not got:
                rows.append(
                    dict(
                        payload=pl,
                        dtype=dt,
                        mode=md,
                        error=(proc.stderr or proc.stdout).strip()[-2000:],
                    )
                )
            else:
                rows.append(json.loads(got[-1][len("RESULT ") :]))
            print(f"# done {pl} {dt} {md}", file=sys.stderr, flush=True)

    print(_summary(rows))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(args=vars(args), results=rows), fh, indent=2)
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
