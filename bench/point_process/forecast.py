"""Point-process forward model through yggdrax statistics: gradients, Fisher, recovery (B3).

``theta = (sigma, beta) -> Thomas-process positions (fixed draws) -> xi and soft
kNN-CDF (tree rebuilt inside every call) -> loss``, all JAX. The script writes
the JSON and figures of the paper's inference demonstration:

1. **Validation.** The ensemble-mean ``xi`` against the analytic Thomas
   ``xi(r)`` with fixed counts (and, to show the correction matters, the Poisson
   one), window-averaged exactly as the soft estimator measures it. A second,
   smaller ensemble at ``beta = --beta-alt`` is checked against the Monte Carlo
   prediction of :func:`profile_xi_fixed_counts_mc`.
2. **Gradient check.** On one realisation: the forward-mode autodiff Jacobian
   ``dS/dtheta``, the reverse-mode one, and central finite differences of the
   full chain (positions, wrap, partition, tree, walk, statistic) over a range
   of steps.
3. **Fisher forecast** from ``xi``, the kNN-CDF and both: the ensemble-mean
   autodiff Jacobian (``--n-jac`` realisations), cross-checked by finite
   differences on the first ``--n-fd-jac`` of them, and the sample covariance of
   ``--n-cov`` independent realisations with the Hartlap factor.
4. **Recovery on independent realisations.** ``--n-fit`` trials: fit the
   parameters of realisation(s) A (seeds never used for the target) to the
   statistic of realisation B, by Levenberg–Marquardt with the autodiff Jacobian.
   Both are noisy, so the expected scatter is ``sqrt(1 + 1/m)`` times the
   Fisher error (``m = --fit-model-realisations``).

Examples::

    # CPU smoke (a couple of minutes; also what the tests' sizes look like)
    python bench/point_process/forecast.py --smoke --out-dir /tmp/pp_smoke

    # CPU, paper default (N = 2048; ~30-40 min on a laptop)
    python bench/point_process/forecast.py --out-dir bench/results/point_process_cpu

    # A100 server: picks a free GPU with autocvd (org rule)
    python bench/point_process/forecast.py --platform gpu --gpu-select free \\
        --n-parents 2048 --n-children 16 --sigma 0.01 --n-query-side 32 \\
        --n-cov 400 --n-jac 64 --n-fit 50 --batch-size 8 \\
        --out-dir bench/results/point_process_gpu

GPU selection happens before JAX starts: ``--gpu-select free`` waits for idle
cards via ``autocvd`` (``least-used`` takes the idlest now, ``none`` keeps
``CUDA_VISIBLE_DEVICES``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--ndev", type=int, default=1, help="devices in the mesh")
    p.add_argument(
        "--gpu-select", choices=["free", "least-used", "none"], default="free"
    )
    p.add_argument("--gpu-wait-timeout", type=int, default=None)
    # model
    p.add_argument("--box", type=float, default=1.0)
    p.add_argument("--n-parents", type=int, default=128)
    p.add_argument("--n-children", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--beta", type=float, default=1.0, help="fiducial profile shape")
    p.add_argument("--beta-alt", type=float, default=1.5, help="MC-validated shape")
    # xi
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--xi-rmin", type=float, default=0.2, help="in units of sigma")
    p.add_argument("--xi-rmax", type=float, default=6.0, help="in units of sigma")
    p.add_argument("--xi-sharpness", type=float, default=20.0)
    p.add_argument("--eps", type=float, default=1e-6)
    # kNN
    p.add_argument("--ks", default="1,2,4")
    p.add_argument("--n-r-knn", type=int, default=5)
    p.add_argument(
        "--knn-rmin", type=float, default=None, help="default 0.5 x mean separation"
    )
    p.add_argument(
        "--knn-rmax", type=float, default=None, help="default 1.6 x mean separation"
    )
    p.add_argument("--knn-sharpness", type=float, default=15.0)
    p.add_argument("--n-query-side", type=int, default=12)
    # ensembles and fits
    p.add_argument("--n-cov", type=int, default=150)
    p.add_argument("--n-jac", type=int, default=24)
    p.add_argument("--n-fd-jac", type=int, default=4)
    p.add_argument("--n-val-alt", type=int, default=16)
    p.add_argument("--n-fit", type=int, default=20)
    p.add_argument("--fit-model-realisations", type=int, default=1)
    p.add_argument("--fit-steps", type=int, default=12)
    p.add_argument("--fit-start", default="1.25,0.8", help="theta0 / theta_true")
    p.add_argument("--fit-stats", default="xi,knn,both")
    p.add_argument("--fd-rel", default="1e-1,3e-2,1e-2,3e-3,1e-3,1e-4,1e-5,1e-6,1e-7")
    p.add_argument("--fd-jac-rel", type=float, default=1e-5)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="realisations per vmap batch (default: 1 on CPU, all on GPU)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None, help="write results.json + figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--figures-from",
        default=None,
        help="only redraw the figures of this results.json (into its directory)",
    )
    p.add_argument("--smoke", action="store_true", help="tiny CPU run")
    args = p.parse_args(argv)
    if args.smoke:
        args.platform = "cpu"
        args.n_parents, args.n_children = 48, 12
        args.sigma = 0.03
        args.nbins, args.n_r_knn, args.ks = 6, 4, "1,2"
        args.n_query_side = 8
        args.n_cov, args.n_jac, args.n_fd_jac, args.n_val_alt = 30, 4, 2, 4
        args.n_fit, args.fit_steps = 3, 6
        args.fd_rel = "1e-2,1e-4,1e-6"
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


def _set_env(args) -> None:
    if args.platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        flags = os.environ.get("XLA_FLAGS", "")
        if args.ndev > 1 and "xla_force_host_platform_device_count" not in flags:
            flags += f" --xla_force_host_platform_device_count={args.ndev}"
        # jax 0.11.2: XLA:CPU's loop vectorizer can stall a compile (see conftest)
        if "xla_backend_extra_options" not in flags:
            flags += " --xla_backend_extra_options=-vectorize-loops=false"
        os.environ["XLA_FLAGS"] = flags.strip()
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_ENABLE_X64"] = "1"
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)


class _Timer:
    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()
                print(f"# {name} ...", file=sys.stderr, flush=True)

            def __exit__(self, *exc):
                dt = time.perf_counter() - self.t
                timer.times[name] = dt
                print(f"#   {name}: {dt:.1f} s", file=sys.stderr, flush=True)

        return _Ctx()


def _boot_ratio_sem(th, expected, seed, n_boot=2000):
    """Bootstrap standard error of std(theta_hat) / expected, per parameter."""
    import numpy as np

    if th.shape[0] < 3:
        return [float("nan")] * th.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, th.shape[0], (n_boot, th.shape[0]))
    ratios = th[idx].std(axis=1, ddof=1) / expected
    return ratios.std(axis=0, ddof=1).tolist()


def run(args) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from yggdrax.applications.point_process import (
        PARAM_NAMES,
        concat_statistics,
        draw_ensemble,
        ensemble_jacobians,
        ensemble_statistics,
        fisher_forecast,
        fit_parameters,
        hartlap_factor,
        make_knn_cdf_statistic,
        make_xi_statistic,
        profile_xi_fixed_counts_mc,
        query_grid,
        sample_covariance,
        statistic_jacobian,
        thomas_positions,
        thomas_xi_fixed_counts,
        thomas_xi_poisson,
        window_average,
    )
    from yggdrax.applications.point_process.inference import (
        central_difference_jacobian,
    )
    from yggdrax.distributed.sharding import make_mesh
    from yggdrax.distributed.stats import soft_window_cutoffs

    timer = _Timer()
    L = float(args.box)
    P, M = int(args.n_parents), int(args.n_children)
    N = P * M
    theta_true = jnp.asarray([args.sigma, args.beta], jnp.float64)
    sep = L / N ** (1.0 / 3.0)
    mesh = make_mesh(args.ndev)
    batch = args.batch_size
    if batch is None:
        batch = 1 if args.platform == "cpu" else None

    edges = np.geomspace(
        args.xi_rmin * args.sigma, args.xi_rmax * args.sigma, args.nbins + 1
    )
    ks = tuple(int(k) for k in args.ks.split(","))
    r_knn = np.geomspace(
        args.knn_rmin or 0.5 * sep, args.knn_rmax or 1.6 * sep, args.n_r_knn
    )
    queries = query_grid(args.n_query_side, L)
    xi_stat = make_xi_statistic(
        edges,
        box_size=L,
        n_points=N,
        sharpness=args.xi_sharpness,
        eps=args.eps,
        mesh=mesh,
    )
    knn_stat = make_knn_cdf_statistic(
        ks,
        r_knn,
        box_size=L,
        n_points=N,
        queries=queries,
        sharpness=args.knn_sharpness,
        mesh=mesh,
    )
    both = concat_statistics(xi_stat, knn_stat)
    p_xi = xi_stat.size
    sl = {
        "xi": slice(0, p_xi),
        "knn": slice(p_xi, both.size),
        "both": slice(0, both.size),
    }
    r_cut = soft_window_cutoffs(edges, args.xi_sharpness, args.eps)

    key = jax.random.PRNGKey(args.seed)
    k_cov, k_jac, k_alt, k_fit_a, k_fit_b, k_mc = jax.random.split(key, 6)
    draws_cov = draw_ensemble(k_cov, args.n_cov, P, M)
    draws_jac = draw_ensemble(k_jac, args.n_jac, P, M)

    def take(d, i):
        return jax.tree.map(lambda a: a[i], d)

    def check(over, what):
        if bool(np.any(np.asarray(over))):
            raise RuntimeError(f"capacity overflow in {what}; raise the config")

    out: dict = {
        "config": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in vars(args).items()
        },
        "derived": {
            "n_points": N,
            "mean_separation": sep,
            "xi_edges": edges.tolist(),
            "xi_cut": list(r_cut),
            "knn_r_grid": r_knn.tolist(),
            "ks": list(ks),
            "n_queries": int(queries.shape[0]),
            "labels": list(both.labels),
            "param_names": list(PARAM_NAMES),
            "theta_true": np.asarray(theta_true).tolist(),
            "devices": [str(d) for d in mesh.devices.ravel()],
            "jax": jax.__version__,
            "batch_size": batch,
        },
    }

    # ---- 2. gradient check on one realisation (first: also warms up the jit) --
    d0 = take(draws_jac, 0)
    jac_fn = jax.jit(lambda t, d: statistic_jacobian(both, t, d, L))
    with timer("jacfwd_compile"):
        v0, j0, o0 = jac_fn(theta_true, d0)
        j0.block_until_ready()
    check(o0, "jacfwd")
    with timer("jacfwd_run"):
        jac_fn(theta_true, d0)[1].block_until_ready()
    jrev_fn = jax.jit(
        jax.jacrev(lambda t, d: both.fn(thomas_positions(t, d, L))[0]),
        static_argnums=(),
    )
    with timer("jacrev_compile_and_run"):
        jr = jrev_fn(theta_true, d0)
    fwd_fn = jax.jit(lambda t, d: both.fn(thomas_positions(t, d, L)))
    fwd_fn(theta_true, d0)[0].block_until_ready()
    with timer("forward_run"):
        fwd_fn(theta_true, d0)[0].block_until_ready()
    fd_fn = jax.jit(lambda t, d, h: central_difference_jacobian(both, t, d, L, h))
    j0n = np.asarray(j0)
    fd_rows = []
    for rel in (float(x) for x in args.fd_rel.split(",")):
        fd, ofd = fd_fn(theta_true, d0, rel * theta_true)
        check(ofd, "finite differences")
        fdn = np.asarray(fd)
        row: dict = {
            "rel_step": rel,
            "abs_step": (rel * np.asarray(theta_true)).tolist(),
        }
        for name, s in sl.items():
            scale = np.abs(j0n[s]).max(axis=0)
            row[name] = (np.abs(fdn[s] - j0n[s]).max(axis=0) / scale).tolist()
        fd_rows.append(row)
        print(f"#   FD rel {rel:.0e}: {row['xi']} {row['knn']}", file=sys.stderr)
    out["gradient_check"] = {
        "description": "max_b |FD - AD| / max_b |AD| per parameter, one realisation",
        "jacobian_ad": j0n.tolist(),
        "values": np.asarray(v0).tolist(),
        "jacrev_vs_jacfwd": float(
            np.abs(np.asarray(jr) - j0n).max() / np.abs(j0n).max()
        ),
        "steps": fd_rows,
    }

    # ---- 1. validation + covariance ensemble -----------------------------------
    stat_fn = jax.jit(lambda t, d: ensemble_statistics(both, t, d, L, batch_size=batch))
    with timer("covariance_ensemble"):
        vals, over = stat_fn(theta_true, draws_cov)
        vals = np.asarray(vals)
    check(over, "covariance ensemble")
    mean = vals.mean(0)
    sem = vals.std(0, ddof=1) / math.sqrt(args.n_cov)
    val: dict = {"measured_mean": mean.tolist(), "measured_sem": sem.tolist()}
    cxi = sample_covariance(jnp.asarray(vals[:, sl["xi"]]))
    if abs(args.beta - 1.0) < 1e-12:
        a_fix = window_average(
            lambda r: thomas_xi_fixed_counts(r, args.sigma, P, M, L**3),
            edges,
            args.xi_sharpness,
            r_cut,
        )
        a_poi = window_average(
            lambda r: thomas_xi_poisson(r, args.sigma, P / L**3),
            edges,
            args.xi_sharpness,
            r_cut,
        )
        pred = {"fixed_counts": a_fix, "poisson": a_poi}
    else:
        pred = {
            "fixed_counts_mc": profile_xi_fixed_counts_mc(
                k_mc, theta_true, edges, args.xi_sharpness, r_cut, P, M, L**3
            )
        }
    h_xi = hartlap_factor(args.n_cov, p_xi)
    prec_mean = h_xi * args.n_cov * np.linalg.inv(np.asarray(cxi))
    for name, a in pred.items():
        d = mean[sl["xi"]] - a
        val[name] = {
            "prediction": np.asarray(a).tolist(),
            "pull": (d / sem[sl["xi"]]).tolist(),
            "chi2": float(d @ prec_mean @ d),
            "dof": p_xi,
        }
    # a second ensemble at another profile shape, against Monte Carlo
    theta_alt = jnp.asarray([args.sigma, args.beta_alt], jnp.float64)
    draws_alt = draw_ensemble(k_alt, args.n_val_alt, P, M)
    xi_fn = jax.jit(
        lambda t, d: ensemble_statistics(xi_stat, t, d, L, batch_size=batch)
    )
    with timer("validation_alt_ensemble"):
        va, oa = xi_fn(theta_alt, draws_alt)
    check(oa, "alt-shape ensemble")
    va = np.asarray(va)
    mc = profile_xi_fixed_counts_mc(
        k_mc, theta_alt, edges, args.xi_sharpness, r_cut, P, M, L**3
    )
    sem_a = va.std(0, ddof=1) / math.sqrt(args.n_val_alt)
    val["alt_shape"] = {
        "beta": args.beta_alt,
        "measured_mean": va.mean(0).tolist(),
        "measured_sem": sem_a.tolist(),
        "prediction_mc": mc.tolist(),
        "pull": ((va.mean(0) - mc) / sem_a).tolist(),
    }
    out["validation"] = val

    # ---- 3. Jacobian ensemble and Fisher ---------------------------------------
    jac_ens = jax.jit(lambda t, d: ensemble_jacobians(both, t, d, L, batch_size=batch))
    with timer("jacobian_ensemble"):
        _, jacs, oj = jac_ens(theta_true, draws_jac)
        jacs = np.asarray(jacs)
    check(oj, "jacobian ensemble")
    j_mean = jacs.mean(0)
    j_sem = jacs.std(0, ddof=1) / math.sqrt(args.n_jac)
    with timer("jacobian_fd_check"):
        fds = []
        for i in range(args.n_fd_jac):
            fd, ofd = fd_fn(
                theta_true, take(draws_jac, i), args.fd_jac_rel * theta_true
            )
            check(ofd, "FD jacobian")
            fds.append(np.asarray(fd))
    fd_mean = np.mean(fds, axis=0)
    ad_sub = jacs[: args.n_fd_jac].mean(0)
    cov_all = np.asarray(sample_covariance(jnp.asarray(vals)))
    fisher: dict = {}
    forecasts = {}
    for name, s in sl.items():
        fc = fisher_forecast(j_mean[s], cov_all[s, s], args.n_cov)
        forecasts[name] = fc
        fisher[name] = {
            "p": int(cov_all[s, s].shape[0]),
            "hartlap": fc.hartlap,
            "fisher": fc.fisher.tolist(),
            "param_covariance": fc.param_covariance.tolist(),
            "errors": fc.errors.tolist(),
            "relative_errors": (fc.errors / np.asarray(theta_true)).tolist(),
            "conditional_errors": fc.conditional_errors.tolist(),
            "correlation": float(fc.correlation[0, 1]),
            "jacobian_fd_vs_ad": float(
                np.abs(fd_mean[s] - ad_sub[s]).max() / np.abs(ad_sub[s]).max()
            ),
            "jacobian_relative_sem": float(
                np.abs(j_sem[s]).max() / np.abs(j_mean[s]).max()
            ),
        }
        print(
            f"#   Fisher {name}: errors {fc.errors} corr {fc.correlation[0, 1]:.3f}",
            file=sys.stderr,
        )
    out["fisher"] = fisher
    out["jacobian_mean"] = j_mean.tolist()
    out["jacobian_sem"] = j_sem.tolist()
    out["data_covariance"] = cov_all.tolist()

    # ---- 4. recovery on independent realisations -------------------------------
    m = args.fit_model_realisations
    start = np.asarray([float(x) for x in args.fit_start.split(",")]) * np.asarray(
        theta_true
    )
    lower = np.asarray([0.2 * args.sigma, 0.3])
    upper = np.asarray([5.0 * args.sigma, 3.0])
    draws_a = draw_ensemble(k_fit_a, args.n_fit * m, P, M)
    draws_b = draw_ensemble(k_fit_b, args.n_fit, P, M)
    with timer("fit_targets"):
        targets, ot = stat_fn(theta_true, draws_b)
        targets = np.asarray(targets)
        # each fit's model at the truth, for chi2(theta_true) >= min chi2
        at_truth, oat = stat_fn(theta_true, draws_a)
        at_truth = np.asarray(at_truth).reshape(args.n_fit, m, -1).mean(1)
    check(ot, "fit targets")
    check(oat, "fit models at the truth")
    recovery: dict = {}
    for name in args.fit_stats.split(","):
        s = sl[name]
        stat = {"xi": xi_stat, "knn": knn_stat, "both": both}[name]
        prec = forecasts[name].hartlap * np.linalg.inv(cov_all[s, s])

        def one_fit(target, d_model, stat=stat, prec=prec):
            return fit_parameters(
                stat,
                target,
                d_model,
                jnp.asarray(start),
                jnp.asarray(prec),
                L,
                n_steps=args.fit_steps,
                lower=jnp.asarray(lower),
                upper=jnp.asarray(upper),
                batch_size=batch,
            )

        fit_jit = jax.jit(one_fit)
        thetas, hist, chi2s = [], [], []
        with timer(f"fits_{name}"):
            for t in range(args.n_fit):
                d_model = jax.tree.map(lambda a: a[t * m : (t + 1) * m], draws_a)
                res = fit_jit(jnp.asarray(targets[t, s]), d_model)
                check(res.overflow, f"fit {name} {t}")
                thetas.append(np.asarray(res.theta))
                hist.append(np.asarray(res.theta_history))
                chi2s.append(np.asarray(res.chi2_history))
        th = np.asarray(thetas)
        r_truth = targets[:, s] - at_truth[:, s]
        chi2_truth = np.einsum("ti,ij,tj->t", r_truth, prec, r_truth)
        final = np.asarray(chi2s)[:, -1]
        fc = forecasts[name]
        expect_cov = (1.0 + 1.0 / m) * fc.param_covariance
        dev = th - np.asarray(theta_true)
        pulls2 = np.einsum("ti,ij,tj->t", dev, np.linalg.inv(expect_cov), dev)
        std = th.std(0, ddof=1) if args.n_fit > 1 else np.full(2, np.nan)
        recovery[name] = {
            "theta_hat": th.tolist(),
            "theta_history": np.asarray(hist).tolist(),
            "chi2_history": np.asarray(chi2s).tolist(),
            "mean": th.mean(0).tolist(),
            "bias_over_expected_sigma": (
                (th.mean(0) - np.asarray(theta_true)) / np.sqrt(np.diag(expect_cov))
            ).tolist(),
            "std": std.tolist(),
            "expected_std": np.sqrt(np.diag(expect_cov)).tolist(),
            "std_over_expected": (std / np.sqrt(np.diag(expect_cov))).tolist(),
            "std_over_expected_bootstrap_sem": _boot_ratio_sem(
                th, np.sqrt(np.diag(expect_cov)), args.seed
            ),
            "std_ratio_1sigma_band": 1.0 / math.sqrt(2.0 * max(args.n_fit - 1, 1)),
            "corr": (
                float(np.corrcoef(th.T)[0, 1]) if args.n_fit > 2 else float("nan")
            ),
            "mean_pull_chi2": float(pulls2.mean()),
            "mean_pull_chi2_expected": 2.0,
            "final_chi2_mean": float(final.mean()),
            "chi2_at_truth_mean": float(chi2_truth.mean()),
            "fraction_final_chi2_le_truth": float(np.mean(final <= chi2_truth)),
            "final_chi2_expected": float((1.0 + 1.0 / m) * (s.stop - s.start - 2)),
        }
        print(
            f"#   recovery {name}: std/expected {recovery[name]['std_over_expected']}"
            f" bias/sigma {recovery[name]['bias_over_expected_sigma']}",
            file=sys.stderr,
        )
    out["recovery"] = {
        "model_realisations_per_fit": m,
        "theta_start": start.tolist(),
        "expected_scale": math.sqrt(1.0 + 1.0 / m),
        **recovery,
    }
    out["timings_s"] = timer.times
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _MUTED, _GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
_COLOR = {"xi": _BLUE, "knn": _ORANGE, "both": _AQUA}
_NAME = {"xi": r"$\xi$", "knn": "kNN-CDF", "both": r"$\xi$ + kNN-CDF"}


def _style(plt):
    plt.rcParams.update(
        {
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def figures(res: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    _style(plt)
    paths = []
    der = res["derived"]
    edges = np.asarray(der["xi_edges"])
    rc = np.sqrt(edges[:-1] * edges[1:])
    p_xi = len(rc)
    val = res["validation"]

    # validation: xi vs analytic
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(4.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    m = np.asarray(val["measured_mean"][:p_xi])
    e = np.asarray(val["measured_sem"][:p_xi])
    beta = res["config"]["beta"]
    ax.errorbar(rc, m, e, fmt="o", ms=4, color=_INK, label=rf"model, $\beta={beta:g}$")
    key = "fixed_counts" if "fixed_counts" in val else "fixed_counts_mc"
    pred = np.asarray(val[key]["prediction"])
    ax.plot(rc, pred, color=_BLUE, label="analytic, fixed counts")
    axr.plot(rc, (m - pred) / e, "o", ms=4, color=_BLUE)
    if "poisson" in val:
        pp = np.asarray(val["poisson"]["prediction"])
        ax.plot(rc, pp, color=_ORANGE, ls="--", label="analytic, Poisson counts")
        axr.plot(rc, (m - pp) / e, "s", ms=4, color=_ORANGE, mfc="none")
    alt = val["alt_shape"]
    ma, ea = np.asarray(alt["measured_mean"]), np.asarray(alt["measured_sem"])
    ax.errorbar(
        rc,
        ma,
        ea,
        fmt="^",
        ms=4,
        color=_MUTED,
        label=rf"model, $\beta={alt['beta']:g}$",
    )
    ax.plot(rc, alt["prediction_mc"], color=_AQUA, ls=":", label="Monte Carlo")
    axr.plot(rc, alt["pull"], "^", ms=4, color=_AQUA)
    ax.set_xscale("log")
    ax.set_yscale("log")
    pos = np.concatenate([m[m > 0], ma[ma > 0]])
    ax.set_ylim(0.5 * pos.min(), 2.0 * pos.max())
    ax.set_ylabel(r"$\bar\xi_b$ (soft window)")
    ax.legend(fontsize=7)
    axr.axhspan(-2, 2, color=_GRID, lw=0)
    axr.set_ylabel("pull")
    axr.set_xlabel(r"$r$ [box units]")
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "xi_validation.pdf"))
    fig.savefig(paths[-1])
    plt.close(fig)

    # gradient check: FD error vs step
    gc = res["gradient_check"]["steps"]
    h = np.asarray([r["rel_step"] for r in gc])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, col in (("xi", _BLUE), ("knn", _ORANGE)):
        err = np.asarray([r[name] for r in gc])
        ax.loglog(
            h,
            err[:, 0],
            "o-",
            color=col,
            ms=4,
            label=rf"{_NAME[name]}, $\partial/\partial\sigma$",
        )
        ax.loglog(
            h,
            err[:, 1],
            "s--",
            color=col,
            ms=4,
            mfc="none",
            label=rf"{_NAME[name]}, $\partial/\partial\beta$",
        )
    # h^2 guide through the xi d/dsigma error at the step nearest 1e-3
    i0 = int(np.argmin(np.abs(np.log(h / 1e-3))))
    ax.loglog(
        h,
        gc[i0]["xi"][0] * (h / h[i0]) ** 2,
        color=_MUTED,
        lw=1,
        ls=":",
        label=r"$\propto h^2$",
    )
    ax.set_ylim(1e-12, 1.0)
    ax.set_xlabel(r"relative step $h/\theta$")
    ax.set_ylabel(r"$\max|{\rm FD}-{\rm AD}|\,/\,\max|{\rm AD}|$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "gradient_fd_check.pdf"))
    fig.savefig(paths[-1])
    plt.close(fig)

    # Fisher ellipses + recovered points
    theta = np.asarray(der["theta_true"])
    rec = res["recovery"]
    scale = rec["expected_scale"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for name in ("xi", "knn", "both"):
        cov = np.asarray(res["fisher"][name]["param_covariance"]) * scale**2
        w, v = np.linalg.eigh(cov)
        ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))
        for nsig, alpha in ((1.0, 1.0), (2.0, 0.45)):
            # 68% / 95% contours of a 2-D Gaussian
            k = math.sqrt(2.30) if nsig == 1.0 else math.sqrt(6.17)
            ax.add_patch(
                Ellipse(
                    theta,
                    2 * k * math.sqrt(w[1]),
                    2 * k * math.sqrt(w[0]),
                    angle=ang,
                    fill=False,
                    color=_COLOR[name],
                    lw=1.6,
                    alpha=alpha,
                    label=_NAME[name] if nsig == 1.0 else None,
                )
            )
        if name in rec:
            th = np.asarray(rec[name]["theta_hat"])
            ax.plot(
                th[:, 0], th[:, 1], "o", ms=4, color=_COLOR[name], mec="white", mew=0.8
            )
    ax.plot(*theta, "+", color=_INK, ms=10, mew=1.5)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(
        rf"Fisher $\times\sqrt{{1+1/m}}$ (68/95%) and fits, $m={rec['model_realisations_per_fit']}$",
        fontsize=8,
        color=_INK,
    )
    ax.legend(fontsize=7, loc="best")
    ax.autoscale_view()
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "fisher_recovery.pdf"))
    fig.savefig(paths[-1])
    plt.close(fig)

    # kNN-CDFs (mean over the covariance ensemble)
    ks = der["ks"]
    rg = np.asarray(der["knn_r_grid"])
    mk = np.asarray(val["measured_mean"][p_xi:]).reshape(len(ks), -1)
    ek = np.asarray(val["measured_sem"][p_xi:]).reshape(len(ks), -1)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for i, k in enumerate(ks):
        ax.errorbar(rg, mk[i], ek[i], fmt="o-", ms=4, color=ramp[i % 4], label=f"k={k}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [box units]")
    ax.set_ylabel(r"$P(r_k \leq r)$, soft")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "knn_cdf.pdf"))
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    args = _parse(argv)
    if args.figures_from:
        with open(args.figures_from) as f:
            res = json.load(f)
        for p in figures(res, os.path.dirname(os.path.abspath(args.figures_from))):
            print(f"# wrote {p}", file=sys.stderr)
        return 0
    _select_gpus(args)
    _set_env(args)
    res = run(args)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "results.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"# wrote {path}", file=sys.stderr)
        if not args.no_figures:
            for p in figures(res, args.out_dir):
                print(f"# wrote {p}", file=sys.stderr)
    summary = {
        "gradient_fd_best": {
            name: min(max(r[name]) for r in res["gradient_check"]["steps"])
            for name in ("xi", "knn")
        },
        "fisher_errors": {k: v["errors"] for k, v in res["fisher"].items()},
        "recovery_std_over_expected": {
            k: v["std_over_expected"]
            for k, v in res["recovery"].items()
            if isinstance(v, dict)
        },
        "timings_s": res["timings_s"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
