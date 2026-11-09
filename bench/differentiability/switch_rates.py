"""A5b: how often the tree's partition switches along a trajectory, and what it costs.

Two realistic paths through position space, generated ONCE with the exact
(brute-force) gradient and then replayed under every acceptance rule, so all
rules see the same steps:

* ``descent``: normalised gradient descent of the particle positions toward
  the target ξ (per-step RMS displacement ``--gd-step``);
* ``leapfrog``: one long leapfrog trajectory of the HMC potential of
  ``hmc_energy.py`` (``U = λ L + wall``) with unit masses.

Per step ``x -> x'`` and rule it records:

* ``membership``: particles whose leaf (as a set of particles) changed -- a
  tree restructuring, the same for every rule;
* ``switched``: node pairs whose far / near status changed, counted when the
  leaf partition did not (node indices are then comparable);
* ``jump``: ``|F(x') - F_x(x')|``, the value of ``x'`` minus the value of the
  continuation ``F_x`` of the estimator from ``x``, which has no switch on the
  step -- so the difference is exactly the step's discontinuous part (zero when
  nothing switched); and the gradient analogue
  ``|∇F(x') - ∇F_x(x')| / |∇F(x')|``. For a hard rule ``F_x`` is the
  estimator on the whole topology (tree and walk lists) of ``x``; for a smooth
  rule, which is continuous at fixed leaf membership, it is the same rule with
  the leaf partition of ``x`` (``membership=x``), so only restructuring jumps;
* ``resid``: the trapezoid residual ``|ΔF - ½ (∇F + ∇F') · Δx|``, a
  rule-independent jump detector: ``O(|Δx|^3)`` for a C¹ function, the jump
  size at a jump. The exact estimator's residual is the floor.

``F`` is the ξ loss ``L`` of ``_a5_common.XiProblem``.

Examples::

    python bench/differentiability/switch_rates.py            # CPU, a few min
    python bench/differentiability/switch_rates.py --smoke
    python bench/differentiability/switch_rates.py --platform gpu --n 16000 \\
        --gpu-select free
"""

from __future__ import annotations

import argparse
import sys
import time

import _a5_common as C


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_args(p)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--gd-step", type=float, default=1e-4)
    p.add_argument("--lf-step", type=float, default=None, help="default: hmc_energy's")
    p.add_argument("--lam", type=float, default=None, help="default: hmc_energy's")
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--sharpness", type=float, default=40.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n, args.steps = 400, 12
    return args


def _paths(args, problem):
    import hmc_energy as H
    import jax
    import jax.numpy as jnp

    x0 = C.uniform(args.n, 7) * 0.9 + 0.05
    g_exact = jax.jit(jax.grad(problem.exact_loss))

    # normalised gradient descent toward the target
    xs = [x0]
    for _ in range(args.steps):
        g = g_exact(xs[-1])
        rms = jnp.sqrt(jnp.mean(jnp.sum(g * g, axis=-1)))
        xs.append(jnp.clip(xs[-1] - args.gd_step * g / rms, 0.0, 1.0))
    descent = jnp.stack(xs)

    # one long leapfrog trajectory of the exact HMC potential
    lam = args.lam if args.lam is not None else H.DEFAULT_LAMBDA
    step = args.lf_step if args.lf_step is not None else H.DEFAULT_STEP
    pot = H.Potential(problem, lam)
    grad_u = jax.jit(jax.grad(pot.exact))
    p = jax.random.normal(jax.random.PRNGKey(3), x0.shape)
    x = C.clustered(args.n, 11) * 0.9 + 0.05
    xs = [x]
    gx = grad_u(x)
    for _ in range(args.steps):
        p = p - 0.5 * step * gx
        x = x + step * p
        gx = grad_u(x)
        p = p - 0.5 * step * gx
        xs.append(x)
    return {"descent": descent, "leapfrog": jnp.stack(xs)}, lam, step


def _replay(problem, path, cfg, lam_pot=None):
    """Per-step switch and jump statistics of ``path`` under ``cfg``."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    def loss(p):
        return problem.loss_from_counts(C.tree_counts(problem, p, cfg).counts)

    if cfg.smooth:
        # continuous at fixed leaf membership: the reference is the same rule
        # with the partition of the step's start
        def frozen_loss(p, at):
            counts = C.tree_counts(problem, p, cfg, membership=at)
            return problem.loss_from_counts(counts.counts)

    else:
        # piecewise smooth between switches: the reference is the whole
        # topology (tree and walk lists) of the step's start
        def frozen_loss(p, at):
            counts = C.frozen_counts(problem, p, cfg, at)
            return problem.loss_from_counts(counts.counts)

    vg = jax.jit(jax.value_and_grad(loss))
    vg_frozen = jax.jit(jax.value_and_grad(frozen_loss))
    lists = jax.jit(lambda p: C.pair_lists(problem, p, cfg))
    leaves = jax.jit(lambda p: C.leaf_signature(problem, p, cfg))
    overflow = jax.jit(lambda p: C.tree_counts(problem, p, cfg).overflow)

    def pairset(a, b):
        a, b = np.asarray(a), np.asarray(b)
        live = a >= 0
        return set(zip(a[live].tolist(), b[live].tolist()))

    rows = []
    prev = None
    for k in range(path.shape[0]):
        x = path[k]
        assert not bool(overflow(x)), k
        f, g = vg(x)
        fa, fb, na, nb = lists(x)
        cur = dict(
            x=x,
            f=float(f),
            g=g,
            far=pairset(fa, fb),
            near=pairset(na, nb),
            leaf=np.asarray(leaves(x)),
        )
        if prev is not None:
            dx = x - prev["x"]
            resid = abs(
                cur["f"] - prev["f"] - 0.5 * float(jnp.sum((g + prev["g"]) * dx))
            )
            ff, gf = vg_frozen(x, prev["x"])
            membership = int(np.sum(cur["leaf"] != prev["leaf"]))
            switched = (
                len(cur["far"] ^ prev["far"]) + len(cur["near"] ^ prev["near"])
                if membership == 0
                else -1
            )
            rows.append(
                dict(
                    membership=membership,
                    switched=switched,
                    jump=abs(cur["f"] - float(ff)),
                    grad_jump=float(jnp.linalg.norm(g - gf) / jnp.linalg.norm(g)),
                    resid=resid,
                    step_change=abs(cur["f"] - prev["f"]),
                )
            )
        prev = cur
    return rows


def _exact_resid(problem, path):
    import jax
    import jax.numpy as jnp

    vg = jax.jit(jax.value_and_grad(problem.exact_loss))
    out, prev = [], None
    for k in range(path.shape[0]):
        f, g = vg(path[k])
        if prev is not None:
            dx = path[k] - prev[2]
            out.append(
                abs(float(f) - prev[0] - 0.5 * float(jnp.sum((g + prev[1]) * dx)))
            )
        prev = (float(f), g, path[k])
    return out


def _summary(rows, floor):
    import numpy as np

    mem = np.array([r["membership"] > 0 for r in rows])
    sw = np.array([r["switched"] > 0 for r in rows])
    jump = np.array([r["jump"] for r in rows])
    gj = np.array([r["grad_jump"] for r in rows])
    res = np.array([r["resid"] for r in rows])
    dF = np.array([r["step_change"] for r in rows])
    fl = float(np.median(floor))

    def stat(a, m):
        return (float(np.median(a[m])), float(np.max(a[m]))) if m.any() else (0.0, 0.0)

    return dict(
        steps=len(rows),
        membership_rate=float(mem.mean()),
        mean_particles_moved_leaf=float(np.mean([r["membership"] for r in rows])),
        mac_switch_rate=float(sw[~mem].mean()) if (~mem).any() else 0.0,
        mean_pairs_switched=float(
            np.mean([r["switched"] for r in rows if r["switched"] >= 0] or [0])
        ),
        median_step_change=float(np.median(dF)),
        resid_floor_exact=fl,
        resid_no_restructure=stat(res, ~mem),
        resid_restructure=stat(res, mem),
        jump_no_restructure=stat(jump, ~mem),
        jump_restructure=stat(jump, mem),
        grad_jump_no_restructure=stat(gj, ~mem),
        grad_jump_restructure=stat(gj, mem),
        steps_resid_above_10x_floor=int(np.sum(res > 10 * max(fl, 1e-300))),
    )


def run(args):
    import numpy as np

    from yggdrax.applications.corrfunc.binning import make_log_edges

    edges = make_log_edges(0.02, 0.3, 10)
    problem = C.XiProblem(args.n, edges, args.sharpness, seed=0, margin=0.5)
    t0 = time.perf_counter()
    paths, lam, step = _paths(args, problem)
    print(f"# paths generated in {time.perf_counter() - t0:.1f}s", flush=True)
    out = {}
    for name, path in paths.items():
        floor = _exact_resid(problem, path)
        probes = [path[k] for k in range(0, path.shape[0], max(1, path.shape[0] // 8))]
        probes.append(path[-1])
        out[name] = {"resid_exact": floor, "rules": {}}
        for label, kw in C.rules(args.theta, args.eps, args.beta):
            cfg = C.plan(problem, probes, leaf_size=args.leaf_size, **kw)
            t0 = time.perf_counter()
            rows = _replay(problem, path, cfg)
            summ = _summary(rows, floor)
            out[name]["rules"][label] = dict(config=kw, steps=rows, summary=summ)
            print(
                f"{name:8s} {label:34s} restructure {summ['membership_rate']:.2f} "
                f"MAC-switch {summ['mac_switch_rate']:.2f} "
                f"({summ['mean_pairs_switched']:.1f} pairs) "
                f"jump(no-restr) med/max {summ['jump_no_restructure'][0]:.1e}/"
                f"{summ['jump_no_restructure'][1]:.1e} "
                f"jump(restr) {summ['jump_restructure'][0]:.1e}/"
                f"{summ['jump_restructure'][1]:.1e} "
                f"resid>10floor {summ['steps_resid_above_10x_floor']} "
                f"| dF {summ['median_step_change']:.1e} floor "
                f"{summ['resid_floor_exact']:.1e} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    return dict(
        bench="A5b switch rates and jump sizes",
        env=C.environment(),
        args=vars(args),
        hmc_lambda=lam,
        leapfrog_step=step,
        edges=[float(e) for e in np.asarray(edges)],
        paths=out,
    )


def plot(payload, path):
    import numpy as np

    plt = C.import_pyplot()
    names = list(payload["paths"])
    fig, axes = plt.subplots(
        2, len(names), figsize=(5.2 * len(names), 6), squeeze=False
    )
    for j, name in enumerate(names):
        data = payload["paths"][name]
        floor = np.array(data["resid_exact"])
        ax = axes[0, j]
        ax.semilogy(floor, "k-", lw=0.8, label="exact (floor)")
        for label, rec in data["rules"].items():
            res = np.array([r["resid"] for r in rec["steps"]])
            ax.semilogy(np.maximum(res, 1e-18), lw=0.8, label=label)
        ax.set_title(f"{name}: trapezoid residual per step")
        ax.set_xlabel("step")
        ax.legend(fontsize=6)
        ax = axes[1, j]
        labels = list(data["rules"])
        xs = np.arange(len(labels))
        for off, key, col in (
            (-0.2, "jump_no_restructure", "C0"),
            (0.2, "jump_restructure", "C3"),
        ):
            med = [data["rules"][lab]["summary"][key][0] for lab in labels]
            mx = [data["rules"][lab]["summary"][key][1] for lab in labels]
            ax.bar(xs + off, np.maximum(mx, 1e-18), 0.4, color=col, alpha=0.35)
            ax.bar(
                xs + off,
                np.maximum(med, 1e-18),
                0.4,
                color=col,
                label=key.replace("_", " ") + " (median; light: max)",
            )
        dF = data["rules"][labels[0]]["summary"]["median_step_change"]
        ax.axhline(dF, color="k", ls=":", lw=0.8, label="median |ΔF| per step")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([lab.replace(", ", "\n") for lab in labels], fontsize=6)
        ax.set_title(f"{name}: jump |F(x') - F_x(x')|")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    print(f"# wrote {path}")


def main(argv=None) -> int:
    args = _parse(argv)
    if args.replot:
        import json

        with open(args.replot) as f:
            plot(json.load(f), args.fig or args.replot.replace(".json", ".png"))
        return 0
    C.select_device(args)
    payload = run(args)
    C.write_json(args.out or C.default_path("switch_rates", "json", args), payload)
    plot(payload, args.fig or C.default_path("switch_rates", "png", args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
