"""Compare B4 optimiser runs at matched evaluations and modelled wall time.

Wall time is modelled from the clean single-process timings of the Adam
baseline (the comparison runs shared the CPU): each loss+gradient evaluation
costs T_VG, each refresh (kNN search) T_SEARCH.
"""

import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(D, "..", "optimise_cpu.json")  # the Adam run
T_VG, T_SEARCH = 2.59, 2.75
FLOOR = None


def load(name):
    """A run's history; ``NAME~`` counts a plain (no-reuse) L-BFGS run as if
    the line search's value had been reused (same trajectory, one evaluation
    fewer on every step but an epoch's first)."""
    adjust = name.endswith("~")
    name = name.rstrip("~")
    path = BASE if name == "adam" else os.path.join(D, f"{name}.json")
    if not os.path.exists(path):
        return None
    p = json.load(open(path))
    h = p["history"]
    k_steps = {e["first_step"] for e in p["epochs"]}
    ev = np.array(
        [
            x.get("evals", 1) - (1 if adjust and x["step"] not in k_steps else 0)
            for x in h
        ],
        float,
    )
    loss = np.array([x["loss"] for x in h])
    refresh = np.array([1.0 if x["step"] in k_steps else 0.0 for x in h])
    cum_ev = np.cumsum(ev)
    t = np.cumsum(ev * T_VG + refresh * T_SEARCH)
    return p, h, loss, cum_ev, t


def at(cum, loss, budget):
    i = np.searchsorted(cum, budget, side="right") - 1
    return loss[i] if i >= 0 else np.nan


def main(names):
    rows = []
    for name in names:
        r = load(name)
        if r is None:
            continue
        p, h, loss, cum_ev, t = r
        s = p["summary"]
        ep = p["epochs"]
        stale = np.array([e["stale_fraction"] for e in ep])
        disp = np.array([e["max_displacement"] for e in ep]) / p["mean_spacing"]
        st = p["statistics"]
        xi_f, xi_t = st["final"]["xi"][-1], st["target"]["xi"][-1]
        rows.append(
            dict(
                name=name,
                steps=len(h),
                evals=int(cum_ev[-1]),
                model_min=t[-1] / 60,
                loss_final=s["final"]["loss"],
                floor=s["independent_realisation"]["loss"],
                at200=at(cum_ev, loss, 200),
                at400=at(cum_ev, loss, 400),
                at800=at(cum_ev, loss, 800),
                cdf=s["final"]["max_abs_cdf_hard_error"],
                lnr=s["final"]["max_abs_ln_ratio_1pxi"],
                xi_out=f"{xi_f:+.3f}/{xi_t:+.3f}",
                stale_mean=stale.mean(),
                stale_max=stale.max(),
                stale_last=stale[-5:].mean(),
                disp_max=disp.max(),
                replans=p["replans"],
                evals_per_step=cum_ev[-1] / len(h),
            )
        )
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(
            " | ".join(
                f"{v:.3g}" if isinstance(v, float) else str(v) for v in r.values()
            )
        )
    return rows


def figure(names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    floor = None
    for name in names:
        r = load(name)
        if r is None:
            continue
        p, h, loss, cum_ev, t = r
        floor = p["summary"]["independent_realisation"]["loss"]
        axes[0].semilogy(cum_ev, loss, label=name, lw=1)
        axes[1].semilogy(t / 60, loss, label=name, lw=1)
        ep = p["epochs"]
        axes[2].plot(
            [e["first_step"] for e in ep],
            [e["stale_fraction"] for e in ep],
            label=name,
            lw=1,
        )
    for ax in axes[:2]:
        ax.axhline(floor, color="k", ls=":", label="independent realisation")
        ax.set_ylabel("loss (at step start)")
        ax.legend(fontsize=7)
    axes[0].set_xlabel("loss+gradient evaluations")
    axes[1].set_xlabel(f"modelled CPU minutes ({T_VG} s/eval, {T_SEARCH} s/kNN search)")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("stale fraction at refresh")
    axes[2].set_yscale("symlog", linthresh=1e-3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "adam",
        "P10~",
        "P5~",
        "P1",
        "C25",
        "C10",
        "C10R",
        "H",
        "HR",
    ]
    main(names)
    figure(names, os.path.join(D, "compare.png"))
