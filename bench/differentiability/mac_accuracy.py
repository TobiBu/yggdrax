"""MAC accuracy: approximation error vs. opening angle ``theta``.

Quantifies the accuracy/cost tradeoff of the Dehnen multipole acceptance
criterion. For each ``theta`` we run the dual-tree traversal, then compare, on
exactly the particle pairs the traversal handled in the *far* field, an exact
direct sum against a monopole (centre-of-mass) approximation of a softened
1/r potential energy:

    U_far_exact = sum over far node-pairs (A,B) of
                      sum_{i in A, j in B} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)
    U_far_mono  = sum over far node-pairs (A,B) of
                      M_A M_B / sqrt(|com_A - com_B|^2 + eps^2)

The reported error is ``|U_far_mono - U_far_exact| / U_total`` where
``U_total`` is the exact energy of the whole system. This is a *monopole*
(order-0) approximation and therefore an upper bound on the error a
higher-order multipole expansion would incur; it isolates the geometric effect
of the MAC. Error is zero when no far pairs are accepted (small ``theta``) and
grows as ``theta`` opens up. Reuses existing yggdrax traversal + mass moments;
no new algorithm code.

Results -> ``results/differentiability/mac_accuracy.json``. This is an accuracy
(not a scaling) measurement -- keep N modest (<~2e4).

Local smoke (CPU):

    conda run -n jaccpot python bench/differentiability/mac_accuracy.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import dump_json, run_metadata, select_free_gpu


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    )
    p.add_argument("--backend", type=str, default="radix")
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--softening", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument(
        "--output", type=str, default="results/differentiability/mac_accuracy.json"
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _exact_potential_energy(pos, mass, eps: float, chunk: int = 512) -> float:
    """Exact softened 1/r potential energy of the full system (chunked)."""
    import numpy as np

    n = pos.shape[0]
    total = 0.0
    for i in range(0, n, chunk):
        block = pos[i : i + chunk]
        block_m = mass[i : i + chunk]
        d = block[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1) + eps * eps)
        contrib = (block_m[:, None] * mass[None, :]) / r
        # Zero the self terms in this block before summing.
        rows = np.arange(block.shape[0])
        contrib[rows, i + rows] = 0.0
        total += float(np.sum(contrib))
    return 0.5 * total  # each unordered pair counted twice


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n = 800
        args.thetas = [0.2, 0.5, 1.0]
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="mac")

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from yggdrax import (
        DualTreeTraversalConfig,
        Tree,
        build_interactions_and_neighbors,
        compute_tree_geometry,
    )
    from yggdrax.tree_moments import compute_tree_mass_moments

    eps = args.softening
    key = jax.random.PRNGKey(args.seed)
    pos = jax.random.uniform(
        key, (args.n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float64
    )
    mass = jnp.ones(args.n, dtype=jnp.float64)

    pos_np = np.asarray(pos)
    mass_np = np.asarray(mass)
    u_total = _exact_potential_energy(pos_np, mass_np, eps)

    tree = Tree.from_particles(
        pos,
        mass,
        tree_type=args.backend,
        build_mode="adaptive",
        leaf_size=args.leaf_size,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    moments = compute_tree_mass_moments(tree, tree.positions_sorted, tree.masses_sorted)

    node_ranges = np.asarray(tree.node_ranges)
    pos_sorted = np.asarray(tree.positions_sorted)
    mass_sorted = np.asarray(tree.masses_sorted)
    node_mass = np.asarray(moments.mass)
    node_com = np.asarray(moments.center_of_mass)

    # Generous fixed capacities so the traversal never truncates far pairs.
    # The tight-theta end (small theta) accepts almost nothing as "far", so the
    # dual-tree walk opens nearly the whole tree: the pair queue and per-leaf
    # near lists must be large enough to hold that near-brute-force frontier at
    # this N. Queues are int32 and cheap in memory, so we size them well above
    # the worst case (num_leaves^2 pending pairs, ~N near particles per leaf).
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 22,
        process_block=64,
        max_interactions_per_node=1 << 15,
        max_neighbors_per_leaf=1 << 15,
    )

    records = []
    for theta in args.thetas:
        interactions, _ = build_interactions_and_neighbors(
            tree, geometry, theta=theta, mac_type="dehnen", traversal_config=cfg
        )
        src = np.asarray(interactions.sources)
        tgt = np.asarray(interactions.targets)

        u_far_exact = 0.0
        u_far_mono = 0.0
        covered_pairs = 0
        for a, b in zip(src.tolist(), tgt.tolist()):
            sa, ea = node_ranges[a]
            sb, eb = node_ranges[b]
            # node_ranges is INCLUSIVE [start, end]; slice end must be +1.
            pa, ma = pos_sorted[sa : ea + 1], mass_sorted[sa : ea + 1]
            pb, mb = pos_sorted[sb : eb + 1], mass_sorted[sb : eb + 1]
            d = pa[:, None, :] - pb[None, :, :]
            r = np.sqrt(np.sum(d * d, axis=-1) + eps * eps)
            u_far_exact += float(np.sum((ma[:, None] * mb[None, :]) / r))
            dc = node_com[a] - node_com[b]
            rc = float(np.sqrt(np.sum(dc * dc) + eps * eps))
            u_far_mono += float(node_mass[a] * node_mass[b] / rc)
            covered_pairs += int(pa.shape[0] * pb.shape[0])

        rel_err = abs(u_far_mono - u_far_exact) / u_total if u_total else 0.0
        far_fraction = u_far_exact / u_total if u_total else 0.0
        records.append(
            {
                "theta": theta,
                "num_far_pairs": int(src.shape[0]),
                "covered_particle_pairs": covered_pairs,
                "u_far_exact": u_far_exact,
                "u_far_monopole": u_far_mono,
                "far_energy_fraction": far_fraction,
                "rel_error_vs_total": rel_err,
            }
        )
        print(
            f"theta={theta:4.2f}  far_pairs={int(src.shape[0]):5d}  "
            f"far_frac={far_fraction:5.2f}  rel_err={rel_err:.3e}"
        )

    payload = {
        "benchmark": "mac_accuracy",
        "params": {
            "n": args.n,
            "thetas": args.thetas,
            "backend": args.backend,
            "leaf_size": args.leaf_size,
            "softening": eps,
            "seed": args.seed,
            "approximation": "monopole (center-of-mass), order 0",
        },
        "u_total_exact": u_total,
        "metadata": run_metadata(),
        "records": records,
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
