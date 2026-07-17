"""Gradients across tree rebuilds: nearest-neighbour spacing under descent.

Existence-proof experiment for the differentiability model (paper Sect. 2): the
tree is rebuilt *inside* the differentiated objective at every optimizer step,
yet reverse-mode gradients drive the system to a target nearest-neighbour
spacing across many rebuilds --- including steps on which the discrete tree
ordering changes.

The objective pushes each particle's nearest-neighbour distance toward a target
``r*``::

    L(x) = mean_i ( d_i(x) - r* )^2 ,

where the neighbour candidates of particle ``i`` are read from the *rebuilt*
tree's Morton ordering (the sorted neighbours at offsets +/-1..k). ``build_tree``
returns an integer ordering, so autodiff assigns it zero cotangent: the gradient
that flows is the gradient of the smooth geometry conditioned on the topology the
rebuild just produced. We step plain gradient descent, rebuild every step, and
record the loss/spacing trajectory together with the number of steps on which
the ordering (topology) actually changed.

Results:
  * summary + scalar trajectories -> ``results/differentiability/nn_rebuild.json``
  * position history (for the figure) -> ``results/differentiability/nn_rebuild.npz``

Production (GPU server):

    micromamba run -n odisseo python bench/differentiability/nn_rebuild.py \
        --num-particles 4096 --steps 200 --gpu-select free

Local smoke (CPU):

    conda run -n jaccpot python bench/differentiability/nn_rebuild.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.differentiability._common import (
    dump_json,
    dump_npz,
    run_metadata,
    select_free_gpu,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-particles", type=int, default=256)
    p.add_argument("--dim", type=int, default=3, help="Spatial dimension.")
    p.add_argument("--target-distance", type=float, default=0.12)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=2.0)
    p.add_argument(
        "--clip",
        type=float,
        default=1.5,
        help="Positions are clipped to [-clip, clip] each step.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-select", choices=("free", "least-used", "none"), default="free"
    )
    p.add_argument(
        "--output", type=str, default="results/differentiability/nn_rebuild.json"
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.num_particles = 256
        args.steps = 30
        args.gpu_select = "none"

    select_free_gpu(args.gpu_select, tag="nn_rebuild")

    import jax
    import jax.numpy as jnp

    from yggdrax import build_tree

    target = args.target_distance
    k = args.k_neighbors
    leaf_size = args.leaf_size

    key = jax.random.PRNGKey(args.seed)
    key_pos, key_mass = jax.random.split(key)
    positions0 = jax.random.uniform(
        key_pos,
        (args.num_particles, args.dim),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    # build_tree requires masses; the Morton ordering depends on positions only,
    # so unit masses suffice for a spacing objective.
    masses = jnp.ones((args.num_particles,), dtype=jnp.float32)

    def morton_nn_stats(positions_unsorted):
        """Return (loss, nn_dist) using the rebuilt tree's Morton neighbours."""
        tree = build_tree(positions_unsorted, masses, leaf_size=leaf_size)
        pos_sorted = positions_unsorted[tree.particle_indices]
        n = pos_sorted.shape[0]
        idx = jnp.arange(n)
        big = jnp.asarray(1e6, dtype=pos_sorted.dtype)

        def safe_norm(delta):
            return jnp.sqrt(jnp.sum(delta * delta, axis=1) + 1e-12)

        candidate_dists = []
        for off in range(1, k + 1):
            idx_plus = jnp.clip(idx + off, 0, n - 1)
            idx_minus = jnp.clip(idx - off, 0, n - 1)
            dist_plus = safe_norm(pos_sorted - pos_sorted[idx_plus])
            dist_minus = safe_norm(pos_sorted - pos_sorted[idx_minus])
            candidate_dists.append(jnp.where(idx + off < n, dist_plus, big))
            candidate_dists.append(jnp.where(idx - off >= 0, dist_minus, big))

        nn_dist = jnp.min(jnp.stack(candidate_dists, axis=1), axis=1)
        loss = jnp.mean((nn_dist - target) ** 2)
        return loss, nn_dist

    def loss_only(positions_unsorted):
        return morton_nn_stats(positions_unsorted)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only))
    stats = jax.jit(morton_nn_stats)

    def ordering(positions_unsorted):
        return build_tree(
            positions_unsorted, masses, leaf_size=leaf_size
        ).particle_indices

    ordering_jit = jax.jit(ordering)

    positions = positions0
    positions_history = [positions0]
    loss_history: list[float] = []
    mean_nn_history: list[float] = []
    topology_changed: list[bool] = []  # per-step: did the ordering change?

    prev_order = ordering_jit(positions)
    _, nn0 = stats(positions)
    initial_mean_nn = float(jnp.mean(nn0))

    for step in range(args.steps):
        loss_val, grad_val = loss_and_grad(positions)
        positions = jnp.clip(
            positions - args.learning_rate * grad_val, -args.clip, args.clip
        )
        positions_history.append(positions)

        _, nn_dist = stats(positions)
        loss_history.append(float(loss_val))
        mean_nn_history.append(float(jnp.mean(nn_dist)))

        order = ordering_jit(positions)
        changed = not bool(jnp.array_equal(order, prev_order))
        topology_changed.append(changed)
        prev_order = order

        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            print(
                f"step={step:3d} | loss={loss_history[-1]:.6f} | "
                f"mean_NN={mean_nn_history[-1]:.6f} | "
                f"topo_changed={'yes' if changed else 'no'}"
            )

    final_loss, final_nn = stats(positions)
    topology_changes = int(sum(topology_changed))

    print(
        f"\ntarget={target}  "
        f"mean_NN {initial_mean_nn:.4f} -> {float(jnp.mean(final_nn)):.4f}  "
        f"loss {loss_history[0]:.3e} -> {float(final_loss):.3e}  "
        f"topology_changes={topology_changes}/{args.steps}"
    )

    payload = {
        "benchmark": "nn_rebuild",
        "params": {
            "num_particles": args.num_particles,
            "dim": args.dim,
            "target_distance": target,
            "k_neighbors": k,
            "leaf_size": leaf_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "clip": args.clip,
            "seed": args.seed,
        },
        "metadata": run_metadata(),
        "summary": {
            "initial_mean_nn": initial_mean_nn,
            "final_mean_nn": float(jnp.mean(final_nn)),
            "initial_loss": loss_history[0],
            "final_loss": float(final_loss),
            "topology_changes": topology_changes,
        },
        "trajectory": {
            "step": list(range(args.steps)),
            "loss": loss_history,
            "mean_nn": mean_nn_history,
            "topology_changed": [bool(c) for c in topology_changed],
        },
    }
    dump_json(payload, args.output)

    npz_path = Path(args.output).with_suffix(".npz")
    dump_npz(
        {
            "positions_initial": positions_history[0],
            "positions_final": positions_history[-1],
            "positions_history": jnp.stack(positions_history),
        },
        npz_path,
    )


if __name__ == "__main__":
    main()
