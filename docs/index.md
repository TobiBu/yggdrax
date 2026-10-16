# Yggdrax

Yggdrax is a JAX-first tree toolkit for hierarchical *N*-body solvers. It
provides Morton ordering, radix / octree / KD-tree builders, per-node geometry,
and dual-tree far-field / near-field interaction traversal primitives designed
for downstream FMM and treecode pipelines.

The pipeline is:

```{eval-rst}
.. code-block:: text

    particles (positions, masses)
        -> Morton encode + sort            (morton)
        -> tree build                      (tree: radix | octree | kdtree)
        -> per-node geometry               (geometry)
        -> dual-tree walk                  (interactions: far / M2L + near / P2P)
        -> interaction / neighbor lists    (NodeInteractionList, CompactTaggedFarPairs, NodeNeighborList)
```

## Installation

```bash
pip install -e ".[dev]"      # library + quality tooling
pip install -e ".[docs]"     # to build this documentation
```

## Requirement: `jax_enable_x64`

yggdrax needs JAX's 64-bit mode: Morton codes are 63-bit `uint64`, and some pair
counts, global ids and centre-of-mass prefix sums are 64-bit. Enable it at the top of
your program, before creating arrays, with `jax.config.update("jax_enable_x64", True)`
or `JAX_ENABLE_X64=1`. yggdrax does not switch it on itself, and the functions that
need it raise a `RuntimeError` when it is off. x64 does not make the computation
float64: the dtype of your inputs sets the working precision.

## Quick start

```python
import jax

jax.config.update("jax_enable_x64", True)

from yggdrax import (
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
    build_octree,
    compute_tree_geometry,
)

key = jax.random.PRNGKey(0)
key_pos, key_mass = jax.random.split(key)
positions = jax.random.uniform(key_pos, (512, 3), minval=-1.0, maxval=1.0)
masses = jax.random.uniform(key_mass, (512,), minval=0.5, maxval=1.5)

tree = build_octree(positions, masses, leaf_size=16)
positions_sorted = positions[tree.particle_indices]
geom = compute_tree_geometry(tree, positions_sorted)

interactions, neighbors = build_interactions_and_neighbors(
    tree,
    geom,
    theta=0.6,
    mac_type="dehnen",
    traversal_config=DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=256,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    ),
)
```

See `examples/getting_started.ipynb` for a runnable walkthrough.

```{toctree}
:maxdepth: 2
:caption: Documentation

concepts
backend_contract
gpu_benchmark_recommended_setup
api
```
