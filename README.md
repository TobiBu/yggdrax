# Yggdrax

![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)
![pydoclint](https://img.shields.io/badge/docstrings-pydoclint-2ea44f.svg)

![Yggdrax Logo](./yggdrax.png)

Yggdrax is a JAX-first tree toolkit for hierarchical N-body solvers. It
provides Morton ordering, radix tree builders, per-node geometry, and dual-tree
interaction traversal primitives designed for downstream FMM and treecode
pipelines. The public `octree` backend now layers explicit octree-cell metadata
on top of the existing Morton/radix construction path so downstream FMM code
can consume both the proven traversal buffers and octree-style child tables.

## Features

- Morton encode/decode and stable Morton sorting for 3D points
- LBVH and fixed-depth radix tree construction
- Explicit octree metadata derived from Morton/radix topology
- Tree geometry extraction (bounds, centers, extents, radii)
- Dual-tree far-field and near-field interaction builders
- Dense and grouped interaction buffer transforms for batched kernels
- Prepared artifact utilities for downstream solver integrations

## Installation

Install from source:

```bash
pip install -e .
```

Install with development tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp

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
traversal_cfg = DualTreeTraversalConfig(
    max_pair_queue=8192,
    process_block=256,
    max_interactions_per_node=2048,
    max_neighbors_per_leaf=2048,
)
interactions, neighbors = build_interactions_and_neighbors(
    tree,
    geom,
    theta=0.6,
    mac_type="dehnen",
    traversal_config=traversal_cfg,
)
```

`build_tree(...)` continues to expose the radix/LBVH backend directly. The
octree wrappers (`build_octree(...)`, `build_fixed_depth_octree(...)`) preserve
the same compatibility fields while additionally exposing explicit octree
buffers such as `oct_children`, `oct_node_depths`, and `radix_node_to_oct`.

Advanced users can override the built-in MAC with a JAX-traceable pair policy:

```python
def pair_policy(policy_state, **pair_data):
    action = ...
    tag = ...
    return action, tag

interactions, neighbors, result = build_interactions_and_neighbors(
    tree,
    geom,
    pair_policy=pair_policy,
    policy_state=...,
    return_result=True,
)
```

The policy receives generic pair geometry/state and returns:
- `action`: one of accept-far / accept-near / refine
- `tag`: integer metadata stored for accepted far pairs

When `return_result=True`, raw far-pair tags are available on
`result.interaction_tags`. This is intended for downstream solvers that need
solver-side scheduling or adaptive-order bucketing without moving solver logic
into `yggdrax`.

See `examples/getting_started.ipynb` for a runnable walkthrough.
For the locked high-performance GPU benchmark configuration, see
`docs/gpu_benchmark_recommended_setup.md` and
`examples/tree_gpu_performance_scaling.ipynb`.

## KD-Tree MAC Note

When comparing Radix vs Octree vs KD-tree traversal outputs, use the same MAC settings as
your downstream solver.

- For FMM-style runs (e.g. jaccpot), `mac_type="dehnen"` is the recommended
  path for apples-to-apples parity checks.
- Octree builds currently share the radix traversal core, so interaction-count
  parity between `radix` and `octree` should hold for the same build settings.
- KD-tree traversal uses a calibrated default effective radius scale for
  Dehnen MAC (`dehnen_radius_scale=1.2`) to match near-field/far-field split
  behavior more closely with radix trees.
- If you benchmark with `mac_type="bh"`, expect different KD/Radix split
  behavior unless you tune parameters explicitly.

## Backend Extensibility

Yggdrax now supports backend-oriented tree dispatch and capability-based
topology contracts:

- Register builders via `register_tree_builder(...)`
- Inspect available builders via `available_tree_types()`
- Use `resolve_tree_topology(...)` for container/topology adapters
- Use derivation helpers (`get_node_levels`, `get_level_offsets`,
  `get_nodes_by_level`) when a backend does not precompute level metadata
- Octree consumers can additionally use explicit buffers like `oct_children`
  and `oct_level_offsets` when level-wise FMM scheduling is preferable to
  binary traversal over `left_child` / `right_child`

Contract details and required/optional fields are documented in
`docs/backend_contract.md`.

## Build And Traversal Configs

Public config dataclasses provide a stable way to reuse tuned settings across
repeated builds and traversals:

- `TreeBuildConfig`: adaptive radix-tree settings (`leaf_size`,
  `return_reordered`, reusable workspace handling)
- `FixedDepthTreeBuildConfig`: fixed-depth tree settings, including local
  Morton refinement controls
- `DualTreeTraversalConfig`: traversal queue, block size, interaction capacity,
  and neighbor capacity

When a `config=...` object is passed to `build_tree(...)` or
`build_fixed_depth_tree(...)`, or their octree counterparts, it takes precedence over the equivalent
individual keyword arguments.

Conformance tests:

```bash
pytest -q --no-cov tests/unit/test_backend_conformance.py
```

## Development

Local quality gates:

```bash
pytest
black --check .
isort --check-only .
pydoclint .
```

Or run the same checks via pre-commit:

```bash
pre-commit run --all-files
```

Coverage is enforced via `pytest-cov`:

```bash
pytest --cov=yggdrax --cov-report=term-missing
```

## Project Structure

- `yggdrax/tree.py`, `yggdrax/_tree_impl.py`: tree building and radix internals
- `yggdrax/octree.py`: explicit octree metadata derived from Morton/radix trees
- `yggdrax/protocols.py`: backend capability protocols
- `yggdrax/geometry.py`, `yggdrax/_geometry_impl.py`: geometry wrappers and implementations
- `yggdrax/interactions.py`, `yggdrax/_interactions_impl.py`: traversal and interaction generation
- `yggdrax/dense_interactions.py`, `yggdrax/grouped_interactions.py`: interaction layout utilities
- `tests/unit`: unit test suite for API and implementation behavior
- `examples`: runnable examples and notebooks

## CI

GitHub Actions runs:

- unit tests with coverage threshold
- `black --check`
- `isort --check-only`
- `pydoclint`

Workflow file: `.github/workflows/ci.yml`.

## Relationship to Rubix

This repository follows the same engineering principles used in the Rubix codebase:

- strict formatting and lint automation
- tested public APIs
- explicit artifact contracts
- examples that reflect real usage paths
