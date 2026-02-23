# Yggdrax

![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)
![pydoclint](https://img.shields.io/badge/docstrings-pydoclint-2ea44f.svg)

![Yggdrax Logo](./yggdrax.png)

Yggdrax is a JAX-first tree toolkit for hierarchical N-body solvers. It
provides Morton ordering, radix tree builders, per-node geometry, and dual-tree
interaction traversal primitives designed for downstream FMM and treecode
pipelines.

## Features

- Morton encode/decode and stable Morton sorting for 3D points
- LBVH and fixed-depth radix tree construction
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
    build_tree,
    compute_tree_geometry,
)

key = jax.random.PRNGKey(0)
key_pos, key_mass = jax.random.split(key)
positions = jax.random.uniform(key_pos, (512, 3), minval=-1.0, maxval=1.0)
masses = jax.random.uniform(key_mass, (512,), minval=0.5, maxval=1.5)

tree = build_tree(positions, masses, leaf_size=16)
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

See `examples/getting_started.ipynb` for a runnable walkthrough.

## Backend Extensibility

Yggdrax now supports backend-oriented tree dispatch and capability-based
topology contracts:

- Register builders via `register_tree_builder(...)`
- Inspect available builders via `available_tree_types()`
- Use `resolve_tree_topology(...)` for container/topology adapters
- Use derivation helpers (`get_node_levels`, `get_level_offsets`,
  `get_nodes_by_level`) when a backend does not precompute level metadata

Contract details and required/optional fields are documented in
`docs/backend_contract.md`.

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
