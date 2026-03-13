# Backend Contract

This document defines the structural contract for tree backends that want to
interoperate with `yggdrax` geometry and traversal APIs.

## Capability Layers

Backends are expected to satisfy capabilities in layers, not as one monolithic
type.

### `TreeStructureProtocol` (core)

Required for all tree traversal logic:

- `parent`
- `left_child`
- `right_child`

### `TreeRangesProtocol` (geometry ranges)

Required for geometry from particle ranges:

- `node_ranges`
- `num_particles`

### `MortonLeafBoundsProtocol` (optional geometry mode)

Required only when `use_morton_geometry=True`:

- `use_morton_geometry`
- `bounds_min`
- `bounds_max`
- `leaf_codes`
- `leaf_depths`

### `TreeLevelIndexProtocol` (optional, derivable)

Used by level-major and interaction grouping paths. If absent, yggdrax derives
these from `parent`:

- `node_level`
- `nodes_by_level`
- `level_offsets`
- `num_levels`

### Explicit octree metadata (optional, octree-oriented consumers)

Backends that want to support level-wise FMM kernels or octant-major scheduling
may additionally expose:

- `oct_parent`
- `oct_children`
- `oct_child_counts`
- `oct_child_mask`
- `oct_valid_mask`
- `oct_node_codes`
- `oct_node_depths`
- `oct_node_ranges`
- `oct_nodes_by_level`
- `oct_level_offsets`
- `oct_num_levels`
- `oct_leaf_mask`
- `oct_leaf_nodes`
- `radix_node_to_oct`
- `radix_leaf_to_oct`

These fields are not required by current yggdrax geometry/traversal APIs, but
they are useful for downstream FMM implementations that prefer explicit octree
cells over the binary radix traversal structure.

## Topology Containers

Public wrappers accept either:

1. A topology object directly, or
2. A container exposing `.topology`.

Use `resolve_tree_topology(...)` if writing custom adapters.

## Derivation Helpers

When level metadata is not precomputed, use:

- `get_node_levels(tree)`
- `get_num_levels(tree)`
- `get_level_offsets(tree)`
- `get_nodes_by_level(tree)`
- `get_num_internal_nodes(tree)`

Precomputing and storing these fields is still recommended for performance.

For octree-aware consumers, precomputing `oct_children` and the octree
level-index buffers is likewise recommended when repeated FMM passes will reuse
the same tree.

## Minimum Practical Backend

A backend that wants geometry + interactions should provide at least:

- `parent`, `left_child`, `right_child`
- `node_ranges`, `num_particles`
- either:
  - Morton leaf fields (`use_morton_geometry` + leaf bounds metadata), or
  - set `use_morton_geometry=False` and provide valid particle range topology.

Level metadata can be omitted initially and added later as an optimization.

## Validation

Run conformance checks:

```bash
pytest -q --no-cov tests/unit/test_backend_conformance.py
```

The suite is adapter-driven and currently runs against radix as the reference
backend.
