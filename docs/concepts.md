# Concepts

This page explains the ideas and vocabulary used throughout Yggdrax.

## Trees and backends

Yggdrax builds a spatial tree over particles and then runs solver-agnostic
traversals over it. Three backends share one build/dispatch path
({func}`yggdrax.build_tree`, {func}`yggdrax.build_octree`,
{class}`yggdrax.KDTree`):

- **radix / LBVH** (`tree_type="radix"`) — a binary radix tree over
  Morton-sorted particles; the proven default.
- **octree** (`tree_type="octree"`) — explicit octree-cell metadata
  (`oct_children`, `oct_level_offsets`, …) layered on the radix topology, for
  level-wise FMM scheduling.
- **kd-tree** (`tree_type="kdtree"`) — a median-split KD-tree; useful for
  neighbor queries and comparison studies.

All backends expose the **FMM-core topology contract**
(`parent`, `left_child`, `right_child`, `node_ranges`, `num_particles`,
`use_morton_geometry`) so that geometry, moments, and traversal work against any
of them. Contract details are in [](backend_contract.md).

## Morton ordering

Particles are mapped to 21-bit-per-axis Z-order (Morton) codes on `uint64`
(x64 is enabled at import for this reason). Sorting by Morton code makes spatial
neighbors contiguous in memory, which is what lets the radix builder and the
prefix-sum moment queries be efficient.

## Multipole Acceptance Criterion (MAC)

The MAC decides, for a target/source node pair, whether the source is
"well-separated" enough to accept as a far-field (M2L) interaction or must be
refined. Yggdrax ships three variants (`mac_type=`):

- **`"bh"`** — the classic Barnes-Hut opening-angle test using the box
  half-extent (L-infinity radius).
- **`"dehnen"`** — Dehnen (2014) criterion using the bounding-sphere radius;
  recommended for FMM-style parity (e.g. jaccpot). `dehnen_radius_scale`
  tunes the effective radius.
- **`"engblom"`** — a jaxFMM-style sphere criterion.

`theta` is the opening-angle parameter: smaller `theta` accepts fewer far pairs
(more accurate, more work). Advanced users can bypass the built-in MAC entirely
with a JAX-traceable `pair_policy` (see {func}`yggdrax.build_interactions_and_neighbors`).

## Fixed-capacity buffers and auto-growth

Everything downstream of the build is written to be `jit`- and
`shard_map`-traceable, which means statically-shaped buffers. Yggdrax therefore
uses **fixed-capacity buffers + a dynamic valid count + overflow flags**. The
public builders start from a small capacity and **retry with a larger one on
overflow** (up to an internal cap), so capacity arguments such as
`max_interactions_per_node`, `max_neighbors_per_leaf`, and `max_pair_queue` are
hints unless pinned via a {class}`yggdrax.DualTreeTraversalConfig`.

## Buffer glossary

```{list-table}
:header-rows: 1

* - Field
  - Meaning
* - `particle_indices`
  - Permutation mapping Morton-sorted order back to the input order.
* - `node_ranges`
  - Per-node inclusive ``[start, end]`` index range into the sorted particles.
* - `parent` / `left_child` / `right_child`
  - Binary topology pointers (``-1`` for missing).
* - `node_level` / `level_offsets` / `nodes_by_level`
  - Level-order metadata (derivable when a backend omits it).
* - `use_morton_geometry`
  - Whether per-node boxes come from Morton leaf codes or particle ranges.
* - `oct_children` / `oct_node_depths` / `radix_node_to_oct`
  - Explicit octree buffers linking radix nodes to octree cells.
```

## Interaction outputs

A single dual-tree walk yields two products:

- **far-field (M2L)** — {class}`yggdrax.NodeInteractionList` (sparse) or
  exact-length compact far pairs; densified for batched kernels by
  `dense_interactions` / `grouped_interactions`.
- **near-field (P2P)** — {class}`yggdrax.NodeNeighborList` of leaf-leaf
  neighbor pairs.
