# Periodic boundaries

Cosmological boxes are periodic. yggdrax supports a periodic cube or cuboid
`[0, L)^d` with the **minimum-image convention**: the distance between two
particles is the length of the shortest of their periodic displacements. Every
periodic entry point takes one optional argument, `box_size`:

- `None` (the default everywhere) is the non-periodic code. This is a trace-time
  branch, so every non-periodic path is the same computation as before the
  option existed.
- a float `L` is a cube.
- a tuple `(L_x, L_y, L_z)` gives one length per dimension.

The walks take `box_size` as a *static* argument. It fixes the acceptance rule at
trace time, so a new box compiles a new kernel. Pass a Python float or a tuple,
not a traced array.

## Building a periodic tree

```python
tree = Tree.from_particles(positions, masses, leaf_size=16, box_size=L)
geometry = compute_tree_geometry(tree, tree.positions_sorted)
```

With `box_size` the positions are wrapped into `[0, L)` (`wrap_positions`) and
the tree is built on the fixed bounds `([0]*d, L)` instead of inferring them from
the particles (`infer_bounds`). `tree.positions_sorted` then holds the wrapped
coordinates. Passing both `bounds` and `box_size` is an error. `build_tree(...)`
and `build_prepared_tree_artifacts(...)` take the same argument, and so does the
capacity-padded `build_static_cells_tree(positions, masses, box_size=L, ...)`,
which traces under `jit` (its Morton domain's upper corner sits `2**-20 L` above
`L`, so that no particle quantises to the code reserved for padding leaves). Every node is a
region inside the box, and no node straddles a face. The walk's radii depend on
that: a node's bounding radius stays small and honest.

## The walks

`build_interactions_and_neighbors` and every other dual-walk builder accept
`box_size`: `build_well_separated_interactions`, `build_leaf_neighbor_lists`,
`build_compact_far_pairs`, `build_compact_far_pairs_and_leaf_neighbor_lists`,
`build_interactions_and_neighbors_split` and `build_prepared_tree_artifacts`. So
does `dual_tree_walk_mutual`. Each route normalises the box in one place, so the
joint walk and the bounded count-pass routes apply the same rule. A test pins
that the routes give identical far and near sets.

With a box, the walk changes in two ways:

1. **Minimum-image node distance.** The node-centre separation `d` used by the
   MAC, by `separation_floor`, by a `pair_policy` (its `dist_sq`) and by a
   mutual-walk `pair_accept` is the minimum-image distance.
2. **Far-pair guard.** A node pair is accepted as far only if, in addition,

   ```
   r_a + r_b + d < min(L) / 2        (strict)
   ```

   Here `r_a` and `r_b` bound every particle's distance from its node centre.
   Every particle pair of the two nodes then has a displacement, taken through
   the centres' image, of length at most `d + r_a + r_b < L/2`. Each component is
   therefore below `L_k/2`, so that displacement is the particle pair's own
   minimum image, and no other image is closer. A far pair stands for one
   interaction through one image, and evaluating it there is exact. When the
   guard fails the pair refines, or becomes a near pair if both nodes are
   leaves. The guard is
   `yggdrax.periodic.far_pair_is_unambiguous`.

   - The dual walk takes these radii from `geometry.radius`, the half-diagonal of
     each node's bounding box. That radius is conservative whatever MAC proxy is
     in use: `bh` uses box half-extents, which are *not* a bound on particle
     distance, and the guard does not rely on them.
   - `dual_tree_walk_mutual` uses `periodic_radii` if given, else the `radii`
     the caller passes. They must bound particle distances, as the mutual FMM's
     exact centre-of-mass radii do. Passing the `bh` MAC proxy as the guard's
     radii voids the guarantee.
   - `mutual_walk_inputs(tree, geometry, mac_type=..., box_size=L)` returns the
     normalised box and `periodic_radii = geometry.radius`; pass both on
     (`box_size=w.box_size, periodic_radii=w.periodic_radii`) and the mutual
     walk gives the periodic dual walk's far and near sets for every MAC.
   - The Pallas backend of `dual_tree_walk_mutual` has no periodic guard:
     `backend="pallas"` with a box raises, and `backend="auto"` walks flat.
   - A `pair_policy` (dual walk) or `pair_accept` (mutual walk) cannot bypass
     the guard: a policy's far acceptance is re-checked against it and demoted
     to near or refine.

**Coverage.** This is unchanged. Every unordered particle pair lies in exactly
one of: a leaf's own block, one near leaf pair, or one accepted far node pair.
The tests check this particle by particle.

**Near field.** Near pairs (and a leaf's own block) are evaluated particle by
particle by the application, with the minimum-image helpers:

| helper | returns |
|---|---|
| `minimum_image(delta, box_size)` | displacements mapped to `[-L/2, L/2]` per component |
| `periodic_displacement(x_from, x_to, box_size)` | minimum-image `x_to - x_from` |
| `periodic_distance_sq(x_a, x_b, box_size)` | squared minimum-image distance |
| `pairwise_periodic_distance_sq(x_a, x_b, box_size)` | `(n, m)` block of the above |
| `wrap_positions(positions, box_size)` | coordinates in `[0, L)` |

All of them return their input unchanged for `box_size=None` and are
differentiable in the positions.

**Far field.** For an accepted pair `(A, B)` the image to use is
`minimum_image(c_B - c_A, box_size)`. By the guard, it is the same image for every
particle pair inside.

For a two-point count up to `r_max` (2PCF, kNN-CDF), set
`separation_floor=r_max`. A far pair then holds no particle pair closer than
`r_max`, so the count needs only the near list and the leaves' own blocks. For
the dual walk this needs sphere MAC extents (`dehnen` or `engblom`); with `bh`
the floor is measured on box half-extents.

## kNN and radius counts

`kdtree.query_neighbors(tree, queries, k=..., box_size=L)` returns
minimum-image neighbours on all three backends (`dense`, `tiled`, `tree`), and
`kdtree.count_neighbors(tree, queries, radius=..., box_size=L)` counts by
minimum-image distance on both of its backends (`tiled`, `tree`). Queries
are wrapped into the box. The tree's points must already lie in `[0, L)`: build
the k-d tree from `wrap_positions(points, L)`. This is checked when the points are
concrete. The `tree` backend prunes a far subtree using the smaller of the
split-plane distance and the distance through the box face behind the query, which
is a valid lower bound for points in the box.

## Limits

- **Minimum image only.** Each particle pair is represented once, at its
  minimum image. Separations beyond `min(L)/2` are ambiguous by definition, and
  a statistic out to such separations (or an Ewald sum over all images) needs
  more than this. Keep `r_max < min(L)/2`. The same holds for kNN: the `k`
  neighbours are well defined only while they are closer than `min(L)/2`.
- **Coarse trees near the box scale.** Nodes whose `r_a + r_b + d` reaches `L/2`
  are never far. A tree with only a few large leaves therefore turns most pairs
  into near pairs. This is correct but slow, and it disappears at realistic `N`.
- **Distributed walks** (`yggdrax.distributed`) do not take `box_size` yet.
  Periodic wraps across domains are a separate item (C1).
- **Pallas mutual walk**: not periodic (see above).
- `rebuild_static_radix_tree_from_template` takes no `box_size`; wrap the
  positions yourself and pass the bounds `build_static_cells_tree(box_size=L)`
  uses, `(0, L * (1 + 2**-20))`.
