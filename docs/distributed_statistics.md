# Distributed clustering statistics (C1, C2)

`yggdrax.distributed.stats` computes two clustering statistics on particles
sharded across devices in a periodic box `[0, L)^3`:

- **soft binned pair counts** for the two-point correlation function ξ, and
- **k-nearest-neighbour distances** for kNN-CDFs.

On 1 device both reduce to a single-device computation. On any number of
devices they give the same answer, to floating-point summation order, whenever
no far node pair is accepted (the default `exact_acceptance()`). The tests check
this at 1, 2, 4 and 8 fake CPU devices against brute force.

Both drivers are differentiable with respect to the particle positions (and the
weights of ξ, and the query positions of kNN) at fixed topology. The gradient
equals the single-device gradient and brute-force O(N²) autodiff to round-off
(≤ 6e-15 measured) on every device count. *Differentiability* lists what the
forward pass was built to allow; *Gradients (C2)* is how the reverse pass works
and how it was checked.

## Data flow (per device, inside one `shard_map`)

```
input shard (any rows) ──wrap──► sfc_partition ──► local shard (cap rows, Morton domain)
                                                        │
                    local cell tree (build_static_cells_tree, box_size=L, num_valid=count)
                                                        │
              summary cells (occupancy_cut): centre, radius + REACH ──all_gather──► every device
                                                        │
  export walk: my tree vs every other device's cells (min-image; prune disjoint spheres;
                never accept far) ──► (cell, my leaf) near list
                                                        │
  build_send_buffers (one copy of each leaf per destination) ─► expand leaves to particles
                                                        │
                       ragged_all_to_all ──► HALO (remote particles within reach)
                                                        │
  combined set [local live ; halo live] ─► cell tree ─► dual_tree_walk_mutual
                (min-image, periodic far guard, pluggable acceptance, ownership pruning)
                                                        │
          ξ: within-leaf + near leaf pairs + far monopoles, weighted by ownership ─► psum
          kNN: near-list edges streamed, segmented top-k merge ─► per-particle
```

Step by step:

1. **Partition.** Positions are wrapped into the box and sample-sorted into
   contiguous Morton domains (`sfc_partition`) on the fixed periodic Morton box
   `(0, L(1 + 2^-20))`. That is the box `build_static_cells_tree(box_size=L)` uses,
   so all devices agree on every code. A global id (the input row) and a
   per-particle flag word travel with each particle.
2. **Local tree and summary.** Each device builds a capacity-padded cell tree
   over its live rows and publishes an occupancy cut of it (`occupancy_cut`, cells
   of at most `summary_max_leaves` leaves). A cell is published as a ball: its
   bounding centre, and its bounding radius **plus the cell's reach**. The reach
   is how far from the cell a remote particle can matter. For ξ it is `r_hi`, the
   upper cut of the binned range. For kNN it is a per-cell bound (below). The
   balls of all devices are `all_gather`ed: O(cells) per device, not O(N).
3. **Export walk (sender push).** Each device walks its own full tree against
   every other device's balls (`export_walk`, one `dual_tree_walk_mutual` over
   `[all cells ; my nodes]`). The walk never accepts a far pair. It prunes a
   (cell, node) pair when the two spheres are disjoint under the minimum image,
   `d > R_cell + reach + r_node`. Every surviving (cell, leaf) pair is a leaf the
   receiver needs. A device never exports to itself.
4. **Send buffers and halo exchange.** `build_send_buffers` deduplicates the
   leaves per destination, so a leaf named by several cells of one receiver is
   shipped once. The leaves are then expanded to their particles, and one ragged
   all-to-all ships position, weight, flags and global id. Each remote particle
   lands in a receiver's halo at most once. This matters: counting is exact only
   because no particle is duplicated.
5. **Combined walk.** Each device builds one cell tree over `[local ; halo]`
   (live rows compacted to the front) and runs the mutual walk with
   `box_size=L`. Each particle has two flags: **query** (`o_i`: owned by this
   device, or for kNN a query point owned by it) and **data**. A node pair is
   pruned when it cannot contain a (query, data) pair. The walk also prunes it
   when its separation range `[d - s, d + s]` (with `s = r_a + r_b`, the bounding
   radii) lies outside `[r_lo, r_hi)`. On top of that runs the payload's
   pluggable acceptance (below).
6. **Reduce.** ξ: per-bin partial sums are `psum`med. kNN: distances stay
   sharded with their global ids, and a CDF reduction on a radius grid is
   `psum`med.

## Payload 1: soft binned pair counts

The statistic is

```
C_b = Σ_{i<j} w_i w_j W_b(r_ij) · 1[r_lo ≤ r_ij < r_hi]
```

Here `W_b` is the soft window of `corrfunc.binning.soft_bin_weights` (log or
linear edges, `sharpness`), and `r_ij` is the minimum-image distance. The hard
cut is where the windows are already below `eps`: `soft_window_cutoffs` puts
`r_hi` at `ln(1/eps)/s` above the last edge in window coordinates, and `r_lo` the
same distance below the first edge. Every pair outside the cut has `W_b ≤ eps`.
So the cut changes the soft count by at most `eps` per pair, and it makes the
statistic a well-defined function of the positions that does not depend on any
tree. That is what "matches the single-device result" means, to round-off.

**Ownership weights.** A pair is seen by every device that holds both particles.
A pair of two owned particles is seen once. A cross-domain pair is seen twice,
on each owner's side, with the other particle in the halo. A halo-halo pair is
never needed. Each emitted pair carries `c_ij = (o_i + o_j) / 2`, which is
1 / ½ / 0 in those three cases, so the `psum` counts every pair exactly once.
This weight is linear in the flags, so a far node pair keeps a closed-form
monopole:

```
Σ_{i∈A, j∈B} c_ij w_i w_j  =  ½ (W°_A W_B + W_A W°_B),     W°_A = Σ_{i∈A} o_i w_i
```

The far pair is evaluated at the minimum-image distance of the two nodes'
centroids. The periodic guard `r_a + r_b + d < L/2` makes that image the same for
every particle pair inside (`docs/periodic_boundaries.md`).

The alternative is to count a cross pair once (on the lower device, say). That
halves the cross-pair arithmetic but needs the same halo. The gradient would
still be right, since C2 differentiates through the halo import (below), but the
½ weighting keeps both owners' work identical and the far monopole closed-form.

**Pluggable acceptance.** `PairAcceptance(accept, discard, data)` uses the
signature of `dual_tree_walk_mutual`'s `pair_accept`:
`f(data, a, b, dist_sq, radius_a, radius_b) -> bool mask`. `accept` marks far
(monopole) pairs. `discard` prunes pairs that contribute nothing. The driver ANDs
`accept` with nothing else and ORs `discard` with its own range and ownership
prunes. Provided rules:

| rule | far pairs | error |
|---|---|---|
| `exact_acceptance()` (default) | none: every pair in range is a near pair | round-off; identical on any device count |
| `geometric_acceptance(theta)` | `(r_a + r_b) < θ d` on bounding radii | monopole error, depends on the tree and so on the partition |

A2/A4a's bin-aware window rule (`claude/a2-traced-rebuild-b7bacc`,
`corrfunc/traced.py::_make_window_predicates`) is already an `(accept, discard)`
pair with this signature and `data = (edges_u, sharpness, eps)`. Once it is on
main, plugging it in is `PairAcceptance(*_WINDOW_PREDICATES[log], data)`.

**The core hook this needs.** `dual_tree_walk_mutual(pair_discard=...)` is added
on this branch with A4a's name, signature and semantics. It is applied after the
(minimum-image) distance is computed, a pruned pair is neither emitted nor
refined, and the Pallas backend refuses it. The A2 branch adds the same
parameter, so merging the two is a textual conflict of a few lines with no
semantic difference.

## Payload 2: kNN distances

The k nearest data neighbours of each query point, by minimum-image distance,
self excluded when the query is a data point. Queries are either the data
themselves or a separate point set (volume-filling randoms, the usual choice for
kNN-CDFs), passed as `queries=`.

**Choice: two passes, capped at `r_max`.** The kNN-CDF is evaluated on a radius
grid, so neighbours beyond the grid's last radius only matter as "beyond". The
caller gives `r_max`. Within it the answer is exact. A particle with fewer than
`k` data neighbours within `r_max` reports `inf` for the missing ranks, which is
the correct `1[r_k ≤ r] = 0` for every `r ≤ r_max`.

- **Pass 1 (local, no communication):** kNN against the device's own data only,
  with the same machinery as pass 2. To keep it cheap, each node looks only
  `first_pass_reach_factor` (default 2) of its own bounding radii beyond its
  sphere. Whatever k-th distance that finds is an upper bound on the true one,
  because its candidates are a subset. A query that finds fewer than `k` gets
  `r_max`. So `b_i = min(found r_k, r_max)`. A first-pass overflow (truncated
  walk) only removes candidates, which can only raise a bound, never make it
  wrong. It is reported as `local_pass_overflow` and is not part of `overflow`.
- **Halo reach per summary cell:** `max_{i ∈ cell} b_i`. Every true neighbour of
  every query in the cell is within that reach of the cell's ball, so the halo
  holds it.
- **Pass 2:** the walk on `[local ; halo]` keeps a node pair only if one side's
  queries can reach the other side's data. The test is
  `gap = d - r_a - r_b < reach[node]`, where a node's reach is the max `b_i` of
  its queries (a bottom-up max over the tree). Every surviving leaf pair, in both
  orientations, plus each leaf with itself, is an edge (query leaf, source leaf).
  Edges are sorted by query leaf and streamed in chunks. Each edge's
  `(leaf, leaf)` distance block gives a per-query top-k. A segmented associative
  scan merges the edges of one query leaf within a chunk, and the segment's last
  edge merges into the running best. The work is proportional to the edges. No
  per-leaf candidate capacity exists, so none can overflow.
- **Values:** the search returns neighbour *indices* (topology). The distances
  are then recomputed from the live positions by a gather and the minimum image.

A plain radius-bounded halo (reach `r_max` everywhere) would also be exact. The
first pass makes both the import and the pass-2 walk adapt to density: in
clustered regions `r_k ≪ r_max` and the halo shell thins accordingly. In voids
it falls back to `r_max`. Measured at N = 8000 uniform, k = 4, `r_max` = 0.15 on
2 and 4 fake devices: the halo is 1.4–1.8× smaller and pass 2 has 4× fewer near
pairs. `adaptive_reach=False` skips pass 1.

A two-pass scheme without a cap would need an unbounded reach for queries in
voids. The cap turns that into a flag the caller can read
(`found < k`), not a capacity blow-up.

## Capacities and overflow

Everything is statically shaped. `DistributedStatsConfig` holds every capacity:

| capacity | what overflows | flag |
|---|---|---|
| `capacity` | partition: a device receives more than `capacity` particles | `partition_overflow` |
| `leaf_capacity`, `halo_leaf_capacity` | cell-tree leaves of the local / combined tree | `leaf_overflow` |
| `summary_capacity` | occupancy-cut cells | `summary_overflow` |
| `export_queue`, `export_near_cap` | export walk wavefront / (cell, leaf) list | `export_overflow` |
| `send_leaf_capacity`, `send_capacity` | deduplicated leaves / particles a device sends | `send_overflow` |
| `halo_capacity` | particles a device receives | `halo_overflow` |
| `max_pair_queue`, `far_cap`, `near_cap` | the combined walk | `walk_overflow` |

The result carries each flag `psum`-reduced over devices (true if any device
overflowed), plus an `overflow` OR of all of them. **They must be read.** A
truncated halo loses pairs silently, at the percent level, and nothing else in
the result shows it. A truncated exchange never reads or writes out of bounds
(`ragged_all_to_all_exchange`'s clamp), so an overflow is detectable and safe to
retry with larger capacities. `DistributedStatsConfig.for_problem` gives
generous defaults from the per-device particle count. The counters (`halo_count`,
`near_count`, ...) come back per device, so a caller can tighten them after a
probe run.

## Periodic treatment

- **Domains.** The Morton box is the periodic box. Domains are contiguous Morton
  ranges and may touch any face.
- **What crosses a face.** Nothing special: the export walk measures the
  minimum-image distance between a sender node's centre and a receiver cell's
  centre. A domain at `x ≈ 0` therefore imports from the domain at `x ≈ L`, and
  a particle near a corner from up to 7 images' worth of neighbouring domains.
  The pruning test `d_min-image > R_cell + reach + r_node` is valid for every
  image: any particle pair's distance through any image is at least the node
  centres' distance through that image minus both radii.
- **Halo coordinates.** Halo particles keep their wrapped coordinates in
  `[0, L)`. The combined tree is built on the box (`box_size=L`), and all
  distances (walk, near blocks, far centroids, kNN) are minimum-image.
- **Limit.** Reaches must stay below `min(L)/2` (`r_hi` for ξ, `r_max` for
  kNN). This is checked at trace time.
- **Non-periodic.** `box_size=None` with explicit `bounds` runs the same pipeline
  without wrapping.

## Differentiability (what C2 inherits)

- **Topology from frozen positions.** The Morton codes, sorts, tree builds,
  walks, cuts and send lists come from `stop_gradient` copies of the positions.
  They are integers.
- **Values from live positions.** Every value the statistics read is the live
  input routed through gathers by those integers and through the two ragged
  exchanges (partition and halo). Both exchanges are linear in the values. On
  CPU they run as `all_gather` + gather (`method="buf"`), which differentiates;
  the native `ragged_all_to_all` has a transpose rule (see *Gradients* for the
  version gate).
- **The hard cut and the minimum image** are piecewise smooth. Away from
  measure-zero ties their Jacobian is that of the plain distance. The hard cut
  sits where `W_b ≤ eps`, so the gradient it drops is at most `eps` per pair.
- **Safe distances.** Masked pairs (padding, self, out of range) have their
  `r²` replaced before the `sqrt`, so no NaN cotangent reaches a coincident pair.
- **The ½ weighting** makes the cross-pair term symmetric between the two
  owners. That left C2 two routes. One returns the halo cotangents to their
  owners. The other has each device differentiate only through its own
  particles and take the cross term at weight 1 instead of ½, which needs no
  reverse exchange. C2 takes the first; *Gradients* says why.

## Gradients (C2)

```python
loss = lambda x, w: f(distributed_pair_counts(x, edges, box_size=L, config=cfg,
                                              weights=w).counts)
g_x, g_w = jax.grad(loss, argnums=(0, 1))(positions, weights)

loss = lambda x, q: f(distributed_knn(x, queries=q, k=k, r_max=r_max, box_size=L,
                                      config=cfg, r_grid=r, sharpness=s).cdf)
g_data, g_queries = jax.grad(loss, argnums=(0, 1))(data, randoms)
```

Both public drivers are traceable: they take device arrays (or tracers), pad the
input shards with `jnp`, and the kNN driver puts its distances back into input
order with a device scatter, not on the host. They work under `jit`, `grad`,
`value_and_grad` and `lax.scan`. The per-device bodies (`pair_counts_shard`,
`knn_shard`) differentiate inside a caller's own `shard_map` too, with
`check_vma=False` and a replicated output, as in the tests.

**What the gradient is.** It is the gradient of the statistic *at fixed
topology*. Every call rebuilds the partition, trees, walks and halo from the
current positions, but none of them carries a cotangent. With the default
`exact_acceptance()` the forward does not depend on the topology, so this is the
gradient of the statistic itself, up to the measure-zero points named in
*Differentiability*:

- a pair crossing the hard cut changes the count by at most `eps`;
- a kNN neighbour set changes at ties, and at `r_max`;
- the minimum image switches at `L/2`, which lies beyond every reach.

With `geometric_acceptance` it is the gradient of the monopole approximation that
the forward evaluates. The centroids are computed from the live positions, so it
is consistent, but this was not tested.

Differentiable inputs:

- ξ: positions and weights.
- kNN: data and query positions. The distance to a neighbour beyond `r_max` is
  `inf` and has no gradient; mask it before it enters a loss, as the soft CDF
  does. The hard CDF is piecewise constant, so its gradient is zero.

`edges`, `sharpness` (ξ) and `eps` set the cut and must be concrete.

### The reverse of the halo: differentiate through the import

C1 left two routes open. C2 takes **(a)**: differentiate through the halo import
and let the transpose of each exchange return the halo cotangents to their owners.

- A cross pair `(i, j)` with `i` on device A and `j` on device B is evaluated on A
  as `½ f(x_i, x̃_j)`, where `x̃_j` is j's halo copy, and on B as `½ f(x̃_i, x_j)`.
- On A, `∂/∂x_i` gives `½ ∂_i f` directly. The cotangent `½ ∂_j f` lands on the
  halo row `x̃_j`.
- The halo row came from `x_j` through `pos[src]` (a gather on B), then the ragged
  exchange, then a gather on A. Its transpose runs backwards: a gather on A, the
  reverse exchange, then a scatter-add into `x_j`'s row on B.
- Device B does the same with the roles swapped. Each side receives two halves,
  so the sum is `∂_i f + ∂_j f`: exactly the single-device gradient, neither half
  of it nor double.
- The partition exchange is transposed the same way, so cotangents end up on the
  caller's input rows.

**Why not (b).** Route (b) has each device differentiate only through its own
particles and give the cross term weight 1. It needs no reverse exchange, and for
ξ it would be correct. It is wrong for kNN: the relation is not symmetric (`j` among
`i`'s k nearest does not put `i` among `j`'s), so the term `d_ij` exists on one
device only, and `x_j`'s share of its cotangent must travel back to `j`'s owner.
One rule for both payloads, with no payload-specific weighting of values against
gradients, is worth one reverse exchange. That exchange is the same size as the
forward one: on GPU (`native`) a ragged all-to-all of the halo, and on CPU
(`buf`) a `psum_scatter` of the gathered send buffers.

`reverse_halo.export_reverse_halo` is not used. JAX's transposes of the gather and
the exchange *are* that reverse halo: a scatter-add of cotangents to their source
rows, built by autodiff. A hand-written `custom_vjp` would only duplicate it.

### The exchange, and the native ragged all-to-all

`DistributedStatsConfig.exchange` (`"auto"`, `"buf"`, `"native"`) selects both
exchanges. It is passed to `sfc_partition(exchange_method=...)` and to the halo.

`"auto"` resolves through `comm.resolve_ragged_method`. It picks the native op
only on GPU/TPU with `jax >= RAGGED_NATIVE_FIXED_JAX = (0, 9, 1)`, and `buf`
elsewhere. That is the same gate jaccpot puts on its gradient path
(`JAX_RAGGED_GRAD_FIXED_VERSION = (0, 9, 1)`), for the same reason. Both are one
XLA defect: `RaggedAllToAllStartThunk::Initialize` cached peer buffer addresses
from the first run, and allocator churn made the op write to stale addresses.
Donation causes such churn in a forward, and running a gradient causes it too. It
was fixed in XLA `4e0cc7e356`, which shipped in jax 0.9.1. jaccpot measured that
on 0.9.0.1 a forward after one gradient loses the whole cross-domain halo, while
every flag stays clean.

CPU has no native lowering, so these tests run `buf`.
`test_forward_survives_a_gradient` asserts that a forward after a gradient is
bit-identical to one before it, for both payloads. On ≥ 2 GPUs,
`YGGDRAX_CHECK_UPSTREAM_RAGGED_FIX=1` adds a test that runs the same check
through `exchange="native"`, and compares its gradient with `buf`'s.

**The jaccpot env's jax is 0.8.3.** There `"auto"` gives `buf` on GPU too: it is
correct, but it all-gathers every device's send buffer, which costs O(ndev ×
send_capacity) memory and bandwidth per device. C3 needs jax ≥ 0.9.1 on the GPUs.

### Memory: the near field is rematerialised

The near field is a `lax.scan` over chunks of leaf pairs. Under reverse mode a
scan stores every chunk's intermediates, which here means `(chunk, leaf, leaf,
nbins)` windows: O(near pairs × leaf² × bins). The chunk function is wrapped in
`jax.checkpoint`, so the reverse pass stores only its integer inputs and
recomputes the windows. The forward is unchanged.

Measured as XLA's compiled per-device temp for ξ at N = 8192 on 4 CPU devices,
fp64, leaf 16, 8 bins:

| | forward | value + grad |
|---|---|---|
| with `jax.checkpoint` | 41 MB | 50 MB |
| without | 41 MB | 13 556 MB |

kNN needs no remat: the top-k search is topology. Its only differentiated values
are the `(rows, k)` distances recomputed at the frozen indices.

`bench/distributed/grad_memory.py` reports the forward's and value+grad's
compiled temp, runtime peak (GPU), compile time and run time per device, for both
payloads in fp32 and fp64. It runs the per-device bodies in a caller's
`shard_map` on pre-sharded input, as a large run would, and sizes the halo from a
forward probe. On a GPU it picks free cards with `autocvd` (`--gpu-select free`).
CPU, `N = 32768`, uniform, defaults (ξ: 8 log bins 0.01–0.05, s = 60; kNN: k = 8,
`r_max` = 0.08, soft CDF):

| payload | dtype | devices | fwd temp | v+g temp | ratio | fwd time | v+g time |
|---|---|---|---|---|---|---|---|
| ξ | fp32 | 4 | 146 MB | 164 MB | 1.13× | 9.7 s | 20.2 s |
| ξ | fp64 | 4 | 157 MB | 186 MB | 1.18× | 11.2 s | 22.0 s |
| kNN | fp32 | 4 | 199 MB | 205 MB | 1.03× | 1.79 s | 1.84 s |
| kNN | fp64 | 4 | 212 MB | 223 MB | 1.05× | 1.50 s | 1.54 s |
| ξ | fp32 | 8 | 78 MB | 97 MB | 1.24× | 11.2 s | 16.1 s |
| ξ | fp64 | 8 | 85 MB | 114 MB | 1.35× | 9.1 s | 18.8 s |
| kNN | fp32 | 8 | 105 MB | 114 MB | 1.08× | 1.24 s | 1.22 s |
| kNN | fp64 | 8 | 112 MB | 129 MB | 1.15× | 1.09 s | 1.05 s |

The temps are XLA's compiled plan per device, which is deterministic. The times
come from a workstation that was running the test suite at the same time, so read
them as rough ratios: about 2× for ξ, about 1× for kNN, whose cost is the
topology search. At this N the halo is 1.4–2.7× a device's own particles (the
probe sized `halo_fraction` to 1.8 on 4 devices and 3.3 on 8), so the capacities,
not N, set the memory. The ratio that will matter at scale is the GPU one, which
C3 measures.

### How it was checked (jaccpot's audit method)

`tests/distributed/test_distributed_stats_grad.py`, on 1, 2, 4 and 8 fake CPU
devices:

- **Equality.** The gradient equals brute-force O(N²) autodiff of the same
  statistic, and the 1-device gradient, to 1e-10. The max error relative to the
  largest component was measured at N = 640 (the kNN-CDF case uses 640 data and
  320 queries):

  | case | vs brute force | vs 1 device |
  |---|---|---|
  | ξ positions, faces/corners | ≤ 1.1e-15 | ≤ 1.1e-15 |
  | ξ weights, faces/corners | ≤ 3.3e-16 | ≤ 2.8e-16 |
  | ξ positions, clustered (one cluster on the corner) | ≤ 6.2e-15 | ≤ 2.1e-15 |
  | ξ weights, clustered | ≤ 3.7e-15 | ≤ 1.9e-15 |
  | kNN distances, both sets | ≤ 2.9e-16 | ≤ 2.4e-16 |
  | soft kNN-CDF of random queries, d/d(data) | ≤ 6.3e-16 | — |
  | soft kNN-CDF, d/d(queries) | ≤ 2.1e-15 | — |

- **Finite differences.** Central differences converge onto AD as h²: the
  relative error is 5.7e-5, 5.7e-7 and 5.6e-9 at h = 1e-5, 1e-6 and 1e-7 for ξ
  positions, identical to FD of the brute-force statistic. The windows are sharp,
  so truncation dominates. The weights' error is 6e-11. For the soft kNN-CDF the
  test runs at a step that leaves every brute-force neighbour set unchanged,
  which it checks first.
- **Trace probe.** `make_jaxpr(grad(loss))` traces in under a second.
- **Liveness sweep** (`tests/distributed/_grad_audit.py`). It propagates
  "reachable from the differentiated inputs" through the forward jaxpr. The
  propagation runs through floats only, stops at `stop_gradient`, and recurses
  into `shard_map`, `jit`, `scan`, `while`, `cond` and `checkpoint`. No `while`,
  `sort` or `top_k` is reached, so no topology decision is differentiated. The
  control sweep without the seam finds 10 live `while` loops for ξ and 17 for
  kNN: the mutual walks, `compute_tree_geometry`, the chunked top-k and
  `_node_max`.
- **Dependency sweep.** It perturbs particles that hug a corner or a face, whose
  partners mostly sit in another Morton domain through the periodic image. The
  change in the full gradient matches brute force to 1e-10 of its scale, rows
  that brute force leaves untouched stay untouched to 1e-12, and some of the
  changed rows lie across a face.
- **Ragged side effect.** A forward after a gradient is bit-identical to one
  before it. The native version is opt-in on GPU (above).
- **`jit(value_and_grad)`** equals the eager gradient. A `lax.scan` of three
  gradient steps equals the same steps taken one at a time; the topology is
  rebuilt inside each step.

### Not covered by C2

- No GPU run, so the native ragged exchange's transpose and the opt-in
  side-effect test have not been run, and no runtime peak memory has been
  measured. Everything above is on fake CPU devices with `buf`.
- Large N: the remat table and the bench are CPU, N ≤ 32768.
- `geometric_acceptance` gradients (consistent by construction, untested) and
  gradients with respect to `sharpness`, `edges` or `r_grid`.
- Second derivatives. The dependency sweep uses finite differences of
  gradients, not forward-over-reverse.

## Not covered (follow-ups)

- Cross-correlations (DD/DR/RR with distinct catalogues) need a "data" and a
  "query" flag per catalogue in the ξ payload. The kNN flags already carry the
  pattern.
- Soft kNN distances: the top-k indices are discrete. A smooth CDF comes from a
  smooth step in `ln r` around each `r_k`, which the CDF reduction offers.
- GPU scaling and capacities at 2048³ (C3), including the gradient's memory
  ceiling (`bench/distributed/grad_memory.py`).
