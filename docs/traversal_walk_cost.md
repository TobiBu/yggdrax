# What the traced dual-tree walk costs, and the flat-emission alternative (2026-09-10)

Measured with `bench/traversal_walk_bench.py` on an idle A100-40GB, N=200k Plummer, theta 0.6, dehnen MAC,
one tree per row, both walks producing **identical far and near pair counts** (and, per
`tests/unit/test_dual_tree_walk_mutual.py`, identical pair sets).

| leaf | leaves | `_dual_tree_walk_impl` (int64, Q as the fused lane runs it) | conds removed | `dual_tree_walk_mutual` int64 | int32 | int32, Q = pow2(1.5 x peak) |
|---|---|---|---|---|---|---|
| 256 | 782 | 79.8 ms (Q 2^18) | -- | -- | -- | 9.2 ms (peak 35k, Q 2^16) |
| 128 | 1563 | 300 ms (int32, Q 2^20) | -- | -- | 23.3 ms (Q 2^20) | -- |
| 64 | 3125 | 503 ms (Q 2^20) | 390 ms | 39.9 ms | 26.4 ms | 13.4 ms (peak 192k, Q 2^18) |
| 32 | 6250 | 1291 ms (Q 2^21) | -- | 48.9 ms | 34.3 ms | 19.3 ms (peak 392k, Q 2^19) |

## Where the dual walk's time goes

`_dual_tree_walk_impl` is a `lax.while_loop` over wavefront generations. Its per-round cost is linear in the
queue capacity (every round masks, gathers and scatters all `max_pair_queue` slots and 4x that for the
refinement candidates), but the bulk of the time at small leaves is the **output layout**:

1. Accepted pairs are written into dense per-node rows, `far_buffer (total_nodes, max_interactions_per_node)`
   -- 820 MB at leaf 64 with a 16384 cap -- and near pairs into `neighbor_buffer (num_leaves,
   max_neighbors_per_leaf)`, both loop-carried.
2. The column for each pair is `count[node] + per-key prefix`, and the prefix needs a full-queue `argsort`
   (`_per_key_prefix`) per round, vmapped over (target, source) for far and again for near, plus four
   `segment_sum`s.
3. Each emission sat inside a `lax.cond(jnp.any(mask), update, identity, buffers)`. The identity branch
   cannot alias its operand to its result, so XLA copied the dense buffers in and out every round and
   synced the predicate to the host. Replacing the conditional by the (exactly equivalent) unconditional
   masked update took 503 -> 390 ms at leaf 64. That change is in this PR for every caller.
4. After the loop the rows are flattened into `total_nodes x K` slots with `repeat`/`tile`/gather/scatter.

What is NOT a lever: promising `unique_indices=True` on the compaction scatters. A 1M-element
`.at[].set(mode="drop")` takes 0.10 ms with or without it on this XLA (the promise is kept where it holds,
for the lowering's benefit elsewhere, but it does not explain the 800 us fusions seen in the fused lane's
kernel table -- those were the dense-row scatters and copies above).

## The flat walk

`dual_tree_walk_mutual` keeps the wavefront skeleton but appends each accepted or near pair **once**
(`a < b`) to a flat `(cap,)` buffer with a cumsum and one scatter (`_flat_append`): no per-node rows, no
sort, no `segment_sum`, no conditional, no post-loop flatten. Its acceptance rule is the dual walk's; with
`mac_type` set it goes through `_compute_mac_ok` so even the equality case matches, and fed
`_build_mac_extents(...)[0]` it reproduces the dual walk's far and near lists as sets. It now reports
`peak_wavefront` (the largest push any round needed, before truncation) and `rounds`, so a traced caller can
size its queue from data instead of from a capacity that merely did not overflow; its index dtype follows
the child arrays (int32 halves every byte).

What was left after that is a per-round floor: 46-61 rounds per walk at ~0.15 ms each, most of them far
below the queue capacity.

## The width ladder (2026-09-11)

Every round used to evaluate all `max_pair_queue` slots. The live wavefront, measured round by round on the
200k Plummer tree (`_wavefront_ladder` commit; eager profile with dynamic shapes):

| leaf / theta | rounds | peak live | rounds with < 4096 live | slot evaluations at full width | with the ladder | live pairs summed |
|---|---|---|---|---|---|---|
| 64 / 0.6 | 55 | 191,798 | first 18, last 3 | 28.8 M (Q = 2^19) | 5.2 M | 2.7 M |
| 32 / 0.6 | 61 | 393,552 | first 18, last 3 | 64.0 M (Q = 2^20) | 13.7 M | 5.6 M |
| 64 / 0.8 | 55 | 92,478 | first 18, last 4 | 14.4 M (Q = 2^18) | 3.3 M | 1.3 M |
| 256 / 0.6 | 46 | 35,366 | first 19, last 3 | 3.0 M (Q = 2^16) | 1.1 M | 0.45 M |

The wavefront rises by ~1.4x per round from the root, sits on a plateau near the peak for ~10 rounds and
decays for ~20 (leaf 64 / 0.6: 3, 5, 12, 28, ..., 191,798, ..., 33,614, 33,222, 31,248, 28,838, 24,522,
19,044, 13,588, 6,762, 1,928, 212). `dual_tree_walk_mutual` now compiles its round body once per width of a
static ladder (powers of 4 from 4096 up to the queue, `_wavefront_ladder`) and a `lax.switch` runs each round
at the narrowest width that holds its live set; the pushed pairs still compact into the full-width queue, so
`peak_wavefront`, `rounds` and the overflow flags are unchanged and every rung yields the same pairs in the
same order (`test_wavefront_ladder_is_bit_identical_to_the_full_width_body`). Slot work drops 4.4-5.5x at
leaf 32-64; the per-round launch floor stays. `wavefront_ladder=False` (or `YGGDRAX_MUTUAL_WALK_LADDER=0`,
read at import) keeps the single-width body for A/B runs; `bench/traversal_walk_bench.py --no-ladder`.

TIMINGS_PLACEHOLDER

## Reproduce

```
python bench/traversal_walk_bench.py --n 200000 --leaf-size 64 --theta 0.6 \
    --max-pair-queue 1048576 --far-cap 2097152 --near-cap 2097152 [--index int32] [--walks dual,mutual,scatter]
python bench/traversal_walk_bench.py --smoke     # tiny CPU run
```
The card is chosen with `autocvd`; the result records the card's utilisation before and after so a contended
row can be recognised.
