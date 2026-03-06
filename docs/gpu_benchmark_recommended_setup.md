# GPU Benchmark Recommended Setup

This document records the currently best-performing runtime setup found in
`examples/tree_gpu_performance_scaling.ipynb` for the radix traversal path
(on A100 in the reported sweep).

## Locked Runtime Settings

- `radix_leaf_size = 64`
- `traversal_process_block = 256`
- keep `theta` and `mac_type` unchanged unless explicitly doing an
  accuracy/performance study

Traversal capacity schedule in the notebook is currently:

- `N <= 8,192`: `(256, 256, 65,536)`
- `N <= 16,384`: `(512, 512, 65,536)`
- `N <= 32,768`: `(1024, 512, 131,072)`
- `N <= 65,536`: `(2048, 1024, 131,072)`
- `N <= 131,072`: `(4096, 1024, 262,144)`

Tuple order is:

`(max_neighbors_per_leaf, max_interactions_per_node, max_pair_queue)`

## Observed Best From Knob Sweep

From the notebook knob sweep (`leaf_size` x `process_block` at large `N`):

- Best traversal runtime: `leaf_size=64`, `process_block=256`
- Best total radix pipeline (`build + geometry + traversal`):
  `leaf_size=64`, `process_block=256`

## Repro Notes

1. Restart kernel before benchmark runs.
2. Run notebook cells from top so config and helper changes are active.
3. Use per-`N` cap selection (`_cap_for_n(...)`) in weak and strong scaling
   sections (already wired in the notebook).
4. If hardware or software stack changes substantially, rerun the knob sweep
   cell and update this file.
