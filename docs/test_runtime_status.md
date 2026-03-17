# Test Runtime Status

This note captures the current state of test-runtime work on this branch.

## Current Baseline

- Environment: `micromamba run -n odisseo`
- GPU pinning for runtime checks: `CUDA_VISIBLE_DEVICES=3`
- Full test command:

```bash
CUDA_VISIBLE_DEVICES=3 micromamba run -n odisseo python -m pytest -o addopts='' --durations=25
```

- Latest verified result: `158 passed in 511.48s (0:08:31)`

## Progress So Far

The runtime-improvement work was done incrementally while keeping the suite green.

### Verified checkpoints

- Initial branch-wide baseline after fixing interaction regressions:
  - `158 passed in 627.92s (0:10:27)`
- After the first runtime reduction pass:
  - `158 passed in 589.61s (0:09:49)`
- After the second runtime reduction pass:
  - `158 passed in 526.33s (0:08:46)`
- Current latest baseline:
  - `158 passed in 511.48s (0:08:31)`

That is a reduction of about `1m 56s` from the first full green baseline, or roughly `18%`.

## What Changed

### Correctness work needed to keep the suite green

- Fixed far-interaction public list ordering in [yggdrax/_interactions_impl.py](/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py) by repacking raw pairs into the documented level-major order using JAX.

### Runtime reduction work already landed in this branch history

- Reduced oversized problem sizes in several smoke/contract tests.
- Standardized traversal configs to reduce repeated JAX compilation for near-identical tests.
- Reduced repeated full traversals in backend conformance coverage.
- Shared dense interaction state through a module-scoped fixture.

### Relevant commits already created

- `2943f97` `Align octree JIT builders with octree path`
- `b9c8c97` `Fix interaction list ordering and trim test overhead`
- `b969778` `Reduce test problem sizes for faster runtime`

## Current Top Runtime Hotspots

From the latest `--durations=25` run:

1. `tests/unit/test_octree_topology.py::test_octree_matches_radix_interaction_counts` — `18.47s`
2. `tests/unit/test_backend_conformance.py::test_backend_conformance[radix]` — `13.63s`
3. `tests/unit/test_interactions_backend_parity.py::test_build_interactions_and_neighbors_contract_holds_for_backends[radix-8]` — `13.16s`
4. `tests/unit/test_dense_interactions.py::test_dense_sources_match_sparse_lists` setup — `13.11s`
5. `tests/unit/test_backend_conformance.py::test_backend_conformance[octree]` — `12.25s`
6. `tests/unit/test_tree.py::test_build_fixed_depth_octree_jit_matches_eager` — `11.31s`
7. `tests/unit/test_interactions_backend_parity.py::test_build_interactions_and_neighbors_contract_holds_for_backends[kdtree-16]` — `11.29s`
8. `tests/unit/test_two_pass_builder.py::test_count_pass_uses_compact_fill_path` — `11.25s`

## Interpretation

The suite is no longer dominated by easy Python-side waste. The remaining large costs are mostly:

- JAX compile + execute time in parity-heavy tests
- repeated backend-wide traversal checks
- dense interaction setup cost
- eager-vs-JIT parity tests that intentionally compile builders

## Best Next Targets

If we continue runtime work, the highest-value next steps are:

1. Revisit overlap between backend conformance and backend parity tests.
2. Reduce or share setup in the remaining dense interaction tests.
3. Inspect slow traversal-policy tests for smaller static configs or reusable fixtures.
4. Decide whether the slowest JIT parity tests should remain in the default local suite or move behind a `slow` marker while staying in CI.

## Notes

- All runtime measurements above were taken with `GPU 3` pinned explicitly.
- The `-o addopts=''` override was required because the current `odisseo` environment does not provide the coverage plugin expected by the default `pyproject.toml` pytest configuration.
