# Pipeline Overview

This repository implements a 3-phase search pipeline for 10-element matroids and pure O-sequence verification.

## Phase 1: C++ Generation

- Program: `cpp/build/gen_nonpaving`
- Input: random seeds, `generation.mode`, and sharding settings.
- Output: JSONL records for unique matroids (`non_paving.jsonl` after merge).
- Dedup: canonical labeling of a bipartite element/base incidence graph via nauty/Traces.
- Connectivity gate: disconnected matroids are filtered out (`generation.require_connected = true` by default).
- Modes:
  - `representable`: one command per field (`GF(2)` and `GF(3)`).
  - `sparse_paving`: one command that samples circuit-hyperplane families.
- Sharding:
  - Pipeline-level config: `generation.shard_index`, `generation.shard_count`.
  - Continuation offset: `generation.trial_index_start` (default `0`).
  - Generator receives `--trial-start` and `--trial-stride`.
  - `trial_start = shard_index + trial_index_start * shard_count`
  - Trial id formula: `trial_id = trial_start + local_idx * trial_stride`.

## Phase 2: Sage h-vector Extraction

- Program: `sage -python python/hvec_extract.py`
- Input: `non_paving.jsonl`
- Output: `hvec.jsonl`
- Work: instantiate `Matroid(bases=...)`, compute `f_vector`, convert to `h_vector`, run prefilters.
- Prefilter gate: CP candidates require `h_rank > 0` (`hvec.check_h_rank_positive = true` by default).

## Phase 3: CP-SAT Pure O-Sequence Check

- Program: `uv run python python/pure_o_cp.py`
- Input: `hvec.jsonl`
- Output: `pure_o_results.jsonl`
- Work: build order-ideal + purity + degree-count CP model and solve with timeout.

## Orchestration

- Program: `uv run python python/run_pipeline.py --config configs/default.toml`
- Behavior:
  - Runs mode-specific Phase 1 commands (`representable` field loop or one `sparse_paving` run).
  - Enforces deterministic sharded trial schedule.
  - Writes per-command progress chunks to `artifacts/progress_chunks/`.
  - Aggregates metrics and writes run manifest.
  - Stops immediately and snapshots if first CP infeasible instance is found.

## Hardness Benchmark

- Program: `uv run python python/hardness_benchmark.py`
- Input: any JSONL with `h_vector` fields, typically `artifacts/hvec.jsonl` or `artifacts/pure_o_results.jsonl`
- Output:
  - `artifacts/hardness_unique_hvectors.jsonl`
  - optional `artifacts/hardness_summary.json`
- Work:
  - Deduplicate by exact h-vector.
  - Run the current CP-SAT formulation once per unique h-vector.
  - Run the top-level-vertex CP-SAT formulation once per unique h-vector.
  - Compute structural metrics such as Macaulay slack and tail-drop pressure.
  - Emit separate normalized scores for current-solver, top-level-solver, and structural difficulty, plus a weighted combined score.
- Interpretation:
  - `current_solver_score` captures empirical struggle in the original formulation.
  - `top_level_solver_score` captures empirical struggle in the maximal-monomial formulation.
  - `structural_score` captures mathematical bottleneck features independent of solver runtime.
  - `combined_score_weights` is recorded per record so downstream tuning can change weights without losing provenance.

## Progress Ledger

- Program: `uv run python python/search_progress.py`
- Input: `artifacts/progress_chunks/*.json`
- Output:
  - `artifacts/search_progress.jsonl` (deduplicated chunk ledger)
  - README section between `<!-- SEARCH_PROGRESS:START -->` and `<!-- SEARCH_PROGRESS:END -->`
