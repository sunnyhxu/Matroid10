# Pipeline Overview

This repository implements a 3-phase search pipeline for non-paving matroids on 10 elements and pure O-sequence verification.

## Phase 1: C++ Generation

- Program: `cpp/build/gen_nonpaving`
- Input: random seeds and generation settings.
- Output: JSONL records for unique non-paving matroids (`non_paving.jsonl`).
- Dedup: canonical labeling of a bipartite element/base incidence graph via nauty/Traces.
- Connectivity gate: disconnected matroids are filtered out (`generation.require_connected = true` by default).

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
  - Runs both fields (`GF(2)` and `GF(3)`).
  - Enforces deterministic seed schedule.
  - Aggregates metrics and writes run manifest.
  - Stops immediately and snapshots if first CP infeasible instance is found.
