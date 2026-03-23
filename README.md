# Matroid10

Pipeline for generating 10-element matroids (representable and sparse paving) and testing whether their h-vectors are pure O-sequences.

## Runtime Model

- Phase 1 (C++): mode switch between representable generation over `GF(2)`/`GF(3)` and sparse paving generation, with nauty/Traces canonical dedup.
- Phase 2 (Sage): reconstruct matroids from bases and extract h-vectors with algebraic prefilters.
- Phase 3 (Python + OR-Tools): CP-SAT feasibility check for pure O-sequence realization.

## Quick Start (WSL)

1. Install dependencies in WSL (Ubuntu): `build-essential`, `cmake`, `nauty`, `libnauty-dev`, `sagemath`, `python3`.
   - On Ubuntu 24.04, `sagemath` is not in apt; `scripts/setup_wsl.sh` installs conda-forge `sage` via `micromamba` automatically.
2. Install `uv` in WSL.
3. Build generator:
   - `bash scripts/build_cpp.sh`
4. Install Python deps:
   - `uv sync`
5. Run full pipeline:
   - `uv run python python/run_pipeline.py --config configs/default.toml`
6. Optional mode/sharding overrides:
   - Sparse paving: `--override generation.mode=sparse_paving`
   - Shard 1/4: `--override generation.shard_index=1 --override generation.shard_count=4`
   - Continue from later trial block: `--override generation.trial_index_start=1000`
     - Trial stream for shard `s` uses `trial_id = trial_start + k * shard_count`
     - `trial_start = shard_index + trial_index_start * shard_count`
     - Example with `shard_count=8`, `max_trials=1000`:
       - first block (`trial_index_start=0`) covers local indices `k=0..999`
       - next block (`trial_index_start=1000`) covers `k=1000..1999` without overlap
7. Update progress ledger + README table:
   - `uv run python python/search_progress.py`

## Main Artifacts

- `artifacts/non_paving.jsonl`
- `artifacts/hvec.jsonl`
- `artifacts/pure_o_results.jsonl`
- `artifacts/metrics.json`
- `artifacts/run_manifest.json`
- `artifacts/counterexample_<runid>.json` (if an infeasible h-vector is found)
- `artifacts/progress_chunks/*.json`
- `artifacts/search_progress.jsonl`

See `docs/pipeline.md` and `docs/schema.md` for details.

## Notes

- `python/hvec_extract.py` computes h-vectors from `Matroid.f_vector()` for Sage 10.7 compatibility.
- Optional Tutte cross-check (`T_M(x,1)`) can be enabled with `hvec.check_h_formula = true`.
- By default, Phase 1 filters disconnected matroids and Phase 2 filters `h_rank <= 0` before CP-SAT.

<!-- SEARCH_PROGRESS:START -->
## Search Progress

| Category | Elements (n) | Status | Method | Coverage |
|---|---:|---|---|---:|
| Representable $\mathbb{F}_2$ | 10 | Not Started | Bitset C++ / CP-SAT | 0.000000% |
| Representable $\mathbb{F}_3$ | 10 | Not Started | GFq C++ / CP-SAT | 0.000000% |
| Sparse Paving | 10 | Not Started | Heuristic Search | 0.000000% |
<!-- SEARCH_PROGRESS:END -->
