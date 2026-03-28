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
5. Run full pipeline (choose one):
   - **Command line:** `uv run python python/run_pipeline.py --config configs/default.toml`
   - **Web UI:** `uv run uvicorn python.web.app:app --host 127.0.0.1 --port 8000` then open http://localhost:8000
6. Optional mode/sharding overrides (command line):
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
8. Continue downstream processing from an existing sparse-paving Phase 1 artifact:
   - `bash scripts/run_sparse_paving_recovery.sh <label>`
9. Extract clean per-run deduplicated outputs from a raw Phase 3 JSONL:
   - `uv run python python/extract_deduplicated_results.py --in artifacts/pure_o_results_<label>.jsonl --matroids-out artifacts/dedup/matroids_<label>.jsonl --hvectors-out artifacts/dedup/hvectors_<label>.jsonl --summary-out artifacts/dedup/summary_<label>.json`
10. Optional: attach one pure O-sequence witness per deduplicated feasible h-vector:
   - `uv run python python/attach_pure_o_witnesses.py --in artifacts/dedup/hvectors_<label>.jsonl --out artifacts/dedup/hvectors_<label>.jsonl --summary-out artifacts/dedup/witness_summary_<label>.json`
11. Optional: remove bulky experimental intermediates while keeping canonical representable outputs and deduplicated sparse outputs:
   - `uv run python python/cleanup_artifacts.py`
   - `uv run python python/cleanup_artifacts.py --apply`

## Web UI

The web interface provides a browser-based way to control the pipeline with real-time progress monitoring.

### Features

- **Start/Stop Controls:** Start the pipeline with a button click; stop gracefully after the current subprocess completes
- **Per-Phase Progress Bars:** Visual progress for Phase 1 (C++ Generation), Phase 2 (Sage h-vector), and Phase 3 (CP-SAT)
- **Live Counters:** Total processed, unique found, feasible, and infeasible counts update in real-time
- **Persistent State:** Browser can close and reopen; status persists to disk (`artifacts/run_state.json`)
- **Accumulating Storage:** Results deduplicated by canonical hash ID across multiple runs (`artifacts/accumulated_results.jsonl`)
- **Auto-Increment Trials:** System tracks `trial_index_start` and automatically continues from the next block on each run

### Running the Web UI

```bash
uv run uvicorn python.web.app:app --host 127.0.0.1 --port 8000
```

Then open http://localhost:8000 in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the main UI page |
| `/api/status` | GET | Get current pipeline status, phases, and counters |
| `/api/start` | POST | Start the pipeline in background |
| `/api/stop` | POST | Request graceful stop |
| `/api/config` | GET | Get current pipeline configuration |
| `/api/accumulated` | GET | Get accumulated dedup store statistics |

## Main Artifacts

- `artifacts/non_paving.jsonl`
- `artifacts/hvec.jsonl`
- `artifacts/pure_o_results.jsonl`
- `artifacts/metrics.json`
- `artifacts/run_manifest.json`
- `artifacts/counterexample_<runid>.json` (if an infeasible h-vector is found)
- `artifacts/progress_chunks/*.json`
- `artifacts/search_progress.jsonl`
- `artifacts/run_state.json` (web UI persistent state)
- `artifacts/accumulated_results.jsonl` (deduplicated results across runs)
- `artifacts/seen_ids.json` (index of seen matroid IDs for fast dedup)
- `artifacts/dedup/matroids_<label>.jsonl` (clean per-run deduplicated matroids)
- `artifacts/dedup/hvectors_<label>.jsonl` (deduplicated h-vectors with feasibility, counts, and optional pure O witness)
- `artifacts/dedup/summary_<label>.json` (per-run dedup summary)
- `artifacts/dedup/witness_summary_<label>.json` (optional witness attachment summary)

See `docs/pipeline.md` and `docs/schema.md` for details.

## Notes

- `python/hvec_extract.py` computes h-vectors from `Matroid.f_vector()` for Sage 10.7 compatibility.
- `scripts/run_sparse_paving_recovery.sh` resumes Phase 2 and Phase 3 from `artifacts/non_paving_sparse_paving.jsonl` and writes labeled recovery outputs.
- `python/extract_deduplicated_results.py` writes clean per-run deduplicated exports under `artifacts/dedup/`.
- `python/attach_pure_o_witnesses.py` enriches the deduplicated h-vector export with one witness pure O-sequence per feasible h-vector.
- `python/cleanup_artifacts.py` removes raw sparse intermediates, probes, logs, and timestamped duplicate exports while preserving canonical outputs.
- Optional Tutte cross-check (`T_M(x,1)`) can be enabled with `hvec.check_h_formula = true`.
- By default, Phase 1 filters disconnected matroids and Phase 2 filters `h_rank <= 0` before CP-SAT.

<!-- SEARCH_PROGRESS:START -->
## Search Progress

| Category | Trials | Unique | Yield | Status |
|---|---:|---:|---:|---|
| Representable $\mathbb{F}_2$ | 95.3M | 639 | 0.0007% | Active |
| Representable $\mathbb{F}_3$ | 14.4M | 6.3K | 0.044% | Active |
| Sparse Paving | 15.0M | 3.0M | 20.2% | Active |
<!-- SEARCH_PROGRESS:END -->
