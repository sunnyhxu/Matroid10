# Matroid10

Pipeline for generating non-paving 10-element matroids and testing whether their h-vectors are pure O-sequences.

## Runtime Model

- Phase 1 (C++): random representable matroid generation over `GF(2)` and `GF(3)`, non-paving filtering, nauty/Traces canonical dedup.
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

## Main Artifacts

- `artifacts/non_paving.jsonl`
- `artifacts/hvec.jsonl`
- `artifacts/pure_o_results.jsonl`
- `artifacts/metrics.json`
- `artifacts/run_manifest.json`
- `artifacts/counterexample_<runid>.json` (if an infeasible h-vector is found)

See `docs/pipeline.md` and `docs/schema.md` for details.

## Notes

- `python/hvec_extract.py` computes h-vectors from `Matroid.f_vector()` for Sage 10.7 compatibility.
- Optional Tutte cross-check (`T_M(x,1)`) can be enabled with `hvec.check_h_formula = true`.
- By default, Phase 1 filters disconnected matroids and Phase 2 filters `h_rank <= 0` before CP-SAT.
