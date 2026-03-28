# Reproducibility Notes

## Deterministic Generation

- Use a fixed base seed from config.
- Pipeline sharding maps each worker to a disjoint trial class:
  - `trial_id = trial_start + local_idx * trial_stride`
  - `trial_start = shard_index + trial_index_start * shard_count`
  - `trial_stride = shard_count`
- Candidate sampling is trial-index seeded, so shard outputs are stable even with different thread counts.

## Replay

To replay a candidate matroid:

1. Read `generator_mode`, `rank`, and `bases`.
2. For `representable` records, also read `field` and `matrix_cols` and decode columns over `GF(field)`.
3. For `sparse_paving` records, use `circuit_hyperplanes` and `sparse_overlap_bound`.
4. Recompute canonical label from the element/base incidence graph and verify `id`.

## Run Manifest

Each pipeline run writes `run_manifest.json` including:

- run id and timestamps
- config path and resolved parameters
- executable/script versions and command lines
- counts and phase durations
- terminal status (`ok`, `counterexample`, or `error`)

## Sparse Paving Recovery Run (2026-03-27)

Recovered downstream processing from the saved sparse-paving Phase 1 artifact:

- Label: `sparse_paving_recovery_20260327`
- Phase 1 source: `artifacts/non_paving_sparse_paving.jsonl`
- Phase 1 stats: `15,023,699` candidates, `3,030,877` unique sparse-paving hits
- Phase 2 stats: `3,151,887` records processed, `3,151,887` passed to CP, `0` extraction errors, `2137.16s`
- Phase 3 stats: `3,151,887` records considered, `62` unique h-vectors solved, `3,151,825` cache hits, `3,151,887` feasible, `0` infeasible, `330.42s`
- Counterexamples found: none

Primary recovery outputs:

- `artifacts/hvec_sparse_paving_recovery_20260327.jsonl`
- `artifacts/pure_o_results_sparse_paving_recovery_20260327.jsonl`
- `artifacts/phase_stats/hvec_extract_sparse_paving_recovery_20260327.json`
- `artifacts/phase_stats/pure_o_cp_sparse_paving_recovery_20260327.json`

Clean deduplicated exports derived from the Phase 3 output:

- `artifacts/dedup/matroids_sparse_paving_recovery_20260327.jsonl` with `3,151,887` unique matroid records
- `artifacts/dedup/hvectors_sparse_paving_recovery_20260327.jsonl` with `62` unique h-vectors, feasibility counts, and one witness pure O-sequence per feasible h-vector
- `artifacts/dedup/summary_sparse_paving_recovery_20260327.json`
- `artifacts/dedup/witness_summary_sparse_paving_recovery_20260327.json`
