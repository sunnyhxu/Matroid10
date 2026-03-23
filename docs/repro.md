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
