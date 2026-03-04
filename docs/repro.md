# Reproducibility Notes

## Deterministic Generation

- Use a fixed base seed from config.
- Each generation worker uses:
  - `worker_seed = base_seed + worker_id * 1000003 + field * 10007`
- Candidate stream order per worker is deterministic.

## Replay

To replay a candidate matroid:

1. Read `field`, `rank`, and `matrix_cols`.
2. Decode each column back to a length-`rank` vector over `GF(field)`.
3. Recompute dependent witness and bases.
4. Recompute canonical label and verify `id`.

## Run Manifest

Each pipeline run writes `run_manifest.json` including:

- run id and timestamps
- config path and resolved parameters
- executable/script versions and command lines
- counts and phase durations
- terminal status (`ok`, `counterexample`, or `error`)
