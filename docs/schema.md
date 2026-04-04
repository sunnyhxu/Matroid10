# JSONL Schemas

## `non_paving.jsonl`

Each line is a JSON object:

- Common fields:
  - `id` (`str`): canonical graph hash
  - `generator_mode` (`str`): `representable` or `sparse_paving`
  - `rank` (`int`)
  - `n` (`int`): always `10`
  - `seed` (`int`): run seed
  - `trial` (`int`): candidate index in sharded trial stream
  - `bases` (`list[int]`): rank-sized bases as 10-bit masks (`0..1023`)
- Representable-only fields:
  - `field` (`int`): `2` or `3`
  - `matrix_cols` (`list[int]`): encoded columns for replay
  - `non_paving_witness` (`int`): dependent subset mask with size `< rank`
- Sparse-paving-only fields:
  - `circuit_hyperplanes` (`list[int]`): rank-sized subsets chosen as circuit-hyperplanes
  - `sparse_overlap_bound` (`int`): overlap cap used during construction (typically `rank - 2`)

Backward compatibility: if `generator_mode` is missing, consumers should interpret the record as `representable`.

## `hvec.jsonl`

Extends each record with:

- `h_vector` (`list[int]`)
- `prefilter_status` (`str`): `reject_precheck|pass_to_cp|error`
- `prefilter_reasons` (`list[str]`)
  - Example reasons: `h1_gt_h2`, `h_rank_nonpositive`, `h_formula_mismatch`

## `pure_o_results.jsonl`

Extends CP-attempted records with:

- `cp_status` (`str`): `FEASIBLE|INFEASIBLE|UNKNOWN|ERROR`
- `cp_time_sec` (`float`)
- `cp_model_size` (`dict`): includes `num_vars`, `num_constraints`, `num_degree_buckets`
- `cp_solver_stats` (`dict`, optional): may include `wall_time`, `num_conflicts`, `num_branches`, and `response_stats`

## `hardness_unique_hvectors.jsonl`

One record per unique h-vector benchmarked across both formulations:

- `h_vector` (`list[int]`)
- `h_vector_key` (`str`)
- `source_records` (`int`): number of source JSONL rows sharing this h-vector
- `example_record_id` (`str|null`)
- `current_solver` (`dict`)
  - `status` (`str`)
  - `wall_time` (`float`)
  - `model_size` (`dict`)
  - `metrics` (`dict`)
  - `score_components` (`dict`)
  - `score_raw` (`float`)
- `top_level_solver` (`dict`): same shape as `current_solver`
- `structural_metrics` (`dict`): includes Macaulay slack/violation summaries, drop summaries, degree/capacity pressure, and `raw_score`
- `current_solver_score` (`float`): normalized empirical difficulty for the current solver
- `top_level_solver_score` (`float`): normalized empirical difficulty for the top-level formulation
- `structural_score` (`float`): normalized structural difficulty
- `combined_score_weights` (`dict`)
- `score_normalization` (`str`): currently `percentile`
- `combined_score` (`float`)

## `artifacts/progress_chunks/*.json`

Per-Phase-1 command summary used for distributed progress tracking:

- `run_id` (`str`)
- `phase1_command_index` (`int`)
- `mode` (`str`): `representable` or `sparse_paving`
- `category` (`str`): e.g. `representable_f2`, `representable_f3`, `sparse_paving`
- `field` (`int|null`)
- `n` (`int`)
- `seed` (`int`)
- `shard_index` (`int`)
- `shard_count` (`int`)
- `trial_start` (`int`)
- `trial_index_start` (`int`)
- `trial_stride` (`int`)
- `candidates` (`int`)
- `unique_hits` (`int`)
