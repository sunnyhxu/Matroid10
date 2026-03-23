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
