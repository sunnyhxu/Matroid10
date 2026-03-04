# JSONL Schemas

## `non_paving.jsonl`

Each line is a JSON object:

- `id` (`str`): canonical graph hash
- `field` (`int`): `2` or `3`
- `rank` (`int`)
- `n` (`int`): always `10`
- `seed` (`int`): run seed
- `trial` (`int`): candidate index in generation stream
- `matrix_cols` (`list[int]`): encoded columns for replay
- `bases` (`list[int]`): rank-sized bases as 10-bit masks (`0..1023`)
- `non_paving_witness` (`int`): dependent subset mask with size `< rank`

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
