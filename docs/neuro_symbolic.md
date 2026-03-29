# Neuro-Symbolic Search

## Overview
The neuro-symbolic search stack lives under `python/neuro_symbolic/` and is organized into six layers:

1. Bootstrap and shared records.
2. Canonicalization plus the search-state graph.
3. Family-specific action spaces for `representable` and `sparse_paving`.
4. Deterministic feature extraction, replay rows, and frozen baseline policies.
5. The controller loop, verifier scheduling, queue routing, and JSONL logs.
6. Replay evaluation, baseline comparisons, and acceptance metrics.

## Bootstrap Inputs
Bootstrap regions come from `artifacts/hardness_unique_hvectors.jsonl`. That file is produced by the existing hardness benchmark pipeline and already contains:

- `h_vector`
- current-solver metrics
- top-level-solver metrics
- structural metrics
- normalized region scores
- at least one witness/seed reference through `example_record_id`

Concrete seed candidates come from `artifacts/hvec.jsonl`, grouped by `h_vector`.

## Canonicalization
Each family `ProblemSpec` owns canonicalization and region-key derivation for its candidates. The controller resolves those identities before insertion, and the graph stores the resolved `RegionKey` and `CanonicalKey` values as state and event-log data.

Canonical identity is computed from the base family under element relabeling. The internal graph key is the full canonical label, and the short display id is the first 16 hex characters of a SHA-256 digest of that label.

This means:

- graph identity is not tied to the 64-bit generator hash alone
- isomorphic candidates merge into one `InstanceNode`
- duplicate-isomorph actions can be logged without increasing node count
- graph-event replay is expected to reconstruct the same seed membership and canonical node set as the original run

## Family-Specific Actions
`representable` candidates stay inside the matrix presentation. The action space includes bounded single-column resamples, small column-batch resamples, and constrained column replacements that are only emitted if the resulting matrix still has full-rank support.

`sparse_paving` candidates stay inside the circuit-hyperplane family. The action space includes admissible add/remove/swap operations plus a bounded batch resample, again emitted only when the overlap constraint is preserved by construction.

## Timeout Interpretation
Timeouts are treated as `unknown_timeout`, not as wins.

- cheap filters run first
- the current reference solver runs next
- the top-level solver runs next only when the cost bucket allows it
- `unknown_timeout` routes to the escalation queue
- `counterexample_found` and `solver_disagreement` route to the terminal-complete queue
- `exact_feasible` routes to the continue-search queue

## Running The Search
Small bounded run:

```bash
python python/neuro_symbolic_search.py \
  --bootstrap-path artifacts/hardness_unique_hvectors.jsonl \
  --seed-candidates-path artifacts/hvec.jsonl \
  --max-steps 10 \
  --action-log-out artifacts/neuro_symbolic/action_log.jsonl \
  --graph-event-out artifacts/neuro_symbolic/graph_events.jsonl \
  --replay-row-out artifacts/neuro_symbolic/replay_rows.jsonl
```

The main outputs are:

- `artifacts/neuro_symbolic/action_log.jsonl`
- `artifacts/neuro_symbolic/graph_events.jsonl`
- `artifacts/neuro_symbolic/replay_rows.jsonl`

## Replay Evaluation
Replay the logged rows and compare baseline strategies:

```bash
python -m python.neuro_symbolic.eval \
  --replay-rows artifacts/neuro_symbolic/replay_rows.jsonl \
  --action-log artifacts/neuro_symbolic/action_log.jsonl \
  --summary-out artifacts/neuro_symbolic/eval_summary.json \
  --include-no-canonical-merge
```

The evaluation summary includes:

- interesting verified outcomes per CPU-hour
- duplicate-isomorph rate
- invalid-action-blocked rate
- escalation yield
- median time to first interesting unknown
- top regions visited
- queue counts
- outcome counts
- verifier spend by stage
- duplicate merge counts

## Manual Benchmark
For a larger manual benchmark on the current hardness artifacts, increase the controller budget rather than changing the workflow:

```bash
python python/neuro_symbolic_search.py \
  --bootstrap-path artifacts/hardness_unique_hvectors.jsonl \
  --seed-candidates-path artifacts/hvec.jsonl \
  --max-steps 200 \
  --action-log-out artifacts/neuro_symbolic/manual_action_log.jsonl \
  --graph-event-out artifacts/neuro_symbolic/manual_graph_events.jsonl \
  --replay-row-out artifacts/neuro_symbolic/manual_replay_rows.jsonl
```

Then compare the replay outputs with:

```bash
python -m python.neuro_symbolic.eval \
  --replay-rows artifacts/neuro_symbolic/manual_replay_rows.jsonl \
  --action-log artifacts/neuro_symbolic/manual_action_log.jsonl \
  --summary-out artifacts/neuro_symbolic/manual_eval_summary.json \
  --include-no-canonical-merge
```

## V1 Non-Goals
- No online policy updates during search.
- No generic family-agnostic matroid mutation API.
- No exact runtime regression target for `CostPolicy`.
