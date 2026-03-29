from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.controller import run_controller_from_paths
from python.neuro_symbolic.policies import CostPolicyModel, LinearScoreModel
from python.neuro_symbolic.problem_specs import RepresentableProblemSpec, SparsePavingProblemSpec


class _HeuristicRegionPolicy:
    def predict_score(self, features):
        return float(features.get("combined_score", 0.0))


class _HeuristicInstancePolicy:
    def predict_score(self, features):
        return float(features.get("parent_combined_score", 0.0))


class _HeuristicCostPolicy:
    def predict(self, features):
        from python.neuro_symbolic.types import CostBucket, CostPrediction

        return CostPrediction(bucket=CostBucket.MEDIUM, timeout_risk=0.2, confidence=0.5, details={"source": "heuristic"})


def _load_json(path: str | None):
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neuro-symbolic search controller on bootstrap artifacts.")
    parser.add_argument("--bootstrap-path", default="artifacts/hardness_unique_hvectors.jsonl")
    parser.add_argument("--seed-candidates-path", default="artifacts/hvec.jsonl")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--action-log-out", default="artifacts/neuro_symbolic/action_log.jsonl")
    parser.add_argument("--graph-event-out", default="artifacts/neuro_symbolic/graph_events.jsonl")
    parser.add_argument("--replay-row-out", default="artifacts/neuro_symbolic/replay_rows.jsonl")
    parser.add_argument("--region-model", default=None)
    parser.add_argument("--instance-model", default=None)
    parser.add_argument("--cost-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    region_model_payload = _load_json(args.region_model)
    instance_model_payload = _load_json(args.instance_model)
    cost_model_payload = _load_json(args.cost_model)

    region_policy = LinearScoreModel.from_dict(region_model_payload) if region_model_payload else _HeuristicRegionPolicy()
    instance_policy = LinearScoreModel.from_dict(instance_model_payload) if instance_model_payload else _HeuristicInstancePolicy()
    cost_policy = CostPolicyModel.from_dict(cost_model_payload) if cost_model_payload else _HeuristicCostPolicy()

    summary = run_controller_from_paths(
        bootstrap_path=args.bootstrap_path,
        seed_candidates_path=args.seed_candidates_path,
        controller_kwargs={
            "problem_specs": {
                "representable": RepresentableProblemSpec(),
                "sparse_paving": SparsePavingProblemSpec(),
            },
            "region_policy": region_policy,
            "instance_policy": instance_policy,
            "cost_policy": cost_policy,
            "seed": args.seed,
        },
        max_steps=args.max_steps,
        action_log_out=args.action_log_out,
        graph_event_out=args.graph_event_out,
        replay_row_out=args.replay_row_out,
    )
    print(json.dumps({"queue_counts": summary.queue_counts, "action_logs": len(summary.action_logs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
