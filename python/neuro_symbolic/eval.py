from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .policy_data import read_replay_rows
from .replay import load_action_logs, summarize_action_logs, write_action_summary


def _bucket_seconds(bucket: str) -> float:
    return {
        "cheap": 1.0,
        "medium": 2.0,
        "expensive": 4.0,
        "likely_timeout": 8.0,
    }[bucket]


def _interesting(label: str) -> bool:
    return label in {"counterexample_found", "solver_disagreement", "unknown_timeout"}


def _row_score(strategy_name: str, row: Mapping[str, Any], index: int) -> tuple[float, int]:
    if strategy_name in {"observed", "uniform", "no_canonical_merge_ablation"}:
        return (1.0, -index)
    region_features = row["region_features"]
    if strategy_name == "structural_score_only":
        return (float(region_features.get("structural_score", 0.0) or 0.0), -index)
    if strategy_name == "dual_solver_hardness_only":
        return (
            float(region_features.get("current_solver_score", 0.0) or 0.0)
            + float(region_features.get("top_level_solver_score", 0.0) or 0.0),
            -index,
        )
    raise ValueError(f"unsupported_strategy:{strategy_name}")


def _normalized_label(strategy_name: str, label: str) -> str:
    if strategy_name == "no_canonical_merge_ablation" and label == "duplicate_isomorph":
        return "exact_feasible"
    return label


def evaluate_replay_rows(rows: Iterable[Mapping[str, Any]], strategy_name: str = "observed") -> Dict[str, Any]:
    materialized = [dict(row) for row in rows]
    ordered = sorted(
        enumerate(materialized),
        key=lambda item: _row_score(strategy_name, item[1], item[0]),
        reverse=True,
    )
    ordered_rows = [row for _, row in ordered]
    normalized_labels = [_normalized_label(strategy_name, str(row["outcome_label"])) for row in ordered_rows]
    total_seconds = sum(_bucket_seconds(str(row["cost_bucket"])) for row in ordered_rows)
    interesting_count = sum(1 for label in normalized_labels if _interesting(label))
    duplicate_count = sum(1 for label in normalized_labels if label == "duplicate_isomorph")
    invalid_count = sum(1 for label in normalized_labels if label == "invalid_action_blocked")
    escalation_rows = sum(
        1
        for row, label in zip(ordered_rows, normalized_labels)
        if bool(row.get("censored_timeout"))
        or str(row["cost_bucket"]) == "likely_timeout"
        or label in {"unknown_timeout", "verifier_error"}
    )
    interesting_unknowns = sum(1 for label in normalized_labels if label == "unknown_timeout")

    cumulative = 0.0
    first_unknown_time = None
    for row, label in zip(ordered_rows, normalized_labels):
        cumulative += _bucket_seconds(str(row["cost_bucket"]))
        if label == "unknown_timeout":
            first_unknown_time = cumulative
            break

    return {
        "strategy_name": strategy_name,
        "rows_evaluated": len(ordered_rows),
        "interesting_outcomes_per_cpu_hour": 0.0 if total_seconds == 0.0 else interesting_count / (total_seconds / 3600.0),
        "duplicate_isomorph_rate": 0.0 if not ordered_rows else duplicate_count / float(len(ordered_rows)),
        "invalid_action_blocked_rate": 0.0 if not ordered_rows else invalid_count / float(len(ordered_rows)),
        "escalation_yield": 0.0 if escalation_rows == 0 else interesting_unknowns / float(escalation_rows),
        "median_time_to_first_interesting_unknown_seconds": first_unknown_time,
    }


def compare_baselines(rows: Iterable[Mapping[str, Any]], include_no_canonical_merge: bool = False) -> Dict[str, Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    strategies = ["uniform", "structural_score_only", "dual_solver_hardness_only"]
    if include_no_canonical_merge:
        strategies.append("no_canonical_merge_ablation")
    return {strategy: evaluate_replay_rows(materialized, strategy_name=strategy) for strategy in strategies}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay evaluation for neuro-symbolic search traces.")
    parser.add_argument("--replay-rows", required=True)
    parser.add_argument("--action-log", default=None)
    parser.add_argument("--summary-out", default="artifacts/neuro_symbolic/eval_summary.json")
    parser.add_argument("--include-no-canonical-merge", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_rows = read_replay_rows(args.replay_rows)
    summary = {
        "observed": evaluate_replay_rows(replay_rows, strategy_name="observed"),
        "baselines": compare_baselines(replay_rows, include_no_canonical_merge=args.include_no_canonical_merge),
    }
    if args.action_log:
        summary["action_log_summary"] = summarize_action_logs(load_action_logs(args.action_log))
    write_action_summary(args.summary_out, summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
