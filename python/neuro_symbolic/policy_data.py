from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:
    from ..common import ensure_parent, read_jsonl
except ImportError:
    from common import ensure_parent, read_jsonl

from .types import CostBucket


OUTCOME_LABELS = {
    "counterexample_found",
    "solver_disagreement",
    "unknown_timeout",
    "verifier_error",
    "exact_feasible",
    "duplicate_isomorph",
    "invalid_action_blocked",
}


def outcome_value(label: str) -> float:
    scores = {
        "counterexample_found": 1.0,
        "solver_disagreement": 0.8,
        "unknown_timeout": 0.4,
        "verifier_error": 0.0,
        "exact_feasible": 0.2,
        "duplicate_isomorph": 0.0,
        "invalid_action_blocked": 0.0,
    }
    if label not in scores:
        raise ValueError(f"unsupported_outcome_label:{label}")
    return scores[label]


def cost_penalty(bucket: CostBucket) -> float:
    penalties = {
        CostBucket.CHEAP: 1.0,
        CostBucket.MEDIUM: 2.0,
        CostBucket.EXPENSIVE: 4.0,
        CostBucket.LIKELY_TIMEOUT: 8.0,
    }
    return penalties[bucket]


def cost_bucket_from_wall_time(
    wall_time: float,
    thresholds: Mapping[str, float] | None = None,
) -> CostBucket:
    thresholds = thresholds or {"cheap": 1.0, "medium": 3.0, "expensive": 6.0}
    if wall_time <= float(thresholds["cheap"]):
        return CostBucket.CHEAP
    if wall_time <= float(thresholds["medium"]):
        return CostBucket.MEDIUM
    if wall_time <= float(thresholds["expensive"]):
        return CostBucket.EXPENSIVE
    return CostBucket.LIKELY_TIMEOUT


def build_replay_row(
    *,
    region_features: Mapping[str, Any],
    instance_action_features: Mapping[str, Any],
    outcome_label: str,
    cost_bucket: CostBucket,
    censored_timeout: bool,
) -> Dict[str, Any]:
    if outcome_label not in OUTCOME_LABELS:
        raise ValueError(f"unsupported_outcome_label:{outcome_label}")
    target = outcome_value(outcome_label) / cost_penalty(cost_bucket)
    return {
        "region_features": dict(region_features),
        "instance_action_features": dict(instance_action_features),
        "outcome_label": outcome_label,
        "cost_bucket": cost_bucket.value,
        "censored_timeout": bool(censored_timeout),
        "region_target": float(target),
        "instance_target": float(target),
        "timeout_target": float(1.0 if censored_timeout or cost_bucket is CostBucket.LIKELY_TIMEOUT else 0.0),
    }


def write_replay_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(__import__("json").dumps(dict(row), separators=(",", ":")))
            handle.write("\n")


def read_replay_rows(path: str | Path) -> List[Dict[str, Any]]:
    return [dict(row) for row in read_jsonl(path)]
