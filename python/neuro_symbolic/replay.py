from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:
    from ..common import ensure_parent, read_jsonl
except ImportError:
    from common import ensure_parent, read_jsonl


def load_action_logs(path: str | Path) -> List[Dict[str, Any]]:
    return [dict(row) for row in read_jsonl(path)]


def summarize_action_logs(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    queue_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    verifier_spend_by_stage: defaultdict[str, float] = defaultdict(float)
    duplicate_merge_count = 0
    total_rows = 0

    for row in rows:
        total_rows += 1
        queue_counts[str(row.get("queue_destination", "unknown"))] += 1
        outcome_label = str(row.get("outcome_label", "unknown"))
        outcome_counts[outcome_label] += 1
        region_counts[str(row.get("region_key", "unknown"))] += 1
        if bool(row.get("duplicate_isomorph")) or outcome_label == "duplicate_isomorph":
            duplicate_merge_count += 1
        for verifier in row.get("verifier_results", []):
            verifier_spend_by_stage[str(verifier.get("name", "unknown"))] += float(verifier.get("wall_time", 0.0) or 0.0)

    top_regions = [
        {"region_key": region_key, "visit_count": count}
        for region_key, count in sorted(region_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "total_rows": total_rows,
        "queue_counts": dict(queue_counts),
        "outcome_counts": dict(outcome_counts),
        "top_regions_visited": top_regions,
        "verifier_spend_by_stage": dict(sorted(verifier_spend_by_stage.items())),
        "duplicate_merge_count": duplicate_merge_count,
    }


def write_action_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    import json

    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
