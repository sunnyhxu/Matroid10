from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .common import dump_json, read_jsonl, write_jsonl
    from .hardness_metrics import compute_empirical_score, compute_structural_metrics, h_vector_key, percentile_normalize
    from .pure_o_cp import solve_h_vector
    from .pure_o_top_level_cp import solve_h_vector_top_level
except ImportError:
    from common import dump_json, read_jsonl, write_jsonl
    from hardness_metrics import compute_empirical_score, compute_structural_metrics, h_vector_key, percentile_normalize
    from pure_o_cp import solve_h_vector
    from pure_o_top_level_cp import solve_h_vector_top_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark unique h-vectors with dual pure O-sequence solvers.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--max-unique", type=int, default=0)
    parser.add_argument("--current-timeout-sec", type=float, default=120.0)
    parser.add_argument("--current-num-workers", type=int, default=8)
    parser.add_argument("--top-timeout-sec", type=float, default=120.0)
    parser.add_argument("--top-num-workers", type=int, default=8)
    parser.add_argument("--weight-current", type=float, default=1.0)
    parser.add_argument("--weight-top-level", type=float, default=1.0)
    parser.add_argument("--weight-structural", type=float, default=1.0)
    return parser.parse_args()


def _serialize_solver_record(result: Any) -> Dict[str, Any]:
    metrics = dict(result.solver_stats or {})
    metrics.setdefault("wall_time", result.wall_time)
    empirical_score = compute_empirical_score(metrics)
    payload: Dict[str, Any] = {
        "status": result.status,
        "wall_time": result.wall_time,
        "model_size": result.model_size,
        "metrics": metrics,
        "score_components": empirical_score,
        "score_raw": empirical_score["raw_score"],
    }
    if result.error:
        payload["error"] = result.error
    return payload


def _top_rank(records: List[Dict[str, Any]], score_field: str, limit: int = 10) -> List[Dict[str, Any]]:
    ranked = sorted(records, key=lambda record: float(record.get(score_field, 0.0)), reverse=True)
    out: List[Dict[str, Any]] = []
    for record in ranked[:limit]:
        out.append(
            {
                "h_vector": record["h_vector"],
                "h_vector_key": record["h_vector_key"],
                score_field: record[score_field],
                "current_solver_status": record["current_solver"]["status"],
                "top_level_solver_status": record["top_level_solver"]["status"],
            }
        )
    return out


def benchmark_unique_h_vectors(
    *,
    input_path: Path,
    output_path: Path,
    summary_out: Path | None = None,
    max_unique: int = 0,
    current_timeout_sec: float = 120.0,
    current_num_workers: int = 8,
    top_timeout_sec: float = 120.0,
    top_num_workers: int = 8,
    weight_current: float = 1.0,
    weight_top_level: float = 1.0,
    weight_structural: float = 1.0,
) -> Dict[str, Any]:
    normalization_mode = "percentile"
    unique_records: Dict[str, Dict[str, Any]] = {}
    input_records = 0

    for record in read_jsonl(input_path):
        input_records += 1
        h_vector = [int(x) for x in record.get("h_vector", [])]
        if not h_vector:
            continue

        key = h_vector_key(h_vector)
        aggregate = unique_records.get(key)
        if aggregate is None:
            unique_records[key] = {
                "h_vector": h_vector,
                "h_vector_key": key,
                "source_records": 1,
                "example_record_id": record.get("id"),
            }
        else:
            aggregate["source_records"] += 1

    ordered_keys = sorted(unique_records.keys())
    if max_unique > 0:
        ordered_keys = ordered_keys[:max_unique]

    output_rows: List[Dict[str, Any]] = []
    for key in ordered_keys:
        aggregate = unique_records[key]
        h_vector = aggregate["h_vector"]

        current_result = solve_h_vector(
            h_vector,
            timeout_sec=current_timeout_sec,
            num_workers=current_num_workers,
        )
        top_level_result = solve_h_vector_top_level(
            h_vector,
            timeout_sec=top_timeout_sec,
            num_workers=top_num_workers,
        )
        structural_metrics = compute_structural_metrics(h_vector)

        output_rows.append(
            {
                "h_vector": h_vector,
                "h_vector_key": key,
                "source_records": aggregate["source_records"],
                "example_record_id": aggregate["example_record_id"],
                "current_solver": _serialize_solver_record(current_result),
                "top_level_solver": _serialize_solver_record(top_level_result),
                "structural_metrics": structural_metrics,
                "structural_score_raw": structural_metrics["raw_score"],
            }
        )

    current_scores = percentile_normalize([row["current_solver"]["score_raw"] for row in output_rows])
    top_level_scores = percentile_normalize([row["top_level_solver"]["score_raw"] for row in output_rows])
    structural_scores = percentile_normalize([row["structural_score_raw"] for row in output_rows])

    weight_total = weight_current + weight_top_level + weight_structural
    if weight_total <= 0:
        raise ValueError("score weights must sum to a positive value")

    for index, row in enumerate(output_rows):
        row["current_solver_score"] = current_scores[index]
        row["top_level_solver_score"] = top_level_scores[index]
        row["structural_score"] = structural_scores[index]
        row["combined_score_weights"] = {
            "current_solver": weight_current,
            "top_level_solver": weight_top_level,
            "structural": weight_structural,
        }
        row["score_normalization"] = normalization_mode
        row["combined_score"] = (
            weight_current * row["current_solver_score"]
            + weight_top_level * row["top_level_solver_score"]
            + weight_structural * row["structural_score"]
        ) / weight_total

    write_jsonl(output_path, output_rows)

    summary = {
        "input_records": input_records,
        "unique_h_vectors": len(output_rows),
        "weights": {
            "current_solver": weight_current,
            "top_level_solver": weight_top_level,
            "structural": weight_structural,
        },
        "score_normalization": normalization_mode,
        "score_distributions": {
            "current_solver_score": {
                "min": min(current_scores) if current_scores else 0.0,
                "max": max(current_scores) if current_scores else 0.0,
            },
            "top_level_solver_score": {
                "min": min(top_level_scores) if top_level_scores else 0.0,
                "max": max(top_level_scores) if top_level_scores else 0.0,
            },
            "structural_score": {
                "min": min(structural_scores) if structural_scores else 0.0,
                "max": max(structural_scores) if structural_scores else 0.0,
            },
        },
        "top_by_current_solver_score": _top_rank(output_rows, "current_solver_score"),
        "top_by_top_level_solver_score": _top_rank(output_rows, "top_level_solver_score"),
        "top_by_structural_score": _top_rank(output_rows, "structural_score"),
        "top_by_combined_score": _top_rank(output_rows, "combined_score"),
    }
    if summary_out is not None:
        dump_json(summary_out, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = benchmark_unique_h_vectors(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        summary_out=Path(args.summary_out) if args.summary_out else None,
        max_unique=args.max_unique,
        current_timeout_sec=args.current_timeout_sec,
        current_num_workers=args.current_num_workers,
        top_timeout_sec=args.top_timeout_sec,
        top_num_workers=args.top_num_workers,
        weight_current=args.weight_current,
        weight_top_level=args.weight_top_level,
        weight_structural=args.weight_structural,
    )
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
