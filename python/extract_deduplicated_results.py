from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from .common import dump_json
    from .pure_o_cp import solve_h_vector
except ImportError:
    from common import dump_json
    from pure_o_cp import solve_h_vector


@dataclass
class HVectorAggregate:
    h_vector: list[int]
    example_matroid_id: str
    num_records: int = 0
    matroid_ids: set[str] = field(default_factory=set)
    cp_status_counts: Counter[str] = field(default_factory=Counter)


def h_vector_key(h_vector: Iterable[int]) -> str:
    return json.dumps([int(x) for x in h_vector], separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deduplicated matroids and h-vectors from pipeline output.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--matroids-out", required=True)
    parser.add_argument("--hvectors-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--include-witness", action="store_true", default=False)
    parser.add_argument("--witness-timeout-sec", type=float, default=120.0)
    parser.add_argument("--witness-num-workers", type=int, default=8)
    return parser.parse_args()


def _canonical_status(counts: Counter[str]) -> str:
    if not counts:
        return "UNKNOWN"
    if len(counts) == 1:
        return next(iter(counts))
    return "MIXED"


def extract_deduplicated_results(
    *,
    input_path: Path,
    matroids_out: Path,
    hvectors_out: Path,
    summary_out: Path,
    include_witness: bool = False,
    witness_timeout_sec: float = 120.0,
    witness_num_workers: int = 8,
) -> Dict[str, int]:
    input_path = Path(input_path)
    matroids_out = Path(matroids_out)
    hvectors_out = Path(hvectors_out)
    summary_out = Path(summary_out)

    matroids_out.parent.mkdir(parents=True, exist_ok=True)
    hvectors_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    seen_matroid_ids: set[str] = set()
    hvector_aggregates: Dict[str, HVectorAggregate] = {}

    input_records = 0
    duplicate_matroid_records_skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, matroids_out.open("w", encoding="utf-8") as matroids_fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            input_records += 1
            record = json.loads(line)

            matroid_id = str(record.get("id", "")).strip()
            h_vector = [int(x) for x in record.get("h_vector", [])]
            key = h_vector_key(h_vector)

            aggregate = hvector_aggregates.get(key)
            if aggregate is None:
                aggregate = HVectorAggregate(h_vector=h_vector, example_matroid_id=matroid_id)
                hvector_aggregates[key] = aggregate
            aggregate.num_records += 1
            if matroid_id:
                aggregate.matroid_ids.add(matroid_id)
            aggregate.cp_status_counts[str(record.get("cp_status", "UNKNOWN"))] += 1

            if matroid_id in seen_matroid_ids:
                duplicate_matroid_records_skipped += 1
                continue

            seen_matroid_ids.add(matroid_id)
            matroids_fout.write(json.dumps(record, separators=(",", ":")))
            matroids_fout.write("\n")

    feasible_h_vectors = 0
    infeasible_h_vectors = 0
    unknown_h_vectors = 0
    error_h_vectors = 0

    with hvectors_out.open("w", encoding="utf-8") as hvectors_fout:
        for key in sorted(hvector_aggregates.keys()):
            aggregate = hvector_aggregates[key]
            status = _canonical_status(aggregate.cp_status_counts)

            if status == "FEASIBLE":
                feasible_h_vectors += 1
            elif status == "INFEASIBLE":
                infeasible_h_vectors += 1
            elif status == "ERROR":
                error_h_vectors += 1
            else:
                unknown_h_vectors += 1

            record_out: Dict[str, Any] = {
                "h_vector": aggregate.h_vector,
                "h_vector_key": key,
                "cp_status": status,
                "cp_status_counts": dict(sorted(aggregate.cp_status_counts.items())),
                "num_records": aggregate.num_records,
                "num_unique_matroids": len(aggregate.matroid_ids),
                "example_matroid_id": aggregate.example_matroid_id,
            }

            if include_witness and status == "FEASIBLE":
                witness_result = solve_h_vector(
                    aggregate.h_vector,
                    timeout_sec=witness_timeout_sec,
                    num_workers=witness_num_workers,
                    emit_witness=True,
                )
                if witness_result.status == "FEASIBLE" and witness_result.witness is not None:
                    record_out["pure_o_witness"] = witness_result.witness
                elif witness_result.error:
                    record_out["pure_o_witness_error"] = witness_result.error

            hvectors_fout.write(json.dumps(record_out, separators=(",", ":")))
            hvectors_fout.write("\n")

    summary = {
        "input_records": input_records,
        "unique_matroids": len(seen_matroid_ids),
        "duplicate_matroid_records_skipped": duplicate_matroid_records_skipped,
        "unique_h_vectors": len(hvector_aggregates),
        "feasible_h_vectors": feasible_h_vectors,
        "infeasible_h_vectors": infeasible_h_vectors,
        "unknown_h_vectors": unknown_h_vectors,
        "error_h_vectors": error_h_vectors,
    }
    dump_json(summary_out, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = extract_deduplicated_results(
        input_path=Path(args.in_path),
        matroids_out=Path(args.matroids_out),
        hvectors_out=Path(args.hvectors_out),
        summary_out=Path(args.summary_out),
        include_witness=args.include_witness,
        witness_timeout_sec=args.witness_timeout_sec,
        witness_num_workers=args.witness_num_workers,
    )
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
