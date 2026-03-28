from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .common import dump_json
    from .pure_o_cp import solve_h_vector
except ImportError:
    from common import dump_json
    from pure_o_cp import solve_h_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach pure O-sequence witnesses to deduplicated h-vectors.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def attach_pure_o_witnesses(
    *,
    input_path: Path,
    output_path: Path,
    summary_out: Path | None = None,
    timeout_sec: float = 120.0,
    num_workers: int = 8,
) -> Dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if summary_out is not None:
        summary_out = Path(summary_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)

    output_tmp = output_path.with_name(output_path.name + ".tmp")

    input_h_vectors = 0
    feasible_input_h_vectors = 0
    witnesses_written = 0
    witness_errors = 0
    witness_infeasible = 0
    witness_unknown = 0

    with input_path.open("r", encoding="utf-8") as fin, output_tmp.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            input_h_vectors += 1
            record = json.loads(line)
            record.pop("pure_o_witness", None)
            record.pop("pure_o_witness_error", None)
            record.pop("pure_o_witness_status", None)

            if record.get("cp_status") == "FEASIBLE":
                feasible_input_h_vectors += 1
                h_vector = [int(x) for x in record.get("h_vector", [])]
                result = solve_h_vector(
                    h_vector,
                    timeout_sec=timeout_sec,
                    num_workers=num_workers,
                    emit_witness=True,
                )
                if result.status == "FEASIBLE" and result.witness is not None:
                    record["pure_o_witness"] = result.witness
                    witnesses_written += 1
                elif result.error:
                    record["pure_o_witness_error"] = result.error
                    witness_errors += 1
                else:
                    record["pure_o_witness_status"] = result.status
                    if result.status == "INFEASIBLE":
                        witness_infeasible += 1
                    else:
                        witness_unknown += 1

            fout.write(json.dumps(record, separators=(",", ":")))
            fout.write("\n")

    output_tmp.replace(output_path)

    summary = {
        "input_h_vectors": input_h_vectors,
        "feasible_input_h_vectors": feasible_input_h_vectors,
        "witnesses_written": witnesses_written,
        "witness_errors": witness_errors,
        "witness_infeasible": witness_infeasible,
        "witness_unknown": witness_unknown,
    }
    if summary_out is not None:
        dump_json(summary_out, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = attach_pure_o_witnesses(
        input_path=Path(args.in_path),
        output_path=Path(args.out_path),
        summary_out=Path(args.summary_out) if args.summary_out else None,
        timeout_sec=args.timeout_sec,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
