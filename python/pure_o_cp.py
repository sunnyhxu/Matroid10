from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from .common import dump_json, monotonic_seconds
except ImportError:
    from common import dump_json, monotonic_seconds

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_COUNTEREXAMPLE = 17


def enumerate_monomials(n_vars: int, max_degree: int) -> Dict[int, List[Tuple[int, ...]]]:
    buckets: Dict[int, List[Tuple[int, ...]]] = {d: [] for d in range(max_degree + 1)}

    def rec(idx: int, remain: int, prefix: List[int]) -> Iterable[Tuple[int, ...]]:
        if idx == n_vars - 1:
            yield tuple(prefix + [remain])
            return
        for v in range(remain + 1):
            prefix.append(v)
            yield from rec(idx + 1, remain - v, prefix)
            prefix.pop()

    for deg in range(max_degree + 1):
        buckets[deg].extend(rec(0, deg, []))
    return buckets


@dataclass
class CPResult:
    status: str
    wall_time: float
    model_size: Dict[str, int]
    error: str | None = None


def solve_h_vector(h_vector: List[int], timeout_sec: float, num_workers: int) -> CPResult:
    try:
        from ortools.sat.python import cp_model
    except Exception as ex:  # noqa: BLE001
        return CPResult(
            status="ERROR",
            wall_time=0.0,
            model_size={"num_vars": 0, "num_constraints": 0, "num_degree_buckets": 0},
            error=f"ortools_import_failed:{ex}",
        )

    if not h_vector or any(x < 0 for x in h_vector):
        return CPResult(
            status="ERROR",
            wall_time=0.0,
            model_size={"num_vars": 0, "num_constraints": 0, "num_degree_buckets": 0},
            error="invalid_h_vector",
        )

    d = len(h_vector) - 1
    n_vars = h_vector[1] if len(h_vector) > 1 else 1
    n_vars = max(1, n_vars)
    by_deg = enumerate_monomials(n_vars, d)

    for k, hk in enumerate(h_vector):
        if hk > len(by_deg.get(k, [])):
            return CPResult(
                status="INFEASIBLE",
                wall_time=0.0,
                model_size={
                    "num_vars": sum(len(v) for v in by_deg.values()),
                    "num_constraints": 0,
                    "num_degree_buckets": len(by_deg),
                },
                error=f"degree_{k}_count_exceeds_capacity",
            )

    model = cp_model.CpModel()
    x: Dict[Tuple[int, ...], cp_model.IntVar] = {}
    for monomials in by_deg.values():
        for alpha in monomials:
            name = "x_" + "_".join(str(v) for v in alpha)
            x[alpha] = model.NewBoolVar(name)

    constraints = 0
    for k, hk in enumerate(h_vector):
        model.Add(sum(x[a] for a in by_deg[k]) == hk)
        constraints += 1

    for deg, monomials in by_deg.items():
        if deg == 0:
            continue
        for alpha in monomials:
            for i in range(n_vars):
                if alpha[i] == 0:
                    continue
                beta = list(alpha)
                beta[i] -= 1
                model.Add(x[alpha] <= x[tuple(beta)])
                constraints += 1

    for deg, monomials in by_deg.items():
        if deg == d:
            continue
        for alpha in monomials:
            exts = []
            for i in range(n_vars):
                beta = list(alpha)
                beta[i] += 1
                beta_t = tuple(beta)
                if beta_t in x:
                    exts.append(x[beta_t])
            if not exts:
                model.Add(x[alpha] == 0)
            else:
                model.Add(sum(exts) >= x[alpha])
            constraints += 1

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_sec
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    wall_time = float(solver.WallTime())

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cp_status = "FEASIBLE"
    elif status == cp_model.INFEASIBLE:
        cp_status = "INFEASIBLE"
    elif status == cp_model.UNKNOWN:
        cp_status = "UNKNOWN"
    else:
        cp_status = "ERROR"

    return CPResult(
        status=cp_status,
        wall_time=wall_time,
        model_size={
            "num_vars": len(model.Proto().variables),
            "num_constraints": constraints,
            "num_degree_buckets": len(by_deg),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP-SAT pure O-sequence verifier.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--counterexample-out", default=None)
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--stop-on-infeasible", action="store_true", default=True)
    parser.add_argument("--no-stop-on-infeasible", action="store_true", default=False)
    return parser.parse_args()


def write_counterexample(path: str, record: Dict[str, object], result: CPResult, timeout_sec: float, num_workers: int) -> None:
    payload = {
        "record": record,
        "cp_result": {
            "status": result.status,
            "wall_time": result.wall_time,
            "model_size": result.model_size,
            "error": result.error,
        },
        "solver_config": {
            "timeout_seconds": timeout_sec,
            "num_workers": num_workers,
        },
    }
    dump_json(path, payload)


def main() -> int:
    args = parse_args()
    stop_on_infeasible = args.stop_on_infeasible and not args.no_stop_on_infeasible
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = monotonic_seconds()
    considered = 0
    attempted = 0
    feasible = 0
    infeasible = 0
    unknown = 0
    errors = 0
    counterexample_path: str | None = None

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            considered += 1
            record = json.loads(line)
            if record.get("prefilter_status") != "pass_to_cp":
                continue
            attempted += 1
            if args.max_instances > 0 and attempted > args.max_instances:
                break

            h_vector = [int(x) for x in record.get("h_vector", [])]
            result = solve_h_vector(h_vector, timeout_sec=args.timeout_sec, num_workers=args.num_workers)

            if result.status == "FEASIBLE":
                feasible += 1
            elif result.status == "INFEASIBLE":
                infeasible += 1
            elif result.status == "UNKNOWN":
                unknown += 1
            else:
                errors += 1

            output_record = dict(record)
            output_record["cp_status"] = result.status
            output_record["cp_time_sec"] = result.wall_time
            output_record["cp_model_size"] = result.model_size
            if result.error:
                output_record["cp_error"] = result.error

            fout.write(json.dumps(output_record, separators=(",", ":")))
            fout.write("\n")

            if result.status == "INFEASIBLE" and stop_on_infeasible:
                if args.counterexample_out:
                    counterexample_path = args.counterexample_out
                else:
                    counterexample_path = f"artifacts/counterexample_{args.run_id}.json"
                write_counterexample(counterexample_path, output_record, result, args.timeout_sec, args.num_workers)
                elapsed = monotonic_seconds() - start
                stats = {
                    "phase": "pure_o_cp",
                    "considered": considered,
                    "attempted": attempted,
                    "feasible": feasible,
                    "infeasible": infeasible,
                    "unknown": unknown,
                    "errors": errors,
                    "counterexample_path": counterexample_path,
                    "elapsed_seconds": elapsed,
                }
                if args.stats_out:
                    dump_json(args.stats_out, stats)
                print(json.dumps(stats))
                return EXIT_COUNTEREXAMPLE

    elapsed = monotonic_seconds() - start
    stats = {
        "phase": "pure_o_cp",
        "considered": considered,
        "attempted": attempted,
        "feasible": feasible,
        "infeasible": infeasible,
        "unknown": unknown,
        "errors": errors,
        "counterexample_path": counterexample_path,
        "elapsed_seconds": elapsed,
    }
    if args.stats_out:
        dump_json(args.stats_out, stats)
    print(json.dumps(stats))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
