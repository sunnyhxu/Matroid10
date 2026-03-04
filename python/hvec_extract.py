from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any, Dict, List

try:
    from .common import dump_json, load_toml, monotonic_seconds
except ImportError:
    from common import dump_json, load_toml, monotonic_seconds


def bitmask_to_base(mask: int, n: int) -> tuple[int, ...]:
    return tuple(i for i in range(n) if (mask >> i) & 1)


def h_from_f_vector(f_vector: List[int], rank: int) -> List[int]:
    # Matroid.f_vector() returns counts of independent sets by size:
    # [f_0, f_1, ..., f_rank], where f_0 = 1 for the empty set.
    f = [int(x) for x in f_vector]
    if not f:
        f = [1]
    out: List[int] = []
    for i in range(rank + 1):
        total = 0
        for j in range(i + 1):
            fj = f[j] if j < len(f) else 0
            total += ((-1) ** (i - j)) * comb(rank - j, i - j) * fj
        out.append(int(total))
    return out


def h_from_tutte_eval_one(matroid: Any, rank: int) -> List[int]:
    # For a rank-r matroid, h_i is the coefficient of x^(r-i) in T_M(x,1).
    tutte = matroid.tutte_polynomial()
    tutte_x = tutte.subs(y=1)
    x_var = tutte.parent().gens()[0]
    out: List[int] = []
    for i in range(rank + 1):
        out.append(int(tutte_x.coefficient({x_var: rank - i})))
    return out


def prefilter_h_vector(
    h_vector: List[int], check_h1_le_h2: bool = True, check_h_rank_positive: bool = True
) -> List[str]:
    reasons: List[str] = []
    if not h_vector:
        reasons.append("empty_h_vector")
        return reasons
    if h_vector[0] != 1:
        reasons.append("h0_not_one")
    if any(x < 0 for x in h_vector):
        reasons.append("negative_entry")
    if check_h1_le_h2 and len(h_vector) >= 3 and h_vector[1] > h_vector[2]:
        reasons.append("h1_gt_h2")
    if check_h_rank_positive and h_vector[-1] <= 0:
        reasons.append("h_rank_nonpositive")
    return reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract h-vectors from matroid bases with Sage.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--stats-out", dest="stats_out", default=None)
    parser.add_argument("--check-formula", action="store_true", default=False)
    parser.add_argument("--no-check-formula", action="store_true", default=False)
    parser.add_argument("--check-h1-le-h2", action="store_true", default=False)
    parser.add_argument("--no-check-h1-le-h2", action="store_true", default=False)
    parser.add_argument("--check-h-rank-positive", action="store_true", default=False)
    parser.add_argument("--no-check-h-rank-positive", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_formula = args.check_formula
    check_h1_le_h2 = True
    check_h_rank_positive = True
    if args.config_path:
        cfg = load_toml(args.config_path)
        hcfg = cfg.get("hvec", {})
        check_formula = bool(hcfg.get("check_h_formula", check_formula))
        check_h1_le_h2 = bool(hcfg.get("check_h1_le_h2", check_h1_le_h2))
        check_h_rank_positive = bool(hcfg.get("check_h_rank_positive", check_h_rank_positive))
    if args.no_check_formula:
        check_formula = False
    if args.check_h1_le_h2:
        check_h1_le_h2 = True
    if args.no_check_h1_le_h2:
        check_h1_le_h2 = False
    if args.check_h_rank_positive:
        check_h_rank_positive = True
    if args.no_check_h_rank_positive:
        check_h_rank_positive = False

    start = monotonic_seconds()
    try:
        from sage.all import Matroid
    except Exception as ex:
        raise RuntimeError(
            "Sage import failed. Run this script via `sage -python python/hvec_extract.py ...`."
        ) from ex

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    reject_precheck = 0
    pass_to_cp = 0
    errors = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            record_out: Dict[str, Any] = dict(record)
            reasons: List[str] = []
            try:
                n = int(record["n"])
                bases = [int(x) for x in record["bases"]]
                base_sets = [bitmask_to_base(mask, n) for mask in bases]
                M = Matroid(groundset=list(range(n)), bases=base_sets)
                rank = int(record["rank"])
                f_vec = [int(x) for x in M.f_vector()]
                h_vec = h_from_f_vector(f_vec, rank)
                if check_formula:
                    h_tutte = h_from_tutte_eval_one(M, rank)
                    if h_tutte != h_vec:
                        reasons.append("h_formula_mismatch")
                reasons.extend(
                    prefilter_h_vector(
                        h_vec,
                        check_h1_le_h2=check_h1_le_h2,
                        check_h_rank_positive=check_h_rank_positive,
                    )
                )
                record_out["h_vector"] = h_vec
                if reasons:
                    record_out["prefilter_status"] = "reject_precheck"
                    reject_precheck += 1
                else:
                    record_out["prefilter_status"] = "pass_to_cp"
                    pass_to_cp += 1
                record_out["prefilter_reasons"] = reasons
            except Exception as ex:  # noqa: BLE001
                errors += 1
                record_out["h_vector"] = []
                record_out["prefilter_status"] = "error"
                record_out["prefilter_reasons"] = [f"exception:{type(ex).__name__}:{ex}"]
            fout.write(json.dumps(record_out, separators=(",", ":")))
            fout.write("\n")

    elapsed = monotonic_seconds() - start
    stats = {
        "phase": "hvec_extract",
        "input_records": total,
        "reject_precheck": reject_precheck,
        "pass_to_cp": pass_to_cp,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "check_formula": check_formula,
        "check_h1_le_h2": check_h1_le_h2,
        "check_h_rank_positive": check_h_rank_positive,
    }
    if args.stats_out:
        dump_json(args.stats_out, stats)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
