from __future__ import annotations

import argparse
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List


KEEP_EXACT = {
    "accumulated_results.jsonl",
    "dedup/hvectors_sparse_paving_recovery_20260327.jsonl",
    "dedup/matroids_sparse_paving_recovery_20260327.jsonl",
    "dedup/summary_sparse_paving_recovery_20260327.json",
    "dedup/witness_summary_sparse_paving_recovery_20260327.json",
    "hvec.jsonl",
    "metrics.json",
    "non_paving.jsonl",
    "non_paving_field2.jsonl",
    "non_paving_field3.jsonl",
    "phase_stats/gen_field2.json",
    "phase_stats/gen_field3.json",
    "phase_stats/gen_sparse_paving.json",
    "phase_stats/hvec_extract.json",
    "phase_stats/hvec_extract_sparse_paving_recovery_20260327.json",
    "phase_stats/pure_o_cp.json",
    "phase_stats/pure_o_cp_sparse_paving_recovery_20260327.json",
    "progress_chunks/matroid10_20260326T023701Z_representable_f2.json",
    "progress_chunks/matroid10_20260326T023701Z_representable_f3.json",
    "progress_chunks/sparse_paving_recovery_20260327.json",
    "pure_o_results.jsonl",
    "run_manifest.json",
    "run_state.json",
    "search_progress.jsonl",
    "seen_ids.json",
}

DELETE_EXACT = {
    "non_paving_field2_connect_test.jsonl": "field-2 connectivity experiment artifact",
    "phase_stats/gen_field2_connect_test.json": "field-2 connectivity experiment stats",
    "phase_stats/hvec_extract_matroid10_20260304T015608Z.json": "timestamped duplicate representable h-vector stats",
    "phase_stats/hvec_extract_sparse_probe_1000.json": "sparse probe h-vector stats",
    "phase_stats/pure_o_cp_matroid10_20260304T015608Z.json": "timestamped duplicate representable CP stats",
    "phase_stats/pure_o_cp_smoke_check.json": "smoke-run CP stats",
    "phase_stats/pure_o_cp_sparse_probe_1000.json": "sparse probe CP stats",
    "phase_stats/pure_o_cp_sparse_probe_1000_cached.json": "cached sparse probe CP stats",
}

DELETE_PATTERNS = {
    "*.stderr.log": "stderr log",
    "*.stdout.log": "stdout log",
    "*.log": "runtime log",
    "hvec_matroid10_*.jsonl": "timestamped duplicate representable h-vector export",
    "hvec_sparse_*.jsonl": "bulky sparse intermediate h-vector export",
    "non_paving_sparse_*.jsonl": "bulky sparse intermediate matroid export",
    "pure_o_results_matroid10_*.jsonl": "timestamped duplicate representable pure O export",
    "pure_o_results_smoke_*.jsonl": "smoke-run pure O export",
    "pure_o_results_sparse_*.jsonl": "bulky sparse intermediate pure O export",
}


@dataclass(frozen=True)
class CleanupAction:
    path: Path
    size: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean bulky experimental artifacts while preserving canonical representable and deduplicated outputs."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--apply", action="store_true", default=False)
    return parser.parse_args()


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def classify_artifact(path: Path, root: Path) -> str | None:
    rel = _relative_posix(path, root)
    if rel in KEEP_EXACT:
        return None
    if rel in DELETE_EXACT:
        return DELETE_EXACT[rel]

    if rel.startswith("dedup/"):
        return None
    if rel.startswith("progress_chunks/"):
        return None

    for pattern, reason in DELETE_PATTERNS.items():
        if fnmatch(rel, pattern):
            return reason
    return None


def build_cleanup_plan(artifacts_dir: Path) -> List[CleanupAction]:
    artifacts_dir = Path(artifacts_dir)
    actions: List[CleanupAction] = []
    for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
        reason = classify_artifact(path, artifacts_dir)
        if reason is None:
            continue
        actions.append(CleanupAction(path=path, size=path.stat().st_size, reason=reason))
    return actions


def apply_cleanup(actions: Iterable[CleanupAction]) -> int:
    reclaimed = 0
    for action in actions:
        reclaimed += action.size
        action.path.unlink(missing_ok=True)
    return reclaimed


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    actions = build_cleanup_plan(artifacts_dir)
    total_bytes = sum(action.size for action in actions)

    print(f"cleanup_candidates={len(actions)} total_bytes={total_bytes} ({_format_size(total_bytes)})")
    for action in actions:
        rel = _relative_posix(action.path, artifacts_dir)
        print(f"{rel}\t{action.reason}\t{action.size}")

    if args.apply:
        reclaimed = apply_cleanup(actions)
        print(f"deleted={len(actions)} reclaimed_bytes={reclaimed} ({_format_size(reclaimed)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
