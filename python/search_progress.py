from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .common import utc_now_iso
except ImportError:
    from common import utc_now_iso

MARKER_START = "<!-- SEARCH_PROGRESS:START -->"
MARKER_END = "<!-- SEARCH_PROGRESS:END -->"

CATEGORY_ORDER = ["representable_f2", "representable_f3", "sparse_paving"]
CATEGORY_INFO = {
    "representable_f2": {"label": "Representable $\\mathbb{F}_2$", "method": "Bitset C++ / CP-SAT"},
    "representable_f3": {"label": "Representable $\\mathbb{F}_3$", "method": "GFq C++ / CP-SAT"},
    "sparse_paving": {"label": "Sparse Paving", "method": "Heuristic Search"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest progress chunks and update README Search Progress table.")
    parser.add_argument("--chunks-glob", default="artifacts/progress_chunks/*.json")
    parser.add_argument("--ledger", default="artifacts/search_progress.jsonl")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--write-readme", action="store_true", default=True)
    parser.add_argument("--no-write-readme", action="store_true", default=False)
    return parser.parse_args()


def compute_chunk_key(row: Dict[str, Any]) -> str:
    key_fields = {
        "mode": row.get("mode"),
        "field": row.get("field"),
        "n": row.get("n"),
        "seed": row.get("seed"),
        "shard_index": row.get("shard_index"),
        "shard_count": row.get("shard_count"),
        "trial_start": row.get("trial_start"),
        "trial_stride": row.get("trial_stride"),
        "run_id": row.get("run_id"),
        "phase1_command_index": row.get("phase1_command_index"),
    }
    payload = json.dumps(key_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            f.write("\n")


def upsert_ledger_rows(existing_rows: List[Dict[str, Any]], chunk_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_by_key: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        key = str(row.get("chunk_key", "")).strip()
        if key:
            rows_by_key[key] = row
    for row in chunk_rows:
        key = compute_chunk_key(row)
        row_out = dict(row)
        row_out["chunk_key"] = key
        row_out["ingested_at"] = utc_now_iso()
        rows_by_key[key] = row_out
    return [rows_by_key[k] for k in sorted(rows_by_key.keys())]


def infer_category(row: Dict[str, Any]) -> str:
    category = str(row.get("category", "")).strip().lower()
    if category in CATEGORY_INFO:
        return category
    mode = str(row.get("mode", "")).strip().lower().replace("-", "_")
    field = row.get("field")
    if mode == "sparse_paving":
        return "sparse_paving"
    if mode == "representable":
        if int(field or 0) == 2:
            return "representable_f2"
        if int(field or 0) == 3:
            return "representable_f3"
    return ""


def universe_for_category(category: str, n: int) -> int:
    if category == "representable_f2":
        return 2 ** (n * n)
    if category == "representable_f3":
        return 3 ** (n * n)
    if category == "sparse_paving":
        return 2_500_000_000_000
    return 0


def format_coverage(percent: float) -> str:
    if percent <= 0:
        return "0.000000%"
    if percent < 0.0001:
        return f"{percent:.3e}%"
    if percent < 1:
        return f"{percent:.6f}%"
    return f"{percent:.2f}%"


def aggregate_progress(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for cat in CATEGORY_ORDER:
        summary[cat] = {"category": cat, "attempted": 0, "n": 10}

    for row in rows:
        category = infer_category(row)
        if not category:
            continue
        attempted = int(row.get("candidates", 0))
        n = int(row.get("n", 10))
        summary[category]["attempted"] += attempted
        summary[category]["n"] = n

    out: List[Dict[str, Any]] = []
    for cat in CATEGORY_ORDER:
        item = summary[cat]
        n = int(item["n"])
        attempted = int(item["attempted"])
        universe = universe_for_category(cat, n)
        coverage = (100.0 * attempted / float(universe)) if universe > 0 else 0.0
        if attempted <= 0:
            status = "Not Started"
        elif attempted >= universe:
            status = "Complete"
        else:
            status = "Sampled"
        out.append(
            {
                "category": cat,
                "category_label": CATEGORY_INFO[cat]["label"],
                "method": CATEGORY_INFO[cat]["method"],
                "n": n,
                "attempted": attempted,
                "universe": universe,
                "coverage_percent": coverage,
                "coverage_display": format_coverage(coverage),
                "status": status,
            }
        )
    return out


def render_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "## Search Progress",
        "",
        "| Category | Elements (n) | Status | Method | Coverage |",
        "|---|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category_label']} | {row['n']} | {row['status']} | {row['method']} | {row['coverage_display']} |"
        )
    return "\n".join(lines)


def replace_readme_section(readme_text: str, table_block: str) -> str:
    section = f"{MARKER_START}\n{table_block}\n{MARKER_END}"
    if MARKER_START in readme_text and MARKER_END in readme_text:
        start = readme_text.index(MARKER_START)
        end = readme_text.index(MARKER_END) + len(MARKER_END)
        return readme_text[:start] + section + readme_text[end:]
    readme_text = readme_text.rstrip() + "\n\n"
    return readme_text + section + "\n"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    chunks_glob = str(resolve_repo_path(repo_root, args.chunks_glob))
    ledger_path = resolve_repo_path(repo_root, args.ledger)
    readme_path = resolve_repo_path(repo_root, args.readme)
    write_readme = args.write_readme and not args.no_write_readme

    chunk_rows: List[Dict[str, Any]] = []
    for chunk_file in sorted(glob.glob(chunks_glob)):
        with open(chunk_file, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["source_path"] = str(Path(chunk_file))
        chunk_rows.append(row)
    ordered_rows = upsert_ledger_rows(load_jsonl(ledger_path), chunk_rows)
    ingested = len(chunk_rows)
    dump_jsonl(ledger_path, ordered_rows)

    aggregate = aggregate_progress(ordered_rows)
    table_block = render_table(aggregate)

    if write_readme:
        readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        updated = replace_readme_section(readme_text, table_block)
        readme_path.write_text(updated, encoding="utf-8")

    print(
        json.dumps(
            {
                "chunks_ingested": ingested,
                "ledger_rows": len(ordered_rows),
                "ledger_path": str(ledger_path),
                "readme_updated": bool(write_readme),
                "readme_path": str(readme_path),
            }
        )
    )
    return 0


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
