"""Hash-based deduplication store for accumulated matroid results."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Set


class DedupStore:
    """Manages accumulated results with hash-based deduplication using matroid IDs."""

    def __init__(self, results_path: Path, seen_ids_path: Path):
        self.results_path = results_path
        self.seen_ids_path = seen_ids_path
        self._lock = threading.Lock()
        self._seen_ids: Set[str] = set()
        self._load_seen_ids()

    def _load_seen_ids(self) -> None:
        """Load existing IDs from disk."""
        if self.seen_ids_path.exists():
            try:
                with self.seen_ids_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._seen_ids = set(data.get("ids", []))
            except (json.JSONDecodeError, OSError):
                self._seen_ids = set()

    def _save_seen_ids(self) -> None:
        """Save seen IDs to disk."""
        self.seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.seen_ids_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump({"ids": list(self._seen_ids)}, f)
        try:
            temp_path.replace(self.seen_ids_path)
        except OSError:
            if self.seen_ids_path.exists():
                self.seen_ids_path.unlink()
            temp_path.rename(self.seen_ids_path)

    def append_deduplicated(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Append new records to accumulated results, skipping duplicates.

        Returns stats about the operation.
        """
        with self._lock:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            new_count = 0
            duplicate_count = 0

            with self.results_path.open("a", encoding="utf-8") as f:
                for record in records:
                    record_id = str(record.get("id", ""))
                    if not record_id:
                        continue
                    if record_id in self._seen_ids:
                        duplicate_count += 1
                        continue
                    self._seen_ids.add(record_id)
                    f.write(json.dumps(record, separators=(",", ":")))
                    f.write("\n")
                    new_count += 1

            self._save_seen_ids()
            return {
                "new_records": new_count,
                "duplicates_skipped": duplicate_count,
            }

    def append_from_jsonl(self, jsonl_path: Path) -> Dict[str, int]:
        """
        Append records from a JSONL file, skipping duplicates.

        Returns stats about the operation.
        """
        if not jsonl_path.exists():
            return {"new_records": 0, "duplicates_skipped": 0}

        records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return self.append_deduplicated(records)

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about the accumulated store."""
        with self._lock:
            return {"unique_count": len(self._seen_ids)}

    def contains(self, record_id: str) -> bool:
        """Check if a record ID has already been seen."""
        with self._lock:
            return record_id in self._seen_ids
