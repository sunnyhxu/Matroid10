"""Tests for DedupStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_paths(tmp_path: Path):
    return {
        "results": tmp_path / "accumulated_results.jsonl",
        "seen_ids": tmp_path / "seen_ids.json",
    }


@pytest.fixture
def dedup_store(temp_paths):
    from python.web.dedup_store import DedupStore
    return DedupStore(temp_paths["results"], temp_paths["seen_ids"])


class TestDedupStore:
    def test_initial_stats_are_zero(self, dedup_store):
        stats = dedup_store.get_stats()
        assert stats["unique_count"] == 0

    def test_append_deduplicated_adds_new_records(self, dedup_store, temp_paths):
        records = [
            {"id": "abc123", "data": "value1"},
            {"id": "def456", "data": "value2"},
        ]
        result = dedup_store.append_deduplicated(records)
        assert result["new_records"] == 2
        assert result["duplicates_skipped"] == 0
        assert dedup_store.get_stats()["unique_count"] == 2

        # Check file contents
        with temp_paths["results"].open() as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_append_deduplicated_skips_duplicates(self, dedup_store):
        records1 = [{"id": "abc123", "data": "value1"}]
        records2 = [{"id": "abc123", "data": "different"}, {"id": "new456", "data": "value2"}]

        dedup_store.append_deduplicated(records1)
        result = dedup_store.append_deduplicated(records2)

        assert result["new_records"] == 1
        assert result["duplicates_skipped"] == 1
        assert dedup_store.get_stats()["unique_count"] == 2

    def test_append_deduplicated_skips_empty_ids(self, dedup_store):
        records = [
            {"id": "", "data": "empty_id"},
            {"data": "no_id"},
            {"id": "valid123", "data": "has_id"},
        ]
        result = dedup_store.append_deduplicated(records)
        assert result["new_records"] == 1

    def test_seen_ids_persist_to_disk(self, temp_paths):
        from python.web.dedup_store import DedupStore

        store1 = DedupStore(temp_paths["results"], temp_paths["seen_ids"])
        store1.append_deduplicated([{"id": "persist123", "data": "test"}])

        # Create new instance to test persistence
        store2 = DedupStore(temp_paths["results"], temp_paths["seen_ids"])
        assert store2.get_stats()["unique_count"] == 1
        assert store2.contains("persist123")

    def test_contains_returns_correct_values(self, dedup_store):
        dedup_store.append_deduplicated([{"id": "exists123", "data": "test"}])
        assert dedup_store.contains("exists123") is True
        assert dedup_store.contains("notexists") is False

    def test_append_from_jsonl(self, dedup_store, tmp_path):
        # Create a temp JSONL file
        jsonl_path = tmp_path / "input.jsonl"
        with jsonl_path.open("w") as f:
            f.write('{"id": "fromfile1", "data": "a"}\n')
            f.write('{"id": "fromfile2", "data": "b"}\n')
            f.write('{"id": "fromfile1", "data": "duplicate"}\n')

        result = dedup_store.append_from_jsonl(jsonl_path)
        assert result["new_records"] == 2
        assert result["duplicates_skipped"] == 1

    def test_append_from_jsonl_handles_missing_file(self, dedup_store, tmp_path):
        result = dedup_store.append_from_jsonl(tmp_path / "nonexistent.jsonl")
        assert result["new_records"] == 0
        assert result["duplicates_skipped"] == 0

    def test_append_from_jsonl_handles_malformed_lines(self, dedup_store, tmp_path):
        jsonl_path = tmp_path / "malformed.jsonl"
        with jsonl_path.open("w") as f:
            f.write('{"id": "valid1", "data": "ok"}\n')
            f.write('not valid json\n')
            f.write('{"id": "valid2", "data": "ok"}\n')

        result = dedup_store.append_from_jsonl(jsonl_path)
        assert result["new_records"] == 2

    def test_results_file_format(self, dedup_store, temp_paths):
        records = [{"id": "format_test", "value": 123, "nested": {"key": "val"}}]
        dedup_store.append_deduplicated(records)

        with temp_paths["results"].open() as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["id"] == "format_test"
        assert parsed["value"] == 123
        assert parsed["nested"]["key"] == "val"

    def test_seen_ids_file_format(self, dedup_store, temp_paths):
        dedup_store.append_deduplicated([
            {"id": "id1", "data": "a"},
            {"id": "id2", "data": "b"},
        ])

        with temp_paths["seen_ids"].open() as f:
            data = json.load(f)
        assert "ids" in data
        assert set(data["ids"]) == {"id1", "id2"}

    def test_atomic_seen_ids_save(self, dedup_store, temp_paths):
        dedup_store.append_deduplicated([{"id": "atomic_test", "data": "x"}])
        # Check no temp files left
        assert not temp_paths["seen_ids"].with_suffix(".tmp").exists()

    def test_handles_corrupted_seen_ids(self, temp_paths):
        from python.web.dedup_store import DedupStore

        # Write corrupted JSON
        temp_paths["seen_ids"].parent.mkdir(parents=True, exist_ok=True)
        with temp_paths["seen_ids"].open("w") as f:
            f.write("not valid json")

        # Should recover gracefully
        store = DedupStore(temp_paths["results"], temp_paths["seen_ids"])
        assert store.get_stats()["unique_count"] == 0

    def test_multiple_appends_accumulate(self, dedup_store):
        for i in range(5):
            dedup_store.append_deduplicated([{"id": f"batch{i}", "data": str(i)}])

        assert dedup_store.get_stats()["unique_count"] == 5
