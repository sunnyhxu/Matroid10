import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python import extract_deduplicated_results as extract_results
from python.pure_o_cp import CPResult


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_deduplicates_matroids_and_h_vectors(tmp_path: Path) -> None:
    input_path = tmp_path / "pure_o_results.jsonl"
    matroids_out = tmp_path / "dedup_matroids.jsonl"
    hvectors_out = tmp_path / "dedup_hvectors.jsonl"
    summary_out = tmp_path / "summary.json"

    records = [
        {"id": "m1", "h_vector": [1, 2, 1], "cp_status": "FEASIBLE", "rank": 3, "bases": [1, 2]},
        {"id": "m1", "h_vector": [1, 2, 1], "cp_status": "FEASIBLE", "rank": 3, "bases": [1, 2]},
        {"id": "m2", "h_vector": [1, 2, 1], "cp_status": "FEASIBLE", "rank": 3, "bases": [3, 4]},
        {"id": "m3", "h_vector": [1, 3, 3, 1], "cp_status": "UNKNOWN", "rank": 4, "bases": [5, 6]},
    ]
    _write_jsonl(input_path, records)

    extract_results.extract_deduplicated_results(
        input_path=input_path,
        matroids_out=matroids_out,
        hvectors_out=hvectors_out,
        summary_out=summary_out,
    )

    dedup_matroids = [json.loads(line) for line in matroids_out.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in dedup_matroids] == ["m1", "m2", "m3"]

    dedup_hvectors = [json.loads(line) for line in hvectors_out.read_text(encoding="utf-8").splitlines()]
    assert dedup_hvectors == [
        {
            "h_vector": [1, 2, 1],
            "h_vector_key": "[1,2,1]",
            "cp_status": "FEASIBLE",
            "cp_status_counts": {"FEASIBLE": 3},
            "num_records": 3,
            "num_unique_matroids": 2,
            "example_matroid_id": "m1",
        },
        {
            "h_vector": [1, 3, 3, 1],
            "h_vector_key": "[1,3,3,1]",
            "cp_status": "UNKNOWN",
            "cp_status_counts": {"UNKNOWN": 1},
            "num_records": 1,
            "num_unique_matroids": 1,
            "example_matroid_id": "m3",
        },
    ]

    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary == {
        "input_records": 4,
        "unique_matroids": 3,
        "duplicate_matroid_records_skipped": 1,
        "unique_h_vectors": 2,
        "feasible_h_vectors": 1,
        "infeasible_h_vectors": 0,
        "unknown_h_vectors": 1,
        "error_h_vectors": 0,
    }


def test_extract_can_add_pure_o_witnesses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "pure_o_results.jsonl"
    matroids_out = tmp_path / "dedup_matroids.jsonl"
    hvectors_out = tmp_path / "dedup_hvectors.jsonl"
    summary_out = tmp_path / "summary.json"

    records = [
        {"id": "m1", "h_vector": [1, 2, 1], "cp_status": "FEASIBLE"},
        {"id": "m2", "h_vector": [1, 2, 1], "cp_status": "FEASIBLE"},
        {"id": "m3", "h_vector": [1, 3, 3, 1], "cp_status": "UNKNOWN"},
    ]
    _write_jsonl(input_path, records)

    calls: list[tuple[int, ...]] = []

    def fake_solve(h_vector: list[int], timeout_sec: float, num_workers: int, emit_witness: bool = False) -> CPResult:
        calls.append(tuple(h_vector))
        return CPResult(
            status="FEASIBLE",
            wall_time=0.1,
            model_size={"num_vars": 1, "num_constraints": 1, "num_degree_buckets": 1},
            witness={
                "n_vars": 2,
                "max_degree": len(h_vector) - 1,
                "selected_monomials": [[0, 0], [1, 0], [0, 1]],
            },
        )

    monkeypatch.setattr(extract_results, "solve_h_vector", fake_solve)

    extract_results.extract_deduplicated_results(
        input_path=input_path,
        matroids_out=matroids_out,
        hvectors_out=hvectors_out,
        summary_out=summary_out,
        include_witness=True,
        witness_timeout_sec=30.0,
        witness_num_workers=2,
    )

    dedup_hvectors = [json.loads(line) for line in hvectors_out.read_text(encoding="utf-8").splitlines()]
    assert dedup_hvectors[0]["pure_o_witness"] == {
        "n_vars": 2,
        "max_degree": 2,
        "selected_monomials": [[0, 0], [1, 0], [0, 1]],
    }
    assert "pure_o_witness" not in dedup_hvectors[1]
    assert calls == [(1, 2, 1)]
