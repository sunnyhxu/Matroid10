import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.bootstrap import (  # noqa: E402
    BootstrapFormatError,
    bootstrap_regions_from_records,
    load_bootstrap_regions,
)


def _make_bootstrap_row(h_vector, *, record_id, current_status="FEASIBLE", top_status="UNKNOWN", combined_score=0.5):
    return {
        "h_vector": list(h_vector),
        "h_vector_key": json.dumps(list(h_vector), separators=(",", ":")),
        "source_records": 3,
        "example_record_id": record_id,
        "current_solver": {
            "status": current_status,
            "wall_time": 1.5,
            "model_size": {"num_vars": 12},
            "metrics": {"num_conflicts": 4},
            "score_components": {"raw_score": 2.0},
            "score_raw": 2.0,
        },
        "top_level_solver": {
            "status": top_status,
            "wall_time": 2.5,
            "model_size": {"num_vars": 9},
            "metrics": {"num_conflicts": 7},
            "score_components": {"raw_score": 3.0},
            "score_raw": 3.0,
        },
        "structural_metrics": {"raw_score": 4.0, "macaulay_total_slack": 2},
        "current_solver_score": 0.2,
        "top_level_solver_score": 0.7,
        "structural_score": 0.9,
        "combined_score_weights": {"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
        "score_normalization": "percentile",
        "combined_score": combined_score,
        "witness_references": [f"witness:{record_id}"],
    }


def test_bootstrap_preserves_hardness_metrics_statuses_and_witness_references():
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / ".pytest_tmp_manual" / "neuro_symbolic_bootstrap"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "hardness_unique_hvectors.jsonl"
    rows = [
        _make_bootstrap_row([1, 2, 3], record_id="seed-a", current_status="FEASIBLE", top_status="UNKNOWN"),
        _make_bootstrap_row([1, 3, 4, 4], record_id="seed-b", current_status="UNKNOWN", top_status="FEASIBLE"),
    ]
    path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")

    records = load_bootstrap_regions(path)

    assert [record.h_vector for record in records] == [[1, 2, 3], [1, 3, 4, 4]]
    assert records[0].current_solver["status"] == "FEASIBLE"
    assert records[0].top_level_solver["status"] == "UNKNOWN"
    assert records[0].witness_references == ["seed-a", "witness:seed-a"]
    assert records[1].current_solver_score == pytest.approx(0.2)
    assert records[1].structural_metrics["macaulay_total_slack"] == 2


def test_bootstrap_ordering_is_deterministic_for_identical_inputs():
    rows = [
        _make_bootstrap_row([1, 4, 6, 4, 1], record_id="seed-c"),
        _make_bootstrap_row([1, 2, 1], record_id="seed-a"),
        _make_bootstrap_row([1, 3, 3, 1], record_id="seed-b"),
    ]

    left = bootstrap_regions_from_records(rows)
    right = bootstrap_regions_from_records(list(reversed(rows)))

    assert [record.region_key.value for record in left] == [record.region_key.value for record in right]
    assert [record.example_record_id for record in left] == ["seed-a", "seed-b", "seed-c"]


def test_bootstrap_from_records_deduplicates_identical_h_vectors():
    rows = [
        _make_bootstrap_row([1, 2, 3], record_id="seed-a"),
        _make_bootstrap_row([1, 2, 3], record_id="seed-b"),
    ]

    records = bootstrap_regions_from_records(rows)

    assert len(records) == 1
    assert records[0].source_records == 3
    assert records[0].witness_references == ["seed-a", "seed-b", "witness:seed-a", "witness:seed-b"]


def test_bootstrap_malformed_row_raises_typed_error():
    rows = [
        {
            "h_vector": [1, 2, 3],
            "h_vector_key": "[1,2,3]",
            "source_records": 1,
        }
    ]

    with pytest.raises(BootstrapFormatError):
        bootstrap_regions_from_records(rows)
