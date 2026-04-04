import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python import hardness_benchmark, hardness_metrics, pure_o_cp


def test_compute_structural_metrics_exposes_expected_fields():
    metrics = hardness_metrics.compute_structural_metrics([1, 3, 6, 10, 14, 15, 12, 6])

    assert metrics["degree"] == 7
    assert metrics["n_vars"] == 3
    assert metrics["macaulay_total_slack"] >= 0
    assert metrics["macaulay_total_violation"] >= 0
    assert metrics["late_drop_sum"] > 0
    assert metrics["raw_score"] > 0


def test_percentile_normalize_handles_ties():
    scores = hardness_metrics.percentile_normalize([1.0, 1.0, 3.0])
    assert scores[0] == scores[1]
    assert scores[2] > scores[0]


def test_benchmark_unique_h_vectors_deduplicates_and_emits_category_scores(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tmp_path = repo_root / ".pytest_tmp_manual" / "hardness_benchmark"
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "hvec.jsonl"
    output_path = tmp_path / "hardness.jsonl"
    summary_path = tmp_path / "summary.json"

    records = [
        {"id": "a", "h_vector": [1, 2, 3]},
        {"id": "b", "h_vector": [1, 2, 3]},
        {"id": "c", "h_vector": [1, 3, 4, 4]},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )

    def fake_current(h_vector, timeout_sec, num_workers):
        return pure_o_cp.CPResult(
            status="FEASIBLE",
            wall_time=0.1 * len(h_vector),
            model_size={"num_vars": 10, "num_constraints": 20, "num_degree_buckets": len(h_vector)},
            solver_stats={"wall_time": 0.1 * len(h_vector), "num_conflicts": len(h_vector), "num_branches": 2 * len(h_vector)},
        )

    def fake_top_level(h_vector, timeout_sec, num_workers):
        return pure_o_cp.CPResult(
            status="FEASIBLE",
            wall_time=0.05 * len(h_vector),
            model_size={
                "num_vars": 7,
                "num_constraints": 11,
                "num_degree_buckets": len(h_vector),
                "num_top_vars": 3,
                "num_coverage_vars": 4,
            },
            solver_stats={"wall_time": 0.05 * len(h_vector), "num_conflicts": 1, "num_branches": len(h_vector)},
        )

    monkeypatch.setattr(hardness_benchmark, "solve_h_vector", fake_current)
    monkeypatch.setattr(hardness_benchmark, "solve_h_vector_top_level", fake_top_level)

    summary = hardness_benchmark.benchmark_unique_h_vectors(
        input_path=input_path,
        output_path=output_path,
        summary_out=summary_path,
        weight_current=2.0,
        weight_top_level=1.0,
        weight_structural=3.0,
    )

    assert summary["unique_h_vectors"] == 2

    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(output_rows) == 2
    assert output_rows[0]["current_solver_score"] >= 0.0
    assert output_rows[0]["top_level_solver_score"] >= 0.0
    assert output_rows[0]["structural_score"] >= 0.0
    assert output_rows[0]["combined_score"] >= 0.0
    assert output_rows[0]["combined_score_weights"] == {
        "current_solver": 2.0,
        "top_level_solver": 1.0,
        "structural": 3.0,
    }

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["top_by_combined_score"]


def test_top_level_solver_matches_reference_on_simple_vectors():
    pytest.importorskip("ortools")

    from python.pure_o_cp import solve_h_vector
    from python.pure_o_top_level_cp import solve_h_vector_top_level

    for h_vector in ([1, 2, 2], [1, 2, 3], [1, 2, 4]):
        reference = solve_h_vector(list(h_vector), timeout_sec=5.0, num_workers=1)
        top_level = solve_h_vector_top_level(list(h_vector), timeout_sec=5.0, num_workers=1)
        assert top_level.status == reference.status
