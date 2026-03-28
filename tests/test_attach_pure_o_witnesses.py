import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python import attach_pure_o_witnesses as attach_witnesses
from python.pure_o_cp import CPResult


def test_attach_witnesses_rewrites_stale_errors_and_preserves_non_feasible(tmp_path: Path, monkeypatch) -> None:
    hvectors_path = tmp_path / "hvectors.jsonl"
    summary_path = tmp_path / "witness_summary.json"
    hvectors_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "h_vector": [1, 2, 1],
                        "cp_status": "FEASIBLE",
                        "pure_o_witness_error": "ortools_import_failed:old",
                    },
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "h_vector": [1, 3, 3, 1],
                        "cp_status": "UNKNOWN",
                    },
                    separators=(",", ":"),
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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

    monkeypatch.setattr(attach_witnesses, "solve_h_vector", fake_solve)

    summary = attach_witnesses.attach_pure_o_witnesses(
        input_path=hvectors_path,
        output_path=hvectors_path,
        summary_out=summary_path,
        timeout_sec=30.0,
        num_workers=2,
    )

    records = [json.loads(line) for line in hvectors_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["pure_o_witness"] == {
        "n_vars": 2,
        "max_degree": 2,
        "selected_monomials": [[0, 0], [1, 0], [0, 1]],
    }
    assert "pure_o_witness_error" not in records[0]
    assert "pure_o_witness" not in records[1]
    assert calls == [(1, 2, 1)]
    assert summary == {
        "input_h_vectors": 2,
        "feasible_input_h_vectors": 1,
        "witnesses_written": 1,
        "witness_errors": 0,
        "witness_infeasible": 0,
        "witness_unknown": 0,
    }
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
