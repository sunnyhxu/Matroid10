import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python import pure_o_cp


def test_main_caches_duplicate_h_vectors(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tmp_path = repo_root / ".pytest_tmp_manual" / "pure_o_cp"
    tmp_path.mkdir(parents=True, exist_ok=True)
    in_path = tmp_path / "hvec.jsonl"
    out_path = tmp_path / "results.jsonl"
    stats_path = tmp_path / "stats.json"

    records = [
        {"id": "a", "prefilter_status": "pass_to_cp", "h_vector": [1, 2, 3]},
        {"id": "b", "prefilter_status": "pass_to_cp", "h_vector": [1, 2, 3]},
        {"id": "c", "prefilter_status": "pass_to_cp", "h_vector": [1, 3, 5]},
    ]
    in_path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )

    calls = []

    def fake_solve(h_vector, timeout_sec, num_workers):
        calls.append(tuple(h_vector))
        return pure_o_cp.CPResult(
            status="FEASIBLE",
            wall_time=0.01,
            model_size={"num_vars": 1, "num_constraints": 1, "num_degree_buckets": 1},
            solver_stats={"wall_time": 0.01, "num_conflicts": 2, "num_branches": 3},
        )

    monkeypatch.setattr(pure_o_cp, "solve_h_vector", fake_solve)
    monkeypatch.setattr(
        pure_o_cp,
        "parse_args",
        lambda: argparse.Namespace(
            in_path=str(in_path),
            out_path=str(out_path),
            timeout_sec=1.0,
            num_workers=1,
            max_instances=0,
            counterexample_out=None,
            run_id="test",
            stats_out=str(stats_path),
            stop_on_infeasible=True,
            no_stop_on_infeasible=False,
        ),
    )

    exit_code = pure_o_cp.main()

    assert exit_code == pure_o_cp.EXIT_OK
    assert calls == [(1, 2, 3), (1, 3, 5)]

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["attempted"] == 3
    assert stats["unique_h_vectors_attempted"] == 2
    assert stats["cache_hits"] == 1

    output_records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [record["cp_status"] for record in output_records] == ["FEASIBLE", "FEASIBLE", "FEASIBLE"]
    assert output_records[0]["cp_solver_stats"]["num_conflicts"] == 2
