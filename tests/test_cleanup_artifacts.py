import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python import cleanup_artifacts


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_cleanup_plan_keeps_canonical_and_dedup_outputs(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"

    keep_paths = [
        artifacts / "non_paving.jsonl",
        artifacts / "hvec.jsonl",
        artifacts / "pure_o_results.jsonl",
        artifacts / "non_paving_field2.jsonl",
        artifacts / "non_paving_field3.jsonl",
        artifacts / "dedup" / "matroids_sparse_paving_recovery_20260327.jsonl",
        artifacts / "dedup" / "hvectors_sparse_paving_recovery_20260327.jsonl",
        artifacts / "phase_stats" / "gen_field2.json",
        artifacts / "progress_chunks" / "sparse_paving_recovery_20260327.json",
    ]
    delete_paths = [
        artifacts / "non_paving_sparse_paving.jsonl",
        artifacts / "hvec_sparse_paving_recovery_20260327.jsonl",
        artifacts / "pure_o_results_sparse_paving_recovery_20260327.jsonl",
        artifacts / "hvec_matroid10_20260304T015608Z.jsonl",
        artifacts / "phase_stats" / "pure_o_cp_sparse_probe_1000.json",
        artifacts / "sparse_paving_recovery_20260327.stdout.log",
    ]

    for path in keep_paths + delete_paths:
        _touch(path)

    actions = cleanup_artifacts.build_cleanup_plan(artifacts)
    delete_rel_paths = {action.path.relative_to(artifacts).as_posix() for action in actions}

    assert "non_paving_sparse_paving.jsonl" in delete_rel_paths
    assert "hvec_sparse_paving_recovery_20260327.jsonl" in delete_rel_paths
    assert "pure_o_results_sparse_paving_recovery_20260327.jsonl" in delete_rel_paths
    assert "hvec_matroid10_20260304T015608Z.jsonl" in delete_rel_paths
    assert "phase_stats/pure_o_cp_sparse_probe_1000.json" in delete_rel_paths
    assert "sparse_paving_recovery_20260327.stdout.log" in delete_rel_paths
    assert "non_paving.jsonl" not in delete_rel_paths
    assert "hvec.jsonl" not in delete_rel_paths
    assert "pure_o_results.jsonl" not in delete_rel_paths
    assert "dedup/matroids_sparse_paving_recovery_20260327.jsonl" not in delete_rel_paths


def test_apply_cleanup_deletes_only_planned_files(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    doomed = artifacts / "pure_o_results_sparse_probe_1000.jsonl"
    survivor = artifacts / "pure_o_results.jsonl"
    _touch(doomed, "abc")
    _touch(survivor, "keep")

    actions = cleanup_artifacts.build_cleanup_plan(artifacts)
    reclaimed = cleanup_artifacts.apply_cleanup(actions)

    assert reclaimed == 3
    assert not doomed.exists()
    assert survivor.exists()
