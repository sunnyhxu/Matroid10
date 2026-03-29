import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.eval import compare_baselines, evaluate_replay_rows  # noqa: E402
from python.neuro_symbolic.replay import load_action_logs, summarize_action_logs  # noqa: E402
from python.neuro_symbolic.policy_data import read_replay_rows  # noqa: E402


def _fixture_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "fixtures" / name


def test_replay_evaluator_reproduces_same_metrics_from_same_fixture():
    rows = read_replay_rows(_fixture_path("neuro_symbolic_replay_rows.jsonl"))

    first = evaluate_replay_rows(rows, strategy_name="observed")
    second = evaluate_replay_rows(rows, strategy_name="observed")

    assert first == second
    assert first["duplicate_isomorph_rate"] >= 0.0
    assert "interesting_outcomes_per_cpu_hour" in first


def test_baseline_comparisons_run_on_small_fixture_without_repo_artifacts():
    rows = read_replay_rows(_fixture_path("neuro_symbolic_replay_rows.jsonl"))

    comparison = compare_baselines(rows, include_no_canonical_merge=True)

    assert "uniform" in comparison
    assert "structural_score_only" in comparison
    assert "dual_solver_hardness_only" in comparison
    assert "no_canonical_merge_ablation" in comparison


def test_action_log_summary_reports_queue_and_verifier_spend():
    rows = load_action_logs(_fixture_path("neuro_symbolic_action_log.jsonl"))

    summary = summarize_action_logs(rows)

    assert summary["queue_counts"]["escalation"] == 2
    assert summary["outcome_counts"]["duplicate_isomorph"] == 1
    assert summary["outcome_counts"]["verifier_error"] == 1
    assert summary["verifier_spend_by_stage"]["reference_solver"] > 0.0


def test_replay_eval_keeps_verifier_error_distinct_and_non_interesting():
    rows = read_replay_rows(_fixture_path("neuro_symbolic_replay_rows.jsonl"))

    summary = evaluate_replay_rows(rows, strategy_name="observed")

    assert summary["rows_evaluated"] == 5
    assert summary["escalation_yield"] < 1.0


def test_docs_reference_real_commands_and_output_files():
    path = Path(__file__).resolve().parents[1] / "docs" / "neuro_symbolic.md"
    text = path.read_text(encoding="utf-8")

    assert "python python/neuro_symbolic_search.py" in text
    assert "artifacts/neuro_symbolic/action_log.jsonl" in text
    assert "artifacts/neuro_symbolic/replay_rows.jsonl" in text
    assert "python -m python.neuro_symbolic.eval" in text
