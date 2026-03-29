import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.budgeting import route_outcome_to_queue, verifier_budget_from_prediction  # noqa: E402
from python.neuro_symbolic.types import CostBucket, CostPrediction  # noqa: E402


def test_verifier_budget_uses_cost_bucket_for_timeout_and_schedule():
    medium = verifier_budget_from_prediction(
        CostPrediction(bucket=CostBucket.MEDIUM, timeout_risk=0.2, confidence=0.8),
        base_timeout_seconds=5.0,
    )
    likely_timeout = verifier_budget_from_prediction(
        CostPrediction(bucket=CostBucket.LIKELY_TIMEOUT, timeout_risk=0.9, confidence=0.9),
        base_timeout_seconds=5.0,
    )

    assert medium.timeout_seconds == 10.0
    assert medium.run_top_level is True
    assert likely_timeout.timeout_seconds == 5.0
    assert likely_timeout.run_top_level is False


def test_unknown_timeout_routes_to_escalation_queue():
    assert route_outcome_to_queue("unknown_timeout") == "escalation"
    assert route_outcome_to_queue("exact_feasible") == "continue_search"
    assert route_outcome_to_queue("counterexample_found") == "terminal_complete"
