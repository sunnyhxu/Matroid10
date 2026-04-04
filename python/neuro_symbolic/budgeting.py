from __future__ import annotations

from dataclasses import dataclass

from .types import CostBucket, CostPrediction


@dataclass(frozen=True)
class VerifierBudgetDecision:
    bucket: CostBucket
    timeout_seconds: float
    run_top_level: bool


def verifier_budget_from_prediction(
    prediction: CostPrediction,
    base_timeout_seconds: float,
) -> VerifierBudgetDecision:
    timeout_multipliers = {
        CostBucket.CHEAP: 1.0,
        CostBucket.MEDIUM: 2.0,
        CostBucket.EXPENSIVE: 4.0,
        CostBucket.LIKELY_TIMEOUT: 1.0,
    }
    run_top_level = prediction.bucket in {CostBucket.MEDIUM, CostBucket.EXPENSIVE}
    return VerifierBudgetDecision(
        bucket=prediction.bucket,
        timeout_seconds=float(base_timeout_seconds) * timeout_multipliers[prediction.bucket],
        run_top_level=run_top_level,
    )


def route_outcome_to_queue(outcome_label: str) -> str:
    if outcome_label in {"unknown_timeout", "verifier_error"}:
        return "escalation"
    if outcome_label == "exact_feasible":
        return "continue_search"
    return "terminal_complete"
