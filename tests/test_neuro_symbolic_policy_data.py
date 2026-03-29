import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.features import extract_instance_action_features, extract_region_features  # noqa: E402
from python.neuro_symbolic.bootstrap import BootstrapRegionRecord  # noqa: E402
from python.neuro_symbolic.policy_data import build_replay_row, cost_bucket_from_wall_time  # noqa: E402
from python.neuro_symbolic.policies import train_cost_policy, train_instance_policy, train_region_policy  # noqa: E402
from python.neuro_symbolic.types import ActionSpec, CandidateFamily, CandidateRecord, CostBucket, RegionKey  # noqa: E402


def _region_record(score: float, status: str = "FEASIBLE") -> BootstrapRegionRecord:
    return BootstrapRegionRecord(
        region_key=RegionKey(value=f"h_vector:[1,2,{int(score * 10)}]", display_id="region", family=None, components=[1, 2, 3]),
        h_vector=[1, 2, 3],
        h_vector_key="[1,2,3]",
        source_records=5,
        example_record_id="seed-a",
        current_solver={"status": status, "wall_time": 1.2, "metrics": {"num_conflicts": 5, "num_branches": 8}},
        top_level_solver={"status": "UNKNOWN", "wall_time": 2.4, "metrics": {"num_conflicts": 3, "num_branches": 4}},
        structural_metrics={"raw_score": 4.0, "macaulay_total_slack": 2, "late_drop_sum": 1},
        current_solver_score=0.3,
        top_level_solver_score=0.6,
        structural_score=0.9,
        combined_score_weights={"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
        score_normalization="percentile",
        combined_score=score,
        witness_references=["seed-a"],
    )


def _candidate(family: CandidateFamily) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"{family.value}-cand",
        family=family,
        rank=3,
        n=6,
        field=2 if family is CandidateFamily.REPRESENTABLE else None,
        matrix_cols=[1, 2, 3, 4, 5, 6] if family is CandidateFamily.REPRESENTABLE else None,
        bases=[7, 11, 13, 14],
        circuit_hyperplanes=[21, 42] if family is CandidateFamily.SPARSE_PAVING else None,
        sparse_overlap_bound=1 if family is CandidateFamily.SPARSE_PAVING else None,
        h_vector=[1, 3, 3, 1],
    )


def _action(family: CandidateFamily, action_type: str) -> ActionSpec:
    return ActionSpec(
        action_id=f"{family.value}:{action_type}",
        family=family,
        action_type=action_type,
        parameters={"column_index": 0},
        locality="single_column",
        cross_region=False,
    )


def test_replay_rows_preserve_labels_and_censored_flags():
    region_features = extract_region_features(_region_record(0.8))
    instance_features = extract_instance_action_features(_candidate(CandidateFamily.REPRESENTABLE), _action(CandidateFamily.REPRESENTABLE, "resample_one_column"), region_features)

    row = build_replay_row(
        region_features=region_features,
        instance_action_features=instance_features,
        outcome_label="unknown_timeout",
        cost_bucket=CostBucket.LIKELY_TIMEOUT,
        censored_timeout=True,
    )

    assert row["outcome_label"] == "unknown_timeout"
    assert row["censored_timeout"] is True
    assert row["cost_bucket"] == "likely_timeout"


def test_cost_bucket_labels_are_stable_under_fixed_thresholds():
    thresholds = {"cheap": 1.0, "medium": 3.0, "expensive": 6.0}

    assert cost_bucket_from_wall_time(0.4, thresholds) is CostBucket.CHEAP
    assert cost_bucket_from_wall_time(2.0, thresholds) is CostBucket.MEDIUM
    assert cost_bucket_from_wall_time(5.0, thresholds) is CostBucket.EXPENSIVE
    assert cost_bucket_from_wall_time(9.0, thresholds) is CostBucket.LIKELY_TIMEOUT


def test_offline_policy_training_produces_serializable_models_and_deterministic_predictions():
    rows = []
    for score, label, wall_time in (
        (0.9, "counterexample_found", 0.5),
        (0.7, "solver_disagreement", 2.0),
        (0.3, "duplicate_isomorph", 8.0),
        (0.5, "exact_feasible", 4.0),
    ):
        region_features = extract_region_features(_region_record(score))
        instance_features = extract_instance_action_features(
            _candidate(CandidateFamily.REPRESENTABLE),
            _action(CandidateFamily.REPRESENTABLE, "resample_one_column"),
            region_features,
        )
        rows.append(
            build_replay_row(
                region_features=region_features,
                instance_action_features=instance_features,
                outcome_label=label,
                cost_bucket=cost_bucket_from_wall_time(wall_time),
                censored_timeout=label == "unknown_timeout",
            )
        )

    region_model = train_region_policy(rows, seed=11)
    instance_model = train_instance_policy(rows, seed=11)
    cost_model = train_cost_policy(rows, seed=11)

    region_prediction = region_model.predict_score(rows[0]["region_features"])
    instance_prediction = instance_model.predict_score(rows[0]["instance_action_features"])
    cost_prediction = cost_model.predict(rows[0]["instance_action_features"])

    reloaded_region = type(region_model).from_dict(json.loads(json.dumps(region_model.to_dict())))
    reloaded_instance = type(instance_model).from_dict(json.loads(json.dumps(instance_model.to_dict())))
    reloaded_cost = type(cost_model).from_dict(json.loads(json.dumps(cost_model.to_dict())))

    assert reloaded_region.predict_score(rows[0]["region_features"]) == region_prediction
    assert reloaded_instance.predict_score(rows[0]["instance_action_features"]) == instance_prediction
    assert reloaded_cost.predict(rows[0]["instance_action_features"]).bucket == cost_prediction.bucket
    assert reloaded_cost.predict(rows[0]["instance_action_features"]).timeout_risk == cost_prediction.timeout_risk


def test_offline_policy_training_fits_intercept_without_mean_target_bias():
    rows = [
        {
            "region_features": {"feature": 0.0},
            "instance_action_features": {"feature": 0.0},
            "region_target": 0.0,
            "instance_target": 0.0,
            "cost_bucket": "cheap",
            "timeout_target": 0.0,
        },
        {
            "region_features": {"feature": 1.0},
            "instance_action_features": {"feature": 1.0},
            "region_target": 1.0,
            "instance_target": 1.0,
            "cost_bucket": "medium",
            "timeout_target": 1.0,
        },
    ]

    region_model = train_region_policy(rows, seed=3)
    instance_model = train_instance_policy(rows, seed=3)

    assert region_model.predict_score({"feature": 0.0}) == 0.0
    assert region_model.predict_score({"feature": 1.0}) == 1.0
    assert instance_model.predict_score({"feature": 0.0}) == 0.0
    assert instance_model.predict_score({"feature": 1.0}) == 1.0
