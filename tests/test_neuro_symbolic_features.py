import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.bootstrap import BootstrapRegionRecord  # noqa: E402
from python.neuro_symbolic.features import extract_instance_action_features, extract_region_features  # noqa: E402
from python.neuro_symbolic.types import ActionSpec, CandidateFamily, CandidateRecord, RegionKey  # noqa: E402


def _region_record() -> BootstrapRegionRecord:
    return BootstrapRegionRecord(
        region_key=RegionKey(value="h_vector:[1,2,3]", display_id="[1,2,3]", family=None, components=[1, 2, 3]),
        h_vector=[1, 2, 3],
        h_vector_key="[1,2,3]",
        source_records=5,
        example_record_id="seed-a",
        current_solver={"status": "FEASIBLE", "wall_time": 1.2, "metrics": {"num_conflicts": 5, "num_branches": 8}},
        top_level_solver={"status": "UNKNOWN", "wall_time": 2.4, "metrics": {"num_conflicts": 3, "num_branches": 4}},
        structural_metrics={"raw_score": 4.0, "macaulay_total_slack": 2, "late_drop_sum": 1},
        current_solver_score=0.3,
        top_level_solver_score=0.6,
        structural_score=0.9,
        combined_score_weights={"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
        score_normalization="percentile",
        combined_score=0.5,
        witness_references=["seed-a"],
    )


def _candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="cand-a",
        family=CandidateFamily.REPRESENTABLE,
        rank=3,
        n=6,
        field=2,
        matrix_cols=[1, 2, 3, 4, 5, 6],
        bases=[7, 11, 13, 14],
        h_vector=[1, 3, 3, 1],
    )


def test_region_feature_extraction_is_deterministic():
    region_record = _region_record()

    first = extract_region_features(region_record, search_stats={"novelty_count": 2, "timeout_count": 1})
    second = extract_region_features(region_record, search_stats={"novelty_count": 2, "timeout_count": 1})

    assert first == second
    assert first["h_vector_0"] == 1.0
    assert first["current_solver_status"] == "FEASIBLE"
    assert first["timeout_count"] == 1.0


def test_instance_action_feature_extraction_is_deterministic():
    candidate = _candidate()
    action = ActionSpec(
        action_id="action-1",
        family=CandidateFamily.REPRESENTABLE,
        action_type="resample_one_column",
        parameters={"column_index": 0},
        locality="single_column",
        cross_region=False,
    )
    region_features = extract_region_features(_region_record())

    first = extract_instance_action_features(candidate, action, region_features, duplicate_isomorph_history=3)
    second = extract_instance_action_features(candidate, action, region_features, duplicate_isomorph_history=3)

    assert first == second
    assert first["family"] == "representable"
    assert first["action_type"] == "resample_one_column"
    assert first["duplicate_isomorph_history"] == 3.0
