import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.types import (  # noqa: E402
    ActionSpec,
    CandidateFamily,
    CandidateRecord,
    CanonicalKey,
    CostBucket,
    CostPrediction,
    FilterResult,
    OutcomeRecord,
    RegionKey,
    VerifierResult,
)


@pytest.mark.parametrize(
    ("value", "expected_type", "assertions"),
    [
        (
            CanonicalKey(key="canon:abc", display_id="abc123", digest="deadbeef"),
            CanonicalKey,
            lambda loaded: (
                loaded.key == "canon:abc",
                loaded.display_id == "abc123",
                loaded.digest == "deadbeef",
            ),
        ),
        (
            RegionKey(value="region:[1,2,3]", display_id="[1,2,3]", family=None, components=[1, 2, 3]),
            RegionKey,
            lambda loaded: (
                loaded.value == "region:[1,2,3]",
                loaded.display_id == "[1,2,3]",
                loaded.family is None,
                loaded.components == [1, 2, 3],
            ),
        ),
        (
            ActionSpec(
                action_id="flip-col-0",
                family=CandidateFamily.REPRESENTABLE,
                action_type="resample_one_column",
                parameters={"column_index": 0, "new_column": 7},
                locality="single_column",
                cross_region=False,
                description="Replace one column with a nearby variant",
            ),
            ActionSpec,
            lambda loaded: (
                loaded.family is CandidateFamily.REPRESENTABLE,
                loaded.action_type == "resample_one_column",
                loaded.parameters["column_index"] == 0,
            ),
        ),
        (
            FilterResult(
                name="cheap-guard",
                passed=True,
                reason=None,
                details={"kept": True},
            ),
            FilterResult,
            lambda loaded: (
                loaded.passed is True,
                loaded.reason is None,
                loaded.details["kept"] is True,
            ),
        ),
        (
            VerifierResult(
                name="reference-solver",
                status="FEASIBLE",
                wall_time=0.25,
                details={"num_branches": 12},
                censored=False,
            ),
            VerifierResult,
            lambda loaded: (
                loaded.status == "FEASIBLE",
                loaded.wall_time == pytest.approx(0.25),
                loaded.censored is False,
            ),
        ),
        (
            OutcomeRecord(
                label="duplicate_isomorph",
                canonical_key="canon:abc",
                display_id="abc123",
                details={"parent": "seed"},
                censored=False,
            ),
            OutcomeRecord,
            lambda loaded: (
                loaded.label == "duplicate_isomorph",
                loaded.canonical_key == "canon:abc",
                loaded.details["parent"] == "seed",
            ),
        ),
        (
            CostPrediction(
                bucket=CostBucket.MEDIUM,
                timeout_risk=0.2,
                confidence=0.8,
                details={"source": "baseline"},
            ),
            CostPrediction,
            lambda loaded: (
                loaded.bucket is CostBucket.MEDIUM,
                loaded.timeout_risk == pytest.approx(0.2),
                loaded.confidence == pytest.approx(0.8),
            ),
        ),
        (
            CandidateRecord(
                candidate_id="matroid-1",
                family=CandidateFamily.SPARSE_PAVING,
                rank=5,
                n=10,
                bases=[31, 47, 55],
                field=None,
                matrix_cols=None,
                circuit_hyperplanes=[63, 95],
                sparse_overlap_bound=3,
                non_paving_witness=None,
                h_vector=[1, 4, 6, 4, 1],
                seed=42,
                trial=7,
                metadata={"origin": "fixture"},
            ),
            CandidateRecord,
            lambda loaded: (
                loaded.family is CandidateFamily.SPARSE_PAVING,
                loaded.rank == 5,
                loaded.circuit_hyperplanes == [63, 95],
                loaded.metadata["origin"] == "fixture",
            ),
        ),
    ],
)
def test_types_round_trip_through_plain_dicts(value, expected_type, assertions):
    payload = value.to_dict()
    loaded = expected_type.from_dict(payload)
    assert isinstance(payload, dict)
    assert isinstance(loaded, expected_type)
    assert all(assertions(loaded))


def test_candidate_record_defaults_missing_generator_mode_to_representable():
    loaded = CandidateRecord.from_dict(
        {
            "id": "legacy-id",
            "rank": 4,
            "n": 10,
            "bases": [15, 23],
            "matrix_cols": [1, 2, 3, 4],
        }
    )

    assert loaded.family is CandidateFamily.REPRESENTABLE
    assert loaded.candidate_id == "legacy-id"
    assert loaded.matrix_cols == [1, 2, 3, 4]
