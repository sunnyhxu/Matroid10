import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.actions import (  # noqa: E402
    compute_h_vector_from_bases,
    enumerate_representable_bases,
    representable_candidate_is_valid,
)
from python.neuro_symbolic.problem_specs.representable import RepresentableProblemSpec  # noqa: E402
from python.neuro_symbolic.types import CandidateFamily, CandidateRecord  # noqa: E402


def _candidate() -> CandidateRecord:
    matrix_cols = [1, 2, 3, 1]
    bases = enumerate_representable_bases(field=2, rank=2, n=4, matrix_cols=matrix_cols)
    h_vector = compute_h_vector_from_bases(bases, rank=2, n=4)
    return CandidateRecord(
        candidate_id="rep-seed",
        family=CandidateFamily.REPRESENTABLE,
        rank=2,
        n=4,
        field=2,
        matrix_cols=matrix_cols,
        bases=bases,
        h_vector=h_vector,
    )


def test_every_emitted_representable_action_yields_a_structurally_valid_candidate():
    spec = RepresentableProblemSpec()
    candidate = _candidate()

    actions = spec.enumerate_valid_actions(candidate)

    assert actions
    for action in actions:
        mutated = spec.apply_action(candidate, action)
        assert representable_candidate_is_valid(mutated)
        assert mutated.family is CandidateFamily.REPRESENTABLE
        assert mutated.matrix_cols is not None


def test_representable_actions_do_not_emit_no_ops_or_duplicate_outputs():
    spec = RepresentableProblemSpec()
    candidate = _candidate()

    actions = spec.enumerate_valid_actions(candidate)
    outputs = [tuple(spec.apply_action(candidate, action).matrix_cols or []) for action in actions]

    assert len(outputs) == len(set(outputs))
    assert tuple(candidate.matrix_cols or []) not in set(outputs)
    assert all(action.action_type in {"resample_one_column", "resample_column_batch", "constrained_column_replacement"} for action in actions)


def test_representable_action_application_keeps_provenance_stable_under_canonicalization():
    spec = RepresentableProblemSpec()
    candidate = _candidate()
    action = spec.enumerate_valid_actions(candidate)[0]

    mutated = spec.apply_action(candidate, action)
    first = spec.canonicalize(mutated)
    second = spec.canonicalize(mutated)

    assert mutated.metadata["provenance"]["action_id"] == action.action_id
    assert mutated.metadata["provenance"]["family"] == "representable"
    assert first.key == second.key
