from itertools import combinations
from dataclasses import replace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.actions import (  # noqa: E402
    compute_h_vector_from_bases,
    sparse_paving_candidate_is_valid,
)
from python.neuro_symbolic.problem_specs.sparse_paving import SparsePavingProblemSpec  # noqa: E402
from python.neuro_symbolic.types import CandidateFamily, CandidateRecord  # noqa: E402


def _mask(values: tuple[int, ...]) -> int:
    out = 0
    for value in values:
        out |= 1 << value
    return out


def _candidate() -> CandidateRecord:
    rank = 3
    n = 5
    circuit_hyperplanes = [_mask((0, 1, 2)), _mask((0, 3, 4))]
    all_rank_subsets = [_mask(indices) for indices in combinations(range(n), rank)]
    bases = [mask for mask in all_rank_subsets if mask not in set(circuit_hyperplanes)]
    h_vector = compute_h_vector_from_bases(bases, rank=rank, n=n)
    return CandidateRecord(
        candidate_id="sp-seed",
        family=CandidateFamily.SPARSE_PAVING,
        rank=rank,
        n=n,
        bases=bases,
        circuit_hyperplanes=circuit_hyperplanes,
        sparse_overlap_bound=1,
        h_vector=h_vector,
    )


def test_every_emitted_sparse_paving_action_preserves_overlap_constraints():
    spec = SparsePavingProblemSpec()
    candidate = _candidate()

    actions = spec.enumerate_valid_actions(candidate)

    assert actions
    for action in actions:
        mutated = spec.apply_action(candidate, action)
        assert sparse_paving_candidate_is_valid(mutated)
        assert mutated.family is CandidateFamily.SPARSE_PAVING


def test_sparse_paving_actions_exclude_obvious_label_permutation_style_actions():
    spec = SparsePavingProblemSpec()
    candidate = _candidate()

    actions = spec.enumerate_valid_actions(candidate)

    assert all(action.action_type in {"add_circuit_hyperplane", "remove_circuit_hyperplane", "swap_circuit_hyperplane", "resample_circuit_hyperplane_batch"} for action in actions)
    assert len({action.action_id for action in actions}) == len(actions)


def test_sparse_paving_action_application_keeps_provenance_stable_under_canonicalization():
    spec = SparsePavingProblemSpec()
    candidate = _candidate()
    action = spec.enumerate_valid_actions(candidate)[0]

    mutated = spec.apply_action(candidate, action)
    first = spec.canonicalize(mutated)
    second = spec.canonicalize(mutated)

    assert mutated.metadata["provenance"]["action_id"] == action.action_id
    assert mutated.metadata["provenance"]["family"] == "sparse_paving"
    assert first.key == second.key


def test_sparse_paving_cheap_filters_block_trailing_zero_h_vectors():
    spec = SparsePavingProblemSpec()
    candidate = _candidate()

    trailing_zero_candidate = replace(candidate, h_vector=[1, 3, 5, 0])
    passing_candidate = replace(candidate, h_vector=[1, 3, 5, 2])
    empty_candidate = replace(candidate, h_vector=[])
    none_candidate = replace(candidate, h_vector=None)

    trailing_zero_results = spec.cheap_filters(trailing_zero_candidate)
    passing_results = spec.cheap_filters(passing_candidate)
    empty_results = spec.cheap_filters(empty_candidate)
    none_results = spec.cheap_filters(none_candidate)

    assert any(result.name == "h_vector_trailing_zeros" and not result.passed for result in trailing_zero_results)
    assert all(result.passed for result in passing_results if result.name == "h_vector_trailing_zeros")
    assert all(result.passed for result in empty_results if result.name == "h_vector_trailing_zeros")
    assert all(result.passed for result in none_results if result.name == "h_vector_trailing_zeros")
