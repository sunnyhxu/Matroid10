from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from ..actions import (
    decode_matrix_col,
    encode_matrix_col,
    first_full_rank_subset,
    rebuild_representable_candidate,
    representable_candidate_is_valid,
)
from ..canonicalize import CanonicalizationService
from ..problem_spec import ProblemSpec
from ..types import ActionSpec, CandidateFamily, CandidateRecord, FilterResult, RegionKey


class RepresentableProblemSpec(ProblemSpec):
    def __init__(self, canonicalizer: CanonicalizationService | None = None) -> None:
        self._canonicalizer = canonicalizer or CanonicalizationService()

    def family_name(self) -> str:
        return CandidateFamily.REPRESENTABLE.value

    def region_key(self, candidate: CandidateRecord) -> RegionKey:
        h_vector = candidate.h_vector or []
        h_vector_key = json.dumps([int(value) for value in h_vector], separators=(",", ":"))
        return RegionKey(
            value=f"h_vector:{h_vector_key}",
            display_id=h_vector_key,
            family=self.family_name(),
            components=[int(value) for value in h_vector],
        )

    def canonicalize(self, candidate: CandidateRecord):
        return self._canonicalizer.canonicalize_candidate(candidate)

    def enumerate_valid_actions(self, candidate: CandidateRecord) -> Sequence[ActionSpec]:
        if not representable_candidate_is_valid(candidate):
            return []

        actions: List[ActionSpec] = []
        seen_outputs: set[tuple[int, ...]] = set()
        assert candidate.matrix_cols is not None
        assert candidate.field is not None

        def maybe_add(action_type: str, parameters: Dict[str, Any], locality: str) -> None:
            action_id = f"{action_type}:{len(actions)}"
            action = ActionSpec(
                action_id=action_id,
                family=CandidateFamily.REPRESENTABLE,
                action_type=action_type,
                parameters=parameters,
                locality=locality,
                cross_region=False,
                description=action_type.replace("_", " "),
            )
            mutated = self.apply_action(candidate, action)
            if not representable_candidate_is_valid(mutated):
                return
            signature = tuple(mutated.matrix_cols or [])
            if signature == tuple(candidate.matrix_cols or []) or signature in seen_outputs:
                return
            seen_outputs.add(signature)
            cross_region = mutated.h_vector != candidate.h_vector
            actions.append(
                ActionSpec(
                    action_id=action.action_id,
                    family=action.family,
                    action_type=action.action_type,
                    parameters=action.parameters,
                    locality=action.locality,
                    cross_region=cross_region,
                    description=action.description,
                )
            )

        for column_index, encoded in enumerate(candidate.matrix_cols):
            digits = decode_matrix_col(encoded, candidate.field, candidate.rank)
            for row_index in range(candidate.rank):
                new_digits = list(digits)
                new_digits[row_index] = (new_digits[row_index] + 1) % candidate.field
                if new_digits == digits:
                    continue
                maybe_add(
                    "resample_one_column",
                    {
                        "column_index": column_index,
                        "row_index": row_index,
                        "new_value": new_digits[row_index],
                    },
                    "single_column",
                )

        for column_index in range(max(0, candidate.n - 1)):
            maybe_add(
                "resample_column_batch",
                {
                    "column_indices": [column_index, column_index + 1],
                    "row_indices": [0, 0],
                },
                "column_batch",
            )

        support_subset = first_full_rank_subset(candidate.matrix_cols, candidate.field, candidate.rank) or ()
        if support_subset:
            source_index = support_subset[0]
            for column_index in range(candidate.n):
                if column_index == source_index:
                    continue
                maybe_add(
                    "constrained_column_replacement",
                    {
                        "column_index": column_index,
                        "replacement_source_index": source_index,
                    },
                    "single_column",
                )

        return actions

    def apply_action(self, candidate: CandidateRecord, action: ActionSpec) -> CandidateRecord:
        if candidate.matrix_cols is None or candidate.field is None:
            raise ValueError("representable_candidate_missing_matrix")

        updated_cols = list(candidate.matrix_cols)
        if action.action_type == "resample_one_column":
            column_index = int(action.parameters["column_index"])
            row_index = int(action.parameters["row_index"])
            digits = decode_matrix_col(updated_cols[column_index], candidate.field, candidate.rank)
            digits[row_index] = int(action.parameters["new_value"])
            updated_cols[column_index] = encode_matrix_col(digits, candidate.field)
        elif action.action_type == "resample_column_batch":
            column_indices = [int(index) for index in action.parameters["column_indices"]]
            row_indices = [int(index) for index in action.parameters["row_indices"]]
            for column_index, row_index in zip(column_indices, row_indices):
                digits = decode_matrix_col(updated_cols[column_index], candidate.field, candidate.rank)
                digits[row_index] = (digits[row_index] + 1) % candidate.field
                updated_cols[column_index] = encode_matrix_col(digits, candidate.field)
        elif action.action_type == "constrained_column_replacement":
            column_index = int(action.parameters["column_index"])
            replacement_source_index = int(action.parameters["replacement_source_index"])
            updated_cols[column_index] = updated_cols[replacement_source_index]
        else:
            raise ValueError(f"unsupported_representable_action:{action.action_type}")

        return rebuild_representable_candidate(
            candidate,
            updated_cols,
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "family": self.family_name(),
            },
        )

    def cheap_filters(self, candidate: CandidateRecord) -> Sequence[FilterResult]:
        passed = representable_candidate_is_valid(candidate)
        h_vector = candidate.h_vector or []
        trailing_zeros_passed = not h_vector or int(h_vector[-1]) != 0
        return [
            FilterResult(
                name="representable_full_rank_support",
                passed=passed,
                reason=None if passed else "full_rank_support_failed",
                details={"family": self.family_name()},
            ),
            FilterResult(
                name="h_vector_trailing_zeros",
                passed=trailing_zeros_passed,
                reason=None if trailing_zeros_passed else "h_vector_trailing_zeros",
                details={"family": self.family_name()},
            ),
        ]

    def exact_verifiers(self) -> Sequence[Dict[str, Any]]:
        return [{"name": "reference_solver"}, {"name": "top_level_solver"}]
