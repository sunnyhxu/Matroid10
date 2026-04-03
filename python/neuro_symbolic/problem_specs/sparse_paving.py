from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from ..actions import rebuild_sparse_paving_candidate, sparse_paving_candidate_is_valid, all_rank_subset_masks
from ..canonicalize import CanonicalizationService
from ..problem_spec import ProblemSpec
from ..types import ActionSpec, CandidateFamily, CandidateRecord, FilterResult, RegionKey


class SparsePavingProblemSpec(ProblemSpec):
    def __init__(self, canonicalizer: CanonicalizationService | None = None) -> None:
        self._canonicalizer = canonicalizer or CanonicalizationService()

    def family_name(self) -> str:
        return CandidateFamily.SPARSE_PAVING.value

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
        if not sparse_paving_candidate_is_valid(candidate):
            return []
        if candidate.circuit_hyperplanes is None or candidate.sparse_overlap_bound is None:
            return []

        actions: List[ActionSpec] = []
        seen_outputs: set[tuple[int, ...]] = set()
        existing = sorted({int(mask) for mask in candidate.circuit_hyperplanes})
        overlap_bound = int(candidate.sparse_overlap_bound)
        all_rank_masks = all_rank_subset_masks(candidate.n, candidate.rank)

        def valid_extension(mask: int, current: Sequence[int]) -> bool:
            return all((mask & other).bit_count() <= overlap_bound for other in current)

        admissible = [mask for mask in all_rank_masks if mask not in set(existing) and valid_extension(mask, existing)]

        def maybe_add(action_type: str, parameters: Dict[str, Any], locality: str) -> None:
            action_id = f"{action_type}:{len(actions)}"
            action = ActionSpec(
                action_id=action_id,
                family=CandidateFamily.SPARSE_PAVING,
                action_type=action_type,
                parameters=parameters,
                locality=locality,
                cross_region=False,
                description=action_type.replace("_", " "),
            )
            mutated = self.apply_action(candidate, action)
            if not sparse_paving_candidate_is_valid(mutated):
                return
            signature = tuple(mutated.circuit_hyperplanes or [])
            if signature == tuple(candidate.circuit_hyperplanes or []) or signature in seen_outputs:
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

        for mask in admissible[:4]:
            maybe_add(
                "add_circuit_hyperplane",
                {"add_mask": mask},
                "single_hyperplane",
            )

        for mask in existing:
            maybe_add(
                "remove_circuit_hyperplane",
                {"remove_mask": mask},
                "single_hyperplane",
            )

        for remove_mask in existing[:2]:
            reduced = [mask for mask in existing if mask != remove_mask]
            for add_mask in all_rank_masks:
                if add_mask in reduced:
                    continue
                if valid_extension(add_mask, reduced):
                    maybe_add(
                        "swap_circuit_hyperplane",
                        {"remove_mask": remove_mask, "add_mask": add_mask},
                        "single_hyperplane",
                    )

        if admissible:
            keep = existing[1:] if existing else []
            batch_adds: List[int] = []
            for mask in admissible:
                if valid_extension(mask, keep + batch_adds):
                    batch_adds.append(mask)
                if len(batch_adds) == 2:
                    break
            if batch_adds:
                maybe_add(
                    "resample_circuit_hyperplane_batch",
                    {"remove_masks": existing[:1], "add_masks": batch_adds},
                    "hyperplane_batch",
                )

        return actions

    def apply_action(self, candidate: CandidateRecord, action: ActionSpec) -> CandidateRecord:
        if candidate.circuit_hyperplanes is None:
            raise ValueError("sparse_paving_candidate_missing_hyperplanes")

        updated = sorted({int(mask) for mask in candidate.circuit_hyperplanes})
        if action.action_type == "add_circuit_hyperplane":
            updated.append(int(action.parameters["add_mask"]))
        elif action.action_type == "remove_circuit_hyperplane":
            remove_mask = int(action.parameters["remove_mask"])
            updated = [mask for mask in updated if mask != remove_mask]
        elif action.action_type == "swap_circuit_hyperplane":
            remove_mask = int(action.parameters["remove_mask"])
            add_mask = int(action.parameters["add_mask"])
            updated = [mask for mask in updated if mask != remove_mask]
            updated.append(add_mask)
        elif action.action_type == "resample_circuit_hyperplane_batch":
            remove_masks = {int(mask) for mask in action.parameters.get("remove_masks", [])}
            add_masks = [int(mask) for mask in action.parameters.get("add_masks", [])]
            updated = [mask for mask in updated if mask not in remove_masks]
            updated.extend(add_masks)
        else:
            raise ValueError(f"unsupported_sparse_paving_action:{action.action_type}")

        return rebuild_sparse_paving_candidate(
            candidate,
            updated,
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "family": self.family_name(),
            },
        )

    def cheap_filters(self, candidate: CandidateRecord) -> Sequence[FilterResult]:
        passed = sparse_paving_candidate_is_valid(candidate)
        h_vector = candidate.h_vector or []
        trailing_zeros_passed = not h_vector or int(h_vector[-1]) != 0
        return [
            FilterResult(
                name="sparse_paving_overlap_constraint",
                passed=passed,
                reason=None if passed else "overlap_constraint_failed",
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
