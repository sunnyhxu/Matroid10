from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

try:
    from ..common import read_jsonl
    from ..pure_o_cp import solve_h_vector
    from ..pure_o_top_level_cp import solve_h_vector_top_level
except ImportError:
    from common import read_jsonl
    from pure_o_cp import solve_h_vector
    from pure_o_top_level_cp import solve_h_vector_top_level

from .bootstrap import BootstrapRegionRecord, load_bootstrap_regions
from .budgeting import route_outcome_to_queue, verifier_budget_from_prediction
from .features import extract_instance_action_features, extract_region_features
from .policy_data import build_replay_row, write_replay_rows
from .queues import QueueEntry, SearchQueues
from .state_graph import SearchStateGraph
from .store import write_graph_events
from .types import CandidateFamily, CandidateRecord, CostPrediction, OutcomeRecord, VerifierResult


VerifierFn = Callable[[CandidateRecord, float], VerifierResult]


@dataclass(frozen=True)
class ControllerRunSummary:
    action_logs: List[Dict[str, Any]]
    replay_rows: List[Dict[str, Any]]
    queue_counts: Dict[str, int]


def _default_reference_verifier(candidate: CandidateRecord, timeout_seconds: float) -> VerifierResult:
    result = solve_h_vector(candidate.h_vector or [], timeout_sec=timeout_seconds, num_workers=1)
    return VerifierResult(
        name="reference_solver",
        status=result.status,
        wall_time=result.wall_time,
        details={"model_size": dict(result.model_size), "solver_stats": dict(result.solver_stats or {})},
        censored=result.status == "UNKNOWN",
    )


def _default_top_level_verifier(candidate: CandidateRecord, timeout_seconds: float) -> VerifierResult:
    result = solve_h_vector_top_level(candidate.h_vector or [], timeout_sec=timeout_seconds, num_workers=1)
    return VerifierResult(
        name="top_level_solver",
        status=result.status,
        wall_time=result.wall_time,
        details={"model_size": dict(result.model_size), "solver_stats": dict(result.solver_stats or {})},
        censored=result.status == "UNKNOWN",
    )


def load_seed_candidates_by_region(path: str | Path) -> Dict[str, List[CandidateRecord]]:
    grouped: Dict[str, List[CandidateRecord]] = {}
    for row in read_jsonl(path):
        candidate = CandidateRecord.from_dict(row)
        h_vector = candidate.h_vector or []
        region_key = f"h_vector:{json.dumps([int(value) for value in h_vector], separators=(',', ':'))}"
        grouped.setdefault(region_key, []).append(candidate)
    return grouped


class NeuroSymbolicController:
    def __init__(
        self,
        *,
        bootstrap_regions: Sequence[BootstrapRegionRecord],
        seed_candidates_by_region: Mapping[str, Sequence[CandidateRecord]],
        problem_specs: Mapping[str, Any],
        region_policy: Any,
        instance_policy: Any,
        cost_policy: Any,
        verifier_functions: Mapping[str, VerifierFn] | None = None,
        base_timeout_seconds: float = 5.0,
        seed: int = 0,
    ) -> None:
        self.bootstrap_regions = list(bootstrap_regions)
        self.seed_candidates_by_region = {key: list(value) for key, value in seed_candidates_by_region.items()}
        self.problem_specs = dict(problem_specs)
        self.region_policy = region_policy
        self.instance_policy = instance_policy
        self.cost_policy = cost_policy
        self.verifier_functions = {
            "reference_solver": _default_reference_verifier,
            "top_level_solver": _default_top_level_verifier,
        }
        if verifier_functions is not None:
            self.verifier_functions.update(verifier_functions)
        self.base_timeout_seconds = float(base_timeout_seconds)
        self.seed = int(seed)
        self.graph = SearchStateGraph()
        self.queues = SearchQueues()
        self.action_logs: List[Dict[str, Any]] = []
        self.replay_rows: List[Dict[str, Any]] = []
        self._candidate_store: Dict[str, CandidateRecord] = {}
        self._seed_offsets: Dict[str, int] = {region.region_key.value: 0 for region in self.bootstrap_regions}

    def _choose_region(self) -> tuple[BootstrapRegionRecord, Dict[str, Any]] | None:
        scored: List[tuple[float, str, BootstrapRegionRecord, Dict[str, Any]]] = []
        for region in self.bootstrap_regions:
            region_features = extract_region_features(
                region,
                search_stats={
                    "novelty_count": len(self._candidate_store),
                    "timeout_count": self.queues.counts()["escalation"],
                    "escalation_count": self.queues.counts()["escalation"],
                    "revisit_count": 0,
                },
            )
            score = float(self.region_policy.predict_score(region_features))
            scored.append((score, region.region_key.value, region, region_features))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        _, _, region, region_features = scored[0]
        return region, region_features

    def _ensure_seed_inserted(self, candidate: CandidateRecord):
        spec = self.problem_specs[candidate.family.value]
        insert_result = self.graph.insert_seed_instance(
            region_key=spec.region_key(candidate),
            canonical_key=spec.canonicalize(candidate),
            family=spec.family_name(),
            candidate_id=candidate.candidate_id,
        )
        self._candidate_store.setdefault(insert_result.canonical_key.key, candidate)
        return insert_result, spec

    def _choose_parent_candidate(self, region_key: str) -> tuple[str, CandidateRecord, Any] | None:
        queued = self.queues.pop_continue_for_region(region_key)
        if queued is not None and queued.canonical_key in self._candidate_store:
            candidate = self._candidate_store[queued.canonical_key]
            spec = self.problem_specs[candidate.family.value]
            return queued.canonical_key, candidate, spec

        seeds = self.seed_candidates_by_region.get(region_key, [])
        if not seeds:
            return None
        offset = self._seed_offsets.get(region_key, 0)
        candidate = seeds[offset % len(seeds)]
        self._seed_offsets[region_key] = offset + 1
        insert_result, spec = self._ensure_seed_inserted(candidate)
        return insert_result.canonical_key.key, candidate, spec

    def _label_from_verifier_results(self, results: Sequence[VerifierResult]) -> str:
        statuses = [result.status for result in results]
        if any(status == "INFEASIBLE" for status in statuses):
            return "solver_disagreement" if len(set(statuses)) > 1 else "counterexample_found"
        if any(status == "UNKNOWN" for status in statuses):
            return "unknown_timeout"
        return "exact_feasible"

    def run(self, max_steps: int) -> ControllerRunSummary:
        for _ in range(max_steps):
            region_choice = self._choose_region()
            if region_choice is None:
                break
            region, region_features = region_choice
            parent_choice = self._choose_parent_candidate(region.region_key.value)
            if parent_choice is None:
                continue
            parent_key, parent_candidate, spec = parent_choice

            actions = list(spec.enumerate_valid_actions(parent_candidate))
            if not actions:
                continue

            scored_actions: List[tuple[float, str, Any, Dict[str, Any], CostPrediction]] = []
            for action in actions:
                action_features = extract_instance_action_features(parent_candidate, action, region_features)
                instance_score = float(self.instance_policy.predict_score(action_features))
                cost_prediction = self.cost_policy.predict(action_features)
                scored_actions.append((instance_score, action.action_id, action, action_features, cost_prediction))
            scored_actions.sort(key=lambda item: (-item[0], item[1]))
            instance_score, _, action, action_features, cost_prediction = scored_actions[0]

            mutated_candidate = spec.apply_action(parent_candidate, action)
            mutated_spec = self.problem_specs[mutated_candidate.family.value]
            insert_result = self.graph.insert_mutated_instance(
                parent_key=parent_key,
                action=action,
                region_key=mutated_spec.region_key(mutated_candidate),
                canonical_key=mutated_spec.canonicalize(mutated_candidate),
                family=mutated_spec.family_name(),
                candidate_id=mutated_candidate.candidate_id,
            )

            filter_results = [result.to_dict() for result in mutated_spec.cheap_filters(mutated_candidate)]
            verifier_results: List[VerifierResult] = []
            if insert_result.created:
                self._candidate_store[insert_result.canonical_key.key] = mutated_candidate

            if insert_result.duplicate_isomorph:
                outcome_label = "duplicate_isomorph"
                queue_destination = route_outcome_to_queue(outcome_label)
                self.graph.record_outcome(
                    insert_result.canonical_key.key,
                    OutcomeRecord(
                        label=outcome_label,
                        canonical_key=insert_result.canonical_key.key,
                        display_id=insert_result.canonical_key.display_id,
                        details={"parent_key": parent_key},
                    ),
                )
            elif any(not result["passed"] for result in filter_results):
                outcome_label = "invalid_action_blocked"
                queue_destination = route_outcome_to_queue(outcome_label)
                self.graph.record_outcome(
                    insert_result.canonical_key.key,
                    OutcomeRecord(
                        label=outcome_label,
                        canonical_key=insert_result.canonical_key.key,
                        display_id=insert_result.canonical_key.display_id,
                        details={"filter_results": filter_results},
                    ),
                )
            else:
                budget = verifier_budget_from_prediction(cost_prediction, self.base_timeout_seconds)
                verifier_schedule = list(mutated_spec.exact_verifiers())
                if verifier_schedule:
                    verifier = self.verifier_functions[verifier_schedule[0]["name"]]
                    verifier_results.append(verifier(mutated_candidate, budget.timeout_seconds))
                if budget.run_top_level and len(verifier_schedule) > 1:
                    verifier = self.verifier_functions[verifier_schedule[1]["name"]]
                    verifier_results.append(verifier(mutated_candidate, budget.timeout_seconds))
                outcome_label = self._label_from_verifier_results(verifier_results)
                queue_destination = route_outcome_to_queue(outcome_label)
                self.graph.record_outcome(
                    insert_result.canonical_key.key,
                    OutcomeRecord(
                        label=outcome_label,
                        canonical_key=insert_result.canonical_key.key,
                        display_id=insert_result.canonical_key.display_id,
                        details={"verifier_results": [result.to_dict() for result in verifier_results]},
                        censored=any(result.censored for result in verifier_results),
                    ),
                )

            entry = QueueEntry(
                canonical_key=insert_result.canonical_key.key,
                region_key=insert_result.region_key.value,
                priority=instance_score,
                reason=outcome_label,
            )
            self.queues.push(queue_destination, entry)

            log_row = {
                "region_key": region.region_key.value,
                "parent_key": parent_key,
                "action": action.to_dict(),
                "region_score": float(self.region_policy.predict_score(region_features)),
                "instance_score": instance_score,
                "cost_prediction": cost_prediction.to_dict(),
                "canonicalization": insert_result.canonical_key.to_dict(),
                "node_created": insert_result.created,
                "duplicate_isomorph": insert_result.duplicate_isomorph,
                "filter_results": filter_results,
                "verifier_results": [result.to_dict() for result in verifier_results],
                "outcome_label": outcome_label,
                "queue_destination": queue_destination,
            }
            self.action_logs.append(log_row)
            self.replay_rows.append(
                build_replay_row(
                    region_features=region_features,
                    instance_action_features=action_features,
                    outcome_label=outcome_label,
                    cost_bucket=cost_prediction.bucket,
                    censored_timeout=any(result.censored for result in verifier_results),
                )
            )

        return ControllerRunSummary(
            action_logs=list(self.action_logs),
            replay_rows=list(self.replay_rows),
            queue_counts=self.queues.counts(),
        )


def write_action_logs(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")))
            handle.write("\n")


def run_controller_from_paths(
    *,
    bootstrap_path: str | Path,
    seed_candidates_path: str | Path,
    controller_kwargs: Mapping[str, Any],
    max_steps: int,
    action_log_out: str | Path | None = None,
    graph_event_out: str | Path | None = None,
    replay_row_out: str | Path | None = None,
) -> ControllerRunSummary:
    controller = NeuroSymbolicController(
        bootstrap_regions=load_bootstrap_regions(bootstrap_path),
        seed_candidates_by_region=load_seed_candidates_by_region(seed_candidates_path),
        **dict(controller_kwargs),
    )
    summary = controller.run(max_steps=max_steps)
    if action_log_out is not None:
        write_action_logs(action_log_out, summary.action_logs)
    if graph_event_out is not None:
        write_graph_events(graph_event_out, controller.graph.event_log())
    if replay_row_out is not None:
        write_replay_rows(replay_row_out, summary.replay_rows)
    return summary
