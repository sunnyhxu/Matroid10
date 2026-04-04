from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Mapping, Sequence

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
from .policies import train_region_policy, train_instance_policy, train_cost_policy
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
        ucb_beta: float = 0.5,
        retrain_interval: int = 25,
        min_retrain_size: int = 10,
    ) -> None:
        self.bootstrap_regions = list(bootstrap_regions)
        self._bootstrap_regions_by_key = {region.region_key.value: region for region in self.bootstrap_regions}
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
        self.ucb_beta = float(ucb_beta)
        self.retrain_interval = int(retrain_interval)
        self.min_retrain_size = int(min_retrain_size)
        self.graph = SearchStateGraph()
        self.queues = SearchQueues()
        self.action_logs: List[Dict[str, Any]] = []
        self.replay_rows: List[Dict[str, Any]] = []
        self._candidate_store: Dict[str, CandidateRecord] = {}
        self._seed_offsets: Dict[str, int] = {region.region_key.value: 0 for region in self.bootstrap_regions}
        self._attempted_action_ids: DefaultDict[str, set[str]] = defaultdict(set)
        self._exhausted_parent_keys: set[str] = set()
        self._region_visit_counts: Dict[str, int] = {}
        self._total_region_visits: int = 0

    def _search_stats(self) -> Dict[str, int]:
        return {
            "novelty_count": len(self._candidate_store),
            "timeout_count": self.queues.counts()["escalation"],
            "escalation_count": self.queues.counts()["escalation"],
            "revisit_count": sum(node.revisit_count for node in self.graph.instance_nodes.values()),
        }

    def _synthetic_region_record(self, region_key: str, candidate: CandidateRecord) -> Dict[str, Any]:
        h_vector = [int(value) for value in (candidate.h_vector or [])]
        return {
            "h_vector": h_vector,
            "h_vector_key": region_key,
            "source_records": 1,
            "current_solver": {"status": "UNKNOWN", "wall_time": 0.0, "metrics": {}},
            "top_level_solver": {"status": "UNKNOWN", "wall_time": 0.0, "metrics": {}},
            "structural_metrics": {"raw_score": 0.0, "macaulay_total_slack": 0.0, "late_drop_sum": 0.0},
            "current_solver_score": 0.0,
            "top_level_solver_score": 0.0,
            "structural_score": 0.0,
            "combined_score": 0.0,
        }

    def _region_features_for_candidate(self, region_key: str, candidate: CandidateRecord) -> Dict[str, Any]:
        region_record = self._bootstrap_regions_by_key.get(region_key)
        if region_record is None:
            region_record = self._synthetic_region_record(region_key, candidate)
        return extract_region_features(region_record, search_stats=self._search_stats())

    def _ranked_regions(self) -> List[tuple[float, str, BootstrapRegionRecord, Dict[str, Any]]]:
        scored: List[tuple[float, str, BootstrapRegionRecord, Dict[str, Any]]] = []
        for region in self.bootstrap_regions:
            region_features = extract_region_features(region, search_stats=self._search_stats())
            score = float(self.region_policy.predict_score(region_features))
            n_i = self._region_visit_counts.get(region.region_key.value, 0)
            ucb_bonus = self.ucb_beta * math.sqrt(math.log(self._total_region_visits + 1) / (n_i + 1))
            effective_score = score + ucb_bonus
            scored.append((effective_score, region.region_key.value, region, region_features))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

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

    def _score_available_actions(
        self,
        *,
        parent_key: str,
        parent_candidate: CandidateRecord,
        spec: Any,
        region_features: Mapping[str, Any],
    ) -> List[tuple[float, str, Any, Dict[str, Any], CostPrediction]]:
        if parent_key in self._exhausted_parent_keys:
            return []

        duplicate_history = 0
        node = self.graph.instance_nodes.get(parent_key)
        if node is not None:
            duplicate_history = sum(1 for outcome in node.outcome_history if outcome.label == "duplicate_isomorph")

        attempted_action_ids = self._attempted_action_ids[parent_key]
        scored_actions: List[tuple[float, str, Any, Dict[str, Any], CostPrediction]] = []
        for action in spec.enumerate_valid_actions(parent_candidate):
            if action.action_id in attempted_action_ids:
                continue
            action_features = extract_instance_action_features(
                parent_candidate,
                action,
                region_features,
                duplicate_isomorph_history=duplicate_history,
            )
            instance_score = float(self.instance_policy.predict_score(action_features))
            cost_prediction = self.cost_policy.predict(action_features)
            scored_actions.append((instance_score, action.action_id, action, action_features, cost_prediction))
        scored_actions.sort(key=lambda item: (-item[0], item[1]))
        if not scored_actions:
            self._exhausted_parent_keys.add(parent_key)
        return scored_actions

    def _pop_queued_parent(
        self,
    ) -> tuple[str, CandidateRecord, Any, str, Dict[str, Any], float, List[tuple[float, str, Any, Dict[str, Any], CostPrediction]]] | None:
        while True:
            queued = self.queues.pop_continue()
            if queued is None:
                return None
            if queued.canonical_key in self._exhausted_parent_keys:
                continue
            candidate = self._candidate_store.get(queued.canonical_key)
            if candidate is None:
                continue
            spec = self.problem_specs[candidate.family.value]
            region_key = spec.region_key(candidate).value
            region_features = self._region_features_for_candidate(region_key, candidate)
            scored_actions = self._score_available_actions(
                parent_key=queued.canonical_key,
                parent_candidate=candidate,
                spec=spec,
                region_features=region_features,
            )
            if not scored_actions:
                continue
            region_score = float(self.region_policy.predict_score(region_features))
            return queued.canonical_key, candidate, spec, region_key, region_features, region_score, scored_actions

    def _choose_bootstrap_parent(
        self,
    ) -> tuple[str, CandidateRecord, Any, str, Dict[str, Any], float, List[tuple[float, str, Any, Dict[str, Any], CostPrediction]]] | None:
        for region_score, _, region, region_features in self._ranked_regions():
            seeds = self.seed_candidates_by_region.get(region.region_key.value, [])
            if not seeds:
                continue
            offset = self._seed_offsets.get(region.region_key.value, 0)
            for index in range(len(seeds)):
                candidate = seeds[(offset + index) % len(seeds)]
                insert_result, spec = self._ensure_seed_inserted(candidate)
                parent_key = insert_result.canonical_key.key
                if parent_key in self._exhausted_parent_keys:
                    continue
                scored_actions = self._score_available_actions(
                    parent_key=parent_key,
                    parent_candidate=candidate,
                    spec=spec,
                    region_features=region_features,
                )
                if not scored_actions:
                    continue
                self._seed_offsets[region.region_key.value] = offset + index + 1
                self._total_region_visits += 1
                self._region_visit_counts[region.region_key.value] = self._region_visit_counts.get(region.region_key.value, 0) + 1
                return parent_key, candidate, spec, region.region_key.value, region_features, region_score, scored_actions
        return None

    def _choose_work_item(
        self,
    ) -> tuple[str, CandidateRecord, Any, str, Dict[str, Any], float, List[tuple[float, str, Any, Dict[str, Any], CostPrediction]]] | None:
        queued = self._pop_queued_parent()
        if queued is not None:
            return queued
        return self._choose_bootstrap_parent()

    def _label_from_verifier_results(self, results: Sequence[VerifierResult]) -> str:
        statuses = [result.status for result in results]
        if any(status == "ERROR" for status in statuses):
            return "verifier_error"
        if any(status == "INFEASIBLE" for status in statuses):
            return "solver_disagreement" if len(set(statuses)) > 1 else "counterexample_found"
        if any(status == "UNKNOWN" for status in statuses):
            return "unknown_timeout"
        return "exact_feasible"

    def _maybe_retrain(self, step: int) -> None:
        if step <= 0:
            return
        if step % self.retrain_interval != 0:
            return
        if len(self.replay_rows) < self.min_retrain_size:
            return
        self.region_policy = train_region_policy(self.replay_rows, seed=self.seed)
        self.instance_policy = train_instance_policy(self.replay_rows, seed=self.seed)
        self.cost_policy = train_cost_policy(self.replay_rows, seed=self.seed)
        self.action_logs.append({
            "event_type": "retrain",
            "step": step,
            "replay_size": len(self.replay_rows),
            "seed": self.seed,
        })

    def run(self, max_steps: int) -> ControllerRunSummary:
        for step_index in range(max_steps):
            work_item = self._choose_work_item()
            if work_item is None:
                break
            parent_key, parent_candidate, spec, parent_region_key, region_features, region_score, scored_actions = work_item
            instance_score, _, action, action_features, cost_prediction = scored_actions[0]
            self._attempted_action_ids[parent_key].add(action.action_id)

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
            if len(scored_actions) > 1:
                self.queues.push(
                    "continue_search",
                    QueueEntry(
                        canonical_key=parent_key,
                        region_key=parent_region_key,
                        priority=float(scored_actions[1][0]),
                        reason="remaining_actions",
                    ),
                )
            else:
                self._exhausted_parent_keys.add(parent_key)

            log_row = {
                "region_key": parent_region_key,
                "parent_key": parent_key,
                "action": action.to_dict(),
                "region_score": region_score,
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
            self._maybe_retrain(step_index)

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
