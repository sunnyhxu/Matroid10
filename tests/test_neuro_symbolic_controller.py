import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.bootstrap import BootstrapRegionRecord  # noqa: E402
from python.neuro_symbolic.canonicalize import CanonicalizationService  # noqa: E402
from python.neuro_symbolic.controller import NeuroSymbolicController  # noqa: E402
from python.neuro_symbolic.problem_spec import ProblemSpec  # noqa: E402
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


class _ScorePolicy:
    def predict_score(self, features):
        return float(features.get("combined_score", 0.0) or features.get("parent_combined_score", 0.0) or 0.0)


class _CostPolicy:
    def __init__(self, bucket: CostBucket) -> None:
        self.bucket = bucket

    def predict(self, features):
        return CostPrediction(bucket=self.bucket, timeout_risk=0.5, confidence=0.8, details={"source": "test"})


class _ActionTypeScorePolicy:
    def __init__(self, scores):
        self.scores = dict(scores)

    def predict_score(self, features):
        action_type = features.get("action_type")
        if action_type is None:
            return float(features.get("combined_score", 0.0) or 0.0)
        return float(self.scores.get(action_type, 0.0))


class _DummySpec(ProblemSpec):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.canonicalizer = CanonicalizationService()

    def family_name(self) -> str:
        return CandidateFamily.REPRESENTABLE.value

    def region_key(self, candidate: CandidateRecord) -> RegionKey:
        return RegionKey(value="h_vector:[1,2,1]", display_id="[1,2,1]", family=None, components=[1, 2, 1])

    def canonicalize(self, candidate: CandidateRecord):
        return self.canonicalizer.canonicalize_candidate(candidate)

    def enumerate_valid_actions(self, candidate: CandidateRecord):
        return [
            ActionSpec(
                action_id=f"{self.mode}:action",
                family=CandidateFamily.REPRESENTABLE,
                action_type="dummy_action",
                parameters={"mode": self.mode},
                locality="single_column",
                cross_region=False,
            )
        ]

    def apply_action(self, candidate: CandidateRecord, action: ActionSpec):
        if self.mode == "duplicate":
            return CandidateRecord(
                candidate_id="dup",
                family=CandidateFamily.REPRESENTABLE,
                rank=2,
                n=4,
                bases=[3, 5, 6, 9, 10],
                h_vector=[1, 2, 1],
            )
        if self.mode == "filter_block":
            return CandidateRecord(
                candidate_id="filter",
                family=CandidateFamily.REPRESENTABLE,
                rank=2,
                n=4,
                bases=[3, 5, 6, 9],
                h_vector=[1, 1, 0],
                metadata={"filter_block": True},
            )
        return CandidateRecord(
            candidate_id="timeout",
            family=CandidateFamily.REPRESENTABLE,
            rank=2,
            n=4,
            bases=[3, 5, 6, 9],
            h_vector=[1, 1, 0],
        )

    def cheap_filters(self, candidate: CandidateRecord):
        if candidate.metadata.get("filter_block"):
            return [FilterResult(name="blocked", passed=False, reason="blocked", details={})]
        return [FilterResult(name="blocked", passed=True, reason=None, details={})]

    def exact_verifiers(self):
        return [{"name": "reference_solver"}, {"name": "top_level_solver"}]


class _SpySpec(_DummySpec):
    def __init__(self, mode: str, *, region_key: RegionKey, canonical_key: str, canonical_display_id: str = "spy-display") -> None:
        super().__init__(mode)
        self._region_key = region_key
        self._canonical_key = canonical_key
        self._canonical_display_id = canonical_display_id

    def region_key(self, candidate: CandidateRecord) -> RegionKey:
        return self._region_key

    def canonicalize(self, candidate: CandidateRecord):
        return CanonicalKey(key=self._canonical_key, display_id=self._canonical_display_id, digest="spy-digest")


class _ExplorationSpec(ProblemSpec):
    def __init__(self, candidates):
        self.candidates = dict(candidates)

    def family_name(self) -> str:
        return CandidateFamily.REPRESENTABLE.value

    def region_key(self, candidate: CandidateRecord) -> RegionKey:
        region_key = str(candidate.metadata["region_key"])
        return RegionKey(value=region_key, display_id=region_key, family=None, components=list(candidate.h_vector or []))

    def canonicalize(self, candidate: CandidateRecord):
        canonical_key = str(candidate.metadata["canonical_key"])
        return CanonicalKey(key=canonical_key, display_id=canonical_key, digest=f"digest:{canonical_key}")

    def enumerate_valid_actions(self, candidate: CandidateRecord):
        actions = []
        for action_row in candidate.metadata.get("actions", []):
            actions.append(
                ActionSpec(
                    action_id=str(action_row["action_id"]),
                    family=CandidateFamily.REPRESENTABLE,
                    action_type=str(action_row["action_type"]),
                    parameters={"next_candidate_id": str(action_row["next_candidate_id"])},
                    locality="single_column",
                    cross_region=False,
                )
            )
        return actions

    def apply_action(self, candidate: CandidateRecord, action: ActionSpec):
        return self.candidates[str(action.parameters["next_candidate_id"])]

    def cheap_filters(self, candidate: CandidateRecord):
        return [FilterResult(name="blocked", passed=True, reason=None, details={})]

    def exact_verifiers(self):
        return [{"name": "reference_solver"}]


def _bootstrap_region() -> BootstrapRegionRecord:
    return BootstrapRegionRecord(
        region_key=RegionKey(value="h_vector:[1,2,1]", display_id="[1,2,1]", family=None, components=[1, 2, 1]),
        h_vector=[1, 2, 1],
        h_vector_key="[1,2,1]",
        source_records=1,
        example_record_id="seed-a",
        current_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
        top_level_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
        structural_metrics={"raw_score": 1.0, "macaulay_total_slack": 0, "late_drop_sum": 0},
        current_solver_score=0.1,
        top_level_solver_score=0.1,
        structural_score=0.1,
        combined_score_weights={"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
        score_normalization="percentile",
        combined_score=0.5,
        witness_references=["seed-a"],
    )


def _bootstrap_region_with_score(
    *,
    region_key: str,
    h_vector,
    combined_score: float,
    example_record_id: str,
) -> BootstrapRegionRecord:
    h_vector_list = list(h_vector)
    return BootstrapRegionRecord(
        region_key=RegionKey(value=region_key, display_id=region_key, family=None, components=h_vector_list),
        h_vector=h_vector_list,
        h_vector_key=region_key,
        source_records=1,
        example_record_id=example_record_id,
        current_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
        top_level_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
        structural_metrics={"raw_score": 1.0, "macaulay_total_slack": 0, "late_drop_sum": 0},
        current_solver_score=0.1,
        top_level_solver_score=0.1,
        structural_score=0.1,
        combined_score_weights={"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
        score_normalization="percentile",
        combined_score=combined_score,
        witness_references=[example_record_id],
    )


def _seed_candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="seed-a",
        family=CandidateFamily.REPRESENTABLE,
        rank=2,
        n=4,
        bases=[5, 6, 9, 10, 12],
        h_vector=[1, 2, 1],
    )


def _exploration_candidate(
    candidate_id: str,
    *,
    region_key: str,
    canonical_key: str,
    h_vector,
    actions,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        family=CandidateFamily.REPRESENTABLE,
        rank=2,
        n=4,
        bases=[5, 6, 9, 10, 12],
        h_vector=list(h_vector),
        metadata={
            "region_key": region_key,
            "canonical_key": canonical_key,
            "actions": list(actions),
        },
    )


def _seed_candidate_for_region(candidate_id: str, *, region_key: str, canonical_key: str, next_candidate_id: str) -> CandidateRecord:
    region_components = [int(value) for value in region_key.split(":")[-1].strip("[]").split(",")]
    return _exploration_candidate(
        candidate_id,
        region_key=region_key,
        canonical_key=canonical_key,
        h_vector=region_components,
        actions=[
            {
                "action_id": f"{candidate_id}:advance",
                "action_type": "advance",
                "next_candidate_id": next_candidate_id,
            }
        ],
    )


def test_controller_skips_duplicate_isomorph_expansions_after_canonical_merge():
    verifier_calls = []

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("duplicate")},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: verifier_calls.append("reference"),
            "top_level_solver": lambda candidate, timeout_seconds: verifier_calls.append("top"),
        },
    )

    summary = controller.run(max_steps=1)

    assert summary.action_logs[0]["outcome_label"] == "duplicate_isomorph"
    assert len(controller.graph.instance_nodes) == 1
    assert verifier_calls == []


def test_candidate_blocked_by_cheap_filters_never_reaches_exact_verifiers():
    verifier_calls = []

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("filter_block")},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: verifier_calls.append("reference"),
            "top_level_solver": lambda candidate, timeout_seconds: verifier_calls.append("top"),
        },
    )

    summary = controller.run(max_steps=1)

    assert summary.action_logs[0]["outcome_label"] == "invalid_action_blocked"
    assert verifier_calls == []


def test_timeout_enters_escalation_queue_and_not_terminal_complete():
    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("timeout")},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.LIKELY_TIMEOUT),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="UNKNOWN",
                wall_time=timeout_seconds,
                details={"timed_out": True},
                censored=True,
            ),
            "top_level_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="top_level_solver",
                status="FEASIBLE",
                wall_time=timeout_seconds,
                details={},
                censored=False,
            ),
        },
    )

    summary = controller.run(max_steps=1)

    assert summary.action_logs[0]["queue_destination"] == "escalation"
    assert summary.queue_counts["escalation"] == 1
    assert summary.queue_counts["terminal_complete"] == 0


def test_controller_expands_discovered_continuation_before_reseeding_bootstrap_parent():
    seed = _exploration_candidate(
        "seed-a",
        region_key="h_vector:[1,2,1]",
        canonical_key="seed",
        h_vector=[1, 2, 1],
        actions=[
            {"action_id": "seed:to_discovered", "action_type": "seed_to_discovered", "next_candidate_id": "child"},
            {"action_id": "seed:backup", "action_type": "seed_backup", "next_candidate_id": "sibling"},
        ],
    )
    child = _exploration_candidate(
        "child",
        region_key="h_vector:[2,2,1]",
        canonical_key="child",
        h_vector=[2, 2, 1],
        actions=[
            {"action_id": "child:advance", "action_type": "child_advance", "next_candidate_id": "leaf"},
        ],
    )
    sibling = _exploration_candidate(
        "sibling",
        region_key="h_vector:[1,2,1]",
        canonical_key="sibling",
        h_vector=[1, 2, 1],
        actions=[],
    )
    leaf = _exploration_candidate(
        "leaf",
        region_key="h_vector:[2,2,2]",
        canonical_key="leaf",
        h_vector=[2, 2, 2],
        actions=[],
    )
    spec = _ExplorationSpec({candidate.candidate_id: candidate for candidate in [seed, child, sibling, leaf]})

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [seed]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: spec},
        region_policy=_ScorePolicy(),
        instance_policy=_ActionTypeScorePolicy(
            {
                "seed_to_discovered": 10.0,
                "seed_backup": 1.0,
                "child_advance": 9.0,
            }
        ),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="FEASIBLE",
                wall_time=timeout_seconds,
                details={},
                censored=False,
            ),
        },
    )

    summary = controller.run(max_steps=3)

    assert [row["parent_key"] for row in summary.action_logs] == ["seed", "child", "seed"]
    assert [row["region_key"] for row in summary.action_logs] == [
        "h_vector:[1,2,1]",
        "h_vector:[2,2,1]",
        "h_vector:[1,2,1]",
    ]
    assert [row["action"]["action_id"] for row in summary.action_logs] == [
        "seed:to_discovered",
        "child:advance",
        "seed:backup",
    ]


def test_controller_never_retries_same_parent_action_pair_and_marks_parent_exhausted():
    seed = _exploration_candidate(
        "seed-a",
        region_key="h_vector:[1,2,1]",
        canonical_key="seed",
        h_vector=[1, 2, 1],
        actions=[
            {"action_id": "seed:only", "action_type": "seed_only", "next_candidate_id": "leaf"},
        ],
    )
    leaf = _exploration_candidate(
        "leaf",
        region_key="h_vector:[2,2,1]",
        canonical_key="leaf",
        h_vector=[2, 2, 1],
        actions=[],
    )
    spec = _ExplorationSpec({candidate.candidate_id: candidate for candidate in [seed, leaf]})

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [seed]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: spec},
        region_policy=_ScorePolicy(),
        instance_policy=_ActionTypeScorePolicy({"seed_only": 5.0}),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="FEASIBLE",
                wall_time=timeout_seconds,
                details={},
                censored=False,
            ),
        },
    )

    summary = controller.run(max_steps=4)

    assert len(summary.action_logs) == 1
    assert summary.action_logs[0]["parent_key"] == "seed"
    assert summary.action_logs[0]["action"]["action_id"] == "seed:only"
    assert controller._attempted_action_ids["seed"] == {"seed:only"}
    assert "seed" in controller._exhausted_parent_keys
    assert "leaf" in controller._exhausted_parent_keys
    assert controller.queues.counts()["continue_search"] == 0


def test_ucb_beta_zero_gives_pure_policy_score_ordering():
    controller = NeuroSymbolicController(
        bootstrap_regions=[
            _bootstrap_region_with_score(
                region_key="region-low",
                h_vector=[1, 1, 1],
                combined_score=0.2,
                example_record_id="seed-low",
            ),
            _bootstrap_region_with_score(
                region_key="region-high",
                h_vector=[2, 1, 1],
                combined_score=0.9,
                example_record_id="seed-high",
            ),
        ],
        seed_candidates_by_region={},
        problem_specs={},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        ucb_beta=0.0,
    )

    ranked_region_keys = [row[2].region_key.value for row in controller._ranked_regions()]

    assert ranked_region_keys == ["region-high", "region-low"]


def test_ucb_bonus_favors_unvisited_regions():
    controller = NeuroSymbolicController(
        bootstrap_regions=[
            _bootstrap_region_with_score(
                region_key="a-visited-region",
                h_vector=[1, 1, 1],
                combined_score=0.5,
                example_record_id="seed-visited",
            ),
            _bootstrap_region_with_score(
                region_key="z-unvisited-region",
                h_vector=[2, 1, 1],
                combined_score=0.5,
                example_record_id="seed-unvisited",
            ),
        ],
        seed_candidates_by_region={},
        problem_specs={},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        ucb_beta=1.0,
    )
    controller._total_region_visits = 10
    controller._region_visit_counts["a-visited-region"] = 10

    ranked_region_keys = [row[2].region_key.value for row in controller._ranked_regions()]

    assert ranked_region_keys == ["z-unvisited-region", "a-visited-region"]


def test_ucb_visits_more_regions_than_greedy():
    regions = [
        _bootstrap_region_with_score(
            region_key="h_vector:[9,1,1]",
            h_vector=[9, 1, 1],
            combined_score=1.0,
            example_record_id="seed-a-0",
        ),
        _bootstrap_region_with_score(
            region_key="h_vector:[8,1,1]",
            h_vector=[8, 1, 1],
            combined_score=0.95,
            example_record_id="seed-b-0",
        ),
        _bootstrap_region_with_score(
            region_key="h_vector:[7,1,1]",
            h_vector=[7, 1, 1],
            combined_score=0.94,
            example_record_id="seed-c-0",
        ),
    ]

    seed_candidates_by_region = {
        "h_vector:[9,1,1]": [
            _seed_candidate_for_region(
                f"seed-a-{index}",
                region_key="h_vector:[9,1,1]",
                canonical_key=f"seed-a-{index}",
                next_candidate_id=f"leaf-a-{index}",
            )
            for index in range(40)
        ],
        "h_vector:[8,1,1]": [
            _seed_candidate_for_region(
                f"seed-b-{index}",
                region_key="h_vector:[8,1,1]",
                canonical_key=f"seed-b-{index}",
                next_candidate_id=f"leaf-b-{index}",
            )
            for index in range(40)
        ],
        "h_vector:[7,1,1]": [
            _seed_candidate_for_region(
                f"seed-c-{index}",
                region_key="h_vector:[7,1,1]",
                canonical_key=f"seed-c-{index}",
                next_candidate_id=f"leaf-c-{index}",
            )
            for index in range(40)
        ],
    }

    candidates = {}
    for region_key, seeds in seed_candidates_by_region.items():
        for seed in seeds:
            candidates[seed.candidate_id] = seed
            next_candidate_id = seed.metadata["actions"][0]["next_candidate_id"]
            candidates[next_candidate_id] = _exploration_candidate(
                next_candidate_id,
                region_key=region_key,
                canonical_key=next_candidate_id,
                h_vector=seed.h_vector or [],
                actions=[],
            )

    problem_spec = _ExplorationSpec(candidates)
    verifier_functions = {
        "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
            name="reference_solver",
            status="FEASIBLE",
            wall_time=timeout_seconds,
            details={},
            censored=False,
        ),
    }

    greedy_controller = NeuroSymbolicController(
        bootstrap_regions=regions,
        seed_candidates_by_region=seed_candidates_by_region,
        problem_specs={CandidateFamily.REPRESENTABLE.value: problem_spec},
        region_policy=_ScorePolicy(),
        instance_policy=_ActionTypeScorePolicy({"advance": 1.0}),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions=verifier_functions,
        seed=17,
        ucb_beta=0.0,
    )
    ucb_controller = NeuroSymbolicController(
        bootstrap_regions=regions,
        seed_candidates_by_region=seed_candidates_by_region,
        problem_specs={CandidateFamily.REPRESENTABLE.value: problem_spec},
        region_policy=_ScorePolicy(),
        instance_policy=_ActionTypeScorePolicy({"advance": 1.0}),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions=verifier_functions,
        seed=17,
        ucb_beta=0.5,
    )

    greedy_summary = greedy_controller.run(max_steps=30)
    ucb_summary = ucb_controller.run(max_steps=30)

    greedy_regions = {row["region_key"] for row in greedy_summary.action_logs if "region_key" in row}
    ucb_regions = {row["region_key"] for row in ucb_summary.action_logs if "region_key" in row}

    assert len(ucb_regions) > len(greedy_regions)


def test_verifier_error_routes_to_escalation_and_stays_distinct_from_exact_feasible():
    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("timeout")},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="ERROR",
                wall_time=timeout_seconds,
                details={"message": "solver crashed"},
                censored=False,
            ),
        },
    )

    summary = controller.run(max_steps=1)

    assert summary.action_logs[0]["outcome_label"] == "verifier_error"
    assert summary.action_logs[0]["queue_destination"] == "escalation"
    assert summary.replay_rows[0]["outcome_label"] == "verifier_error"
    assert summary.queue_counts["escalation"] == 1
    assert summary.queue_counts["continue_search"] == 0


def test_end_to_end_controller_logs_are_deterministic_under_fixed_seed():
    kwargs = dict(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("timeout")},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.LIKELY_TIMEOUT),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="UNKNOWN",
                wall_time=timeout_seconds,
                details={"timed_out": True},
                censored=True,
            ),
            "top_level_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="top_level_solver",
                status="FEASIBLE",
                wall_time=timeout_seconds,
                details={},
                censored=False,
            ),
        },
        seed=17,
    )

    first = NeuroSymbolicController(**kwargs).run(max_steps=1)
    second = NeuroSymbolicController(**kwargs).run(max_steps=1)

    assert first.action_logs == second.action_logs
    assert first.replay_rows == second.replay_rows


def test_controller_uses_problem_spec_region_and_canonical_keys_for_graph_and_logs():
    custom_region = RegionKey(
        value="custom-region",
        display_id="custom-region",
        family="representable",
        components=[9, 9, 9],
    )
    mutated_spec = _SpySpec("timeout", region_key=custom_region, canonical_key="custom-canonical-key")

    controller = NeuroSymbolicController(
        bootstrap_regions=[
            BootstrapRegionRecord(
                region_key=custom_region,
                h_vector=[1, 2, 1],
                h_vector_key="[1,2,1]",
                source_records=1,
                example_record_id="seed-a",
                current_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
                top_level_solver={"status": "FEASIBLE", "wall_time": 0.1, "metrics": {"num_conflicts": 0, "num_branches": 0}},
                structural_metrics={"raw_score": 1.0, "macaulay_total_slack": 0, "late_drop_sum": 0},
                current_solver_score=0.1,
                top_level_solver_score=0.1,
                structural_score=0.1,
                combined_score_weights={"current_solver": 1.0, "top_level_solver": 1.0, "structural": 1.0},
                score_normalization="percentile",
                combined_score=0.5,
                witness_references=["seed-a"],
            )
        ],
        seed_candidates_by_region={"custom-region": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: mutated_spec},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.LIKELY_TIMEOUT),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="UNKNOWN",
                wall_time=timeout_seconds,
                details={"timed_out": True},
                censored=True,
            ),
            "top_level_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="top_level_solver",
                status="FEASIBLE",
                wall_time=timeout_seconds,
                details={},
                censored=False,
            ),
        },
    )

    summary = controller.run(max_steps=1)

    assert "custom-region" in controller.graph.region_nodes
    assert "custom-canonical-key" in controller.graph.instance_nodes
    assert summary.action_logs[0]["region_key"] == "custom-region"
    assert summary.action_logs[0]["canonicalization"]["key"] == "custom-canonical-key"


def test_controller_retrains_all_policies_after_interval_when_buffer_ready():
    from python.neuro_symbolic.policies import LinearScoreModel, CostPolicyModel

    original_region_policy = _ScorePolicy()
    original_instance_policy = _ScorePolicy()
    original_cost_policy = _CostPolicy(CostBucket.CHEAP)

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("exact_feasible")},
        region_policy=original_region_policy,
        instance_policy=original_instance_policy,
        cost_policy=original_cost_policy,
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="FEASIBLE",
                wall_time=0.1,
                details={},
                censored=False,
            ),
            "top_level_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="top_level_solver",
                status="FEASIBLE",
                wall_time=0.1,
                details={},
                censored=False,
            ),
        },
        retrain_interval=1,
        min_retrain_size=1,
    )

    controller.run(max_steps=2)

    assert controller.region_policy is not original_region_policy
    assert controller.instance_policy is not original_instance_policy
    assert controller.cost_policy is not original_cost_policy
    assert isinstance(controller.region_policy, LinearScoreModel)
    assert isinstance(controller.instance_policy, LinearScoreModel)
    assert isinstance(controller.cost_policy, CostPolicyModel)


def test_controller_does_not_retrain_before_min_replay_size():
    original_region_policy = _ScorePolicy()
    original_instance_policy = _ScorePolicy()
    original_cost_policy = _CostPolicy(CostBucket.CHEAP)

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [_seed_candidate()]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _DummySpec("exact_feasible")},
        region_policy=original_region_policy,
        instance_policy=original_instance_policy,
        cost_policy=original_cost_policy,
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="FEASIBLE",
                wall_time=0.1,
                details={},
                censored=False,
            ),
            "top_level_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="top_level_solver",
                status="FEASIBLE",
                wall_time=0.1,
                details={},
                censored=False,
            ),
        },
        retrain_interval=1,
        min_retrain_size=3,
    )

    controller.run(max_steps=2)

    assert controller.region_policy is original_region_policy
    assert controller.instance_policy is original_instance_policy
    assert controller.cost_policy is original_cost_policy
    retrain_events = [e for e in controller.action_logs if e.get("event_type") == "retrain"]
    assert retrain_events == []


def test_controller_logs_retrain_event_at_expected_step():
    # Use _ExplorationSpec with 3 distinct children so the loop runs 3 full steps.
    root = _exploration_candidate(
        "root",
        region_key="h_vector:[1,2,1]",
        canonical_key="root",
        h_vector=[1, 2, 1],
        actions=[
            {"action_id": "root:to_a", "action_type": "seed_to_a", "next_candidate_id": "child_a"},
            {"action_id": "root:to_b", "action_type": "seed_to_b", "next_candidate_id": "child_b"},
            {"action_id": "root:to_c", "action_type": "seed_to_c", "next_candidate_id": "child_c"},
        ],
    )
    child_a = _exploration_candidate("child_a", region_key="h_vector:[1,2,1]", canonical_key="child_a", h_vector=[1, 2, 1], actions=[])
    child_b = _exploration_candidate("child_b", region_key="h_vector:[1,2,1]", canonical_key="child_b", h_vector=[1, 2, 1], actions=[])
    child_c = _exploration_candidate("child_c", region_key="h_vector:[1,2,1]", canonical_key="child_c", h_vector=[1, 2, 1], actions=[])

    controller = NeuroSymbolicController(
        bootstrap_regions=[_bootstrap_region()],
        seed_candidates_by_region={"h_vector:[1,2,1]": [root]},
        problem_specs={CandidateFamily.REPRESENTABLE.value: _ExplorationSpec({
            "child_a": child_a,
            "child_b": child_b,
            "child_c": child_c,
        })},
        region_policy=_ScorePolicy(),
        instance_policy=_ScorePolicy(),
        cost_policy=_CostPolicy(CostBucket.CHEAP),
        verifier_functions={
            "reference_solver": lambda candidate, timeout_seconds: VerifierResult(
                name="reference_solver",
                status="FEASIBLE",
                wall_time=0.1,
                details={},
                censored=False,
            ),
        },
        retrain_interval=2,
        min_retrain_size=2,
        seed=17,
    )

    controller.run(max_steps=3)

    retrain_events = [e for e in controller.action_logs if e.get("event_type") == "retrain"]
    assert len(retrain_events) == 1
    event = retrain_events[0]
    assert event["step"] == 2
    assert event["replay_size"] == 3
    assert event["seed"] == 17
