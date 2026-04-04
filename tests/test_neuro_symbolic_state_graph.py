import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.canonicalize import CanonicalizationService  # noqa: E402
from python.neuro_symbolic.state_graph import SearchStateGraph  # noqa: E402
from python.neuro_symbolic.store import read_graph_events, replay_graph_events, write_graph_events  # noqa: E402
from python.neuro_symbolic.types import ActionSpec, CandidateFamily, CandidateRecord, OutcomeRecord, RegionKey  # noqa: E402


def _candidate(candidate_id: str, bases: list[int], *, h_vector: list[int]) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        family=CandidateFamily.REPRESENTABLE,
        rank=2,
        n=4,
        bases=bases,
        h_vector=h_vector,
    )


def _region_key(h_vector: list[int]) -> RegionKey:
    display = json.dumps(h_vector, separators=(",", ":"))
    return RegionKey(value=f"h_vector:{display}", display_id=display, family=None, components=h_vector)


def _insert_seed(graph: SearchStateGraph, canonicalizer: CanonicalizationService, candidate: CandidateRecord):
    return graph.insert_seed_instance(
        region_key=_region_key(candidate.h_vector or []),
        canonical_key=canonicalizer.canonicalize_candidate(candidate),
        family=candidate.family.value,
        candidate_id=candidate.candidate_id,
    )


def _insert_mutated(
    graph: SearchStateGraph,
    canonicalizer: CanonicalizationService,
    parent_key: str,
    action: ActionSpec,
    candidate: CandidateRecord,
):
    return graph.insert_mutated_instance(
        parent_key=parent_key,
        action=action,
        region_key=_region_key(candidate.h_vector or []),
        canonical_key=canonicalizer.canonicalize_candidate(candidate),
        family=candidate.family.value,
        candidate_id=candidate.candidate_id,
    )


def test_reinserting_same_canonical_candidate_updates_revisit_count_without_duplicate_node():
    canonicalizer = CanonicalizationService()
    graph = SearchStateGraph()

    created = _insert_seed(graph, canonicalizer, _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))
    revisited = _insert_seed(graph, canonicalizer, _candidate("seed-right", [3, 5, 6, 9, 10], h_vector=[1, 2, 1]))

    assert created.created is True
    assert revisited.created is False
    assert len(graph.instance_nodes) == 1
    instance_node = next(iter(graph.instance_nodes.values()))
    assert instance_node.revisit_count == 1


def test_insert_mutated_instance_records_duplicate_isomorph_event():
    canonicalizer = CanonicalizationService()
    graph = SearchStateGraph()
    seed = _insert_seed(graph, canonicalizer, _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))
    action = ActionSpec(
        action_id="mut-1",
        family=CandidateFamily.REPRESENTABLE,
        action_type="resample_one_column",
        parameters={"column_index": 0},
        locality="single_column",
    )

    duplicate = _insert_mutated(
        graph,
        canonicalizer,
        seed.canonical_key.key,
        action,
        _candidate("seed-right", [3, 5, 6, 9, 10], h_vector=[1, 2, 1]),
    )

    assert duplicate.created is False
    assert duplicate.duplicate_isomorph is True
    assert len(graph.instance_nodes) == 1
    assert any(event["event_type"] == "duplicate_isomorph" for event in graph.event_log())


def test_non_isomorphic_candidate_creates_a_new_node_and_records_outcome():
    canonicalizer = CanonicalizationService()
    graph = SearchStateGraph()
    seed = _insert_seed(graph, canonicalizer, _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))
    action = ActionSpec(
        action_id="mut-2",
        family=CandidateFamily.REPRESENTABLE,
        action_type="resample_one_column",
        parameters={"column_index": 1},
        locality="single_column",
    )
    created = _insert_mutated(
        graph,
        canonicalizer,
        seed.canonical_key.key,
        action,
        _candidate("other", [3, 5, 6, 9], h_vector=[1, 1, 0]),
    )

    graph.record_outcome(
        created.canonical_key.key,
        OutcomeRecord(
            label="exact_feasible",
            canonical_key=created.canonical_key.key,
            display_id=created.canonical_key.display_id,
            details={"verified": True},
        ),
    )

    assert created.created is True
    assert len(graph.instance_nodes) == 2
    assert graph.instance_nodes[created.canonical_key.key].outcome_history[0].label == "exact_feasible"


def test_graph_event_logs_replay_to_the_same_final_node_counts_and_keys():
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / ".pytest_tmp_manual" / "neuro_symbolic_state_graph"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "graph_events.jsonl"

    canonicalizer = CanonicalizationService()
    graph = SearchStateGraph()
    seed = _insert_seed(graph, canonicalizer, _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))
    _insert_mutated(
        graph,
        canonicalizer,
        seed.canonical_key.key,
        ActionSpec(
            action_id="mut-2",
            family=CandidateFamily.REPRESENTABLE,
            action_type="resample_one_column",
            parameters={"column_index": 1},
            locality="single_column",
        ),
        _candidate("other", [3, 5, 6, 9], h_vector=[1, 1, 0]),
    )

    write_graph_events(path, graph.event_log())
    replayed = replay_graph_events(read_graph_events(path))

    assert set(replayed.instance_nodes) == set(graph.instance_nodes)
    assert set(replayed.region_nodes) == set(graph.region_nodes)
    assert replayed.region_nodes["h_vector:[1,2,1]"].seed_instance_keys == graph.region_nodes["h_vector:[1,2,1]"].seed_instance_keys
    assert len(replayed.event_log()) == len(graph.event_log())


def test_graph_events_are_jsonl_friendly():
    graph = SearchStateGraph()
    _insert_seed(graph, CanonicalizationService(), _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))

    payload = graph.event_log()[0]

    assert json.loads(json.dumps(payload))["event_type"] == payload["event_type"]


def test_graph_replay_preserves_seed_membership_without_duplicate_seed_entries():
    canonicalizer = CanonicalizationService()
    graph = SearchStateGraph()

    _insert_seed(graph, canonicalizer, _candidate("seed-left", [5, 6, 9, 10, 12], h_vector=[1, 2, 1]))
    _insert_seed(graph, canonicalizer, _candidate("seed-right", [3, 5, 6, 9, 10], h_vector=[1, 2, 1]))

    replayed = replay_graph_events(graph.event_log())

    assert replayed.region_nodes["h_vector:[1,2,1]"].seed_instance_keys == graph.region_nodes["h_vector:[1,2,1]"].seed_instance_keys
    assert len(replayed.region_nodes["h_vector:[1,2,1]"].seed_instance_keys) == 1
    instance_node = next(iter(replayed.instance_nodes.values()))
    assert instance_node.revisit_count == 1
