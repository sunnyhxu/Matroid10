from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:
    from ..common import ensure_parent, read_jsonl
except ImportError:
    from common import ensure_parent, read_jsonl

from .state_graph import SearchStateGraph


def write_graph_events(path: str | Path, events: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), separators=(",", ":")))
            handle.write("\n")


def read_graph_events(path: str | Path) -> List[Dict[str, Any]]:
    return [dict(event) for event in read_jsonl(path)]


def write_node_snapshots(path: str | Path, graph: SearchStateGraph) -> None:
    ensure_parent(path)
    payload = {
        "regions": [node.to_dict() for node in graph.region_nodes.values()],
        "instances": [node.to_dict() for node in graph.instance_nodes.values()],
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def replay_graph_events(events: Iterable[Mapping[str, Any]]) -> SearchStateGraph:
    graph = SearchStateGraph()
    for event in events:
        graph.apply_event(event)
    return graph
