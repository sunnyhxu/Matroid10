from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from .types import ActionSpec, CanonicalKey, OutcomeRecord, RegionKey


@dataclass
class RegionNode:
    region_key: RegionKey
    seed_instance_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_key": self.region_key.to_dict(),
            "seed_instance_keys": list(self.seed_instance_keys),
        }


@dataclass
class InstanceNode:
    canonical_key: CanonicalKey
    region_key: RegionKey
    family: str
    candidate_id: str | None = None
    revisit_count: int = 0
    outcome_history: List[OutcomeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_key": self.canonical_key.to_dict(),
            "region_key": self.region_key.to_dict(),
            "family": self.family,
            "candidate_id": self.candidate_id,
            "revisit_count": self.revisit_count,
            "outcome_history": [outcome.to_dict() for outcome in self.outcome_history],
        }


@dataclass(frozen=True)
class InsertResult:
    canonical_key: CanonicalKey
    region_key: RegionKey
    created: bool
    duplicate_isomorph: bool


class SearchStateGraph:
    def __init__(self) -> None:
        self.region_nodes: Dict[str, RegionNode] = {}
        self.instance_nodes: Dict[str, InstanceNode] = {}
        self._events: List[Dict[str, Any]] = []

    def event_log(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._events]

    def get_or_create_region(self, region_key: RegionKey) -> RegionNode:
        existing = self.region_nodes.get(region_key.value)
        if existing is not None:
            return existing

        node = RegionNode(region_key=region_key)
        self.region_nodes[region_key.value] = node
        self._events.append(
            {
                "event_type": "region_created",
                "region_key": region_key.to_dict(),
            }
        )
        return node

    def insert_seed_instance(
        self,
        *,
        region_key: RegionKey,
        canonical_key: CanonicalKey,
        family: str,
        candidate_id: str | None = None,
    ) -> InsertResult:
        region = self.get_or_create_region(region_key)
        existing = self.instance_nodes.get(canonical_key.key)
        if existing is not None:
            existing.revisit_count += 1
            self._events.append(
                {
                    "event_type": "seed_instance_revisited",
                    "canonical_key": canonical_key.to_dict(),
                    "region_key": region_key.to_dict(),
                    "family": family,
                    "candidate_id": candidate_id,
                    "revisit_count": existing.revisit_count,
                }
            )
            return InsertResult(
                canonical_key=canonical_key,
                region_key=region_key,
                created=False,
                duplicate_isomorph=False,
            )

        node = InstanceNode(
            canonical_key=canonical_key,
            region_key=region_key,
            family=family,
            candidate_id=candidate_id,
        )
        self.instance_nodes[canonical_key.key] = node
        if canonical_key.key not in region.seed_instance_keys:
            region.seed_instance_keys.append(canonical_key.key)
        self._events.append(
            {
                "event_type": "seed_instance_inserted",
                "canonical_key": canonical_key.to_dict(),
                "region_key": region_key.to_dict(),
                "family": family,
                "candidate_id": candidate_id,
            }
        )
        return InsertResult(
            canonical_key=canonical_key,
            region_key=region_key,
            created=True,
            duplicate_isomorph=False,
        )

    def insert_mutated_instance(
        self,
        *,
        parent_key: str,
        action: ActionSpec,
        region_key: RegionKey,
        canonical_key: CanonicalKey,
        family: str,
        candidate_id: str | None = None,
    ) -> InsertResult:
        self.get_or_create_region(region_key)
        existing = self.instance_nodes.get(canonical_key.key)
        if existing is not None:
            existing.revisit_count += 1
            self.record_duplicate_isomorph(parent_key, action, canonical_key, region_key, family, candidate_id)
            return InsertResult(
                canonical_key=canonical_key,
                region_key=region_key,
                created=False,
                duplicate_isomorph=True,
            )

        node = InstanceNode(
            canonical_key=canonical_key,
            region_key=region_key,
            family=family,
            candidate_id=candidate_id,
        )
        self.instance_nodes[canonical_key.key] = node
        self._events.append(
            {
                "event_type": "mutated_instance_inserted",
                "canonical_key": canonical_key.to_dict(),
                "region_key": region_key.to_dict(),
                "family": family,
                "candidate_id": candidate_id,
                "parent_key": parent_key,
                "action": action.to_dict(),
            }
        )
        return InsertResult(
            canonical_key=canonical_key,
            region_key=region_key,
            created=True,
            duplicate_isomorph=False,
        )

    def record_duplicate_isomorph(
        self,
        parent_key: str,
        action: ActionSpec,
        canonical_key: CanonicalKey,
        region_key: RegionKey | None = None,
        family: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        self._events.append(
            {
                "event_type": "duplicate_isomorph",
                "canonical_key": canonical_key.to_dict(),
                "region_key": None if region_key is None else region_key.to_dict(),
                "family": family,
                "candidate_id": candidate_id,
                "parent_key": parent_key,
                "action": action.to_dict(),
                "revisit_count": self.instance_nodes[canonical_key.key].revisit_count,
            }
        )

    def record_outcome(self, canonical_key: str, outcome: OutcomeRecord) -> None:
        node = self.instance_nodes[canonical_key]
        node.outcome_history.append(outcome)
        self._events.append(
            {
                "event_type": "outcome_recorded",
                "canonical_key": canonical_key,
                "outcome": outcome.to_dict(),
            }
        )

    def apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event["event_type"])

        if event_type == "region_created":
            region_key = RegionKey.from_dict(event["region_key"])
            self.region_nodes.setdefault(region_key.value, RegionNode(region_key=region_key))
            self._events.append(dict(event))
            return

        if event_type in {"seed_instance_inserted", "mutated_instance_inserted"}:
            region_key = RegionKey.from_dict(event["region_key"])
            canonical_key = CanonicalKey.from_dict(event["canonical_key"])
            region = self.region_nodes.setdefault(region_key.value, RegionNode(region_key=region_key))
            self.instance_nodes.setdefault(
                canonical_key.key,
                InstanceNode(
                    canonical_key=canonical_key,
                    region_key=region_key,
                    family=str(event["family"]),
                    candidate_id=None if event.get("candidate_id") is None else str(event["candidate_id"]),
                ),
            )
            if event_type == "seed_instance_inserted" and canonical_key.key not in region.seed_instance_keys:
                region.seed_instance_keys.append(canonical_key.key)
            self._events.append(dict(event))
            return

        if event_type in {"seed_instance_revisited", "duplicate_isomorph"}:
            canonical_key = CanonicalKey.from_dict(event["canonical_key"])
            node = self.instance_nodes[canonical_key.key]
            node.revisit_count = max(node.revisit_count, int(event.get("revisit_count", node.revisit_count)))
            self._events.append(dict(event))
            return

        if event_type == "outcome_recorded":
            canonical_key = str(event["canonical_key"])
            self.instance_nodes[canonical_key].outcome_history.append(OutcomeRecord.from_dict(event["outcome"]))
            self._events.append(dict(event))
            return

        raise ValueError(f"unsupported_event_type:{event_type}")
