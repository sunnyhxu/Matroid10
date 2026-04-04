from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Mapping


class CandidateFamily(str, Enum):
    REPRESENTABLE = "representable"
    SPARSE_PAVING = "sparse_paving"


class CostBucket(str, Enum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"
    LIKELY_TIMEOUT = "likely_timeout"


def _copy_mapping(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if payload is None:
        return {}
    return dict(payload)


def _copy_int_list(values: List[int] | None) -> List[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


@dataclass(frozen=True)
class CanonicalKey:
    key: str
    display_id: str | None = None
    digest: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display_id": self.display_id,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalKey":
        return cls(
            key=str(payload["key"]),
            display_id=None if payload.get("display_id") is None else str(payload["display_id"]),
            digest=None if payload.get("digest") is None else str(payload["digest"]),
        )


@dataclass(frozen=True)
class RegionKey:
    value: str
    display_id: str | None = None
    family: str | None = None
    components: List[int] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "display_id": self.display_id,
            "family": self.family,
            "components": _copy_int_list(self.components),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionKey":
        components = payload.get("components")
        return cls(
            value=str(payload["value"]),
            display_id=None if payload.get("display_id") is None else str(payload["display_id"]),
            family=None if payload.get("family") is None else str(payload["family"]),
            components=_copy_int_list(components),
        )


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    family: CandidateFamily
    action_type: str
    parameters: Dict[str, Any] = dc_field(default_factory=dict)
    locality: str = "local"
    cross_region: bool = False
    description: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "family": self.family.value,
            "action_type": self.action_type,
            "parameters": _copy_mapping(self.parameters),
            "locality": self.locality,
            "cross_region": self.cross_region,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSpec":
        return cls(
            action_id=str(payload["action_id"]),
            family=CandidateFamily(str(payload["family"])),
            action_type=str(payload["action_type"]),
            parameters=_copy_mapping(payload.get("parameters")),
            locality=str(payload.get("locality", "local")),
            cross_region=bool(payload.get("cross_region", False)),
            description=None if payload.get("description") is None else str(payload["description"]),
        )


@dataclass(frozen=True)
class FilterResult:
    name: str
    passed: bool
    reason: str | None = None
    details: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "details": _copy_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FilterResult":
        return cls(
            name=str(payload["name"]),
            passed=bool(payload["passed"]),
            reason=None if payload.get("reason") is None else str(payload["reason"]),
            details=_copy_mapping(payload.get("details")),
        )


@dataclass(frozen=True)
class VerifierResult:
    name: str
    status: str
    wall_time: float
    details: Dict[str, Any] = dc_field(default_factory=dict)
    censored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "wall_time": float(self.wall_time),
            "details": _copy_mapping(self.details),
            "censored": self.censored,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerifierResult":
        return cls(
            name=str(payload["name"]),
            status=str(payload["status"]),
            wall_time=float(payload["wall_time"]),
            details=_copy_mapping(payload.get("details")),
            censored=bool(payload.get("censored", False)),
        )


@dataclass(frozen=True)
class OutcomeRecord:
    label: str
    canonical_key: str | None = None
    display_id: str | None = None
    details: Dict[str, Any] = dc_field(default_factory=dict)
    censored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "canonical_key": self.canonical_key,
            "display_id": self.display_id,
            "details": _copy_mapping(self.details),
            "censored": self.censored,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeRecord":
        return cls(
            label=str(payload["label"]),
            canonical_key=None if payload.get("canonical_key") is None else str(payload["canonical_key"]),
            display_id=None if payload.get("display_id") is None else str(payload["display_id"]),
            details=_copy_mapping(payload.get("details")),
            censored=bool(payload.get("censored", False)),
        )


@dataclass(frozen=True)
class CostPrediction:
    bucket: CostBucket
    timeout_risk: float
    confidence: float
    details: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket.value,
            "timeout_risk": float(self.timeout_risk),
            "confidence": float(self.confidence),
            "details": _copy_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostPrediction":
        return cls(
            bucket=CostBucket(str(payload["bucket"])),
            timeout_risk=float(payload["timeout_risk"]),
            confidence=float(payload["confidence"]),
            details=_copy_mapping(payload.get("details")),
        )


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str | None
    family: CandidateFamily
    rank: int
    n: int
    bases: List[int]
    field: int | None = None
    matrix_cols: List[int] | None = None
    circuit_hyperplanes: List[int] | None = None
    sparse_overlap_bound: int | None = None
    non_paving_witness: int | None = None
    h_vector: List[int] | None = None
    seed: int | None = None
    trial: int | None = None
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.candidate_id,
            "generator_mode": self.family.value,
            "rank": int(self.rank),
            "n": int(self.n),
            "bases": _copy_int_list(self.bases) or [],
            "field": self.field,
            "matrix_cols": _copy_int_list(self.matrix_cols),
            "circuit_hyperplanes": _copy_int_list(self.circuit_hyperplanes),
            "sparse_overlap_bound": self.sparse_overlap_bound,
            "non_paving_witness": self.non_paving_witness,
            "h_vector": _copy_int_list(self.h_vector),
            "seed": self.seed,
            "trial": self.trial,
        }
        if self.metadata:
            payload["metadata"] = _copy_mapping(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        family_value = str(payload.get("generator_mode", CandidateFamily.REPRESENTABLE.value))
        return cls(
            candidate_id=None if payload.get("id") is None else str(payload["id"]),
            family=CandidateFamily(family_value),
            rank=int(payload["rank"]),
            n=int(payload["n"]),
            bases=_copy_int_list(payload.get("bases")) or [],
            field=None if payload.get("field") is None else int(payload["field"]),
            matrix_cols=_copy_int_list(payload.get("matrix_cols")),
            circuit_hyperplanes=_copy_int_list(payload.get("circuit_hyperplanes")),
            sparse_overlap_bound=None
            if payload.get("sparse_overlap_bound") is None
            else int(payload["sparse_overlap_bound"]),
            non_paving_witness=None
            if payload.get("non_paving_witness") is None
            else int(payload["non_paving_witness"]),
            h_vector=_copy_int_list(payload.get("h_vector")),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            trial=None if payload.get("trial") is None else int(payload["trial"]),
            metadata=_copy_mapping(payload.get("metadata")),
        )
