from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:
    from ..common import read_jsonl
except ImportError:
    from common import read_jsonl

from .types import RegionKey


DEFAULT_BOOTSTRAP_ARTIFACT = Path("artifacts/hardness_unique_hvectors.jsonl")


class BootstrapFormatError(ValueError):
    """Raised when a hardness bootstrap row is missing required structure."""


def _h_vector_key(h_vector: Iterable[int]) -> str:
    return json.dumps([int(value) for value in h_vector], separators=(",", ":"))


def _require_mapping(payload: Mapping[str, Any], field_name: str) -> Dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise BootstrapFormatError(f"missing_or_invalid_{field_name}")
    return dict(value)


def _require_int_list(payload: Mapping[str, Any], field_name: str) -> List[int]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not value:
        raise BootstrapFormatError(f"missing_or_invalid_{field_name}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise BootstrapFormatError(f"missing_or_invalid_{field_name}") from exc


def _optional_str(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return str(value)


def _optional_float(payload: Mapping[str, Any], field_name: str, default: float = 0.0) -> float:
    value = payload.get(field_name, default)
    if value is None:
        return default
    return float(value)


def _witness_references(rows: List[Mapping[str, Any]]) -> List[str]:
    example_ids: List[str] = []
    other_refs: List[str] = []
    for row in rows:
        example_record_id = _optional_str(row, "example_record_id")
        if example_record_id is not None:
            example_ids.append(example_record_id)
        for key in ("witness_references", "witness_ids"):
            raw_list = row.get(key)
            if isinstance(raw_list, list):
                other_refs.extend(str(item) for item in raw_list)
        for key in ("witness_reference", "witness_id"):
            raw_value = row.get(key)
            if raw_value is not None:
                other_refs.append(str(raw_value))

    ordered_examples = sorted(set(example_ids))
    ordered_other_refs = sorted(ref for ref in set(other_refs) if ref not in ordered_examples)
    return ordered_examples + ordered_other_refs


@dataclass(frozen=True)
class BootstrapRegionRecord:
    region_key: RegionKey
    h_vector: List[int]
    h_vector_key: str
    source_records: int
    example_record_id: str | None
    current_solver: Dict[str, Any]
    top_level_solver: Dict[str, Any]
    structural_metrics: Dict[str, Any]
    current_solver_score: float
    top_level_solver_score: float
    structural_score: float
    combined_score_weights: Dict[str, float]
    score_normalization: str
    combined_score: float
    witness_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_key": self.region_key.to_dict(),
            "h_vector": [int(value) for value in self.h_vector],
            "h_vector_key": self.h_vector_key,
            "source_records": int(self.source_records),
            "example_record_id": self.example_record_id,
            "current_solver": dict(self.current_solver),
            "top_level_solver": dict(self.top_level_solver),
            "structural_metrics": dict(self.structural_metrics),
            "current_solver_score": float(self.current_solver_score),
            "top_level_solver_score": float(self.top_level_solver_score),
            "structural_score": float(self.structural_score),
            "combined_score_weights": dict(self.combined_score_weights),
            "score_normalization": self.score_normalization,
            "combined_score": float(self.combined_score),
            "witness_references": list(self.witness_references),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BootstrapRegionRecord":
        return cls(
            region_key=RegionKey.from_dict(payload["region_key"]),
            h_vector=[int(value) for value in payload["h_vector"]],
            h_vector_key=str(payload["h_vector_key"]),
            source_records=int(payload["source_records"]),
            example_record_id=_optional_str(payload, "example_record_id"),
            current_solver=dict(payload["current_solver"]),
            top_level_solver=dict(payload["top_level_solver"]),
            structural_metrics=dict(payload["structural_metrics"]),
            current_solver_score=float(payload["current_solver_score"]),
            top_level_solver_score=float(payload["top_level_solver_score"]),
            structural_score=float(payload["structural_score"]),
            combined_score_weights={str(key): float(value) for key, value in dict(payload["combined_score_weights"]).items()},
            score_normalization=str(payload["score_normalization"]),
            combined_score=float(payload["combined_score"]),
            witness_references=[str(value) for value in payload.get("witness_references", [])],
        )


def _coerce_bootstrap_row(row: Mapping[str, Any], grouped_rows: List[Mapping[str, Any]]) -> BootstrapRegionRecord:
    h_vector = _require_int_list(row, "h_vector")
    h_vector_key = str(row.get("h_vector_key") or _h_vector_key(h_vector))
    current_solver = _require_mapping(row, "current_solver")
    top_level_solver = _require_mapping(row, "top_level_solver")
    structural_metrics = _require_mapping(row, "structural_metrics")
    combined_score_weights = {
        str(key): float(value) for key, value in _require_mapping(row, "combined_score_weights").items()
    }
    region_key = RegionKey(
        value=f"h_vector:{h_vector_key}",
        display_id=h_vector_key,
        family=None,
        components=h_vector,
    )
    return BootstrapRegionRecord(
        region_key=region_key,
        h_vector=h_vector,
        h_vector_key=h_vector_key,
        source_records=max(int(candidate.get("source_records", 0) or 0) for candidate in grouped_rows),
        example_record_id=_optional_str(row, "example_record_id"),
        current_solver=current_solver,
        top_level_solver=top_level_solver,
        structural_metrics=structural_metrics,
        current_solver_score=_optional_float(row, "current_solver_score"),
        top_level_solver_score=_optional_float(row, "top_level_solver_score"),
        structural_score=_optional_float(row, "structural_score"),
        combined_score_weights=combined_score_weights,
        score_normalization=str(row.get("score_normalization", "percentile")),
        combined_score=_optional_float(row, "combined_score"),
        witness_references=_witness_references(grouped_rows),
    )


def bootstrap_regions_from_records(records: Iterable[Mapping[str, Any]]) -> List[BootstrapRegionRecord]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        h_vector = _require_int_list(row, "h_vector")
        key = str(row.get("h_vector_key") or _h_vector_key(h_vector))
        grouped.setdefault(key, []).append(row)

    output: List[BootstrapRegionRecord] = []
    for key in sorted(grouped):
        grouped_rows = sorted(grouped[key], key=lambda row: _optional_str(row, "example_record_id") or "")
        representative = grouped_rows[0]
        output.append(_coerce_bootstrap_row(representative, grouped_rows))
    return output


def load_bootstrap_regions(path: str | Path = DEFAULT_BOOTSTRAP_ARTIFACT) -> List[BootstrapRegionRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return bootstrap_regions_from_records(list(read_jsonl(path)))
