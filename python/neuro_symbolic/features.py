from __future__ import annotations

from typing import Any, Dict, Mapping

from .bootstrap import BootstrapRegionRecord
from .types import ActionSpec, CandidateRecord


def _region_mapping(region_record: BootstrapRegionRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(region_record, BootstrapRegionRecord):
        return region_record.to_dict()
    return region_record


def extract_region_features(
    region_record: BootstrapRegionRecord | Mapping[str, Any],
    search_stats: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    search_stats = search_stats or {}
    row = _region_mapping(region_record)
    h_vector = [int(value) for value in row["h_vector"]]
    current_solver = dict(row["current_solver"])
    top_level_solver = dict(row["top_level_solver"])
    structural_metrics = dict(row["structural_metrics"])
    current_metrics = dict(current_solver.get("metrics", {}))
    top_metrics = dict(top_level_solver.get("metrics", {}))

    features: Dict[str, Any] = {
        "region_key": str(row["h_vector_key"]),
        "h_vector_len": float(len(h_vector)),
        "h_vector_sum": float(sum(h_vector)),
        "source_records": float(row["source_records"]),
        "current_solver_status": str(current_solver.get("status", "UNKNOWN")),
        "current_solver_wall_time": float(current_solver.get("wall_time", 0.0) or 0.0),
        "current_solver_num_conflicts": float(current_metrics.get("num_conflicts", 0.0) or 0.0),
        "current_solver_num_branches": float(current_metrics.get("num_branches", 0.0) or 0.0),
        "top_level_solver_status": str(top_level_solver.get("status", "UNKNOWN")),
        "top_level_solver_wall_time": float(top_level_solver.get("wall_time", 0.0) or 0.0),
        "top_level_solver_num_conflicts": float(top_metrics.get("num_conflicts", 0.0) or 0.0),
        "top_level_solver_num_branches": float(top_metrics.get("num_branches", 0.0) or 0.0),
        "structural_raw_score": float(structural_metrics.get("raw_score", 0.0) or 0.0),
        "macaulay_total_slack": float(structural_metrics.get("macaulay_total_slack", 0.0) or 0.0),
        "late_drop_sum": float(structural_metrics.get("late_drop_sum", 0.0) or 0.0),
        "current_solver_score": float(row.get("current_solver_score", 0.0) or 0.0),
        "top_level_solver_score": float(row.get("top_level_solver_score", 0.0) or 0.0),
        "structural_score": float(row.get("structural_score", 0.0) or 0.0),
        "combined_score": float(row.get("combined_score", 0.0) or 0.0),
        "novelty_count": float(search_stats.get("novelty_count", 0.0) or 0.0),
        "timeout_count": float(search_stats.get("timeout_count", 0.0) or 0.0),
        "escalation_count": float(search_stats.get("escalation_count", 0.0) or 0.0),
        "revisit_count": float(search_stats.get("revisit_count", 0.0) or 0.0),
    }
    for index, value in enumerate(h_vector):
        features[f"h_vector_{index}"] = float(value)
    return features


def extract_instance_action_features(
    candidate: CandidateRecord,
    action: ActionSpec,
    region_features: Mapping[str, Any],
    duplicate_isomorph_history: int = 0,
) -> Dict[str, Any]:
    h_vector = candidate.h_vector or []
    features: Dict[str, Any] = {
        "family": candidate.family.value,
        "rank": float(candidate.rank),
        "n": float(candidate.n),
        "bases_count": float(len(candidate.bases)),
        "h_vector_len": float(len(h_vector)),
        "h_vector_sum": float(sum(h_vector)),
        "field": float(candidate.field or 0),
        "matrix_cols_count": float(len(candidate.matrix_cols or [])),
        "circuit_hyperplanes_count": float(len(candidate.circuit_hyperplanes or [])),
        "parent_combined_score": float(region_features.get("combined_score", 0.0) or 0.0),
        "parent_current_solver_score": float(region_features.get("current_solver_score", 0.0) or 0.0),
        "parent_structural_score": float(region_features.get("structural_score", 0.0) or 0.0),
        "action_type": action.action_type,
        "action_locality": action.locality,
        "action_cross_region": float(1.0 if action.cross_region else 0.0),
        "duplicate_isomorph_history": float(duplicate_isomorph_history),
        "region_key": str(region_features.get("region_key", "")),
    }
    for index, value in enumerate(h_vector):
        features[f"candidate_h_vector_{index}"] = float(value)
    return features
