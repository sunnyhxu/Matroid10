from __future__ import annotations

import json
from math import comb, log1p
from typing import Any, Dict, Iterable, List, Sequence


def h_vector_key(h_vector: Iterable[int]) -> str:
    return json.dumps([int(x) for x in h_vector], separators=(",", ":"))


def degree_capacity(n_vars: int, degree: int) -> int:
    return comb(n_vars + degree - 1, degree)


def macaulay_next(value: int, degree: int) -> int:
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        return 0

    remainder = value
    upper_bound: int | None = None
    total = 0

    for current_degree in range(degree, 0, -1):
        candidate = current_degree
        while True:
            next_candidate = candidate + 1
            if upper_bound is not None and next_candidate >= upper_bound:
                break
            if comb(next_candidate, current_degree) > remainder:
                break
            candidate = next_candidate

        if candidate >= current_degree and comb(candidate, current_degree) <= remainder:
            remainder -= comb(candidate, current_degree)
            upper_bound = candidate
            total += comb(candidate + 1, current_degree + 1)
        else:
            upper_bound = current_degree

    return total


def percentile_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    indexed = sorted((float(value), idx) for idx, value in enumerate(values))
    size = len(indexed)
    out = [0.0] * size
    start = 0

    while start < size:
        end = start + 1
        while end < size and indexed[end][0] == indexed[start][0]:
            end += 1

        average_rank = (start + end - 1) / 2.0
        if size == 1:
            score = 0.5
        else:
            score = average_rank / float(size - 1)

        for _, original_index in indexed[start:end]:
            out[original_index] = score
        start = end

    return out


def compute_empirical_score(metrics: Dict[str, Any]) -> Dict[str, float]:
    wall_time = float(metrics.get("wall_time", 0.0) or 0.0)
    wall_millis = max(0.0, wall_time) * 1000.0
    num_conflicts = int(metrics.get("num_conflicts", 0) or 0)
    num_branches = int(metrics.get("num_branches", 0) or 0)

    components = {
        "log_num_conflicts": log1p(num_conflicts) if num_conflicts else 0.0,
        "log_num_branches": log1p(num_branches) if num_branches else 0.0,
        "log_wall_millis": log1p(wall_millis),
    }
    components["raw_score"] = sum(components.values())
    return components


def compute_structural_metrics(h_vector: Sequence[int]) -> Dict[str, Any]:
    h = [int(x) for x in h_vector]
    if not h or any(x < 0 for x in h):
        raise ValueError("invalid_h_vector")

    degree = len(h) - 1
    n_vars = max(1, h[1] if len(h) > 1 else 1)

    capacities = [degree_capacity(n_vars, current_degree) for current_degree in range(len(h))]
    capacity_pressure = [h[current_degree] / capacities[current_degree] for current_degree in range(len(h))]

    macaulay_bounds: List[int] = []
    macaulay_slack: List[int] = []
    macaulay_violation: List[int] = []
    for current_degree in range(1, degree):
        bound = macaulay_next(h[current_degree], current_degree)
        next_value = h[current_degree + 1]
        macaulay_bounds.append(bound)
        macaulay_slack.append(max(0, bound - next_value))
        macaulay_violation.append(max(0, next_value - bound))

    drops = [max(0, h[current_degree] - h[current_degree + 1]) for current_degree in range(len(h) - 1)]
    peak_value = max(h)
    peak_degree = min(index for index, value in enumerate(h) if value == peak_value)
    late_drops = drops[peak_degree:]

    tightness = 1.0 / (1.0 + float(sum(macaulay_slack)))
    late_drop_ratio = sum(late_drops) / float(max(1, peak_value))
    max_capacity_pressure = max(capacity_pressure)
    structural_raw_score = float(sum(macaulay_violation)) + tightness + late_drop_ratio + max_capacity_pressure

    return {
        "degree": degree,
        "n_vars": n_vars,
        "peak_degree": peak_degree,
        "peak_value": peak_value,
        "capacity_by_degree": capacities,
        "capacity_pressure_by_degree": capacity_pressure,
        "macaulay_bounds": macaulay_bounds,
        "macaulay_slack_by_degree": macaulay_slack,
        "macaulay_violation_by_degree": macaulay_violation,
        "macaulay_total_slack": sum(macaulay_slack),
        "macaulay_total_violation": sum(macaulay_violation),
        "drop_by_degree": drops,
        "tail_drop_sum": sum(drops),
        "late_drop_sum": sum(late_drops),
        "tightness": tightness,
        "max_capacity_pressure": max_capacity_pressure,
        "late_drop_ratio": late_drop_ratio,
        "raw_score": structural_raw_score,
    }
