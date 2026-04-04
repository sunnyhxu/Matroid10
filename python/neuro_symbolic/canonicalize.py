from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import CandidateRecord, CanonicalKey


def _sorted_unique_bases(bases: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(base) for base in bases}))


def _apply_permutation(mask: int, inverse_permutation: Dict[int, int], n: int) -> int:
    out = 0
    for element in range(n):
        if (mask >> element) & 1:
            out |= 1 << inverse_permutation[element]
    return out


def _canonical_label_from_order(n: int, rank: int, bases: Sequence[int], order: Sequence[int]) -> str:
    inverse = {element: new_index for new_index, element in enumerate(order)}
    canonical_bases = sorted(_apply_permutation(mask, inverse, n) for mask in bases)
    return f"n={n}|r={rank}|bases={json.dumps(canonical_bases, separators=(',', ':'))}"


def _base_color(mask: int, partition: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    return tuple(sum(1 for element in cell if (mask >> element) & 1) for cell in partition)


def _refine_partition(n: int, bases: Sequence[int], partition: Sequence[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    current = tuple(tuple(cell) for cell in partition)

    while True:
        base_colors = {mask: _base_color(mask, current) for mask in bases}
        refined: List[Tuple[int, ...]] = []
        changed = False

        for cell in current:
            if len(cell) == 1:
                refined.append(tuple(cell))
                continue

            grouped: Dict[Tuple[Tuple[Tuple[int, ...], int], ...], List[int]] = {}
            for element in cell:
                incident_histogram = Counter(
                    base_colors[mask] for mask in bases if (mask >> element) & 1
                )
                signature = tuple(sorted(incident_histogram.items()))
                grouped.setdefault(signature, []).append(element)

            if len(grouped) == 1:
                refined.append(tuple(sorted(cell)))
                continue

            changed = True
            for signature in sorted(grouped):
                refined.append(tuple(sorted(grouped[signature])))

        next_partition = tuple(refined)
        if not changed or next_partition == current:
            return next_partition
        current = next_partition


def _canonical_label_search(n: int, rank: int, bases: Sequence[int], partition: Sequence[Tuple[int, ...]]) -> str:
    refined = _refine_partition(n, bases, partition)
    if all(len(cell) == 1 for cell in refined):
        return _canonical_label_from_order(n, rank, bases, [cell[0] for cell in refined])

    branch_index = next(index for index, cell in enumerate(refined) if len(cell) > 1)
    branch_cell = refined[branch_index]

    best_label: str | None = None
    for chosen in branch_cell:
        individualized = list(refined[:branch_index])
        individualized.append((chosen,))
        individualized.append(tuple(element for element in branch_cell if element != chosen))
        individualized.extend(refined[branch_index + 1 :])
        label = _canonical_label_search(n, rank, bases, tuple(individualized))
        if best_label is None or label < best_label:
            best_label = label

    assert best_label is not None
    return best_label


class CanonicalizationService:
    """Canonicalize matroid candidates by minimizing the base family under element relabeling."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, Tuple[int, ...]], CanonicalKey] = {}

    def canonicalize_candidate(self, candidate: CandidateRecord) -> CanonicalKey:
        bases = _sorted_unique_bases(candidate.bases)
        cache_key = (int(candidate.n), int(candidate.rank), bases)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        partition = (tuple(range(int(candidate.n))),)
        full_label = _canonical_label_search(int(candidate.n), int(candidate.rank), bases, partition)
        digest = hashlib.sha256(full_label.encode("utf-8")).hexdigest()
        canonical_key = CanonicalKey(
            key=full_label,
            display_id=digest[:16],
            digest=digest,
        )
        self._cache[cache_key] = canonical_key
        return canonical_key
