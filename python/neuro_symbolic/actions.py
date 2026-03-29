from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from ..hvec_extract import h_from_f_vector
except ImportError:
    from hvec_extract import h_from_f_vector

from .types import CandidateFamily, CandidateRecord


def decode_matrix_col(encoded: int, field: int, rank: int) -> List[int]:
    value = int(encoded)
    digits: List[int] = []
    for _ in range(rank):
        digits.append(value % field)
        value //= field
    return digits


def encode_matrix_col(entries: Sequence[int], field: int) -> int:
    multiplier = 1
    encoded = 0
    for value in entries:
        encoded += int(value) * multiplier
        multiplier *= field
    return encoded


def decode_matrix_cols(matrix_cols: Sequence[int], field: int, rank: int) -> List[List[int]]:
    return [decode_matrix_col(value, field, rank) for value in matrix_cols]


def _modular_inverse(value: int, field: int) -> int:
    reduced = value % field
    if reduced == 0:
        raise ValueError("zero_has_no_inverse")
    if field == 2:
        return 1
    if field == 3:
        return 1 if reduced == 1 else 2
    raise ValueError(f"unsupported_field:{field}")


def matrix_rank(matrix_cols: Sequence[int], field: int, rank: int, subset_indices: Sequence[int] | None = None) -> int:
    if subset_indices is None:
        subset_indices = list(range(len(matrix_cols)))
    decoded = decode_matrix_cols([matrix_cols[index] for index in subset_indices], field, rank)
    if not decoded:
        return 0

    width = len(decoded)
    matrix = [[decoded[column][row] for column in range(width)] for row in range(rank)]
    row_rank = 0
    for column in range(width):
        pivot = None
        for row_index in range(row_rank, rank):
            if matrix[row_index][column] % field != 0:
                pivot = row_index
                break
        if pivot is None:
            continue
        if pivot != row_rank:
            matrix[pivot], matrix[row_rank] = matrix[row_rank], matrix[pivot]
        inverse = _modular_inverse(matrix[row_rank][column], field)
        for current_column in range(column, width):
            matrix[row_rank][current_column] = (matrix[row_rank][current_column] * inverse) % field
        for row_index in range(rank):
            if row_index == row_rank:
                continue
            factor = matrix[row_index][column] % field
            if factor == 0:
                continue
            for current_column in range(column, width):
                matrix[row_index][current_column] = (matrix[row_index][current_column] - factor * matrix[row_rank][current_column]) % field
        row_rank += 1
        if row_rank == rank:
            break
    return row_rank


def full_rank_support(matrix_cols: Sequence[int], field: int, rank: int) -> bool:
    return matrix_rank(matrix_cols, field, rank) == rank


def enumerate_representable_bases(field: int, rank: int, n: int, matrix_cols: Sequence[int]) -> List[int]:
    bases: List[int] = []
    for subset in combinations(range(n), rank):
        if matrix_rank(matrix_cols, field, rank, list(subset)) == rank:
            mask = 0
            for index in subset:
                mask |= 1 << index
            bases.append(mask)
    return bases


def _independent_sets_by_size_from_bases(bases: Sequence[int], rank: int) -> List[set[int]]:
    independent_sets = [set() for _ in range(rank + 1)]
    independent_sets[0].add(0)
    for base in bases:
        elements = [index for index in range(base.bit_length()) if (base >> index) & 1]
        for subset_size in range(1, rank + 1):
            for subset in combinations(elements, subset_size):
                mask = 0
                for index in subset:
                    mask |= 1 << index
                independent_sets[subset_size].add(mask)
    return independent_sets


def compute_h_vector_from_bases(bases: Sequence[int], rank: int, n: int) -> List[int]:
    _ = n
    independent_sets = _independent_sets_by_size_from_bases(bases, rank)
    f_vector = [len(independent_sets[size]) for size in range(rank + 1)]
    return h_from_f_vector(f_vector, rank)


def all_rank_subset_masks(n: int, rank: int) -> List[int]:
    masks: List[int] = []
    for subset in combinations(range(n), rank):
        mask = 0
        for index in subset:
            mask |= 1 << index
        masks.append(mask)
    return masks


def representable_candidate_is_valid(candidate: CandidateRecord) -> bool:
    if candidate.family is not CandidateFamily.REPRESENTABLE:
        return False
    if candidate.field is None or candidate.matrix_cols is None:
        return False
    if len(candidate.matrix_cols) != candidate.n:
        return False
    if not full_rank_support(candidate.matrix_cols, candidate.field, candidate.rank):
        return False
    expected_bases = enumerate_representable_bases(candidate.field, candidate.rank, candidate.n, candidate.matrix_cols)
    return sorted(expected_bases) == sorted(candidate.bases)


def sparse_paving_candidate_is_valid(candidate: CandidateRecord) -> bool:
    if candidate.family is not CandidateFamily.SPARSE_PAVING:
        return False
    if candidate.circuit_hyperplanes is None or candidate.sparse_overlap_bound is None:
        return False

    overlap_bound = int(candidate.sparse_overlap_bound)
    circuit_hyperplanes = [int(mask) for mask in candidate.circuit_hyperplanes]
    for left, right in combinations(circuit_hyperplanes, 2):
        if (left & right).bit_count() > overlap_bound:
            return False

    all_subsets = all_rank_subset_masks(candidate.n, candidate.rank)
    expected_bases = [mask for mask in all_subsets if mask not in set(circuit_hyperplanes)]
    return sorted(expected_bases) == sorted(candidate.bases)


def first_full_rank_subset(matrix_cols: Sequence[int], field: int, rank: int) -> Tuple[int, ...] | None:
    for subset in combinations(range(len(matrix_cols)), rank):
        if matrix_rank(matrix_cols, field, rank, list(subset)) == rank:
            return tuple(subset)
    return None


def rebuild_representable_candidate(candidate: CandidateRecord, matrix_cols: Sequence[int], provenance: Dict[str, object]) -> CandidateRecord:
    field = int(candidate.field or 0)
    updated_cols = [int(value) for value in matrix_cols]
    bases = enumerate_representable_bases(field, candidate.rank, candidate.n, updated_cols)
    h_vector = compute_h_vector_from_bases(bases, candidate.rank, candidate.n)
    metadata = dict(candidate.metadata)
    metadata["provenance"] = dict(provenance)
    return CandidateRecord(
        candidate_id=None if candidate.candidate_id is None else f"{candidate.candidate_id}:{provenance['action_id']}",
        family=CandidateFamily.REPRESENTABLE,
        rank=candidate.rank,
        n=candidate.n,
        bases=bases,
        field=field,
        matrix_cols=updated_cols,
        circuit_hyperplanes=None,
        sparse_overlap_bound=None,
        non_paving_witness=candidate.non_paving_witness,
        h_vector=h_vector,
        seed=candidate.seed,
        trial=candidate.trial,
        metadata=metadata,
    )


def rebuild_sparse_paving_candidate(
    candidate: CandidateRecord,
    circuit_hyperplanes: Iterable[int],
    provenance: Dict[str, object],
) -> CandidateRecord:
    updated_hyperplanes = sorted({int(mask) for mask in circuit_hyperplanes})
    bases = [mask for mask in all_rank_subset_masks(candidate.n, candidate.rank) if mask not in set(updated_hyperplanes)]
    h_vector = compute_h_vector_from_bases(bases, candidate.rank, candidate.n)
    metadata = dict(candidate.metadata)
    metadata["provenance"] = dict(provenance)
    return CandidateRecord(
        candidate_id=None if candidate.candidate_id is None else f"{candidate.candidate_id}:{provenance['action_id']}",
        family=CandidateFamily.SPARSE_PAVING,
        rank=candidate.rank,
        n=candidate.n,
        bases=bases,
        field=None,
        matrix_cols=None,
        circuit_hyperplanes=updated_hyperplanes,
        sparse_overlap_bound=candidate.sparse_overlap_bound,
        non_paving_witness=None,
        h_vector=h_vector,
        seed=candidate.seed,
        trial=candidate.trial,
        metadata=metadata,
    )
