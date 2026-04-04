import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.neuro_symbolic.canonicalize import CanonicalizationService  # noqa: E402
from python.neuro_symbolic.types import CandidateFamily, CandidateRecord  # noqa: E402


def _candidate(candidate_id: str, bases: list[int]) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        family=CandidateFamily.REPRESENTABLE,
        rank=2,
        n=4,
        bases=bases,
        h_vector=[1, 2, 1],
    )


def test_isomorphic_candidates_share_a_canonical_key():
    service = CanonicalizationService()
    left = _candidate("left", [5, 6, 9, 10, 12])
    right = _candidate("right", [3, 5, 6, 9, 10])

    left_key = service.canonicalize_candidate(left)
    right_key = service.canonicalize_candidate(right)

    assert left_key.key == right_key.key
    assert left_key.display_id == right_key.display_id


def test_non_isomorphic_candidate_creates_new_key():
    service = CanonicalizationService()
    left = _candidate("left", [5, 6, 9, 10, 12])
    other = _candidate("other", [3, 5, 6, 9])

    left_key = service.canonicalize_candidate(left)
    other_key = service.canonicalize_candidate(other)

    assert left_key.key != other_key.key


def test_internal_identity_uses_full_canonical_label_and_short_display_digest():
    service = CanonicalizationService()
    candidate = _candidate("left", [5, 6, 9, 10, 12])

    canonical = service.canonicalize_candidate(candidate)

    assert canonical.key.startswith("n=4|r=2|")
    assert canonical.display_id == hashlib.sha256(canonical.key.encode("utf-8")).hexdigest()[:16]
    assert canonical.digest == hashlib.sha256(canonical.key.encode("utf-8")).hexdigest()
    assert canonical.display_id != canonical.key
