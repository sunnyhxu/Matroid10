from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from .types import ActionSpec, CandidateRecord, CanonicalKey, FilterResult, RegionKey


class ProblemSpec(ABC):
    """Family-specific interface for neuro-symbolic search candidates."""

    @abstractmethod
    def family_name(self) -> str:
        """Return the stable family identifier."""

    @abstractmethod
    def region_key(self, candidate: CandidateRecord) -> RegionKey:
        """Return the stable, serializable region key for a candidate."""

    @abstractmethod
    def canonicalize(self, candidate: CandidateRecord) -> CanonicalKey:
        """Return the internal canonical identity plus a display identifier."""

    @abstractmethod
    def enumerate_valid_actions(self, candidate: CandidateRecord) -> Sequence[ActionSpec]:
        """Return inspectable action descriptors without mutating the candidate."""

    @abstractmethod
    def apply_action(self, candidate: CandidateRecord, action: ActionSpec) -> CandidateRecord:
        """Apply one valid action and return the resulting candidate."""

    @abstractmethod
    def cheap_filters(self, candidate: CandidateRecord) -> Sequence[FilterResult]:
        """Run deterministic cheap symbolic guards before exact verification."""

    @abstractmethod
    def exact_verifiers(self) -> Sequence[Dict[str, Any]]:
        """Return deterministic verifier descriptors in execution order."""
