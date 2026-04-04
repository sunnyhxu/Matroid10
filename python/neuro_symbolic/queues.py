from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class QueueEntry:
    canonical_key: str
    region_key: str
    priority: float
    reason: str


@dataclass
class SearchQueues:
    continue_search: List[QueueEntry] = field(default_factory=list)
    escalation: List[QueueEntry] = field(default_factory=list)
    terminal_complete: List[QueueEntry] = field(default_factory=list)

    def push(self, destination: str, entry: QueueEntry) -> None:
        target = getattr(self, destination)
        target.append(entry)
        target.sort(key=lambda item: (-item.priority, item.canonical_key, item.reason))

    def pop_continue(self) -> QueueEntry | None:
        if not self.continue_search:
            return None
        return self.continue_search.pop(0)

    def counts(self) -> Dict[str, int]:
        return {
            "continue_search": len(self.continue_search),
            "escalation": len(self.escalation),
            "terminal_complete": len(self.terminal_complete),
        }
