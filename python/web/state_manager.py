"""Persistent state manager for pipeline runs."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StateManager:
    """Manages persistent pipeline run state with atomic writes and thread safety."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._lock = threading.Lock()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "run_id": None,
            "status": "idle",
            "stop_requested": False,
            "current_phase": None,
            "next_trial_index_start": 0,
            "phase_status": {
                "phase1": {"status": "pending", "progress": {}},
                "phase2": {"status": "pending", "progress": {}},
                "phase3": {"status": "pending", "progress": {}},
            },
            "counters": {
                "total_processed": 0,
                "unique_found": 0,
                "feasible": 0,
                "infeasible": 0,
            },
            "last_updated": utc_now_iso(),
        }

    def load(self) -> Dict[str, Any]:
        """Load state from disk, return default state if file doesn't exist."""
        if not self.state_path.exists():
            return self._default_state()
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return self._default_state()

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically save state to disk using temp file + rename."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = utc_now_iso()
        temp_path = self.state_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        # Atomic rename (works on POSIX; on Windows may fail if target exists)
        try:
            temp_path.replace(self.state_path)
        except OSError:
            # Windows fallback: remove target first
            if self.state_path.exists():
                self.state_path.unlink()
            temp_path.rename(self.state_path)

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        """Thread-safe load, merge updates, save atomically."""
        with self._lock:
            state = self.load()
            for key, value in kwargs.items():
                if key == "counters" and isinstance(value, dict):
                    state.setdefault("counters", {})
                    state["counters"].update(value)
                elif key == "phase_status" and isinstance(value, dict):
                    state.setdefault("phase_status", {})
                    for phase, phase_data in value.items():
                        state["phase_status"].setdefault(phase, {})
                        state["phase_status"][phase].update(phase_data)
                else:
                    state[key] = value
            self.save(state)
            return state

    def request_stop(self) -> None:
        """Set the stop_requested flag."""
        self.update(stop_requested=True)

    def is_stop_requested(self) -> bool:
        """Check if stop was requested."""
        state = self.load()
        return state.get("stop_requested", False)

    def reset_for_new_run(self, run_id: str, trial_index_start: Optional[int] = None) -> Dict[str, Any]:
        """Reset state for a new pipeline run."""
        with self._lock:
            current = self.load()
            next_trial = trial_index_start if trial_index_start is not None else current.get("next_trial_index_start", 0)
            state = self._default_state()
            state["run_id"] = run_id
            state["status"] = "running"
            state["next_trial_index_start"] = next_trial
            self.save(state)
            return state

    def increment_trial_index(self, max_trials: int) -> None:
        """Increment next_trial_index_start after a run completes."""
        with self._lock:
            state = self.load()
            current = state.get("next_trial_index_start", 0)
            state["next_trial_index_start"] = current + max_trials
            self.save(state)
