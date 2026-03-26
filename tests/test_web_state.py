"""Tests for StateManager."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def temp_state_path(tmp_path: Path) -> Path:
    return tmp_path / "run_state.json"


@pytest.fixture
def state_manager(temp_state_path: Path):
    from python.web.state_manager import StateManager
    return StateManager(temp_state_path)


class TestStateManager:
    def test_load_returns_default_when_file_missing(self, state_manager):
        state = state_manager.load()
        assert state["status"] == "idle"
        assert state["stop_requested"] is False
        assert state["run_id"] is None
        assert state["next_trial_index_start"] == 0

    def test_save_creates_file(self, state_manager, temp_state_path):
        state = {"status": "running", "run_id": "test123"}
        state_manager.save(state)
        assert temp_state_path.exists()

    def test_save_is_atomic(self, state_manager, temp_state_path):
        state_manager.save({"status": "running"})
        # Check no temp files left behind
        assert not temp_state_path.with_suffix(".tmp").exists()

    def test_load_reads_saved_state(self, state_manager):
        state_manager.save({"status": "running", "run_id": "test456"})
        loaded = state_manager.load()
        assert loaded["status"] == "running"
        assert loaded["run_id"] == "test456"

    def test_update_merges_values(self, state_manager):
        state_manager.save({"status": "idle", "run_id": None})
        result = state_manager.update(status="running", run_id="test789")
        assert result["status"] == "running"
        assert result["run_id"] == "test789"

    def test_update_merges_counters(self, state_manager):
        state_manager.save({
            "status": "running",
            "counters": {"total_processed": 100, "unique_found": 50}
        })
        result = state_manager.update(counters={"unique_found": 75, "feasible": 10})
        assert result["counters"]["total_processed"] == 100
        assert result["counters"]["unique_found"] == 75
        assert result["counters"]["feasible"] == 10

    def test_update_merges_phase_status(self, state_manager):
        state_manager.save({
            "status": "running",
            "phase_status": {
                "phase1": {"status": "completed"},
                "phase2": {"status": "pending"}
            }
        })
        result = state_manager.update(phase_status={"phase2": {"status": "running", "progress": {"step": 1}}})
        assert result["phase_status"]["phase1"]["status"] == "completed"
        assert result["phase_status"]["phase2"]["status"] == "running"
        assert result["phase_status"]["phase2"]["progress"]["step"] == 1

    def test_request_stop_sets_flag(self, state_manager):
        state_manager.save({"status": "running", "stop_requested": False})
        state_manager.request_stop()
        assert state_manager.is_stop_requested() is True

    def test_is_stop_requested_returns_false_initially(self, state_manager):
        state_manager.save({"status": "running"})
        assert state_manager.is_stop_requested() is False

    def test_reset_for_new_run(self, state_manager):
        state_manager.save({
            "status": "completed",
            "run_id": "old_run",
            "stop_requested": True,
            "next_trial_index_start": 1000
        })
        result = state_manager.reset_for_new_run("new_run_123")
        assert result["run_id"] == "new_run_123"
        assert result["status"] == "running"
        assert result["stop_requested"] is False
        assert result["next_trial_index_start"] == 1000  # Preserved

    def test_reset_for_new_run_with_trial_override(self, state_manager):
        state_manager.save({"next_trial_index_start": 500})
        result = state_manager.reset_for_new_run("test_run", trial_index_start=100)
        assert result["next_trial_index_start"] == 100

    def test_increment_trial_index(self, state_manager):
        state_manager.save({"next_trial_index_start": 0})
        state_manager.increment_trial_index(1000)
        state = state_manager.load()
        assert state["next_trial_index_start"] == 1000

        state_manager.increment_trial_index(1000)
        state = state_manager.load()
        assert state["next_trial_index_start"] == 2000

    def test_thread_safety(self, state_manager):
        """Test that concurrent updates don't corrupt state."""
        state_manager.save({"counters": {"total": 0}})
        errors = []

        def increment():
            try:
                for _ in range(100):
                    state = state_manager.load()
                    current = state.get("counters", {}).get("total", 0)
                    state_manager.update(counters={"total": current + 1})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Note: Due to race conditions, final count may not be exactly 500
        # but state should not be corrupted
        state = state_manager.load()
        assert "counters" in state
        assert "total" in state["counters"]

    def test_last_updated_is_set_on_save(self, state_manager):
        state_manager.save({"status": "running"})
        state = state_manager.load()
        assert "last_updated" in state
        assert state["last_updated"] is not None
