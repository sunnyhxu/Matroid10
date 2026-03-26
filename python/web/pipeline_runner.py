"""Background pipeline execution with stop checks and progress tracking."""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .state_manager import StateManager
from .dedup_store import DedupStore

try:
    from ..run_pipeline import (
        EXIT_OK,
        EXIT_COUNTEREXAMPLE,
        EXIT_ERROR,
        PipelineContext,
        create_context,
        build_phase1_commands,
        build_phase2_command,
        build_phase3_command,
        load_json_if_exists,
        merge_non_paving,
        write_progress_chunk,
        resolve_executable,
    )
    from ..common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from run_pipeline import (
        EXIT_OK,
        EXIT_COUNTEREXAMPLE,
        EXIT_ERROR,
        PipelineContext,
        create_context,
        build_phase1_commands,
        build_phase2_command,
        build_phase3_command,
        load_json_if_exists,
        merge_non_paving,
        write_progress_chunk,
        resolve_executable,
    )
    from common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso


class PipelineRunner:
    """Runs the pipeline in a background thread with stop support."""

    def __init__(
        self,
        state_manager: StateManager,
        dedup_store: DedupStore,
        config_path: Path,
        repo_root: Optional[Path] = None,
    ):
        self.state_manager = state_manager
        self.dedup_store = dedup_store
        self.config_path = config_path
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self._thread: Optional[threading.Thread] = None
        self._current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, trial_index_start: Optional[int] = None) -> bool:
        """
        Start the pipeline in a background thread.

        Args:
            trial_index_start: Optional override for trial_index_start.
                              If None, uses value from state_manager.

        Returns:
            True if started successfully, False if already running.
        """
        if self.is_running():
            return False

        self._thread = threading.Thread(
            target=self._run_pipeline,
            args=(trial_index_start,),
            daemon=True,
        )
        self._thread.start()
        return True

    def request_stop(self) -> None:
        """Request graceful stop of the pipeline."""
        self.state_manager.request_stop()

    def _check_stop(self) -> bool:
        """Check if stop was requested."""
        return self.state_manager.is_stop_requested()

    def _run_subprocess(self, cmd: List[str], phase: str) -> subprocess.CompletedProcess:
        """Run a subprocess with tracking for potential interruption."""
        print(f"[{phase}] {' '.join(cmd)}")

        with self._process_lock:
            self._current_process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        stdout, stderr = self._current_process.communicate()
        returncode = self._current_process.returncode

        with self._process_lock:
            self._current_process = None

        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    def _update_counters_from_stats(self, phase: str, stats: Dict[str, Any]) -> None:
        """Update counters based on phase statistics."""
        counters: Dict[str, int] = {}

        if phase == "phase1":
            counters["total_processed"] = int(stats.get("candidates", 0))
            counters["unique_found"] = int(stats.get("unique_hits", 0))
        elif phase == "phase2":
            counters["total_processed"] = int(stats.get("input_count", 0))
        elif phase == "phase3":
            counters["feasible"] = int(stats.get("feasible", 0))
            counters["infeasible"] = int(stats.get("infeasible", 0))

        if counters:
            self.state_manager.update(counters=counters)

    def _run_pipeline(self, trial_index_start: Optional[int] = None) -> None:
        """Execute the pipeline phases with stop checks between subprocesses."""
        try:
            # Load config
            cfg = load_toml(self.config_path)

            # Get trial_index_start from state if not provided
            if trial_index_start is None:
                state = self.state_manager.load()
                trial_index_start = state.get("next_trial_index_start", 0)

            # Create context
            ctx = create_context(
                self.config_path,
                cfg,
                self.repo_root,
                trial_index_start_override=trial_index_start,
            )

            # Reset state for new run
            self.state_manager.reset_for_new_run(ctx.run_id, trial_index_start)

            # Ensure directories exist
            ensure_dir(ctx.artifacts_dir)
            ensure_dir(ctx.phase_stats_dir)
            ensure_dir(ctx.progress_chunks_dir)

            # Check binary exists
            if not ctx.binary.exists():
                self.state_manager.update(
                    status="error",
                    error=f"missing_binary:{ctx.binary}",
                )
                return

            # Run Phase 1
            phase1_result = self._run_phase1(ctx)
            if phase1_result["status"] != "ok":
                return

            # Run Phase 2
            phase2_result = self._run_phase2(ctx)
            if phase2_result["status"] != "ok":
                return

            # Run Phase 3
            phase3_result = self._run_phase3(ctx)

            # Pipeline completed
            final_status = "counterexample" if phase3_result.get("counterexample_found") else "completed"
            self.state_manager.update(
                status=final_status,
                current_phase=None,
            )

            # Increment trial index for next run
            if ctx.max_trials > 0:
                self.state_manager.increment_trial_index(ctx.max_trials)

            # Merge results into accumulated store
            self._merge_to_accumulated_store(ctx)

        except Exception as e:
            self.state_manager.update(
                status="error",
                error=str(e),
                current_phase=None,
            )

    def _run_phase1(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Execute Phase 1 with stop checks between each target."""
        self.state_manager.update(
            current_phase="phase1",
            phase_status={"phase1": {"status": "running", "progress": {}}},
        )

        commands_info = build_phase1_commands(ctx)
        phase1_outputs: List[Path] = []
        total_candidates = 0
        total_unique = 0

        for idx, info in enumerate(commands_info):
            # Check for stop request
            if self._check_stop():
                self.state_manager.update(status="stopped", current_phase=None)
                return {"status": "stopped"}

            cmd = info["command"]
            target = info["target"]

            # Update progress
            self.state_manager.update(
                phase_status={
                    "phase1": {
                        "status": "running",
                        "progress": {
                            "current_target": idx + 1,
                            "total_targets": len(commands_info),
                            "category": target["category"],
                        },
                    }
                }
            )

            proc = self._run_subprocess(cmd, f"phase1.{target['category']}")

            # Check for stop after subprocess
            if self._check_stop():
                self.state_manager.update(status="stopped", current_phase=None)
                return {"status": "stopped"}

            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                self.state_manager.update(
                    status="error",
                    error=f"phase1_failed_{target['category']}_rc_{proc.returncode}",
                    current_phase=None,
                )
                return {"status": "error"}

            phase1_outputs.append(target["out"])

            # Update counters from stats
            stats = load_json_if_exists(target["stats"])
            total_candidates += int(stats.get("candidates", 0))
            total_unique += int(stats.get("unique_hits", 0))

            self.state_manager.update(
                counters={
                    "total_processed": total_candidates,
                    "unique_found": total_unique,
                }
            )

            # Write progress chunk
            chunk_path = ctx.progress_chunks_dir / f"{ctx.run_id}_{target['category']}.json"
            chunk_payload = {
                "run_id": ctx.run_id,
                "phase1_command_index": idx,
                "created_at": utc_now_iso(),
                "mode": ctx.generation_mode,
                "category": target["category"],
                "field": target["field"],
                "n": ctx.n,
                "seed": target["seed"],
                "candidates": int(stats.get("candidates", 0)),
                "unique_hits": int(stats.get("unique_hits", 0)),
                "status": "ok",
            }
            write_progress_chunk(chunk_path, chunk_payload)

        # Merge outputs
        merge_stats = merge_non_paving(phase1_outputs, ctx.non_paving_out, dedup_global=ctx.dedup_global)

        self.state_manager.update(
            phase_status={"phase1": {"status": "completed", "progress": merge_stats}},
            counters={"unique_found": merge_stats.get("output_records", total_unique)},
        )

        return {"status": "ok", "outputs": phase1_outputs, "merge_stats": merge_stats}

    def _run_phase2(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Execute Phase 2 with stop check."""
        if self._check_stop():
            self.state_manager.update(status="stopped", current_phase=None)
            return {"status": "stopped"}

        self.state_manager.update(
            current_phase="phase2",
            phase_status={"phase2": {"status": "running", "progress": {}}},
        )

        cmd = build_phase2_command(ctx)
        proc = self._run_subprocess(cmd, "phase2")

        if self._check_stop():
            self.state_manager.update(status="stopped", current_phase=None)
            return {"status": "stopped"}

        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            self.state_manager.update(
                status="error",
                error=f"phase2_failed_rc_{proc.returncode}",
                current_phase=None,
            )
            return {"status": "error"}

        hvec_stats = ctx.phase_stats_dir / "hvec_extract.json"
        stats = load_json_if_exists(hvec_stats)

        self.state_manager.update(
            phase_status={"phase2": {"status": "completed", "progress": stats}},
        )

        return {"status": "ok", "stats": stats}

    def _run_phase3(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Execute Phase 3 with stop check."""
        if self._check_stop():
            self.state_manager.update(status="stopped", current_phase=None)
            return {"status": "stopped"}

        self.state_manager.update(
            current_phase="phase3",
            phase_status={"phase3": {"status": "running", "progress": {}}},
        )

        cmd = build_phase3_command(ctx)
        proc = self._run_subprocess(cmd, "phase3")

        if self._check_stop():
            self.state_manager.update(status="stopped", current_phase=None)
            return {"status": "stopped"}

        if proc.returncode not in (EXIT_OK, EXIT_COUNTEREXAMPLE):
            print(proc.stdout)
            print(proc.stderr)
            self.state_manager.update(
                status="error",
                error=f"phase3_failed_rc_{proc.returncode}",
                current_phase=None,
            )
            return {"status": "error"}

        cp_stats = ctx.phase_stats_dir / "pure_o_cp.json"
        stats = load_json_if_exists(cp_stats)

        self.state_manager.update(
            phase_status={"phase3": {"status": "completed", "progress": stats}},
            counters={
                "feasible": int(stats.get("feasible", 0)),
                "infeasible": int(stats.get("infeasible", 0)),
            },
        )

        is_counterexample = proc.returncode == EXIT_COUNTEREXAMPLE
        return {
            "status": "counterexample" if is_counterexample else "ok",
            "stats": stats,
            "counterexample_found": is_counterexample,
        }

    def _merge_to_accumulated_store(self, ctx: PipelineContext) -> None:
        """Merge pipeline results into the accumulated dedup store."""
        # Merge from the pure_o results (final output)
        if ctx.pure_o_out.exists():
            stats = self.dedup_store.append_from_jsonl(ctx.pure_o_out)
            print(f"Merged to accumulated store: {stats}")
