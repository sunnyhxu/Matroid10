from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from .common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso
except ImportError:
    from common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_COUNTEREXAMPLE = 17

MODE_REPRESENTABLE = "representable"
MODE_SPARSE_PAVING = "sparse_paving"


@dataclass
class PipelineContext:
    """Holds all pipeline configuration and paths."""

    repo_root: Path
    cfg_path: Path
    cfg: Dict[str, Any]
    run_id: str
    artifacts_dir: Path
    phase_stats_dir: Path
    progress_chunks_dir: Path

    # Generation config
    generation_mode: str
    binary: Path
    rank_min: int
    rank_max: int
    n: int
    threads: int
    global_seed: int
    max_seconds_total: int
    max_trials: int
    dedup_global: bool
    require_connected: bool
    sparse_accept_prob: float
    sparse_min_ch: int
    sparse_max_ch: int

    # Shard config
    shard_index: int
    shard_count: int
    trial_index_start: int
    trial_start: int
    trial_stride: int

    # Output paths
    non_paving_out: Path
    hvec_out: Path
    pure_o_out: Path
    metrics_out: Path
    manifest_out: Path
    counterexample_out: Path

    # Hvec config
    hvec_script: Path

    # CP config
    cp_script: Path
    cp_timeout: float
    cp_workers: int
    cp_max_instances: int
    stop_on_counterexample: bool

    # Runtime
    max_wall_seconds: int
    start_time: float = field(default_factory=monotonic_seconds)


def parse_override_value(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_override_value(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def apply_override(cfg: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override (missing '='): {override}")
    path, raw = override.split("=", 1)
    parts = [p.strip() for p in path.split(".") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Override must be section.key=value: {override}")
    cursor: Dict[str, Any] = cfg
    for p in parts[:-1]:
        if p not in cursor or not isinstance(cursor[p], dict):
            cursor[p] = {}
        cursor = cursor[p]
    cursor[parts[-1]] = parse_override_value(raw.strip())


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def run_cmd(cmd: List[str], cwd: Path, phase: str) -> subprocess.CompletedProcess[str]:
    print(f"[{phase}] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    if fallback.exists():
        return str(fallback)
    return name


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_non_paving(field_outputs: List[Path], out_path: Path, dedup_global: bool) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    input_records = 0
    output_records = 0
    duplicates = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for fp in field_outputs:
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as fin:
                for raw in fin:
                    line = raw.strip()
                    if not line:
                        continue
                    input_records += 1
                    if dedup_global:
                        record = json.loads(line)
                        key = str(record.get("id", ""))
                        if key in seen:
                            duplicates += 1
                            continue
                        seen.add(key)
                    fout.write(line)
                    fout.write("\n")
                    output_records += 1

    return {
        "input_records": input_records,
        "output_records": output_records,
        "duplicates_removed": duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Matroid10 pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def normalize_mode(raw_mode: str) -> str:
    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {MODE_REPRESENTABLE, MODE_SPARSE_PAVING}:
        return mode
    raise ValueError(f"Unsupported generation.mode: {raw_mode}")


def validate_shard_config(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("generation.shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("generation.shard_index must satisfy 0 <= shard_index < shard_count")


def validate_trial_index_start(trial_index_start: int) -> None:
    if trial_index_start < 0:
        raise ValueError("generation.trial_index_start must be >= 0")


def write_progress_chunk(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, payload)


def create_context(
    cfg_path: Path,
    cfg: Dict[str, Any],
    repo_root: Optional[Path] = None,
    trial_index_start_override: Optional[int] = None,
) -> PipelineContext:
    """Create a PipelineContext from configuration."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    pipeline_cfg = cfg.get("pipeline", {})
    gen_cfg = cfg.get("generation", {})
    hvec_cfg = cfg.get("hvec", {})
    cp_cfg = cfg.get("cp", {})
    paths_cfg = cfg.get("paths", {})

    run_prefix = str(pipeline_cfg.get("run_id_prefix", "matroid10"))
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_prefix}_{run_stamp}"

    generation_mode = normalize_mode(str(gen_cfg.get("mode", MODE_REPRESENTABLE)))
    shard_index = int(gen_cfg.get("shard_index", 0))
    shard_count = int(gen_cfg.get("shard_count", 1))
    trial_index_start = trial_index_start_override if trial_index_start_override is not None else int(gen_cfg.get("trial_index_start", 0))
    validate_shard_config(shard_index, shard_count)
    validate_trial_index_start(trial_index_start)
    trial_start = shard_index + trial_index_start * shard_count
    trial_stride = shard_count

    artifacts_dir = resolve_path(repo_root, str(pipeline_cfg.get("artifacts_dir", "artifacts")))
    phase_stats_dir = artifacts_dir / "phase_stats"
    progress_chunks_dir = artifacts_dir / "progress_chunks"

    counterexample_prefix = resolve_path(
        repo_root, str(paths_cfg.get("counterexample_prefix", "artifacts/counterexample"))
    )

    return PipelineContext(
        repo_root=repo_root,
        cfg_path=cfg_path,
        cfg=cfg,
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        phase_stats_dir=phase_stats_dir,
        progress_chunks_dir=progress_chunks_dir,
        generation_mode=generation_mode,
        binary=resolve_path(repo_root, str(gen_cfg.get("binary", "cpp/build/gen_nonpaving"))),
        rank_min=int(gen_cfg.get("rank_min", 4)),
        rank_max=int(gen_cfg.get("rank_max", 9)),
        n=int(gen_cfg.get("n", 10)),
        threads=int(gen_cfg.get("threads", 1)),
        global_seed=int(gen_cfg.get("seed", 42)),
        max_seconds_total=int(gen_cfg.get("max_seconds_total", 7200)),
        max_trials=int(gen_cfg.get("max_trials", 0)),
        dedup_global=bool(gen_cfg.get("dedup_global", True)),
        require_connected=bool(gen_cfg.get("require_connected", True)),
        sparse_accept_prob=float(gen_cfg.get("sparse_accept_prob", 0.05)),
        sparse_min_ch=int(gen_cfg.get("sparse_min_circuit_hyperplanes", 1)),
        sparse_max_ch=int(gen_cfg.get("sparse_max_circuit_hyperplanes", 0)),
        shard_index=shard_index,
        shard_count=shard_count,
        trial_index_start=trial_index_start,
        trial_start=trial_start,
        trial_stride=trial_stride,
        non_paving_out=resolve_path(repo_root, str(paths_cfg.get("non_paving_jsonl", "artifacts/non_paving.jsonl"))),
        hvec_out=resolve_path(repo_root, str(paths_cfg.get("hvec_jsonl", "artifacts/hvec.jsonl"))),
        pure_o_out=resolve_path(repo_root, str(paths_cfg.get("pure_o_jsonl", "artifacts/pure_o_results.jsonl"))),
        metrics_out=resolve_path(repo_root, str(paths_cfg.get("metrics_json", "artifacts/metrics.json"))),
        manifest_out=resolve_path(repo_root, str(paths_cfg.get("manifest_json", "artifacts/run_manifest.json"))),
        counterexample_out=Path(str(counterexample_prefix) + f"_{run_id}.json"),
        hvec_script=resolve_path(repo_root, str(hvec_cfg.get("script", "python/hvec_extract.py"))),
        cp_script=resolve_path(repo_root, str(cp_cfg.get("script", "python/pure_o_cp.py"))),
        cp_timeout=float(cp_cfg.get("timeout_seconds", 120)),
        cp_workers=int(cp_cfg.get("num_workers", 8)),
        cp_max_instances=int(cp_cfg.get("max_instances", 0)),
        stop_on_counterexample=bool(pipeline_cfg.get("stop_on_counterexample", True)),
        max_wall_seconds=int(pipeline_cfg.get("max_wall_seconds", 7200)),
    )


def build_phase1_targets(ctx: PipelineContext) -> List[Dict[str, Any]]:
    """Build list of phase 1 generation targets."""
    gen_cfg = ctx.cfg.get("generation", {})
    targets: List[Dict[str, Any]] = []

    if ctx.generation_mode == MODE_REPRESENTABLE:
        fields = [int(x) for x in gen_cfg.get("fields", [2, 3])]
        if not fields:
            raise ValueError("generation.fields must not be empty for representable mode")
        for idx, fld in enumerate(fields):
            targets.append(
                {
                    "category": f"representable_f{fld}",
                    "field": fld,
                    "seed": ctx.global_seed + idx * 1000003 + fld * 10007,
                    "out": ctx.artifacts_dir / f"non_paving_field{fld}.jsonl",
                    "stats": ctx.phase_stats_dir / f"gen_field{fld}.json",
                }
            )
    else:
        targets.append(
            {
                "category": "sparse_paving",
                "field": None,
                "seed": ctx.global_seed,
                "out": ctx.artifacts_dir / "non_paving_sparse_paving.jsonl",
                "stats": ctx.phase_stats_dir / "gen_sparse_paving.json",
            }
        )

    return targets


def build_phase1_command(ctx: PipelineContext, target: Dict[str, Any], per_target_seconds: int) -> List[str]:
    """Build a single phase 1 command for a target."""
    cmd = [
        str(ctx.binary),
        "--config",
        str(ctx.cfg_path),
        "--mode",
        "representable" if ctx.generation_mode == MODE_REPRESENTABLE else "sparse-paving",
        "--rank-min",
        str(ctx.rank_min),
        "--rank-max",
        str(ctx.rank_max),
        "--n",
        str(ctx.n),
        "--threads",
        str(ctx.threads),
        "--seed",
        str(target["seed"]),
        "--max-seconds",
        str(per_target_seconds),
        "--trial-start",
        str(ctx.trial_start),
        "--trial-stride",
        str(ctx.trial_stride),
        "--out",
        str(target["out"]),
        "--stats-out",
        str(target["stats"]),
    ]
    if target["field"] is not None:
        cmd.extend(["--field", str(target["field"])])
    if ctx.require_connected:
        cmd.append("--require-connected")
    else:
        cmd.append("--allow-disconnected")
    if ctx.max_trials > 0:
        cmd.extend(["--max-trials", str(ctx.max_trials)])
    if ctx.generation_mode == MODE_SPARSE_PAVING:
        cmd.extend(
            [
                "--sparse-accept-prob",
                str(ctx.sparse_accept_prob),
                "--sparse-min-ch",
                str(ctx.sparse_min_ch),
                "--sparse-max-ch",
                str(ctx.sparse_max_ch),
            ]
        )
    return cmd


def build_phase1_commands(ctx: PipelineContext) -> List[Dict[str, Any]]:
    """Build all phase 1 commands with their targets."""
    targets = build_phase1_targets(ctx)
    per_target_seconds = max(1, ctx.max_seconds_total // max(1, len(targets)))

    result = []
    for target in targets:
        cmd = build_phase1_command(ctx, target, per_target_seconds)
        result.append({
            "command": cmd,
            "target": target,
        })
    return result


def run_phase1(
    ctx: PipelineContext,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 1 (C++ generation).

    Args:
        ctx: Pipeline context
        stop_check: Optional callable that returns True if stop was requested
        progress_callback: Optional callable to report progress updates

    Returns:
        Dict with phase results including outputs, stats, and any errors
    """
    ensure_dir(ctx.artifacts_dir)
    ensure_dir(ctx.phase_stats_dir)
    ensure_dir(ctx.progress_chunks_dir)

    commands_info = build_phase1_commands(ctx)
    phase1_outputs: List[Path] = []
    phase1_commands: List[List[str]] = []
    phase1_stats_files: List[Path] = []
    phase1_chunk_files: List[Path] = []

    for idx, info in enumerate(commands_info):
        # Check for stop request
        if stop_check and stop_check():
            return {
                "status": "stopped",
                "outputs": phase1_outputs,
                "commands": phase1_commands,
                "stats_files": phase1_stats_files,
            }

        # Check wall time
        if monotonic_seconds() - ctx.start_time > ctx.max_wall_seconds:
            return {
                "status": "timeout",
                "error": "pipeline_wall_timeout_before_phase1_complete",
                "outputs": phase1_outputs,
                "commands": phase1_commands,
            }

        cmd = info["command"]
        target = info["target"]
        phase1_commands.append(cmd)

        proc = run_cmd(cmd, cwd=ctx.repo_root, phase=f"phase1.{target['category']}")
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return {
                "status": "error",
                "error": f"phase1_failed_{target['category']}_rc_{proc.returncode}",
                "commands": phase1_commands,
                "outputs": phase1_outputs,
            }

        phase1_outputs.append(target["out"])
        phase1_stats_files.append(target["stats"])
        stats_payload = load_json_if_exists(target["stats"])

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
            "shard_index": ctx.shard_index,
            "shard_count": ctx.shard_count,
            "trial_index_start": ctx.trial_index_start,
            "trial_start": ctx.trial_start,
            "trial_stride": ctx.trial_stride,
            "stats_path": str(target["stats"]),
            "output_path": str(target["out"]),
            "candidates": int(stats_payload.get("candidates", 0)),
            "unique_hits": int(stats_payload.get("unique_hits", 0)),
            "status": "ok",
        }
        write_progress_chunk(chunk_path, chunk_payload)
        phase1_chunk_files.append(chunk_path)

        if progress_callback:
            progress_callback("phase1", {
                "target_index": idx,
                "total_targets": len(commands_info),
                "category": target["category"],
                "stats": stats_payload,
            })

    # Merge outputs
    merge_stats = merge_non_paving(phase1_outputs, ctx.non_paving_out, dedup_global=ctx.dedup_global)

    return {
        "status": "ok",
        "mode": ctx.generation_mode,
        "commands": phase1_commands,
        "outputs": [str(p) for p in phase1_outputs],
        "merged_output": str(ctx.non_paving_out),
        "merge_stats": merge_stats,
        "stats": {str(p): load_json_if_exists(p) for p in phase1_stats_files},
        "progress_chunks": [str(p) for p in phase1_chunk_files],
    }


def build_phase2_command(ctx: PipelineContext) -> List[str]:
    """Build the phase 2 (h-vector extraction) command."""
    sage_exe = resolve_executable("sage")
    hvec_stats = ctx.phase_stats_dir / "hvec_extract.json"
    return [
        sage_exe,
        "-python",
        str(ctx.hvec_script),
        "--in",
        str(ctx.non_paving_out),
        "--out",
        str(ctx.hvec_out),
        "--config",
        str(ctx.cfg_path),
        "--stats-out",
        str(hvec_stats),
    ]


def run_phase2(
    ctx: PipelineContext,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2 (Sage h-vector extraction).

    Returns:
        Dict with phase results
    """
    if stop_check and stop_check():
        return {"status": "stopped"}

    if monotonic_seconds() - ctx.start_time > ctx.max_wall_seconds:
        return {"status": "timeout", "error": "pipeline_wall_timeout_before_phase2"}

    cmd = build_phase2_command(ctx)
    hvec_stats = ctx.phase_stats_dir / "hvec_extract.json"

    proc = run_cmd(cmd, cwd=ctx.repo_root, phase="phase2")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return {
            "status": "error",
            "error": f"phase2_failed_rc_{proc.returncode}",
            "command": cmd,
        }

    stats = load_json_if_exists(hvec_stats)
    if progress_callback:
        progress_callback("phase2", {"stats": stats})

    return {
        "status": "ok",
        "command": cmd,
        "stats": stats,
    }


def build_phase3_command(ctx: PipelineContext) -> List[str]:
    """Build the phase 3 (CP-SAT) command."""
    uv_exe = resolve_executable("uv")
    cp_stats = ctx.phase_stats_dir / "pure_o_cp.json"
    cmd = [
        uv_exe,
        "run",
        "python",
        str(ctx.cp_script),
        "--in",
        str(ctx.hvec_out),
        "--out",
        str(ctx.pure_o_out),
        "--timeout-sec",
        str(ctx.cp_timeout),
        "--num-workers",
        str(ctx.cp_workers),
        "--run-id",
        ctx.run_id,
        "--counterexample-out",
        str(ctx.counterexample_out),
        "--stats-out",
        str(cp_stats),
    ]
    if ctx.cp_max_instances > 0:
        cmd.extend(["--max-instances", str(ctx.cp_max_instances)])
    if not ctx.stop_on_counterexample:
        cmd.append("--no-stop-on-infeasible")
    return cmd


def run_phase3(
    ctx: PipelineContext,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3 (CP-SAT pure O-sequence verification).

    Returns:
        Dict with phase results
    """
    if stop_check and stop_check():
        return {"status": "stopped"}

    if monotonic_seconds() - ctx.start_time > ctx.max_wall_seconds:
        return {"status": "timeout", "error": "pipeline_wall_timeout_before_phase3"}

    cmd = build_phase3_command(ctx)
    cp_stats = ctx.phase_stats_dir / "pure_o_cp.json"

    proc = run_cmd(cmd, cwd=ctx.repo_root, phase="phase3")

    if proc.returncode not in (EXIT_OK, EXIT_COUNTEREXAMPLE):
        print(proc.stdout)
        print(proc.stderr)
        return {
            "status": "error",
            "error": f"phase3_failed_rc_{proc.returncode}",
            "command": cmd,
        }

    stats = load_json_if_exists(cp_stats)
    if progress_callback:
        progress_callback("phase3", {"stats": stats})

    is_counterexample = proc.returncode == EXIT_COUNTEREXAMPLE
    return {
        "status": "counterexample" if is_counterexample else "ok",
        "command": cmd,
        "stats": stats,
        "counterexample_found": is_counterexample,
    }


def run_full_pipeline(
    ctx: PipelineContext,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete pipeline (all 3 phases).

    Returns:
        Dict with full pipeline results
    """
    ensure_dir(ctx.artifacts_dir)
    ensure_dir(ctx.phase_stats_dir)
    ensure_dir(ctx.progress_chunks_dir)

    manifest: Dict[str, Any] = {
        "run_id": ctx.run_id,
        "started_at": utc_now_iso(),
        "config_path": str(ctx.cfg_path),
        "config": ctx.cfg,
        "status": "running",
        "shard": {
            "index": ctx.shard_index,
            "count": ctx.shard_count,
            "strategy": "trial_mod",
            "trial_index_start": ctx.trial_index_start,
            "trial_start": ctx.trial_start,
            "trial_stride": ctx.trial_stride,
        },
        "phases": {},
    }
    dump_json(ctx.manifest_out, manifest)

    # Check binary exists
    if not ctx.binary.exists():
        print(f"Generator binary not found at {ctx.binary}. Build it with: bash scripts/build_cpp.sh")
        manifest["status"] = "error"
        manifest["error"] = f"missing_binary:{ctx.binary}"
        manifest["ended_at"] = utc_now_iso()
        dump_json(ctx.manifest_out, manifest)
        return {"status": "error", "error": manifest["error"], "exit_code": EXIT_ERROR}

    # Phase 1
    phase1_result = run_phase1(ctx, stop_check, progress_callback)
    manifest["phases"]["phase1"] = phase1_result
    manifest["phases"]["phase1"]["shard"] = manifest["shard"]

    if phase1_result["status"] != "ok":
        manifest["status"] = phase1_result["status"]
        if "error" in phase1_result:
            manifest["error"] = phase1_result["error"]
        manifest["ended_at"] = utc_now_iso()
        dump_json(ctx.manifest_out, manifest)
        return {"status": phase1_result["status"], "manifest": manifest, "exit_code": EXIT_ERROR}

    # Phase 2
    phase2_result = run_phase2(ctx, stop_check, progress_callback)
    manifest["phases"]["phase2"] = phase2_result

    if phase2_result["status"] != "ok":
        manifest["status"] = phase2_result["status"]
        if "error" in phase2_result:
            manifest["error"] = phase2_result["error"]
        manifest["ended_at"] = utc_now_iso()
        dump_json(ctx.manifest_out, manifest)
        return {"status": phase2_result["status"], "manifest": manifest, "exit_code": EXIT_ERROR}

    # Phase 3
    phase3_result = run_phase3(ctx, stop_check, progress_callback)
    manifest["phases"]["phase3"] = phase3_result

    elapsed = monotonic_seconds() - ctx.start_time
    status = "counterexample" if phase3_result.get("counterexample_found") else phase3_result["status"]
    manifest["status"] = status
    manifest["ended_at"] = utc_now_iso()
    manifest["elapsed_seconds"] = elapsed

    metrics = {
        "run_id": ctx.run_id,
        "status": status,
        "elapsed_seconds": elapsed,
        "shard": manifest["shard"],
        "phase1": manifest["phases"].get("phase1", {}),
        "phase2": manifest["phases"].get("phase2", {}),
        "phase3": manifest["phases"].get("phase3", {}),
    }
    if status == "counterexample" and ctx.counterexample_out.exists():
        metrics["counterexample_path"] = str(ctx.counterexample_out)

    dump_json(ctx.metrics_out, metrics)
    dump_json(ctx.manifest_out, manifest)

    exit_code = EXIT_COUNTEREXAMPLE if status == "counterexample" else EXIT_OK
    return {"status": status, "manifest": manifest, "exit_code": exit_code}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = resolve_path(repo_root, args.config)
    cfg = load_toml(cfg_path)
    for ovr in args.override:
        apply_override(cfg, ovr)

    ctx = create_context(cfg_path, cfg, repo_root)
    result = run_full_pipeline(ctx)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
