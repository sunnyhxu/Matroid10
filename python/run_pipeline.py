from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from .common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso
except ImportError:
    from common import dump_json, ensure_dir, load_toml, monotonic_seconds, utc_now_iso

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_COUNTEREXAMPLE = 17

MODE_REPRESENTABLE = "representable"
MODE_SPARSE_PAVING = "sparse_paving"


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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = resolve_path(repo_root, args.config)
    cfg = load_toml(cfg_path)
    for ovr in args.override:
        apply_override(cfg, ovr)

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
    trial_index_start = int(gen_cfg.get("trial_index_start", 0))
    validate_shard_config(shard_index, shard_count)
    validate_trial_index_start(trial_index_start)
    trial_start = shard_index + trial_index_start * shard_count
    trial_stride = shard_count

    artifacts_dir = resolve_path(repo_root, str(pipeline_cfg.get("artifacts_dir", "artifacts")))
    ensure_dir(artifacts_dir)

    non_paving_out = resolve_path(repo_root, str(paths_cfg.get("non_paving_jsonl", "artifacts/non_paving.jsonl")))
    hvec_out = resolve_path(repo_root, str(paths_cfg.get("hvec_jsonl", "artifacts/hvec.jsonl")))
    pure_o_out = resolve_path(repo_root, str(paths_cfg.get("pure_o_jsonl", "artifacts/pure_o_results.jsonl")))
    metrics_out = resolve_path(repo_root, str(paths_cfg.get("metrics_json", "artifacts/metrics.json")))
    manifest_out = resolve_path(repo_root, str(paths_cfg.get("manifest_json", "artifacts/run_manifest.json")))
    counterexample_prefix = resolve_path(
        repo_root, str(paths_cfg.get("counterexample_prefix", "artifacts/counterexample"))
    )
    counterexample_out = Path(str(counterexample_prefix) + f"_{run_id}.json")

    phase_stats_dir = artifacts_dir / "phase_stats"
    progress_chunks_dir = artifacts_dir / "progress_chunks"
    ensure_dir(phase_stats_dir)
    ensure_dir(progress_chunks_dir)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "config_path": str(cfg_path),
        "config": cfg,
        "status": "running",
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "strategy": "trial_mod",
            "trial_index_start": trial_index_start,
            "trial_start": trial_start,
            "trial_stride": trial_stride,
        },
        "phases": {},
    }
    dump_json(manifest_out, manifest)

    start = monotonic_seconds()
    max_wall = int(pipeline_cfg.get("max_wall_seconds", 7200))

    binary = resolve_path(repo_root, str(gen_cfg.get("binary", "cpp/build/gen_nonpaving")))
    if not binary.exists():
        print(f"Generator binary not found at {binary}. Build it with: bash scripts/build_cpp.sh")
        manifest["status"] = "error"
        manifest["error"] = f"missing_binary:{binary}"
        manifest["ended_at"] = utc_now_iso()
        dump_json(manifest_out, manifest)
        return EXIT_ERROR

    max_seconds_total = int(gen_cfg.get("max_seconds_total", 7200))
    global_seed = int(gen_cfg.get("seed", 42))
    threads = int(gen_cfg.get("threads", 1))
    rank_min = int(gen_cfg.get("rank_min", 4))
    rank_max = int(gen_cfg.get("rank_max", 9))
    n = int(gen_cfg.get("n", 10))
    max_trials = int(gen_cfg.get("max_trials", 0))
    dedup_global = bool(gen_cfg.get("dedup_global", True))
    require_connected = bool(gen_cfg.get("require_connected", True))

    sparse_accept_prob = float(gen_cfg.get("sparse_accept_prob", 0.05))
    sparse_min_ch = int(gen_cfg.get("sparse_min_circuit_hyperplanes", 1))
    sparse_max_ch = int(gen_cfg.get("sparse_max_circuit_hyperplanes", 0))

    phase1_targets: List[Dict[str, Any]] = []
    if generation_mode == MODE_REPRESENTABLE:
        fields = [int(x) for x in gen_cfg.get("fields", [2, 3])]
        if not fields:
            raise ValueError("generation.fields must not be empty for representable mode")
        for idx, field in enumerate(fields):
            phase1_targets.append(
                {
                    "category": f"representable_f{field}",
                    "field": field,
                    "seed": global_seed + idx * 1000003 + field * 10007,
                    "out": artifacts_dir / f"non_paving_field{field}.jsonl",
                    "stats": phase_stats_dir / f"gen_field{field}.json",
                }
            )
    else:
        phase1_targets.append(
            {
                "category": "sparse_paving",
                "field": None,
                "seed": global_seed,
                "out": artifacts_dir / "non_paving_sparse_paving.jsonl",
                "stats": phase_stats_dir / "gen_sparse_paving.json",
            }
        )

    per_target_seconds = max(1, max_seconds_total // max(1, len(phase1_targets)))
    phase1_outputs: List[Path] = []
    phase1_commands: List[List[str]] = []
    phase1_stats_files: List[Path] = []
    phase1_chunk_files: List[Path] = []

    for idx, target in enumerate(phase1_targets):
        if monotonic_seconds() - start > max_wall:
            manifest["status"] = "error"
            manifest["error"] = "pipeline_wall_timeout_before_phase1_complete"
            manifest["ended_at"] = utc_now_iso()
            dump_json(manifest_out, manifest)
            return EXIT_ERROR

        cmd = [
            str(binary),
            "--config",
            str(cfg_path),
            "--mode",
            "representable" if generation_mode == MODE_REPRESENTABLE else "sparse-paving",
            "--rank-min",
            str(rank_min),
            "--rank-max",
            str(rank_max),
            "--n",
            str(n),
            "--threads",
            str(threads),
            "--seed",
            str(target["seed"]),
            "--max-seconds",
            str(per_target_seconds),
            "--trial-start",
            str(trial_start),
            "--trial-stride",
            str(trial_stride),
            "--out",
            str(target["out"]),
            "--stats-out",
            str(target["stats"]),
        ]
        if target["field"] is not None:
            cmd.extend(["--field", str(target["field"])])
        if require_connected:
            cmd.append("--require-connected")
        else:
            cmd.append("--allow-disconnected")
        if max_trials > 0:
            cmd.extend(["--max-trials", str(max_trials)])
        if generation_mode == MODE_SPARSE_PAVING:
            cmd.extend(
                [
                    "--sparse-accept-prob",
                    str(sparse_accept_prob),
                    "--sparse-min-ch",
                    str(sparse_min_ch),
                    "--sparse-max-ch",
                    str(sparse_max_ch),
                ]
            )

        phase1_commands.append(cmd)
        proc = run_cmd(cmd, cwd=repo_root, phase=f"phase1.{target['category']}")
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            manifest["status"] = "error"
            manifest["error"] = f"phase1_failed_{target['category']}_rc_{proc.returncode}"
            manifest["phases"]["phase1"] = {"commands": phase1_commands}
            manifest["ended_at"] = utc_now_iso()
            dump_json(manifest_out, manifest)
            return EXIT_ERROR

        phase1_outputs.append(target["out"])
        phase1_stats_files.append(target["stats"])
        stats_payload = load_json_if_exists(target["stats"])
        chunk_path = progress_chunks_dir / f"{run_id}_{target['category']}.json"
        chunk_payload = {
            "run_id": run_id,
            "phase1_command_index": idx,
            "created_at": utc_now_iso(),
            "mode": generation_mode,
            "category": target["category"],
            "field": target["field"],
            "n": n,
            "seed": target["seed"],
            "shard_index": shard_index,
            "shard_count": shard_count,
            "trial_index_start": trial_index_start,
            "trial_start": trial_start,
            "trial_stride": trial_stride,
            "stats_path": str(target["stats"]),
            "output_path": str(target["out"]),
            "candidates": int(stats_payload.get("candidates", 0)),
            "unique_hits": int(stats_payload.get("unique_hits", 0)),
            "status": "ok",
        }
        write_progress_chunk(chunk_path, chunk_payload)
        phase1_chunk_files.append(chunk_path)

    merge_stats = merge_non_paving(phase1_outputs, non_paving_out, dedup_global=dedup_global)
    manifest["phases"]["phase1"] = {
        "mode": generation_mode,
        "commands": phase1_commands,
        "outputs": [str(p) for p in phase1_outputs],
        "merged_output": str(non_paving_out),
        "merge_stats": merge_stats,
        "stats": {str(p): load_json_if_exists(p) for p in phase1_stats_files},
        "progress_chunks": [str(p) for p in phase1_chunk_files],
        "shard": manifest["shard"],
    }

    if monotonic_seconds() - start > max_wall:
        manifest["status"] = "error"
        manifest["error"] = "pipeline_wall_timeout_before_phase2"
        manifest["ended_at"] = utc_now_iso()
        dump_json(manifest_out, manifest)
        return EXIT_ERROR

    hvec_script = resolve_path(repo_root, str(hvec_cfg.get("script", "python/hvec_extract.py")))
    hvec_stats = phase_stats_dir / "hvec_extract.json"
    sage_exe = resolve_executable("sage")
    phase2_cmd = [
        sage_exe,
        "-python",
        str(hvec_script),
        "--in",
        str(non_paving_out),
        "--out",
        str(hvec_out),
        "--config",
        str(cfg_path),
        "--stats-out",
        str(hvec_stats),
    ]
    proc2 = run_cmd(phase2_cmd, cwd=repo_root, phase="phase2")
    if proc2.returncode != 0:
        print(proc2.stdout)
        print(proc2.stderr)
        manifest["status"] = "error"
        manifest["error"] = f"phase2_failed_rc_{proc2.returncode}"
        manifest["phases"]["phase2"] = {"command": phase2_cmd}
        manifest["ended_at"] = utc_now_iso()
        dump_json(manifest_out, manifest)
        return EXIT_ERROR
    manifest["phases"]["phase2"] = {"command": phase2_cmd, "stats": load_json_if_exists(hvec_stats)}

    if monotonic_seconds() - start > max_wall:
        manifest["status"] = "error"
        manifest["error"] = "pipeline_wall_timeout_before_phase3"
        manifest["ended_at"] = utc_now_iso()
        dump_json(manifest_out, manifest)
        return EXIT_ERROR

    cp_script = resolve_path(repo_root, str(cp_cfg.get("script", "python/pure_o_cp.py")))
    cp_stats = phase_stats_dir / "pure_o_cp.json"
    cp_timeout = float(cp_cfg.get("timeout_seconds", 120))
    cp_workers = int(cp_cfg.get("num_workers", 8))
    cp_max_instances = int(cp_cfg.get("max_instances", 0))
    stop_on_counterexample = bool(pipeline_cfg.get("stop_on_counterexample", True))

    uv_exe = resolve_executable("uv")
    phase3_cmd = [
        uv_exe,
        "run",
        "python",
        str(cp_script),
        "--in",
        str(hvec_out),
        "--out",
        str(pure_o_out),
        "--timeout-sec",
        str(cp_timeout),
        "--num-workers",
        str(cp_workers),
        "--run-id",
        run_id,
        "--counterexample-out",
        str(counterexample_out),
        "--stats-out",
        str(cp_stats),
    ]
    if cp_max_instances > 0:
        phase3_cmd.extend(["--max-instances", str(cp_max_instances)])
    if not stop_on_counterexample:
        phase3_cmd.append("--no-stop-on-infeasible")
    proc3 = run_cmd(phase3_cmd, cwd=repo_root, phase="phase3")

    if proc3.returncode not in (EXIT_OK, EXIT_COUNTEREXAMPLE):
        print(proc3.stdout)
        print(proc3.stderr)
        manifest["status"] = "error"
        manifest["error"] = f"phase3_failed_rc_{proc3.returncode}"
        manifest["phases"]["phase3"] = {"command": phase3_cmd}
        manifest["ended_at"] = utc_now_iso()
        dump_json(manifest_out, manifest)
        return EXIT_ERROR

    phase3_stats = load_json_if_exists(cp_stats)
    manifest["phases"]["phase3"] = {"command": phase3_cmd, "stats": phase3_stats}

    elapsed = monotonic_seconds() - start
    status = "counterexample" if proc3.returncode == EXIT_COUNTEREXAMPLE else "ok"
    manifest["status"] = status
    manifest["ended_at"] = utc_now_iso()
    manifest["elapsed_seconds"] = elapsed

    metrics = {
        "run_id": run_id,
        "status": status,
        "elapsed_seconds": elapsed,
        "shard": manifest["shard"],
        "phase1": manifest["phases"].get("phase1", {}),
        "phase2": manifest["phases"].get("phase2", {}),
        "phase3": manifest["phases"].get("phase3", {}),
    }
    if status == "counterexample" and counterexample_out.exists():
        metrics["counterexample_path"] = str(counterexample_out)

    dump_json(metrics_out, metrics)
    dump_json(manifest_out, manifest)
    return EXIT_COUNTEREXAMPLE if status == "counterexample" else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
