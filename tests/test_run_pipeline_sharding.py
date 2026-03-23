from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from python import run_pipeline


def _write_config(path: Path, mode: str, binary: Path, artifacts_dir: Path, trial_index_start: int = 0) -> None:
    artifacts = artifacts_dir.as_posix()
    binary_path = binary.as_posix()
    cfg_path = artifacts_dir / "counterexample"
    path.write_text(
        f"""
[pipeline]
run_id_prefix = "test"
artifacts_dir = "{artifacts}"
max_wall_seconds = 120
stop_on_counterexample = true

[generation]
binary = "{binary_path}"
mode = "{mode}"
fields = [2, 3]
rank_min = 4
rank_max = 9
n = 10
threads = 1
seed = 42
max_seconds_total = 10
max_trials = 5
shard_index = 1
shard_count = 4
trial_index_start = {trial_index_start}
sparse_accept_prob = 0.2
sparse_min_circuit_hyperplanes = 2
sparse_max_circuit_hyperplanes = 6
dedup_global = true
require_connected = true

[hvec]
script = "python/hvec_extract.py"

[cp]
script = "python/pure_o_cp.py"
timeout_seconds = 1
num_workers = 1
max_instances = 0

[paths]
non_paving_jsonl = "{(artifacts_dir / "non_paving.jsonl").as_posix()}"
hvec_jsonl = "{(artifacts_dir / "hvec.jsonl").as_posix()}"
pure_o_jsonl = "{(artifacts_dir / "pure_o_results.jsonl").as_posix()}"
metrics_json = "{(artifacts_dir / "metrics.json").as_posix()}"
manifest_json = "{(artifacts_dir / "run_manifest.json").as_posix()}"
counterexample_prefix = "{cfg_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )


def _run_main_with_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, trial_index_start: int = 0
) -> list[tuple[str, list[str]]]:
    cfg = tmp_path / "cfg.toml"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "gen_nonpaving"
    binary.write_text("", encoding="utf-8")
    _write_config(cfg, mode=mode, binary=binary, artifacts_dir=artifacts_dir, trial_index_start=trial_index_start)

    calls: list[tuple[str, list[str]]] = []

    def fake_run_cmd(cmd: list[str], cwd: Path, phase: str) -> SimpleNamespace:  # noqa: ARG001
        calls.append((phase, cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_pipeline, "run_cmd", fake_run_cmd)
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--config", str(cfg)])

    rc = run_pipeline.main()
    assert rc == run_pipeline.EXIT_OK
    return calls


def test_validate_shard_config_bounds():
    run_pipeline.validate_shard_config(0, 1)
    run_pipeline.validate_shard_config(2, 3)
    with pytest.raises(ValueError):
        run_pipeline.validate_shard_config(-1, 3)
    with pytest.raises(ValueError):
        run_pipeline.validate_shard_config(3, 3)
    with pytest.raises(ValueError):
        run_pipeline.validate_shard_config(0, 0)


def test_validate_trial_index_start_bounds():
    run_pipeline.validate_trial_index_start(0)
    run_pipeline.validate_trial_index_start(1000)
    with pytest.raises(ValueError):
        run_pipeline.validate_trial_index_start(-1)


def test_representable_phase1_commands_include_shard_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = _run_main_with_mode(monkeypatch, tmp_path, mode="representable")
    phase1_cmds = [cmd for phase, cmd in calls if phase.startswith("phase1.")]
    assert len(phase1_cmds) == 2
    for cmd in phase1_cmds:
        assert "--mode" in cmd
        assert cmd[cmd.index("--mode") + 1] == "representable"
        assert "--trial-start" in cmd
        assert cmd[cmd.index("--trial-start") + 1] == "1"
        assert "--trial-stride" in cmd
        assert cmd[cmd.index("--trial-stride") + 1] == "4"
        assert "--field" in cmd


def test_sparse_paving_phase1_command_routing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = _run_main_with_mode(monkeypatch, tmp_path, mode="sparse_paving")
    phase1_cmds = [cmd for phase, cmd in calls if phase.startswith("phase1.")]
    assert len(phase1_cmds) == 1
    cmd = phase1_cmds[0]
    assert "--mode" in cmd
    assert cmd[cmd.index("--mode") + 1] == "sparse-paving"
    assert "--field" not in cmd
    assert "--trial-start" in cmd
    assert cmd[cmd.index("--trial-start") + 1] == "1"
    assert "--trial-stride" in cmd
    assert cmd[cmd.index("--trial-stride") + 1] == "4"
    assert "--sparse-accept-prob" in cmd
    assert "--sparse-min-ch" in cmd
    assert "--sparse-max-ch" in cmd


def test_trial_index_start_offsets_trial_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = _run_main_with_mode(monkeypatch, tmp_path, mode="representable", trial_index_start=1000)
    phase1_cmds = [cmd for phase, cmd in calls if phase.startswith("phase1.")]
    assert len(phase1_cmds) == 2
    for cmd in phase1_cmds:
        assert "--trial-start" in cmd
        # shard_index=1, shard_count=4, trial_index_start=1000 => 1 + 1000*4 = 4001
        assert cmd[cmd.index("--trial-start") + 1] == "4001"
        assert "--trial-stride" in cmd
        assert cmd[cmd.index("--trial-stride") + 1] == "4"
