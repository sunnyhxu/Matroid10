import os
import subprocess
import sys
from pathlib import Path


def test_script_entrypoint_runs_without_py_path():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "python/neuro_symbolic_search.py", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Run the neuro-symbolic search controller on bootstrap artifacts." in result.stdout
