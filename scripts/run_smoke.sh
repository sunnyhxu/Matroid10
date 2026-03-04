#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv not found. Install with scripts/setup_wsl.sh or add ~/.local/bin to PATH."
  exit 2
fi

set +e
"$UV_BIN" run python "$ROOT_DIR/python/run_pipeline.py" \
  --config "$ROOT_DIR/configs/default.toml" \
  --override pipeline.max_wall_seconds=300 \
  --override generation.max_seconds_total=240 \
  --override cp.timeout_seconds=30 \
  --override cp.max_instances=50
rc=$?
set -e
if [[ $rc -eq 17 ]]; then
  echo "Smoke run found a counterexample candidate (exit 17). Treating as successful smoke execution."
  exit 0
fi
exit $rc
