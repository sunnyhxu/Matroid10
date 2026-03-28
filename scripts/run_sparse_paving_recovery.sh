#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LABEL="${1:-sparse_paving_recovery}"
NON_PAVING_IN="artifacts/non_paving_sparse_paving.jsonl"
HVEC_OUT="artifacts/hvec_${LABEL}.jsonl"
PURE_O_OUT="artifacts/pure_o_results_${LABEL}.jsonl"
HVEC_STATS="artifacts/phase_stats/hvec_extract_${LABEL}.json"
PURE_O_STATS="artifacts/phase_stats/pure_o_cp_${LABEL}.json"
COUNTEREXAMPLE_OUT="artifacts/counterexample_${LABEL}.json"
SAGE_BIN="$(command -v sage || true)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${SAGE_BIN}" && -x "${HOME}/.local/bin/sage" ]]; then
  SAGE_BIN="${HOME}/.local/bin/sage"
fi

if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi

if [[ ! -f "${NON_PAVING_IN}" ]]; then
  echo "Missing phase-1 input: ${NON_PAVING_IN}" >&2
  exit 1
fi

if [[ -z "${SAGE_BIN}" ]]; then
  echo "Unable to locate sage executable" >&2
  exit 1
fi

if [[ -z "${UV_BIN}" ]]; then
  echo "Unable to locate uv executable" >&2
  exit 1
fi

mkdir -p artifacts/phase_stats

echo "[$(date -Is)] phase2 start"
"${SAGE_BIN}" -python python/hvec_extract.py \
  --in "${NON_PAVING_IN}" \
  --out "${HVEC_OUT}" \
  --config configs/default.toml \
  --stats-out "${HVEC_STATS}"

echo "[$(date -Is)] phase3 start"
"${UV_BIN}" run python python/pure_o_cp.py \
  --in "${HVEC_OUT}" \
  --out "${PURE_O_OUT}" \
  --timeout-sec 120 \
  --num-workers 8 \
  --run-id "${LABEL}" \
  --counterexample-out "${COUNTEREXAMPLE_OUT}" \
  --stats-out "${PURE_O_STATS}"

echo "[$(date -Is)] recovery complete"
