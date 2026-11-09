#!/usr/bin/env bash
# Recompute manuscript tables/figures from local data + frozen code.
# GEO is scored from the already-downloaded LFC parquet (no network).
# Writes only to artifacts/results/cand6_deconftf/audit/r8_repro_scratch/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv-bio/bin/python" ]]; then
  echo "missing $ROOT/.venv-bio/bin/python" >&2
  exit 1
fi
echo "[reproduce] root=$ROOT  no GEO re-download"
exec "$ROOT/.venv-bio/bin/python" "$ROOT/artifacts/src/r8_reproduce_verify.py"
