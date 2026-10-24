#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/consent-ai-matplotlib"
data_root="${CONSENT_DATA_ROOT:-data/raw}"
output_root="${CONSENT_RESULTS_DIR:-results}"
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python -m consent_ai.run --data-root "$data_root" --output "$output_root"
