#!/usr/bin/env bash
# ODE single-track overfit + Weights & Biases (Lightning WandbLogger).
#
# Usage:
#   ./scripts/run_ode_overfit_wandb.sh /path/to/metrics.parquet [--epochs N] [--name RUN] ... [--quick]
# Defaults (in ode.overfit_test): 40 epochs (quick). Long overfit: --epochs 300
# Examples:
#   ./scripts/quick_overfit_wandb.sh /path/to/metrics.parquet
#   ./scripts/run_ode_overfit_wandb.sh /path/to/metrics.parquet --epochs 300 --name overfit-full
#
# Optional env:
#   WANDB_PROJECT   — default: latent-ode-overfit
#   WANDB_ENTITY    — your W&B team/username (if not set in ~/.netrc or wandb login)
#
# One-time: log in so runs upload to the UI
#   ./venv/bin/wandb login

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet [--epochs N] [--name RUN] ..." >&2
  exit 1
fi
shift

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

VENV_PY="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Expected venv at ${REPO_ROOT}/venv — adjust VENV_PY in this script if needed." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/Thesis"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-overfit}"
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

exec "${VENV_PY}" -m ode.overfit_test \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  "$@"
