#!/bin/bash -l
#SBATCH --job-name="ODE_Overfit"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/overfit_%j.log

# Usage (interactive):  bash scripts/run_overfit.sh <parquet> [--epochs N] [--name RUN] ...
# Usage (batch):        sbatch scripts/run_overfit.sh <parquet> [--epochs N] [--name RUN] ...
# Example:
#   sbatch scripts/run_overfit.sh metrics_with_features.parquet \
#     --sigma-mode z_score --geometry-size-loss linear \
#     --no-normalize-z0 --time-weighted-loss --size-only-loss \
#     --name overfit-run7 --epochs 500

conda activate venv
# Ensure we use the venv Python, not the base one
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
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

export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-overfit}"
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

exec "${PYTHON}" -m ode.overfit_test \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --solver dopri5 \
  --rtol 1e-6 \
  --atol 1e-6 \
  "$@"
