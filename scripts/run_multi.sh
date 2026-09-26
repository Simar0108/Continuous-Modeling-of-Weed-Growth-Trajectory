#!/bin/bash -l
#SBATCH --job-name="ODE_Multi"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/multi_%j.log

# Multi-track Latent Neural ODE training
#
# Usage (interactive):  bash scripts/run_multi.sh <parquet> [flags]
# Usage (batch):        sbatch scripts/run_multi.sh <parquet> [flags]
#
# First multi-track test (10 tracks, full trajectory):
#   sbatch scripts/run_multi.sh Thesis/metrics_with_features.parquet \
#     --sigma-mode z_score --geometry-size-loss log \
#     --max-tracks 10 --epochs 400 \
#     --name multi-run1 --wandb
#
# With horizon curriculum (establishment -> full trajectory):
#   sbatch scripts/run_multi.sh Thesis/metrics_with_features.parquet \
#     --sigma-mode z_score --geometry-size-loss log \
#     --max-tracks 10 --epochs 400 \
#     --horizon-start-frac 0.3 --horizon-ramp-start 100 --horizon-ramp-end 250 \
#     --name multi-run2 --wandb

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet [--max-tracks N] [--name RUN] ..." >&2
  exit 1
fi
shift

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

mkdir -p logs checkpoints

export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-multi}"
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

exec "${PYTHON}" -m ode.train_multi \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --solver dopri5 \
  --rtol 1e-6 \
  --atol 1e-6 \
  "$@"
