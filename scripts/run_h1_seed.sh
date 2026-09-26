#!/bin/bash -l
#SBATCH --job-name=h1-seed
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/h1_seed_%j.log

# Train one clean H1 ODE seed on Maize 70/15/15.
# Hyperparameters match checkpoints/h1_final_best/best.ckpt.
# Does not write under checkpoints/h1_final_best/.
#
# Usage:
#   sbatch scripts/run_h1_seed.sh Thesis/metrics_with_features.parquet --seed 0
#   sbatch scripts/run_h1_seed.sh Thesis/metrics_with_features.parquet --seed 1 --train-time-frac 0.6

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet [--seed N] ..." >&2
  exit 1
fi
shift

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

mkdir -p logs checkpoints
export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-maize-100}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

exec "${PYTHON}" -m ode.train_h1_seed \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --species Maize \
  --max-tracks 100 \
  --epochs 400 \
  --allow-dirty \
  "$@"
