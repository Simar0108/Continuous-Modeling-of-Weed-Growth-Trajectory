#!/bin/bash -l
#SBATCH --job-name=h1-stab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/h1_stab_%j.log

# Step 11b: H1 architecture + 70/15/15 split, fixed-horizon val selection + EMA.
# Git-clean required (no --allow-dirty). Does not write h1_final_best.
#
# Usage:
#   sbatch scripts/run_h1_stab.sh Thesis/metrics_with_features.parquet --seed 0

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
  --run-tag h1_stab_seed \
  --ema-decay 0.999 \
  --fixed-horizon-val-every 1 \
  "$@"
