#!/bin/bash -l
#SBATCH --job-name="DiscreteBaseline"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/baseline_%x_%j.log

# Train one discrete baseline from the thesis comparison suite.
#
# Usage (batch):
#   sbatch scripts/run_baselines.sh lstm Thesis/metrics_with_features.parquet [extra flags...]
#   sbatch scripts/run_baselines.sh gru Thesis/metrics_with_features.parquet --epochs 300 --wandb
#   sbatch scripts/run_baselines.sh transformer Thesis/metrics_with_features.parquet --species Maize
#
# Usage (interactive):
#   bash scripts/run_baselines.sh lstm Thesis/metrics_with_features.parquet --epochs 1

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

MODEL_TYPE="${1:-}"
PARQUET="${2:-}"

if [[ -z "${MODEL_TYPE}" || -z "${PARQUET}" ]]; then
  echo "Usage: $0 {lstm|gru|transformer} /path/to/metrics.parquet [extra args...]" >&2
  exit 1
fi
shift 2

case "${MODEL_TYPE}" in
  lstm|gru|transformer) ;;
  *)
    echo "Invalid model_type: ${MODEL_TYPE}. Expected one of: lstm, gru, transformer" >&2
    exit 1
    ;;
esac

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

mkdir -p logs checkpoints

export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-baselines}"
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

exec "${PYTHON}" -m ode.train_baselines \
  --model_type "${MODEL_TYPE}" \
  --parquet "${PARQUET}" \
  --sigma-mode z_score \
  --species Maize \
  --max-tracks 100 \
  --epochs 300 \
  --batch-size 32 \
  --wandb \
  --project "${WANDB_PROJECT}" \
  "$@"
