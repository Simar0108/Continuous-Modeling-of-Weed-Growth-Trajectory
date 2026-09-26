#!/bin/bash -l
#SBATCH --job-name="Hypothesis1"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/hypothesis_one_%j.log

# Final evidence suite for Hypothesis 1 (ODE vs LSTM/GRU/Transformer).
#
# Usage (batch):
#   sbatch scripts/run_hypothesis_one.sh Thesis/metrics_with_features.parquet
#
# Checkpoints are auto-discovered (ODE, LSTM, GRU, Transformer). Override only if needed:
#   bash scripts/run_hypothesis_one.sh Thesis/metrics_with_features.parquet \
#     --ode-checkpoint checkpoints/overfit_best_overfit-sanity.ckpt

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-Thesis/metrics_with_features.parquet}"
shift

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

mkdir -p logs figures checkpoints

export PYTHONPATH="${REPO_ROOT}"

exec "${PYTHON}" -m ode.hypothesis_one_final \
  --parquet "${PARQUET}" \
  --sigma-mode z_score \
  --species Maize \
  --max-tracks 100 \
  --split test \
  --solver dopri5 \
  --rtol 1e-7 \
  --atol 1e-7 \
  --output-dir figures \
  "$@"
