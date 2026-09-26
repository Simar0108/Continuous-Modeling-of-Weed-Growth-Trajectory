#!/bin/bash -l
#SBATCH --job-name=h1-s7
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/step7_%j.log

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet --rhs ... --seed N --name RUN" >&2
  exit 1
fi
shift

mkdir -p logs checkpoints
export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-maize-100}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

exec "${PYTHON}" -m ode.train_step7 \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --track-embed \
  --head-lr-mult 10 \
  --max-tracks 100 \
  --epochs 300 \
  "$@"
