#!/bin/bash -l
#SBATCH --job-name=h1-ncdex
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ncde_extrap_%j.log

# One prefix-60 NCDE seed. Primary D-017 endpoint. Kill date 2026-10-25.
# Usage: sbatch scripts/run_ncde_extrap.sh Thesis/metrics_with_features.parquet --seed N

conda activate venv
set -euo pipefail
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet --seed N" >&2
  exit 1
fi
shift
if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

SEED=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done
if [[ -z "${SEED}" ]]; then
  echo "REFUSE: --seed is required (one GPU per seed)." >&2
  exit 1
fi

mkdir -p logs checkpoints
export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-maize-100}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${REPO_ROOT}/wandb/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${REPO_ROOT}/wandb/artifacts}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

RUN_TAG="h1_ncde_seed"
echo "[ncde-extrap] seed=${SEED} tag=${RUN_TAG}"
"${PYTHON}" -m ode.train_ncde \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --species Maize \
  --max-tracks 100 \
  --epochs 400 \
  --ema-decay 0.999 \
  --fixed-horizon-val-every 1 \
  --nfe-limit 150 \
  --nfe-sustain 3 \
  --train-time-frac 0.6 \
  --run-tag "${RUN_TAG}" \
  --name "h1-ncde-s${SEED}-extrap60" \
  "${EXTRA[@]}"
echo "[ncde-extrap] seed ${SEED} trainer-exit"
