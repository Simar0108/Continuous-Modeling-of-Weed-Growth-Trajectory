#!/bin/bash -l
#SBATCH --job-name=h1-ncde
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ncde_%j.log

# Neural CDE in-window arm. Kill date 2026-10-25. Does not write h1_final_best.
# Usage: sbatch scripts/run_ncde.sh Thesis/metrics_with_features.parquet

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
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${REPO_ROOT}/wandb/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${REPO_ROOT}/wandb/artifacts}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

RUN_TAG="h1_ncde_seed"
OUT_DIR="${REPO_ROOT}/figures/ncde/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

for s in 0 1 2 3 4; do
  echo "[ncde] in-window seed=${s} tag=${RUN_TAG}"
  "${PYTHON}" -m ode.train_ncde \
    --parquet "${PARQUET}" \
    --wandb \
    --project "${WANDB_PROJECT}" \
    --species Maize \
    --max-tracks 100 \
    --epochs 400 \
    --ema-decay 0.999 \
    --fixed-horizon-val-every 1 \
    --run-tag "${RUN_TAG}" \
    --seed "${s}" \
    "$@"
done

"${PYTHON}" eval/evaluateh1.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/evaluate_drop.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
echo "[ncde] in-window done ${RUN_TAG}"
