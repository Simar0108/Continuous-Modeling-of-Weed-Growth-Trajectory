#!/bin/bash -l
#SBATCH --job-name=h1-pr
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/pathreg_%j.log

# Pathreg in-window arm. Does not write h1_final_best or h1_stab.
# Usage:
#   sbatch scripts/run_pathreg.sh Thesis/metrics_with_features.parquet --pathreg-lambda 0.1

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet --pathreg-lambda L [--seed N] ..." >&2
  exit 1
fi
shift
if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

LAM=""
REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pathreg-lambda)
      LAM="$2"; shift 2 ;;
    --pathreg-lambda=*)
      LAM="${1#*=}"; shift ;;
    *)
      REST+=("$1"); shift ;;
  esac
done
if [[ -z "${LAM}" ]]; then
  echo "REFUSE: --pathreg-lambda is required" >&2
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

TAG="$("${PYTHON}" -c "from ode.pathreg import lambda_tag; print(lambda_tag(float('${LAM}')))")"
RUN_TAG="h1_pathreg_l${TAG}_seed"
OUT_DIR="${REPO_ROOT}/figures/pathreg/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

for s in 0 1 2 3 4; do
  echo "[pathreg] in-window lambda=${LAM} seed=${s} tag=${RUN_TAG}"
  "${PYTHON}" -m ode.train_pathreg \
    --parquet "${PARQUET}" \
    --wandb \
    --project "${WANDB_PROJECT}" \
    --species Maize \
    --max-tracks 100 \
    --epochs 400 \
    --ema-decay 0.999 \
    --fixed-horizon-val-every 1 \
    --pathreg-lambda "${LAM}" \
    --run-tag "${RUN_TAG}" \
    --seed "${s}" \
    "${REST[@]}"
done

"${PYTHON}" eval/evaluateh1.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/evaluate_drop.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/summarize_pathreg.py --out-root "${REPO_ROOT}/figures/pathreg"
echo "[pathreg] in-window done lambda=${LAM} ${RUN_TAG}"
