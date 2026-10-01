#!/bin/bash -l
#SBATCH --job-name=h1-prx
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/pathreg_extrap_%j.log

# Pathreg prefix-60 / tail-40. Does not write h1_final_best or h1_stab.
# Usage:
#   sbatch scripts/run_pathreg_extrap.sh Thesis/metrics_with_features.parquet --pathreg-lambda 0.1

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet --pathreg-lambda L" >&2
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

# Inline .4g tag — do not import ode.pathreg (bootstrap stdout emptied TAG).
TAG="$("${PYTHON}" -c "lam=float('${LAM}'); print(f'{lam:.4g}'.replace('-','m').replace('.','p'))")"
if [[ -z "${TAG}" || "${TAG}" == *$'\n'* ]]; then
  echo "REFUSE: empty/multiline lambda tag for lambda=${LAM} (got ${TAG!r})" >&2
  exit 1
fi
RUN_TAG="h1_pathreg_l${TAG}_seed"
OUT_DIR="${REPO_ROOT}/figures/pathreg/${RUN_TAG}_extrap60"
mkdir -p "${OUT_DIR}"
echo "[pathreg-extrap] lambda=${LAM} tag=${TAG} run_tag=${RUN_TAG} out=${OUT_DIR}"

for s in 0 1 2 3 4; do
  echo "[pathreg-extrap] lambda=${LAM} seed=${s} tag=${RUN_TAG}"
  "${PYTHON}" -m ode.train_pathreg \
    --parquet "${PARQUET}" \
    --wandb \
    --project "${WANDB_PROJECT}" \
    --species Maize \
    --max-tracks 100 \
    --epochs 400 \
    --ema-decay 0.999 \
    --fixed-horizon-val-every 1 \
    --train-time-frac 0.6 \
    --pathreg-lambda "${LAM}" \
    --run-tag "${RUN_TAG}" \
    --seed "${s}" \
    --name "h1-pathreg-l${TAG}-s${s}-extrap60" \
    "${REST[@]}"
done

"${PYTHON}" eval/evaluate_extrap.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/summarize_pathreg.py --out-root "${REPO_ROOT}/figures/pathreg"
echo "[pathreg-extrap] done lambda=${LAM} ${RUN_TAG}"
