#!/bin/bash -l
#SBATCH --job-name=h1-ncde-ev
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ncde_eval_%j.log

# In-window NCDE eval. Writes figures/ncde/ (gitignored). Not figures/h1_lock.
conda activate venv
set -euo pipefail
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

RUN_TAG="h1_ncde_seed"
OUT_DIR="${REPO_ROOT}/figures/ncde/${RUN_TAG}"
mkdir -p "${OUT_DIR}" logs

for s in 0 1 2 3 4; do
  d="${REPO_ROOT}/checkpoints/${RUN_TAG}${s}"
  if [[ ! -f "${d}/COMPLETE" || ! -f "${d}/best.ckpt" ]]; then
    echo "REFUSE: incomplete ${d} (need COMPLETE and best.ckpt)" >&2
    exit 1
  fi
done

"${PYTHON}" eval/evaluateh1.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/evaluate_drop.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/collect_pathreg_z0.py \
  --run-tag "${RUN_TAG}" --ckpt-name best.ckpt --n-seeds 5 \
  --out "${OUT_DIR}/z0.json"
echo "[ncde-eval] done ${OUT_DIR}"
