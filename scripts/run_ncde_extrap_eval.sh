#!/bin/bash -l
#SBATCH --job-name=h1-ncdex-ev
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ncde_extrap_eval_%j.log

# Prefix-60 NCDE eval. Writes figures/ncde/ (gitignored). Not figures/h1_lock.
conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

RUN_TAG="h1_ncde_seed"
OUT_DIR="${REPO_ROOT}/figures/ncde/${RUN_TAG}_extrap60"
mkdir -p "${OUT_DIR}" logs

"${PYTHON}" eval/evaluate_extrap.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 --out-dir "${OUT_DIR}"
"${PYTHON}" eval/collect_pathreg_z0.py \
  --run-tag "${RUN_TAG}" --ckpt-name best.ckpt --n-seeds 5 --extrap \
  --out "${OUT_DIR}/z0.json"
echo "[ncde-extrap-eval] done ${OUT_DIR}"
