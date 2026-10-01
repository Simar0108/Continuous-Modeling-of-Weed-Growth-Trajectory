#!/bin/bash -l
#SBATCH --job-name=pr-repair
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/pathreg_repair_%j.log

# Re-score colliding pathreg ckpts into unique --out-dir. No training.
# Does not write figures/h1_lock or checkpoints/h1_final_best.

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

COLLIDE_TAG="h1_pathreg_l_seed"

echo "[repair] λ=1.0 in-window best-v2.ckpt"
mkdir -p "${REPO_ROOT}/figures/pathreg/h1_pathreg_l1_seed"
"${PYTHON}" eval/evaluateh1.py --device auto --run-tag "${COLLIDE_TAG}" \
  --n-seeds 5 --ckpt-name best-v2.ckpt \
  --out-dir "${REPO_ROOT}/figures/pathreg/h1_pathreg_l1_seed"

echo "[repair] λ=0.1 prefix-60 best-v1.ckpt"
mkdir -p "${REPO_ROOT}/figures/pathreg/h1_pathreg_l0p1_seed_extrap60"
"${PYTHON}" eval/evaluate_extrap.py --device auto --run-tag "${COLLIDE_TAG}" \
  --n-seeds 5 --ckpt-name best-v1.ckpt \
  --out-dir "${REPO_ROOT}/figures/pathreg/h1_pathreg_l0p1_seed_extrap60"

"${PYTHON}" eval/collect_pathreg_z0.py \
  --run-tag "${COLLIDE_TAG}" --ckpt-name best-v2.ckpt \
  --out "${REPO_ROOT}/figures/pathreg/h1_pathreg_l1_seed/z0_from_bestv2.json"

# Place already-valid columns into unique dirs so summarize can read CSVs.
mkdir -p "${REPO_ROOT}/figures/pathreg/h1_pathreg_l0p1_seed"
mkdir -p "${REPO_ROOT}/figures/pathreg/h1_pathreg_l0p01_seed_extrap60"
mkdir -p "${REPO_ROOT}/figures/pathreg/h1_pathreg_l1_seed_extrap60"
if [[ -f "${REPO_ROOT}/figures/pathreg/${COLLIDE_TAG}/table1_v2.csv" ]]; then
  cp -n "${REPO_ROOT}/figures/pathreg/${COLLIDE_TAG}/table1_v2.csv" \
    "${REPO_ROOT}/figures/pathreg/h1_pathreg_l0p1_seed/table1_v2.csv" || true
fi
if [[ -f "${REPO_ROOT}/figures/pathreg/${COLLIDE_TAG}/extrap_summary.csv" ]]; then
  cp -n "${REPO_ROOT}/figures/pathreg/${COLLIDE_TAG}/extrap_summary.csv" \
    "${REPO_ROOT}/figures/pathreg/h1_pathreg_l1_seed_extrap60/extrap_summary.csv" || true
fi

"${PYTHON}" eval/summarize_pathreg.py \
  --out-root "${REPO_ROOT}/figures/pathreg" \
  --table-out "${REPO_ROOT}/logs/step12_pathreg_stopping.json"
echo "[repair] done"
