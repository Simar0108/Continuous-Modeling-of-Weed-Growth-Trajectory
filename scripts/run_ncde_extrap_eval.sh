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

# Prefix-60 NCDE eval. COMPLETE+best.ckpt is the per-seed gate (afterany).
# Full 5/5 writes figures/ncde/. Partial survivors write figures/ncdediagnostic/.
conda activate venv
set -euo pipefail
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

RUN_TAG="h1_ncde_seed"
OUT_DIR="${REPO_ROOT}/figures/ncde/${RUN_TAG}_extrap60"
mkdir -p "${OUT_DIR}" logs

complete=()
for s in 0 1 2 3 4; do
  d="${REPO_ROOT}/checkpoints/${RUN_TAG}${s}_extrap60"
  if [[ -f "${d}/COMPLETE" && -f "${d}/best.ckpt" ]]; then
    complete+=("${s}")
  else
    echo "skip seed ${s}: no COMPLETE+best.ckpt (${d})"
  fi
done
if [[ ${#complete[@]} -eq 0 ]]; then
  echo "REFUSE: no COMPLETE prefix-60 NCDE seeds" >&2
  exit 1
fi
echo "[ncde-extrap-eval] COMPLETE seeds: ${complete[*]} (${#complete[@]}/5)"

if [[ ${#complete[@]} -lt 5 ]]; then
  echo "PARTIAL n=${#complete[@]}/5 diagnostic, not pre-registered protocol"
  "${PYTHON}" eval/evaluate_ncde_diagnostic.py --device auto \
    --out-dir "${REPO_ROOT}/figures/ncdediagnostic"
  echo "[ncde-extrap-eval] diagnostic done figures/ncdediagnostic"
  exit 0
fi

"${PYTHON}" eval/evaluate_extrap.py --device auto --run-tag "${RUN_TAG}" --n-seeds 5 \
  --require-complete --out-dir "${OUT_DIR}"
"${PYTHON}" eval/collect_pathreg_z0.py \
  --run-tag "${RUN_TAG}" --ckpt-name best.ckpt --n-seeds 5 --extrap --require-complete \
  --out "${OUT_DIR}/z0.json"
echo "[ncde-extrap-eval] done ${OUT_DIR}"
