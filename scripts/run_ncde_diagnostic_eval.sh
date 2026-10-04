#!/bin/bash -l
#SBATCH --job-name=h1-ncde-diag
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ncde_diagnostic_%j.log

# PARTIAL diagnostic only. Not D-017. Does not write figures/ncde/.
conda activate venv
set -euo pipefail
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"
mkdir -p logs
"${PYTHON}" eval/evaluate_ncde_diagnostic.py --device auto \
  --out-dir "${REPO_ROOT}/figures/ncdediagnostic"
echo "[ncde-diag] trainer-exit"
