#!/bin/bash -l
#SBATCH --job-name=s11b-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/step11b_eval_%j.log

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

"${PYTHON}" eval/evaluateh1.py --device auto --run-tag h1_stab_seed --n-seeds 5
"${PYTHON}" eval/evaluate_drop.py --device auto --run-tag h1_stab_seed --n-seeds 5
echo "[step11b_eval] done"
