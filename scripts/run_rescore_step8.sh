#!/bin/bash -l
#SBATCH --job-name=s9-rescore
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/step9_rescore_%j.log

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"
exec "${PYTHON}" eval/rescore_step8_early.py
