#!/bin/bash -l
#SBATCH --job-name=s11c-ex5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/step11c_extrap_train_%j.log

# Train five prefix-60% ODE seeds sequentially, then score tails.
conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"
REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-maize-100}"
export WANDB_DIR="${WANDB_DIR:-/scratch/ssing226/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/ssing226/wandb-cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/scratch/ssing226/wandb-artifacts}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" logs checkpoints
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"

PARQUET="${REPO_ROOT}/Thesis/metrics_with_features.parquet"
for s in 0 1 2 3 4; do
  echo "[extrap-train] seed ${s}"
  "${PYTHON}" -m ode.train_h1_seed \
    --parquet "${PARQUET}" \
    --wandb \
    --project "${WANDB_PROJECT}" \
    --species Maize \
    --max-tracks 100 \
    --epochs 400 \
    --run-tag h1_stab_seed \
    --ema-decay 0.999 \
    --fixed-horizon-val-every 1 \
    --seed "${s}" \
    --train-time-frac 0.6 \
    --name "h1-stab-s${s}-extrap60"
done
"${PYTHON}" eval/evaluate_extrap.py --device auto --run-tag h1_stab_seed --n-seeds 5
echo "[step11c_extrap] done"
