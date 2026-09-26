#!/bin/bash -l
#SBATCH --job-name=maize-100-tracks
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssing226@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/maize_100_%j.log

# ─────────────────────────────────────────────────────────────────────────────
# 100-Track Maize run — Magnitude Boost + Physiology Regularisation
#
# Key changes vs run_multi.sh:
#   • latent_dim 64     (scaled up from 32 for richer population coverage)
#   • batch-size 32     (mini-batches prevent OOM at 100 simultaneous trajectories)
#   • affine-scale 6.0  (boosted decoder amplitude to reach J-curve peaks)
#   • phys-dropout 0.2  (prevents Z channel from memorising individual tracks)
#   • size-loss-weight 1.0 / phys-loss-weight 0.5 (surgical loss weighting)
#   • horizon curriculum: 30% → 100% over epochs 100-250
#
# Usage (batch):
#   sbatch scripts/run_100.sh Thesis/metrics_with_features.parquet [--name RUN]
#
# Usage (dry-run / interactive check):
#   bash scripts/run_100.sh Thesis/metrics_with_features.parquet --name maize-test
# ─────────────────────────────────────────────────────────────────────────────

conda activate venv
PYTHON="${CONDA_PREFIX}/bin/python"

REPO_ROOT="/rhome/ssing226/MastersThesis"
cd "$REPO_ROOT"

PARQUET="${1:-}"
if [[ -z "${PARQUET}" ]]; then
  echo "Usage: $0 /path/to/metrics.parquet [--name RUN] ..." >&2
  exit 1
fi
shift

if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi

mkdir -p logs checkpoints

export PYTHONPATH="${REPO_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-ode-maize-100}"
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

# Optional: a run name can be passed with --name; default below is used otherwise.
RUN_NAME="${RUN_NAME:-maize-century-run}"

exec "${PYTHON}" -m ode.train_multi \
  --parquet "${PARQUET}" \
  --wandb \
  --project "${WANDB_PROJECT}" \
  --solver dopri5 \
  --rtol 1e-6 \
  --atol 1e-6 \
  --species    Maize \
  --max-tracks 100 \
  --batch-size 32 \
  --latent-dim 64 \
  --epochs     400 \
  --sigma-mode z_score \
  --geometry-size-loss log \
  --size-loss-weight 1.0 \
  --phys-loss-weight 0.5 \
  --phys-dropout 0.2 \
  --affine-scale-init 6.0 \
  --freeze-affine-epochs 50 \
  --horizon-start-frac 0.3 \
  --horizon-ramp-start 100 \
  --horizon-ramp-end 250 \
  --name "${RUN_NAME}" \
  "$@"
