#!/bin/bash
# Launch the 3-point pathreg sweep (in-window + prefix-60), 6 GPU jobs.
# Does not touch h1_final_best or h1_stab.
set -euo pipefail
REPO_ROOT="/rhome/ssing226/MastersThesis"
PARQUET="${1:-${REPO_ROOT}/Thesis/metrics_with_features.parquet}"
cd "$REPO_ROOT"
if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi
mkdir -p logs
for lam in 0.01 0.1 1.0; do
  echo "[pathreg] submit lambda=${lam}"
  sbatch --job-name="pr-in-${lam}" scripts/run_pathreg.sh "${PARQUET}" --pathreg-lambda "${lam}"
  sbatch --job-name="pr-ex-${lam}" scripts/run_pathreg_extrap.sh "${PARQUET}" --pathreg-lambda "${lam}"
done
squeue -u "$USER"
