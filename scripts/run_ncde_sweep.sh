#!/bin/bash
# Submit NCDE in-window + prefix-60. Git-clean required. Kill date 2026-10-25.
set -euo pipefail
REPO_ROOT="/rhome/ssing226/MastersThesis"
PARQUET="${1:-${REPO_ROOT}/Thesis/metrics_with_features.parquet}"
cd "$REPO_ROOT"
if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi
mkdir -p logs
in_id="$(sbatch --parsable --job-name=h1-ncde scripts/run_ncde.sh "${PARQUET}")"
sbatch --dependency="afterok:${in_id}" --job-name=h1-ncdex scripts/run_ncde_extrap.sh "${PARQUET}"
echo "[ncde] submitted in-window ${in_id}; prefix-60 afterok"
squeue -u "$USER"
