#!/bin/bash
# Submit 5 in-window + 5 prefix-60 NCDE seeds, then eval afterany.
# COMPLETE+best.ckpt inside the eval script is the real per-seed gate.
# afterok cannot express "score what survived."
# Git-clean required. Kill date 2026-10-25. Does not write figures/h1_lock.
set -euo pipefail
REPO_ROOT="/rhome/ssing226/MastersThesis"
PARQUET="${1:-${REPO_ROOT}/Thesis/metrics_with_features.parquet}"
cd "$REPO_ROOT"
if [[ ! -f "${PARQUET}" ]]; then
  echo "File not found: ${PARQUET}" >&2
  exit 1
fi
mkdir -p logs

in_ids=()
ex_ids=()
for s in 0 1 2 3 4; do
  in_ids+=("$(sbatch --parsable --job-name="h1-ncde-s${s}" scripts/run_ncde.sh "${PARQUET}" --seed "${s}")")
  ex_ids+=("$(sbatch --parsable --job-name="h1-ncdex-s${s}" scripts/run_ncde_extrap.sh "${PARQUET}" --seed "${s}")")
done
in_dep="$(IFS=:; echo "${in_ids[*]}")"
ex_dep="$(IFS=:; echo "${ex_ids[*]}")"
ev="$(sbatch --parsable --dependency="afterany:${in_dep}" --job-name=h1-ncde-ev scripts/run_ncde_eval.sh)"
exev="$(sbatch --parsable --dependency="afterany:${ex_dep}" --job-name=h1-ncdex-ev scripts/run_ncde_extrap_eval.sh)"
echo "[ncde] in-window ${in_ids[*]} -> eval ${ev}"
echo "[ncde] prefix-60 ${ex_ids[*]} -> eval ${exev}"
squeue -u "$USER"
