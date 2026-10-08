#!/bin/bash -l
set -euo pipefail

# Extract one MFWD species with the verified data/ extractor.
# Usage: scripts/extract_one_species.sh SPECIES
# Output and COMPLETE sentinel: /scratch/ssing226/mfwd/extract/SPECIES/

SPECIES="${1:?usage: $0 SPECIES}"
REPO_ROOT="/rhome/ssing226/MastersThesis"
GT="/scratch/ssing226/mfwd/gt.csv"
IMAGE_ROOT="/scratch/ssing226/mfwd/extracted/trays"
OUT="/scratch/ssing226/mfwd/extract/${SPECIES}"
PYTHON="${CONDA_PREFIX:-/rhome/ssing226/.conda/envs/venv}/bin/python"

mkdir -p "${OUT}" /scratch/ssing226/mfwd/verify/mpl
export MPLBACKEND=Agg
export MPLCONFIGDIR="/scratch/ssing226/mfwd/verify/mpl"
export PYTHONUNBUFFERED=1

exec "${PYTHON}" "${REPO_ROOT}/scripts/extract_one_species.py" \
  --species "${SPECIES}" \
  --gt "${GT}" \
  --image-root "${IMAGE_ROOT}" \
  --out-dir "${OUT}" \
  --data-dir "${REPO_ROOT}/data"
