#!/usr/bin/env bash
# Quick single-track overfit (default 40 epochs in ode.overfit_test) + W&B.
#
# Usage:
#   ./scripts/quick_overfit_wandb.sh /path/to/metrics.parquet [extra ode.overfit_test args...]
# Examples:
#   ./scripts/quick_overfit_wandb.sh ~/data/metrics.parquet
#   ./scripts/quick_overfit_wandb.sh ~/data/metrics.parquet --kinetic-reg-weight 0
#
# For a long run: use run_ode_overfit_wandb.sh with --epochs 300

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/scripts/run_ode_overfit_wandb.sh" "$@" --quick
