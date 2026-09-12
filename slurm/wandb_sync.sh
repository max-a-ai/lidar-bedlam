#!/usr/bin/env bash
# Sync offline wandb runs from the login node (which has internet).
#   slurm/wandb_sync.sh [outputs]
set -euo pipefail
ROOT=${1:-outputs}
for run in "$ROOT"/*/wandb/offline-run-*; do
  [ -d "$run" ] || continue
  uv run wandb sync "$run"
done
