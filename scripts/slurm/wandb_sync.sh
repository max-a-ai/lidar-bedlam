#!/usr/bin/env bash
# Sync offline wandb runs from the login node (which has internet).
#   scripts/slurm/wandb_sync.sh [checkpoints]
set -euo pipefail
ROOT=${1:-checkpoints}
for run in "$ROOT"/*/wandb/offline-run-*; do
  [ -d "$run" ] || continue
  uv run wandb sync "$run"
done
