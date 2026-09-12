#!/usr/bin/env bash
# Keep syncing offline wandb runs from the login node (which has internet)
# every INTERVAL seconds. Start once, detached:
#   setsid nohup slurm/wandb_sync_loop.sh > outputs/wandb_sync_loop.log 2>&1 &
set -uo pipefail
ROOT=${1:-outputs}
INTERVAL=${INTERVAL:-600}
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
while true; do
  for run in "$ROOT"/*/wandb/offline-run-*; do
    [ -d "$run" ] || continue
    uv run wandb sync "$run" 2>&1 | grep -v "^$" | tail -1
  done
  sleep "$INTERVAL"
done
