#!/usr/bin/env bash
# Login-node loop: mirror outputs/*/metrics.jsonl to wandb (live) and sync
# any finished offline wandb runs left by older jobs. Start once, detached:
#   setsid nohup lidar_bedlam/slurm/wandb_mirror.sh > outputs/wandb_mirror.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
INTERVAL=${INTERVAL:-120}
while true; do
  uv run python lidar_bedlam/scripts/wandb_mirror.py --root outputs --once 2>&1 | grep -v "^wandb: " | grep -v "^$"
  for run in outputs/*/wandb/offline-run-*; do
    [ -d "$run" ] || continue
    [ -f "$(dirname "$(dirname "$run")")/DONE" ] || continue
    [ -f "$run/.synced" ] && continue
    uv run wandb sync "$run" 2>&1 | tail -1 && touch "$run/.synced"
  done
  sleep "$INTERVAL"
done
