#!/usr/bin/env bash
# Login-node loop: mirror outputs/*/metrics.jsonl to wandb (live) and sync
# any finished offline wandb runs left by older jobs. Start once, detached:
#   setsid nohup lidar_bedlam/slurm/wandb_mirror.sh > outputs/wandb_mirror.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
INTERVAL=${INTERVAL:-120}
export WANDB_MIRROR_DIR=${WANDB_MIRROR_DIR:-$HOME/wandb-mirror}  # not in the workspace (file quota)
mkdir -p "$WANDB_MIRROR_DIR"
while true; do
  # the venv binaries directly: uv run exits 120 without a TTY (nohup)
  "$UV_PROJECT_ENVIRONMENT/bin/python" lidar_bedlam/scripts/wandb_mirror.py --root outputs --once 2>&1 | grep -v "^wandb: " | grep -v "^$"
  for run in outputs/*/wandb/offline-run-*; do
    [ -d "$run" ] || continue
    [ -f "$(dirname "$(dirname "$run")")/DONE" ] || continue
    [ -f "$run/.synced" ] && continue
    "$UV_PROJECT_ENVIRONMENT/bin/wandb" sync "$run" 2>&1 | tail -1 && touch "$run/.synced"
  done
  sleep "$INTERVAL"
done
