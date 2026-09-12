#!/usr/bin/env bash
#SBATCH --job-name=lidar-bedlam
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=outputs/slurm-%x-%j.out
#
# Self-resubmitting training job for the Helma cluster.
#
#   sbatch --export=ALL,CONFIG=configs/main_mixed.yaml slurm/train.sbatch
#
# Submit from the repo root (the Slurm log goes to outputs/). Environment
# (all optional):
#   CONFIG        config file (default configs/main_mixed.yaml)
#   RUN_NAME      run name; empty = enumerate <experiment>-NNN once and keep
#                 it for every resubmission (passed on via --export)
#   DATA_ROOT     shard root on the cluster (default <REPO>/data/generated)
#   STAGE_TO_TMP  1 = copy the shards to node-local /tmp first (default 1)
#   STAGE_DIRS    sub-directories of DATA_ROOT to stage (default: all shard dirs)
#   MAX_RESUBMITS safety cap on chain length (default 20)
#   EXTRA_SET     extra "section.key=value" overrides
#
# Notes for Helma (see docs/cluster.md): only $HOME, /hnvme/workspace and
# node-local /tmp are visible on compute nodes ($WORK on /home/atuin and
# /anvme are not); wandb has no internet there, so the trainer writes
# outputs/<run>/metrics.jsonl and slurm/wandb_mirror.sh (login node) pushes
# it to wandb live.
set -euo pipefail

REPO=${REPO:-/hnvme/workspace/v103fe17-lidar-bedlam}
CONFIG=${CONFIG:-configs/main_mixed.yaml}
DATA_ROOT=${DATA_ROOT:-/hnvme/workspace/v103fe17-lidar-bedlam/data/generated}
STAGE_TO_TMP=${STAGE_TO_TMP:-1}
MAX_RESUBMITS=${MAX_RESUBMITS:-20}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
EXTRA_SET=${EXTRA_SET:-}
WANDB_PROJECT=${WANDB_PROJECT:-lidar-bedlam}

cd "$REPO"
# the venv lives in $HOME: workspace filesystems are inode-limited
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
export WANDB_MODE=disabled  # metrics.jsonl is mirrored to wandb from the login node
export OMP_NUM_THREADS=8

if [ -z "${RUN_NAME:-}" ]; then
  RUN_NAME=$(uv run python scripts/next_run_name.py --config "$CONFIG")
fi
export RUN_NAME
echo "run $RUN_NAME config $CONFIG resubmit $RESUBMIT_COUNT job $SLURM_JOB_ID"

STAGE_DIRS=${STAGE_DIRS:-"body_models real/v1 synth/v1 synth/v1_groups02-11 synth/v1_group12"}
if [ "$STAGE_TO_TMP" = "1" ]; then
  LOCAL=${TMPDIR:-/tmp/$USER}/lidar-bedlam-data  # $TMPDIR is job-scoped and cleaned by Slurm
  mkdir -p "$LOCAL"
  for d in $STAGE_DIRS; do
    [ -d "$DATA_ROOT/$d" ] || continue
    rsync -a --relative --info=progress2 "$DATA_ROOT/./$d" "$LOCAL/" | tail -1
  done
  export DATA_ROOT=$LOCAL
else
  export DATA_ROOT
fi

# Re-queue on the wall-time signal: the trainer checkpoints on SIGUSR1 and
# exits, then a fresh job with the same RUN_NAME resumes from last.pt.
resubmit() {
  if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
    echo "resubmit cap reached"; return
  fi
  if [ -f "outputs/$RUN_NAME/DONE" ]; then
    echo "run finished, no resubmit"; return
  fi
  sbatch --export=ALL,CONFIG="$CONFIG",RUN_NAME="$RUN_NAME",DATA_ROOT="${DATA_ROOT_ORIG}",RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),EXTRA_SET="$EXTRA_SET" \
    "$REPO/slurm/train.sbatch"
}
DATA_ROOT_ORIG=${DATA_ROOT_ORIG:-${DATA_ROOT}}
export DATA_ROOT_ORIG

forward_usr1() {
  echo "USR1 from Slurm: forwarding to the trainer"
  kill -USR1 "$TRAIN_PID" 2>/dev/null || true
}
trap forward_usr1 USR1

# shellcheck disable=SC2086
uv run torchrun --standalone --nproc_per_node 4 scripts/lidar-bedlam-main.py \
  --config "$CONFIG" \
  --wandb-project "$WANDB_PROJECT" --wandb-name "$RUN_NAME" \
  --wandb-mode disabled \
  --set $EXTRA_SET &
TRAIN_PID=$!
wait $TRAIN_PID || true
wait $TRAIN_PID 2>/dev/null || true
STATUS=$?
echo "trainer exited with $STATUS"
resubmit
