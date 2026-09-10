#!/usr/bin/env bash
#SBATCH --job-name=lidar-bedlam
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=%x-%j.out
#
# Self-resubmitting training job for the Helma cluster.
#
#   sbatch --export=ALL,CONFIG=configs/main_mixed.yaml scripts/slurm/train.sbatch
#
# Environment (all optional):
#   CONFIG        config file (default configs/main_mixed.yaml)
#   RUN_NAME      run name; empty = enumerate <experiment>-NNN once and keep
#                 it for every resubmission (passed on via --export)
#   DATA_ROOT     shard root on the cluster (default <REPO>/data/generated)
#   STAGE_TO_TMP  1 = copy the shards to node-local /tmp first (default 1)
#   MAX_RESUBMITS safety cap on chain length (default 20)
#   EXTRA_SET     extra "section.key=value" overrides
#
# Notes for Helma (see docs/cluster.md): only $HOME, /hnvme/workspace and
# node-local /tmp are visible on compute nodes ($WORK on /home/atuin and
# /anvme are not); wandb has no internet there, so runs are offline and
# synced from the login node (scripts/slurm/wandb_sync.sh).
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
export WANDB_MODE=offline
export OMP_NUM_THREADS=8

if [ -z "${RUN_NAME:-}" ]; then
  RUN_NAME=$(uv run python scripts/next_run_name.py --config "$CONFIG")
fi
export RUN_NAME
echo "run $RUN_NAME config $CONFIG resubmit $RESUBMIT_COUNT job $SLURM_JOB_ID"

if [ "$STAGE_TO_TMP" = "1" ]; then
  LOCAL=/tmp/$USER/lidar-bedlam-data
  mkdir -p "$LOCAL"
  rsync -a --info=progress2 "$DATA_ROOT/" "$LOCAL/"
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
  if [ -f "checkpoints/$RUN_NAME/DONE" ]; then
    echo "run finished, no resubmit"; return
  fi
  sbatch --export=ALL,CONFIG="$CONFIG",RUN_NAME="$RUN_NAME",DATA_ROOT="${DATA_ROOT_ORIG}",RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),EXTRA_SET="$EXTRA_SET" \
    "$REPO/scripts/slurm/train.sbatch"
}
DATA_ROOT_ORIG=${DATA_ROOT_ORIG:-${DATA_ROOT}}
export DATA_ROOT_ORIG

forward_usr1() {
  echo "USR1 from Slurm: forwarding to the trainer"
  kill -USR1 "$TRAIN_PID" 2>/dev/null || true
}
trap forward_usr1 USR1

# shellcheck disable=SC2086
uv run torchrun --standalone --nproc_per_node 4 main.py \
  --config "$CONFIG" \
  --wandb-project "$WANDB_PROJECT" --wandb-name "$RUN_NAME" \
  --wandb-mode offline \
  --set $EXTRA_SET &
TRAIN_PID=$!
wait $TRAIN_PID || true
wait $TRAIN_PID 2>/dev/null || true
STATUS=$?
echo "trainer exited with $STATUS"
resubmit
