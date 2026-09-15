#!/usr/bin/env bash
#SBATCH --job-name=lidar-bedlam
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm-%x-%j.out
#
# Training job for cluster2 (ivlcluster02: 7x RTX 4090 24 GB, 96 CPUs, single
# node Slurm, internet, NAS at /mnt/nas). Data and outputs live on the NAS
# (resources/data and outputs are symlinks into /mnt/nas/methods/max/lidar-bedlam);
# the venv is in $HOME/venvs, the uv cache on the NAS (the local disk is small).
#
#   sbatch --export=ALL,CONFIG=configs/ablation_loss_mesh.yaml lidar_bedlam/slurm/train_cluster2.sbatch
#   sbatch --gres=gpu:rtx4090:4 --cpus-per-task=48 --export=ALL,CONFIG=...,NPROC=4 lidar_bedlam/slurm/train_cluster2.sbatch
#
# Environment (all optional):
#   CONFIG       config file (default configs/main_mixed.yaml)
#   RUN_NAME     run name; empty = enumerate <experiment>-NNN
#   NPROC        processes (GPUs) for torchrun, match --gres (default 1)
#   EXTRA_SET    extra "section.key=value" overrides
#   OMP_THREADS  OMP_NUM_THREADS (default 2)
# wandb logs online here (no mirror needed); auth via ~/.netrc.
set -euo pipefail

REPO=${REPO:-$HOME/lidar-bedlam}
CONFIG=${CONFIG:-configs/main_mixed.yaml}
NPROC=${NPROC:-1}
EXTRA_SET=${EXTRA_SET:-}
WANDB_PROJECT=${WANDB_PROJECT:-lidar-bedlam}

cd "$REPO"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/nas/methods/max/uv-cache}
export DATA_ROOT=${DATA_ROOT:-/mnt/nas/methods/max/lidar-bedlam/data/generated}
export OMP_NUM_THREADS=${OMP_THREADS:-2}
export PATH=$HOME/.local/bin:$PATH

if [ -z "${RUN_NAME:-}" ]; then
  RUN_NAME=$(uv run python lidar_bedlam/scripts/next_run_name.py --config "$CONFIG")
fi
export RUN_NAME
echo "run $RUN_NAME config $CONFIG job $SLURM_JOB_ID gpus $NPROC host $(hostname)"

# shellcheck disable=SC2086
uv run torchrun --standalone --nproc_per_node "$NPROC" lidar_bedlam/scripts/lidar-bedlam-main.py \
  --config "$CONFIG" \
  --wandb-project "$WANDB_PROJECT" --wandb-name "$RUN_NAME" \
  --set $EXTRA_SET
echo "trainer exited with $?"
