#!/usr/bin/env bash
#SBATCH --job-name=eval-runs
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm-%x-%j.out
#
# Re-evaluate finished runs with the current protocol (full validation sets,
# best.pt and last.pt) and write outputs/eval/<run>-<ckpt>.json.
#
#   sbatch lidar_bedlam/slurm/eval_runs.sbatch            # every run with a DONE marker
#   sbatch --export=ALL,RUNS="full-main-mixed-000 abl-ball025-000" lidar_bedlam/slurm/eval_runs.sbatch
#
# The config of a run is derived from its name: abl-scale-16x-001 ->
# configs/ablation_scale_16x.yaml, full-mix80-000 -> configs/full_mix80.yaml,
# abl-gate-hard-s1 -> configs/ablation_gate_hard.yaml.
set -uo pipefail

REPO=${REPO:-/hnvme/workspace/v103fe17-lidar-bedlam}
cd "$REPO"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/lidar-bedlam}
export DATA_ROOT=${DATA_ROOT:-/hnvme/workspace/v103fe17-lidar-bedlam/resources/data/generated}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
mkdir -p outputs/eval

if [ -z "${RUNS:-}" ]; then
  RUNS=$(for d in outputs/*/DONE; do basename "$(dirname "$d")"; done)
fi

config_of() {
  local name=$1
  name=${name%-[0-9][0-9][0-9]}  # -000
  name=${name%-s[0-9]}           # -s1
  name=${name/#abl-/ablation-}
  name=${name/#a-abl-/a-ablation-}  # point-anchored copies (a- prefix)
  echo "configs/${name//-/_}.yaml"
}

for run in $RUNS; do
  cfg=$(config_of "$run")
  if [ ! -f "$cfg" ]; then echo "$run: no config $cfg, skipped"; continue; fi
  for ckpt in best last; do
    out="outputs/eval/$run-$ckpt.json"
    if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then echo "$out exists"; continue; fi
    [ -f "outputs/$run/$ckpt.pt" ] || { echo "$run: no $ckpt.pt"; continue; }
    echo "== $run $ckpt ($cfg)"
    uv run python lidar_bedlam/scripts/lidar-bedlam-eval.py \
      --checkpoint "outputs/$run/$ckpt.pt" --config "$cfg" \
      --set data.val_max_samples=10000 data.num_workers=12 \
      --out "$out" || echo "$run $ckpt failed"
  done
done
echo "eval done"
