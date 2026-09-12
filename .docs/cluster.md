# Cluster notes (Helma, NHR@FAU)

Facts verified with probe jobs on 2026-09-10 (`~/probe4.out`, `~/probe5.out`
on the login node).

| Item | Value |
|---|---|
| Login | `ssh helma` (from `~/.ssh/config`), account `v103fe`, user `v103fe17` |
| Partitions | `h100` and `h200` (4 GPUs per node, 24 h), `preempt` (48 h, preemptible), `cpu` (48-core NUMA multiples only) |
| Repo | `/hnvme/workspace/v103fe17-lidar-bedlam` (Lustre NVMe workspace, unlimited space, 61k/81k file quota) |
| Data | `/hnvme/workspace/v103fe17-lidar-bedlam/resources/data/generated` (`DATA_ROOT`) |
| Visible on compute nodes | `$HOME` (`/home/hpc`, 100 GB quota), `/hnvme/workspace`, `/tmp` (14 TB node-local NVMe, `$TMPDIR`) |
| **Not** visible on compute nodes | `$WORK` (`/home/atuin`), `/anvme/workspace` (the Alex NVMe) |
| Internet | login node: GitHub, PyPI, wandb reachable; compute nodes: PyPI only, no wandb |
| Tooling | `uv 0.12` in `~/.local/bin`, system Python 3.9 (uv fetches 3.12) |

## Workflow

```bash
# local: push code and data
rsync -az --exclude .venv --exclude /data --exclude /checkpoints --exclude /outputs . helma:/hnvme/workspace/v103fe17-lidar-bedlam/
rsync -a resources/data/generated/{body_models,synth,real} helma:/hnvme/workspace/v103fe17-lidar-bedlam/resources/data/generated/

# helma login node
cd /hnvme/workspace/v103fe17-lidar-bedlam
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/lidar-bedlam   # venv outside the inode-limited workspace
uv sync
setsid nohup lidar_bedlam/slurm/wandb_mirror.sh > outputs/wandb_mirror.log 2>&1 < /dev/null &   # once per login session: live wandb
sbatch --export=ALL,CONFIG=configs/main_mixed.yaml lidar_bedlam/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/ablation_gate_none.yaml lidar_bedlam/slurm/train.sbatch
squeue -u $USER
```

The job script enumerates the run name (`<experiment>-NNN`) once, stages the
shards to `/tmp`, runs `torchrun` on 4 GPUs, checkpoints on Slurm's `USR1`
15 min before the wall time, and resubmits itself with the same run name
until `outputs/<run>/DONE` exists or the resubmit cap is hit. Metrics go
to `outputs/<run>/metrics.jsonl`; `lidar_bedlam/slurm/wandb_mirror.sh` on the login node
mirrors them to wandb every 2 minutes (charts use `train/epoch` as x axis).
