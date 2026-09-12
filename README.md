# lidar-bedlam

Synthetic LiDAR-camera SMPL pose and shape dataset from BEDLAM depth and a
selective-attention fusion model, evaluated on Waymo and SLOPER4D.

## Quickstart

```bash
uv sync
git submodule update --init
python3 lidar_bedlam/scripts/dm_link.py --smoke      # resources/, outputs/ for this machine
uv run python -m lidar_bedlam.body.convert_smpl \
    resources/data/body_models/smpl/SMPL_NEUTRAL.pkl \
    resources/data/generated/body_models/smpl/SMPL_NEUTRAL.pkl
uv run pytest
uv run jupyter lab notebooks/capabilities.ipynb   # quick visual checks on real data
```

Training (local smoke; the cluster job is `lidar_bedlam/slurm/train.sbatch`):

```bash
uv run torchrun --standalone --nproc_per_node 1 lidar_bedlam/scripts/lidar-bedlam-main.py \
    --config configs/smoke.yaml --wandb-project lidar-bedlam \
    --wandb-name smoke-local --wandb-mode offline
uv run python lidar_bedlam/scripts/lidar-bedlam-eval.py --config configs/smoke.yaml \
    --checkpoint outputs/smoke-local/best.pt --out outputs/smoke-local/eval.json
```

## Verify

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Layout

See [HANDOFF.md](HANDOFF.md) for the full tree and
[.docs/progress.md](.docs/progress.md) for the timetable, log and todos.
The package is `lidar_bedlam/` (flat layout), the paper is
`.docs/latex-draft/` (build with `latexmk` there).

`resources/` and `outputs/` are gitignored and machine-dependent.
Build them for this machine with `python3 lidar_bedlam/scripts/dm_link.py` (hosts and
sources in `config-global.json`).
