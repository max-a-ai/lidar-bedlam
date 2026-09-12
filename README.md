# lidar-bedlam

Synthetic LiDAR-camera SMPL pose and shape dataset from BEDLAM depth and a
selective-attention fusion model, evaluated on Waymo and SLOPER4D.

## Quickstart

```bash
uv sync
git submodule update --init
python3 scripts/dm_link.py --smoke      # data/, checkpoints/, outputs/ for this machine
uv run python -m lidar_bedlam.body.convert_smpl \
    data/body_models/smpl/SMPL_NEUTRAL.pkl \
    data/generated/body_models/smpl/SMPL_NEUTRAL.pkl
uv run pytest
uv run jupyter lab debug/capabilities.ipynb   # what the repo can do, on real data
```

Training (local smoke; the cluster job is `slurm/train.sbatch`):

```bash
uv run torchrun --standalone --nproc_per_node 1 main.py \
    --config configs/smoke.yaml --wandb-project lidar-bedlam \
    --wandb-name smoke-local --wandb-mode offline
```

## Verify

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run mypy src/
```

## Layout

See [HANDOFF.md](HANDOFF.md) for the full picture and
[.docs/progress.md](.docs/progress.md) for the timetable, log and todos.
The paper lives in `.docs/latex-draft/` (build with `latexmk` there).

`data/`, `checkpoints/` and `outputs/` are gitignored and machine-dependent.
Build them for this machine with `python3 scripts/dm_link.py` (hosts and
sources in `config-global.json`).
