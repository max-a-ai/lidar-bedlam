# lidar-bedlam

Synthetic LiDAR-camera SMPL pose and shape dataset from BEDLAM and a
selective-attention fusion model.

## Quickstart

```bash
uv sync
git submodule update --init
bash scripts/link_data.sh          # data/ symlinks to the NAS and local RAID
uv run python -m lidar_bedlam.body.convert_smpl \
    data/body_models/smpl/SMPL_NEUTRAL.pkl \
    data/generated/body_models/smpl/SMPL_NEUTRAL.pkl
uv run pytest
```

## Verify

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run mypy src/
```

## Layout

See [HANDOFF.md](HANDOFF.md) for the full picture and [PROJECT.md](PROJECT.md)
for the daily log and todos. Live dashboard: open
`.vault-lidar-bedlam/lidar-bedlam.canvas` in Obsidian.
