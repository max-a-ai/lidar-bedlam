# lidar-bedlam

Synthetic LiDAR-camera SMPL pose and shape dataset from BEDLAM and a
selective-attention fusion model.

## Quickstart

```bash
uv sync
git submodule update --init
uv run lidar-bedlam
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
