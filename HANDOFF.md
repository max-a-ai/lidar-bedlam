# lidar-bedlam — Handoff

Generated: 2026-09-08

---

## Status Snapshot

Scaffold in place, five baseline submodules cloned, dataset audit done. All
three quality gates pass:

- `uv run ruff check src/`
- `uv run ruff format --check src/`
- `uv run mypy src/` (strict)

Next up: BEDLAM body-data download, then `bedlam.camera` / `bedlam.bodies`
parsers (see `PROJECT.md`).

---

## Goal

Generate a LiDAR + camera SMPL pose-and-shape dataset from BEDLAM v1 depth
renders, train a selective-attention fusion model on it (image cues for
hands, ankles, head orientation; LiDAR cues for body pose, shape and 3D
placement), and show on real data (SLOPER4D, LiDARHuman26M, Waymo) that
synthetic data improves real SMPL estimation. Paper for a small conference;
trainings must run within two weeks of 2026-09-08.

---

## Confirmed Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python version | **3.12** | torch / smplx / pytorch3d wheels; baselines pin <=3.10 in their own envs |
| Package manager | **UV** | Reproducibility. `uv sync`, `uv run`. |
| Build backend | **hatchling** | Stable, well-documented. |
| Project layout | **src layout** | `src/lidar_bedlam/` |
| Linter / formatter | **Ruff** (79 cols) | |
| Type checker | **mypy** (strict) | |
| Synthetic source | **BEDLAM v1 only** | full 32-bit depth on every frame; BEDLAM 2.0 (29 TB) skipped |
| Body model | **SMPL** | matches real GT (SLOPER4D, LiDARHuman26M) and LIF-Net/TokenHMR/CameraHMR |
| LiDAR simulation | **32 / 64 / 128 / 256 beams** + occlusion augmentation | resolution and occlusion robustness are the key ablations |
| Data location | `/mnt/md0/lidar-bedlam` | 8.2 TB free local RAID; NAS is sshfs and slow |
| Experiment tracking | wandb entity `erik_hm`, project `lidar-bedlam` | standing rule, auth via `~/.netrc` |
| Baselines | git submodules in `third_party/` | CameraHMR, TokenHMR, LiDAR-HMR, sam-3d-body, lif (own LIF-Net) |
| Progress log / todos | `PROJECT.md` | user-requested format; vault files point to it |

---

## Repository Structure

```
lidar-bedlam/
├── .vault-lidar-bedlam/
│   ├── lidar-bedlam.canvas      # Obsidian canvas — dashboard
│   ├── progress.md              # Gantt; log lives in PROJECT.md
│   └── todo.md                  # pointer; todos live in PROJECT.md
├── configs/                     # training / simulation configs (yaml)
├── docs/
│   ├── bedlam_audit.md          # what is on the NAS, depth semantics
│   ├── datasets.md              # real + synthetic dataset survey
│   └── research_notes.md        # sourced web research (BEDLAM2, baselines)
├── scripts/
│   └── validate_bedlam_on_nas.sh
├── src/lidar_bedlam/
│   ├── __init__.py
│   ├── __main__.py
│   └── app.py
├── third_party/                 # submodules: CameraHMR TokenHMR LiDAR-HMR sam-3d-body lif
├── PROJECT.md                   # daily log + structured todos (source of truth)
├── instructions.md              # agent coding standards
├── pyproject.toml
├── .python-version
├── README.md
├── .gitignore
└── HANDOFF.md                   # this file
```

### Vault

Open `.vault-lidar-bedlam/lidar-bedlam.canvas` in Obsidian for a live
dashboard. If Obsidian hides the dot-prefixed directory, enable "Show hidden
files" or open the canvas by path.

---

## Key facts future-you needs

- BEDLAM depth EXR: one `Depth` float channel, **planar z-depth in cm**,
  sky = 1e8, rendered from clothed characters. `fx = W/2 / tan(hfov/2)`.
- BEDLAM camera CSV is in Unreal coordinates (cm, yaw/pitch/roll in deg,
  left-handed, Z up); conversion notes in
  `third_party/CameraHMR` and the bedlam_render repo
  (`unreal/render/unreal_coordinate_system.md`).
- SMPL-X params are NOT in the image gt tars; the body-data download is
  required and must be done by the user (MPI login).
- Body models: `~/nas_drive/methods/max/data/body_models` (never commit).
- Baseline weights: TokenHMR in `~/nas_drive/methods/max/data/checkpoints/tokenhmr`,
  SAM 3D Body in the HF cache, LiDAR-HMR only via Baidu pan (local checkout
  `~/Documents/LiDAR-HMR` has `models/graphormer/data`).
- The NAS is sshfs: read archives with streaming `tar`, never `du` or copy.

---

## For the Next Handoff

1. Read this file top-to-bottom, then `PROJECT.md`.
2. Open the canvas for the visual index.
3. Read `instructions.md` for coding standards (enforced by ruff + mypy).
4. Run `uv sync` and `git submodule update --init` to reproduce the env.
5. When you finish a session, add one sentence to the `PROJECT.md` log,
   move finished todos to Done, and update this file if you learned
   something future-you would need to pick up cold.
