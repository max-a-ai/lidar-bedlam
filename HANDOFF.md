# lidar-bedlam — Handoff

Generated: 2026-09-08, layout finalised 2026-09-12 ("preparation finished")

---

## Status Snapshot

Everything up to training is done and on the cluster: BEDLAM extraction
(12 groups), synthetic shards (390,373 records with precomputed ViT-H
tokens), real shards (Waymo 4,591 / 894, SLOPER4D 21,062 / ~10k), the
selective-attention model, the trainer, the Slurm chain job, and the
batch-size scaling tests. From here on the work happens on Helma
(`/hnvme/workspace/v103fe17-lidar-bedlam`); this checkout is the template.

Gates: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy`, `uv run pytest` (70 tests) all pass.

Running / next: `mix80` (80 % BEDLAM, 10 % SLOPER4D, 10 % Waymo) and
`synth100` (100 % BEDLAM) at batch 2048, then the ablation ladder in
`.docs/story.md`; results and todos in `.docs/progress.md`.

---

## Goal

Generate a LiDAR + camera SMPL pose-and-shape dataset from BEDLAM v1 depth
renders, train a selective-attention fusion model on it (image cues for
head, arms, hands, legs, feet; LiDAR cues for body centre, placement,
orientation, shape), and show on Waymo and SLOPER4D that synthetic data
improves in-the-wild SMPL estimation including 3D placement. Eurographics
2027 full paper: abstract 25 Sep 2026, paper 1 Oct 2026, double-blind.

---

## Confirmed Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python version | **3.12** | torch cu126 / smplx wheels; cluster |
| Package manager | **UV** | `uv sync`, `uv run`; default groups `dev` + `debug` |
| Build backend | **hatchling** | stable, well-documented |
| Project layout | **flat**: `lidar-bedlam/lidar_bedlam/` with `scripts/` and `slurm/` inside the package; `resources/` (data, pretrained checkpoints) and `outputs/` gitignored | top level stays minimal; Python import names cannot contain `-` |
| Package layout | fixed: `data/ models/ losses/ metrics/ train/ utils/` + domain packages | see "Package structure" below |
| Linter / formatter | **Ruff** (79 cols), excludes `third_party/`, `notebooks/` | |
| Type checker | **mypy** (strict) on `lidar_bedlam/ tests/ lidar_bedlam/scripts/` | |
| Entry points | `lidar_bedlam/scripts/lidar-bedlam-main.py` (train), `lidar_bedlam/scripts/lidar-bedlam-eval.py` | named after the repo; **the wandb hook matches `main.py` only, extend its regex to `*-main.py`** |
| Synthetic source | **BEDLAM v1 only**, 12 groups | full 32-bit depth per frame; BEDLAM 2.0 (29 TB) skipped |
| Body model | **SMPL** | matches real GT and the baselines |
| LiDAR simulation | Ouster families 32-256 ch x 512-2048 steps, placement ball 1 m, occlusion + sensor augmentation | resolution and placement transfer are the key ablations |
| Real targets | **Waymo** (headline), **SLOPER4D** (secondary) | LiDARHuman26M, PedX dropped |
| Backbone | ViT-H (TokenHMR weights) **frozen**, tokens precomputed per shard | 29 M trainable params |
| Shard reader | memory-mapped npz members | `np.load` re-read 96 MB per row; now 5,600 samples/s on 4 GPUs |
| Training | torchrun DDP, AdamW, warm-up + cosine, bf16; batch 2048, lr 3e-4, 12k steps for the full runs | batch tests 2026-09-12: throughput flat above 1024, 13.7 GiB/GPU |
| Cluster | Helma h100, self-resubmitting chain, data staged to job `$TMPDIR` | `lidar_bedlam/slurm/train.sbatch`, `.docs/cluster.md` |
| Live metrics | trainer writes `outputs/<run>/metrics.jsonl`; `lidar_bedlam/slurm/wandb_mirror.sh` on the login node mirrors it to wandb every 2 min, charts on `train/epoch` | compute nodes have no internet and no proxy; `wandb sync` cannot read a live offline run |
| Schedule unit | `optim.max_epochs` (epoch = one pass over all training records, ~416k); main 50, ablations 17 | `max_steps` derived; both logged |
| Data (local) | `/mnt/md0/lidar-bedlam` -> `resources/data/generated` | 8 TB RAID; the NAS is sshfs |
| Data (Helma) | `/hnvme/workspace/v103fe17-lidar-bedlam/resources/data/generated` | `$WORK`, `/anvme` not on compute nodes |
| Experiment tracking | wandb `erik_hm/lidar-bedlam` | auth via `~/.netrc` (also on Helma) |
| Baselines | git submodules in `third_party/` | CameraHMR, TokenHMR, LiDAR-HMR, sam-3d-body, lif |
| Paper build | `latexmk` in `.docs/latex-draft/` | style files vendored; venue macros `bib/bib-short.def` |

---

## Repository Structure (`tree -L 2 -a`, gitignored folders marked)

```
lidar-bedlam/
├── lidar_bedlam/            # THE package, flat layout, named after the repo (hyphen -> underscore)
│   ├── data/  models/  losses/  metrics/  train/  utils/     # mandatory subpackages
│   ├── body/  geometry/  lidar/  rigs/  generate/            # project-domain subpackages
│   ├── scripts/             # CLIs: lidar-bedlam-main.py, lidar-bedlam-eval.py, dm_link.py, data preparation
│   ├── slurm/               # cluster jobs: train.sbatch (chain job), wandb_mirror.sh
│   └── __init__.py  __main__.py  app.py  py.typed
├── configs/                 # one yaml per experiment (main, real/synth only, ablations, smoke)
├── notebooks/               # capabilities.ipynb: quick visual checks of what the repo can do
├── tests/                   # pytest, one file per module group
├── third_party/             # baseline repos as git submodules
├── .docs/                   # the record (hidden: needs `tree -a`): progress.md, notes, figures/, latex-draft/, runs/
├── resources/               # gitignored, built per machine by lidar_bedlam/scripts/dm_link.py
│   ├── data/                #   dataset links + generated/ (shards, tokens, extracted frames)
│   └── pretrained-checkpoints/  # links to pretrained weights (TokenHMR ViT-H)
├── outputs/                 # gitignored: outputs/<run-name>/ (last.pt, best.pt, metrics.jsonl, DONE), outputs/logs/
├── config-global.json       # hosts, datasets, checkpoints, methods, smoke
├── pyproject.toml  uv.lock  .python-version
├── README.md  HANDOFF.md  instructions.md
└── .gitignore  .gitmodules
```

### Package structure (fixed, the template for new projects)

```
lidar_bedlam/
├── data/        # datasets, schema, loaders, shards, augmentation          [mandatory]
├── models/      # architectures                                          [mandatory]
├── losses/      # training losses                                        [mandatory]
├── metrics/     # evaluation metrics and protocol                        [mandatory]
├── train/       # config, sampler, trainer                               [mandatory]
├── utils/       # io, viz, small helpers                                 [mandatory]
├── scripts/     # command-line entry points (no __init__.py)             [mandatory]
├── slurm/       # cluster job scripts                                    [mandatory if a cluster is used]
├── body/  geometry/  lidar/  rigs/  generate/   # project domains
├── __init__.py  # __version__
├── __main__.py  # python -m lidar_bedlam
├── app.py       # console entry (pyproject [project.scripts])
└── py.typed
```

`.docs/` is dot-prefixed and hidden by default in Obsidian and file
browsers; enable "Show hidden files" to see it.

---

## Key facts future-you needs

- BEDLAM depth EXR: one `Depth` float channel, **planar z-depth in cm**,
  sky = 1e8. `fx = W/2 / tan(hfov/2)`.
- SMPL labels: `resources/data/generated/bedlam_labels/smpl/`; translation =
  `trans_cam + cam_ext[:3,3]`; matched to mask persons by silhouette overlap.
- Shards: npz of 512 records with named LiDAR scan variants (`main_0`,
  `main_1`, `ball025`, `target_waymo`, `rig_*`, `check`; real: `real`);
  tokens next to each shard as `<shard>.tokens.npy` (N, 2, 256, 1280) fp16;
  the trainer requires the token file. Layout: `.docs/data_pipeline.md`.
- Body models: `~/nas_drive/methods/max/data/body_models` (never commit);
  chumpy-free SMPL pkl in `resources/data/generated/body_models/smpl/`.
- Baseline weights: TokenHMR ckpt linked as `resources/pretrained-checkpoints/tokenhmr_vith`;
  SAM 3D Body in the HF cache; LiDAR-HMR via Baidu pan only.
- Helma: venv in `$HOME/venvs/lidar-bedlam` (`UV_PROJECT_ENVIRONMENT`),
  never inside the inode-limited workspace; `lidar_bedlam/slurm/wandb_mirror.sh` started
  once per login session (`.docs/cluster.md`).
- Submit from the repo root on Helma:
  `sbatch --partition=h100 --job-name=<name> --export=ALL,CONFIG=configs/<x>.yaml lidar_bedlam/slurm/train.sbatch`.
- The notebook is generated code: edit `lidar_bedlam/scripts/build_debug_notebook.py`,
  rebuild, verify with `lidar_bedlam/scripts/verify_notebook.py`.
- The paper must stay anonymous: no rig names (FUSE-Bike, AVA), no group
  or first-person naming of own datasets.

---

## For the Next Handoff

1. Read this file top-to-bottom.
2. Read `.docs/progress.md` — timetable, log, and the open todos.
3. Read `instructions.md` for coding standards (enforced by ruff + mypy).
4. Run `uv sync` to reproduce the env.
5. Run `python3 lidar_bedlam/scripts/dm_link.py --check` to confirm this machine's
   `resources/` is wired up.
6. When you finish a session, append a dated entry under the right `# Log`
   section of `.docs/progress.md`, prune its `# Todos`, and update this
   file if you learned something future-you would need to pick up cold.
