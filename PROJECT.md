# lidar-bedlam — Project Document

Single source of truth for progress and todos. Log: newest first, one sentence
per day. Todos: grouped by contribution, in execution order; done items move to
the bottom. Hard deadline target: data built and trainings running within two
weeks of 2026-09-08; ablations after hand-in.

## Log

- 2026-09-10 (training) — WP2 training stack: `main.py` (torchrun, mixture sampler, AMP, checkpoint/resume, offline wandb, gate modes), `evaluate.py`, 16 configs, self-resubmitting Helma job script; smoke run on the 4090 passed; token precompute and the full synthetic generation running; repo and data syncing to the Helma workspace.
- 2026-09-10 (plan) — Grilled the full plan: Waymo is the headline (5,559 usable samples), SLOPER4D secondary, LiDARHuman26M and PedX dropped, SAM 3 masks for Waymo points, precomputed ViT-H tokens, Slurm/H100 training with auto-resume; written to `docs/plan.md`.
- 2026-09-10 (late) — Decided the ablation plan (`docs/ablations.md`): merged the extrinsics studies into a ball-radius sweep, added the resolution ablation, dropped rolling shutter as an ablation; routed legs and arms to the camera prior; EG 2027 LaTeX skeleton started.
- 2026-09-10 (night) — Ego speed settings (10 km/h steps, Waymo-measured preset 20 ± 20 km/h) and LiDAR rolling-shutter distortion, exposed in the notebook.
- 2026-09-10 (night) — Added the own AVA car and FUSE-Bike rigs (nuScenes exports) and exported all five rigs to the Obsidian vault as PNG, interactive HTML, GLB and tree text with a `rigs.md` note.
- 2026-09-10 (later) — Sensor-rig capability: calibration trees for Waymo, nuScenes and SLOPER4D with per-sensor axes and camera frustums, plus a checkbox-table notebook section; other NAS datasets surveyed for calibration.
- 2026-09-10 — Added sensor-level and image augmentations (cover, channel dropout, jitter, outliers, miscalibration, erasing, colour, blur, JPEG, bbox jitter), wired them into the dataset, and an interactive notebook section with the 4x3 resolution grid.
- 2026-09-09 (evening) — LiDAR presets renamed to Ouster families (OS0/OS1/OS2, 32-256 channels, 512/1024/2048 steps per revolution) with the camera HFOV mapped to a column window; notebook updated.
- 2026-09-09 (later) — Implemented and tested the selective-attention fusion model (ViT-H with TokenHMR weights, point tokenizer, gated decoder, SMPL heads) and the executed `debug/capabilities.ipynb` visualising BEDLAM, simulated LiDAR at 4 resolutions and 2 viewpoints, real datasets and the model.
- 2026-09-09 — Located the missing SMPL/SMPL-X labels (project server only), wrote the fetch script, finished extracting the first BEDLAM group (17,778 person-frames), and built the tested LiDAR simulator (32/64/128/256 beams, sensor offsets) plus occlusion augmentation.
- 2026-09-08 — Audited BEDLAM v1 (complete, planar z-depth in cm), skipped BEDLAM 2.0, scaffolded the repo with five baseline submodules, pushed to GitHub, and built the verified data pipeline (SLOPER4D, LiDARHuman26M, Waymo, BEDLAM loaders, torch dataset, losses, metrics, 21 tests).

## Contributions

- **C1** — A synthetic LiDAR-camera SMPL pose-and-shape dataset generated from BEDLAM.
- **C2** — A selective-attention fusion model: image cues drive hands, ankles and head orientation; LiDAR cues drive body pose, shape and 3D placement.
- **C3** — Experiments and ablations showing synthetic data improves real-world SMPL estimation, with 3D placement as the headline metric.

## Todos

### C1 — Dataset from BEDLAM

- [ ] Run the official xxh128 validation on the NAS itself for `png`, `depth`, `masks`, `gt` (`scripts/validate_bedlam_on_nas.sh`).
- [ ] Extraction of the 11 remaining paper groups running in the background (`scripts/extract_bedlam_groups.sh`, logs in `logs/`).
- [ ] Apply `SpeedSetting` + `apply_rolling_shutter` in the generation pipeline (per-sample speed, sweep over the camera window); default speed 0 for v1.
- [ ] Generate v1: groups 02-11 running in the background (117,657 frames); group 12 after its extraction finishes; then precompute tokens for the new shards and rsync them to Helma.
- [ ] Dataset statistics for the paper (persons, frames, distance histogram, beams, occlusion levels).

### C2 — Selective-attention fusion model

- [ ] Launch the main run on Helma once the data sync is complete (`sbatch --export=ALL,CONFIG=configs/main_mixed.yaml scripts/slurm/train.sbatch`), verify the resume chain on the first wall-time hit, sync wandb from the login node.
- [ ] Waymo LiDAR-only samples (5,006) as `has_image=False` records (after the main run).
- [ ] Add the SMPL mesh overlays to the BEDLAM cells of `debug/capabilities.ipynb` once the labels are attached.

### C3 — Experiments and ablations (plan: `docs/ablations.md`)

- [ ] Baseline runners for CameraHMR, TokenHMR, LiDAR-HMR, SAM 3D Body (MHR to joints) on the real validation sets (LiDAR-HMR checkpoint arrives on the NAS).
- [ ] Main table: `configs/synth_only.yaml`, `real_only.yaml`, `main_mixed.yaml` on Waymo val and SLOPER4D test.
- [ ] Ablation 6.1/6.2 selective attention: `configs/ablation_gate_{none,hard}.yaml`, `ablation_{image,lidar}_only.yaml` vs `ablation_mixed_short.yaml` (1/3 schedule).
- [ ] Ablation 6.3 extrinsics: `configs/ablation_rig_waymo.yaml`, `ablation_ball025.yaml` vs the 1 m ball; 6.6 data scaling: `ablation_scale_{2,4,8,16,32}x.yaml`.
- [ ] Ablation 6.4 resolution: `configs/ablation_target_waymo.yaml` (target setting only) vs the random resolutions of the main run.
- [ ] After hand-in, ablation 6.5 realism: no augmentation, no occlusion (2 extra runs).
- [ ] Paper: fill `paper/main.tex` (Eurographics 2027 skeleton) with the tables above.

### C1b — Data for specific sensor setups

- [ ] Decide which additional rigs to add (see the heads-up list in the chat / `docs/datasets.md`).
- [ ] Generate synthetic samples for a chosen rig: place the BEDLAM camera at a rig camera, simulate the rig's LiDAR(s) from their calibrated offsets.

### Housekeeping

- [ ] Rotate the BEDLAM password and remove it from the shared NAS `be_download.sh`.
- [ ] Complete the Human-M3 download (final zip part missing) and the remaining 9 SLOPER4D sequences.
- [ ] Fix dangling symlinks in `publicdatasets/LiDARHuman26M`.
- [ ] Consider deleting the 1.4 TB of redundant Waymo tarballs on the NAS.

## Done

- [x] 2026-09-10 — WP2: `train/` (config, mixture sampler, trainer with DDP/AMP/resume/SIGUSR1/offline wandb), `metrics/protocol.py` (Waymo-15 vs COCO-17 mapping, AP/mAP summary), gate modes in the decoder, IoU-confidence head, `main.py`, `evaluate.py`, 16 configs, `scripts/slurm/train.sbatch` + `wandb_sync.sh`; 5 tests; 40-step smoke run on the 4090.
- [x] 2026-09-10 — Helma probed: `$WORK`/`/anvme` invisible on compute nodes, `/hnvme/workspace` and `/tmp` visible, wandb offline only (`docs/cluster.md`).
- [x] 2026-09-10 — BEDLAM SMPL labels fetched and attached; label-to-mask matching 98 %.
- [x] 2026-09-08 — BEDLAM v1 archive audit (names, sizes, EOF blocks, gz integrity).
- [x] 2026-09-08 — BEDLAM 2.0 size research and decision to skip.
- [x] 2026-09-08 — Real dataset survey (`docs/datasets.md`).
- [x] 2026-09-08 — Depth semantics verified: planar z-depth, cm, sky 1e8 (`docs/bedlam_audit.md`).
- [x] 2026-09-08 — Repo scaffold (uv, hatchling, ruff, mypy strict, vault) and five baseline submodules; pushed to github.com/max-a-ai/lidar-bedlam.
- [x] 2026-09-08 — `data/` symlinks (`scripts/link_data.sh`), chumpy-free SMPL conversion, SMPL frame-change verified.
- [x] 2026-09-08 — Loaders for SLOPER4D, LiDARHuman26M, Waymo (pose_complete_4) and BEDLAM raw frames, sharing one schema; reprojection checks pass.
- [x] 2026-09-08 — `HumanPoseDataset` (torch), `FusionLoss`, pose metrics (MPJPE, PA-MPJPE, PVE) and 3D box AP/mAP + translation error; 21 unit tests.
- [x] 2026-09-08 — `scripts/extract_bedlam.py` streaming extraction; first group extracted at 6 fps (17,778 person-frames).
- [x] 2026-09-09 — `lidar/simulate.py` ray-marching LiDAR simulator with presets os32/os64/os128/os256/waymo64, sensor offsets, noise, dropout; `lidar/augment.py` occlusions; 6 tests.
- [x] 2026-09-09 — `scripts/fetch_bedlam_labels.sh` and documentation of where the SMPL/SMPL-X labels live.
- [x] 2026-09-09 — `models/`: ViT (loads TokenHMR ViT-H), point tokenizer, selective gated decoder, SMPL heads with differentiable SMPL, 10 tests; losses already covered.
- [x] 2026-09-09 — `debug/capabilities.ipynb` generated and executed without errors (22 cells).
- [x] 2026-09-10 — `lidar/motion.py`: `SpeedSetting`, `EgoMotion`, `apply_rolling_shutter`; Waymo ego-speed statistics measured; 4 tests.
- [x] 2026-09-10 — Generator: shard records, sensor placement, virtual distance, `scripts/generate_synthetic.py`; trial verified visually.
- [x] 2026-09-10 — BEDLAM SMPL labels attached and verified (`data/bedlam_labels.py`).
- [x] 2026-09-10 — `rigs/`: AVA and FUSE-Bike loaders via the nuScenes format; `export.py` + `scripts/export_rigs.py` (PNG/HTML/GLB into the vault); 2 tests.
- [x] 2026-09-10 — `rigs/`: Waymo, nuScenes, SLOPER4D calibration loaders, tree print, plotly rig figures, 5 tests; notebook section 9.
- [x] 2026-09-10 — Point and image augmentation configs (`PointAugmentConfig`, `ImageAugmentConfig`) wired into `HumanPoseDataset`; 6 tests; interactive notebook section 5b.
