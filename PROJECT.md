# lidar-bedlam — Project Document

Single source of truth for progress and todos. Log: newest first, one sentence
per day. Todos: grouped by contribution, in execution order; done items move to
the bottom. Hard deadline target: data built and trainings running within two
weeks of 2026-09-08; ablations after hand-in.

## Log

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

- [ ] Download the BEDLAM training labels with `bash scripts/fetch_bedlam_labels.sh` (needs the BEDLAM login; `bedlam-labels-smpl.zip` = SMPL, `all_npz_12_training.zip` = SMPL-X; per-image params already in the camera frame, so no body placement is needed).
- [ ] Write `data/bedlam.py` label attachment: join the npz records (`imgname`, `center`, `scale`, `pose_cam`, `shape`, `trans_cam`, `cam_int`, `gtkps`) to the extracted frames and person masks; verify by projecting SMPL joints onto the png.
- [ ] Run the official xxh128 validation on the NAS itself for `png`, `depth`, `masks`, `gt` (`scripts/validate_bedlam_on_nas.sh`).
- [ ] Extract the remaining groups at 6 fps with `scripts/extract_bedlam.py` (first group done: 100 sequences, 17,778 person-frames, 9.8 GB).
- [ ] Match label records to mask person ids (labels are per body, masks are per person index) via projected-joint-in-mask tests.
- [ ] Apply `SpeedSetting` + `apply_rolling_shutter` in the generation pipeline (per-sample speed, sweep over the camera window).
- [ ] Write the generated sample format (`.npz` per person per frame: image crop, K, points (N,4: xyz + beam id), SMPL params, box3d, beam count) and a `SampleSource` for it.
- [ ] Generate the v0 dataset (one group), inspect 20 samples visually, then generate the full subset.
- [ ] Dataset statistics for the paper (persons, frames, distance histogram, beams, occlusion levels).

### C2 — Selective-attention fusion model

- [ ] Add the SMPL mesh overlays to the BEDLAM cells of `debug/capabilities.ipynb` once the labels are attached.
- [ ] Training script `main.py` with `--wandb-project lidar-bedlam --wandb-name <run>` (entity `erik_hm`), config in `configs/`.
- [ ] Weighted multi-source sampler over `HumanPoseDataset` (synthetic vs real weights).
- [ ] Evaluation script (`lidar_bedlam/evaluation/`) that runs a model over a source and reports MPJPE / PA-MPJPE / PVE / translation error / box mAP.
- [ ] Smoke train on 1 k samples, then full run on the synthetic set.

### C3 — Experiments and ablations (plan: `docs/ablations.md`)

- [ ] Freeze the evaluation protocol (COCO-17 joints for cross-dataset comparison, distance bins, IoU thresholds) in `docs/data_pipeline.md`.
- [ ] Baseline runners for CameraHMR, TokenHMR, LiDAR-HMR, SAM 3D Body (MHR to joints), LIF-Net on the real validation sets.
- [ ] Main table: synthetic-only, real-only, synthetic + real (weighted), on SLOPER4D, LiDARHuman26M and Waymo (keypoints).
- [ ] Ablation 6.1/6.2 selective attention: learned gates, no gate, hard routing, image only, LiDAR only (4 extra runs, 1/3 schedule).
- [ ] Ablation 6.3 extrinsics: LiDAR fixed to the target rig vs ball r = 0.25 m vs r = 1.0 m, evaluated on Waymo and SLOPER4D (2 extra runs).
- [ ] Ablation 6.4 resolution: all 12 channel x step settings vs the target's setting only (1 extra run).
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
- [x] 2026-09-10 — `rigs/`: AVA and FUSE-Bike loaders via the nuScenes format; `export.py` + `scripts/export_rigs.py` (PNG/HTML/GLB into the vault); 2 tests.
- [x] 2026-09-10 — `rigs/`: Waymo, nuScenes, SLOPER4D calibration loaders, tree print, plotly rig figures, 5 tests; notebook section 9.
- [x] 2026-09-10 — Point and image augmentation configs (`PointAugmentConfig`, `ImageAugmentConfig`) wired into `HumanPoseDataset`; 6 tests; interactive notebook section 5b.
