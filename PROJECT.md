# lidar-bedlam — Project Document

Single source of truth for progress and todos. Log: newest first, one sentence
per day. Todos: grouped by contribution, in execution order; done items move to
the bottom. Hard deadline target: data built and trainings running within two
weeks of 2026-09-08; ablations after hand-in.

## Log

- 2026-09-09 — Located the missing SMPL/SMPL-X labels (BEDLAM project server only, not on Hugging Face), wrote the fetch script, and finished extracting the first BEDLAM group (17,778 person-frames at 6 fps).
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
- [ ] Implement `lidar.simulate`: depth (planar z, cm) to camera-frame point cloud; ray-cast a spinning LiDAR pattern (32/64/128/256 beams, configurable vertical FOV, azimuth resolution, range, noise, dropout) from a LiDAR origin at a configurable extrinsic offset to the camera.
- [ ] Implement occlusion augmentation for point clouds (random box/plane cut-outs, self-occlusion from a shifted origin, partial-body crops).
- [ ] Write the generated sample format (`.npz` per person per frame: image crop, K, points (N,4: xyz + beam id), SMPL params, box3d, beam count) and a `SampleSource` for it.
- [ ] Generate the v0 dataset (one group), inspect 20 samples visually, then generate the full subset.
- [ ] Dataset statistics for the paper (persons, frames, distance histogram, beams, occlusion levels).

### C2 — Selective-attention fusion model

- [ ] Port the ViT image encoder + TokenHMR/HMR2 head loading from `third_party/lif` into `src/lidar_bedlam/models` (typed, no global config object).
- [ ] Point encoder for sparse human LiDAR (PointNet++ from lif or a small point transformer).
- [ ] Selective cross-attention head: joint-group routing masks (hands/ankles/head from image tokens; torso/legs/global orient/betas/translation from LiDAR tokens) with a learnable gate; log gate values.
- [ ] Losses: SMPL params, 3D joints, 2D reprojection, and an explicit 3D translation loss in the camera frame.
- [ ] Training script `main.py` with `--wandb-project lidar-bedlam --wandb-name <run>` (entity `erik_hm`), config in `configs/`.
- [ ] Weighted multi-source sampler over `HumanPoseDataset` (synthetic vs real weights).
- [ ] Evaluation script (`lidar_bedlam/evaluation/`) that runs a model over a source and reports MPJPE / PA-MPJPE / PVE / translation error / box mAP.
- [ ] Smoke train on 1 k samples, then full run on the synthetic set.

### C3 — Experiments and ablations

- [ ] Freeze the evaluation protocol (COCO-17 joints for cross-dataset comparison, distance bins, IoU thresholds) in `docs/data_pipeline.md`.
- [ ] Baseline runners for CameraHMR, TokenHMR, LiDAR-HMR, SAM 3D Body (MHR to joints), LIF-Net on the real validation sets.
- [ ] Main table: synthetic-only, real-only, synthetic + real (weighted), on SLOPER4D and LiDARHuman26M; Waymo keypoint eval.
- [ ] Ablation after hand-in: beam count 32/64/128/256, occlusion augmentation on/off, LiDAR extrinsic shift augmentation, routing masks off, image-only vs LiDAR-only.
- [ ] Ablation after hand-in: clothing vs body-only points (BEDLAM masks make this possible).

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
- [x] 2026-09-08 — `scripts/extract_bedlam.py` streaming extraction; first group extracting at 6 fps.
