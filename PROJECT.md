# lidar-bedlam — Project Document

Single source of truth for progress and todos. Log: newest first, one sentence
per day. Todos: grouped by contribution, in execution order; done items move to
the bottom. Hard deadline target: data built and trainings running within two
weeks of 2026-09-08; ablations after hand-in.

## Log

- 2026-09-08 — Audited BEDLAM v1 on the NAS (complete, 6.54 TB, depth = planar z-depth in cm), decided to skip BEDLAM 2.0, surveyed real datasets, scaffolded the uv repo with five baseline submodules and wrote the plan.

## Contributions

- **C1** — A synthetic LiDAR-camera SMPL pose-and-shape dataset generated from BEDLAM.
- **C2** — A selective-attention fusion model: image cues drive hands, ankles and head orientation; LiDAR cues drive body pose, shape and 3D placement.
- **C3** — Experiments and ablations showing synthetic data improves real-world SMPL estimation, with 3D placement as the headline metric.

## Todos

### C1 — Dataset from BEDLAM

- [ ] Download BEDLAM body data (SMPL-X animations + neutral ground truth) from bedlam.is.tue.mpg.de into `/mnt/md0/lidar-bedlam/bedlam_body` (user runs it; not on the NAS).
- [ ] Run the official xxh128 validation on the NAS itself for `png`, `depth`, `masks`, `gt` (`scripts/validate_bedlam_on_nas.sh`).
- [ ] Choose the sequence subset (start with `20221024_*` groups at 6 fps, then scale).
- [ ] Extract png + depth + masks + gt for the subset from the NAS tars to `/mnt/md0/lidar-bedlam/raw` (stream through `tar`, never copy whole tars).
- [ ] Implement `bedlam.camera`: parse camera CSV (Unreal cm, yaw/pitch/roll) into OpenCV K, R, t; unit test against the be_seq comments.
- [ ] Implement `bedlam.bodies`: map `be_seq.csv` body rows + start_frame to SMPL-X animation frames; convert SMPL-X to SMPL with `smplx2smpl.pkl`; produce per-person SMPL params in camera frame.
- [ ] Verify GT alignment: project SMPL joints with K/R/t onto the png and compare with body masks (IoU > 0.9 on a few frames).
- [ ] Implement `lidar.simulate`: depth (planar z, cm) to camera-frame point cloud; ray-cast a spinning LiDAR pattern (32/64/128/256 beams, configurable vertical FOV, azimuth resolution, range, noise, dropout) from a LiDAR origin at a configurable extrinsic offset to the camera.
- [ ] Implement occlusion augmentation for point clouds (random box/plane cut-outs, self-occlusion from a shifted origin, partial-body crops) and per-person cropping via body+clothing masks.
- [ ] Write the sample format (`.npz` per person per frame: image crop, K, points (N,4: xyz + beam id), SMPL params, cam translation, visibility flags, beam count) and the dataset index.
- [ ] Generate the v0 dataset (one group), inspect 20 samples visually, then generate the full subset.
- [ ] Dataset statistics for the paper (persons, frames, distance histogram, beams, occlusion levels).

### C2 — Selective-attention fusion model

- [ ] Port the ViT image encoder + TokenHMR/HMR2 head loading from `third_party/lif` into `src/lidar_bedlam/models` (typed, no global config object).
- [ ] Point encoder for sparse human LiDAR (PointNet++ from lif or a small point transformer).
- [ ] Selective cross-attention head: joint-group routing masks (hands/ankles/head from image tokens; torso/legs/global orient/betas/translation from LiDAR tokens) with a learnable gate; log gate values.
- [ ] Losses: SMPL params, 3D joints, 2D reprojection, and an explicit 3D translation loss in the camera frame.
- [ ] Training script `main.py` with `--wandb-project lidar-bedlam --wandb-name <run>` (entity `erik_hm`), config in `configs/`.
- [ ] Data loaders: synthetic (C1) + real (Waymo keypoints, SLOPER4D, LiDARHuman26M) with per-source sampling weights.
- [ ] Smoke train on 1 k samples, then full run on the synthetic set.

### C3 — Experiments and ablations

- [ ] Evaluation protocol: MPJPE, PA-MPJPE, PVE, and 3D placement error (root translation error, per-distance bins).
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
- [x] 2026-09-08 — Repo scaffold (uv, hatchling, ruff, mypy strict, vault) and five baseline submodules.
