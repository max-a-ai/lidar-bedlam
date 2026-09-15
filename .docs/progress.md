# lidar-bedlam — progress

Single source of truth: timetable at the top, log in the middle, todos at
the bottom. All three use the same sections: **Data**, **Model**,
**Experiments**, **Paper**, **Housekeeping**. Log entries are newest first
within a section and carry the commit that made them.

Target: Eurographics 2027 full paper (abstract 25 Sep 2026, paper 1 Oct
2026, double-blind, 10 pages). Claim and run ladder: `story.md`.

# Timetable

```mermaid
gantt
    title lidar-bedlam to the EG 2027 deadline
    dateFormat  YYYY-MM-DD
    axisFormat  %d.%m.
    section Data
    BEDLAM audit, loaders, simulator      :done,    d1, 2026-09-08, 2026-09-10
    Labels, generator, real shards, tokens:done,    d2, 2026-09-10, 2026-09-12
    Pool on Helma, pseudo-GT, 3DPW shards :done,    d3, 2026-09-12, 2026-09-15
    Simulator fix, synth v2 regeneration  :         d4, 2026-09-16, 2026-09-19
    Backbone tokens S/B/L                 :         d5, 2026-09-18, 2026-09-21
    section Model
    Fusion model, trainer, Slurm chain    :done,    m1, 2026-09-09, 2026-09-11
    Surface and mesh losses, early stop   :done,    m2, 2026-09-14, 2026-09-15
    Localisation, point anchor, fusion    :active,  m3, 2026-09-15, 2026-09-19
    Backbone ladder, inference benchmark  :         m4, 2026-09-19, 2026-09-24
    section Experiments
    Finals, fusion, synthesis, scaling    :done,    e1, 2026-09-13, 2026-09-15
    Baselines scored with our protocol    :done,    e2, 2026-09-14, 2026-09-15
    Loss and anchor ablations, 3DPW eval  :active,  e3, 2026-09-15, 2026-09-18
    SAM 3D Body, 3DPW in the mixture      :         e4, 2026-09-18, 2026-09-24
    section Paper
    Skeleton, story, related work         :done,    p1, 2026-09-10, 2026-09-11
    Metrics paragraph, architecture figure:done,    p2, 2026-09-14, 2026-09-15
    Data + method sections                :         p3, 2026-09-16, 2026-09-21
    Abstract deadline                     :milestone, 2026-09-25, 0d
    Results, ablations, final             :         p4, 2026-09-21, 2026-10-01
    Paper deadline                        :milestone, 2026-10-01, 0d
```

# Log

## Data

### 2026-09-15 — 3DPW test split, tokens, LiDAR-HMR pseudo-GT
`threedpw_test` shards (6,617 records, 16 shards) with ViT-H tokens, on
Helma, listed as a validation source in all 50 configs. LiDAR-HMR's Waymo
pseudo-GT downloaded (`resources/data/external/lidar_hmr_waymov2`, vehicle
frame); ours fits tighter (22 vs 27 mm keypoints, 2.7 vs 3.1 cm points).
Short-run shard set (~100 GB) copied to the NAS for cluster2.

### 2026-09-14 — simulator fix
Offset sensors were clipped by the frustum window and there was no facing
test (`lidar/simulate.py`); shards in use predate the fix, `synth/v2` not
regenerated yet. Details under Experiments, 2026-09-14.

### 2026-09-13 — 3DPW shards final, both new ablations queued
3,444 frames -> 4,468 records in 12 shards (`meshlidar/v1`), tokens
computed, copied to Helma; `ablation-3dpw` (job 847706) and
`ablation-pseudo-waymo` (847660) queued. Notebook executed with 34 cells
and 0 errors; sections 10 and 11 render (figures live inside the
re-roll Output widgets). Detached runs must use `.venv/bin/python`:
`uv run` exits 120 without a terminal.

### 2026-09-13 — 3DPW images are numbered by index, not by video id
The first 3DPW pass paired labels with `image_<img_frame_ids[i]>.jpg`;
those ids are 60 Hz video ids (2i), so half the frames were missing and
the other half paired with the wrong image. Images are numbered by the
sequence index (as in the SPIN loader); shards regenerated.

### 2026-09-13 — pseudo-GT SMPL for Waymo and mesh-LiDAR samples from 3DPW
Two new data sources for later ablations. (1) `generate/pseudo_smpl.py` +
`scripts/pseudo_smpl_waymo.py`: SMPL fitted per Waymo training record,
initialised by `main-mixed-000`, optimised on the 13 shared keypoints,
the 2D keypoints and the LiDAR returns (pulled to 3 cm outside the mesh),
with a pose prior on the init; accepted when the keypoint error is below
8 cm; written as `real/v1_pseudo` (keypoints stay the joint labels,
`has_smpl=True` on accepted fits). First shards: 100 % accepted, keypoint
error 105 -> 21 mm. (2) `data/threedpw.py` (world-to-camera verified
against `poses2d`, gendered SMPL), `lidar/mesh_depth.py` (z-buffer
rasteriser, vertices pushed 1-4 cm along their normals = clothing),
`generate/mesh_synth.py` (same scan plan and record builder as BEDLAM,
`dataset="3dpw"`), `scripts/generate_3dpw.py`; returns sit 3.0 cm from
the skin mesh (median). Configs `ablation_pseudo_waymo.yaml`,
`ablation_3dpw.yaml`. Notebook sections 10 and 11 (seeded re-roll
buttons). Trainer now writes an evaluation figure per source and step
(`outputs/<run>/vis/`), mirrored to wandb as images. 3 tests.

### 2026-09-12 — person masks now include the hair
BEDLAM ships a `hair` mask part in the `*handhair*` groups; the loader
unioned only `body` and `clothing`, so long-haired heads were cut and
their LiDAR returns dropped (found on
`20221024…handhair…/seq_000099/0075/00`). `PERSON_MASK_PARTS` gained
`hair` (1 test); notebook cell 1b shows old vs new mask and the returns on
the hair for that sample (+395 mask pixels, +5 returns at OS1-128).
Group 01 (the only hair group, 10,101 records) regenerated as `synth/v1`
(old copy `synth/v1_nohair`), tokens recomputed, to be copied to Helma.

### 2026-09-12 — repo mapped onto the fixed project structure
`.docs/` (progress, figures, latex-draft, runs), `lidar_bedlam/slurm/`, `outputs/` for
runs, `config-global.json` + `lidar_bedlam/scripts/dm_link.py` replacing
`link_data.sh`; `PROJECT.md` and `CHANGELOG.md` merged into this file.

### 2026-09-12 — synthetic pool complete
Groups 02-11: 304,930 records in 601 shards (tokens done, syncing to
Helma); group 12: 75,342 records in 156 shards (tokens computing). With
group 01 (10,101) the pool is 390,373 records, 85x the Waymo train count.
Body models, group 01 and all real shards (85 GB) already on Helma.

### 2026-09-10 10:46 — BEDLAM frames without depth are skipped (dc8a628)
`BedlamFramesSource` indexes only frames whose png and depth exist
(`incomplete` counter); the notebook failed on the group whose extraction
was still running. 1 test.

### 2026-09-10 09:16 — fixes found at scale (1429370)
Per-frame mask lookup ignored the group; SLOPER4D seq009 stores 4 values
per 2D joint; bash `GROUPS` is reserved. Waymo shards: 4,591 train / 894
val records; SLOPER4D train 21,062 records.

### 2026-09-10 03:28 — real shards, shard dataset, SAM 3 masks, tokens (b3bf5eb)
`lidar_bedlam/scripts/sam3_waymo_masks.py` (5,560 Waymo crops, verified visually),
`generate/real.py`, `data/shards.py` (`ShardDataset`: scan-variant
selection, precomputed tokens, point augmentation),
`lidar_bedlam/scripts/precompute_tokens.py` (frozen ViT-H tokens, fp16).

### 2026-09-10 03:19 — synthetic shard generator (926e3a5)
`generate/records.py` (fixed-shape records, npz shards with named scan
variants), `lidar/placement.py` (ball placement, rig poses),
`lidar/distance.py` (virtual distance by moving the camera back with
z-buffer re-rendering; the first attempt scaled the scene, which keeps
angles and was wrong), `generate/synth.py` (8 scan variants per frame,
visibility rule 90x35 px + 30 returns, augmented copy, distance
augmentation on 30 % of frames), multiprocess CLI. 5 tests.

### 2026-09-10 03:10 — BEDLAM SMPL labels attached (d54a96a)
`data/bedlam_labels.py`: translation = `trans_cam + cam_ext[:3,3]`
(verified: 85 % of vertices inside the masks); silhouette-overlap matching
(98 % of labels matched). Extraction queue for the 11 remaining groups.

### 2026-09-10 00:55 — sensor rigs and rolling shutter (d46f545)
`rigs/`: Waymo, nuScenes, SLOPER4D, AVA, FUSE-Bike loaders, plotly
figures, exports (PNG, HTML, GLB, tree) into `figures/rigs/`;
`lidar/motion.py`: `SpeedSetting` (10 km/h steps, Waymo preset 20 +- 20
km/h) and `apply_rolling_shutter`.

### 2026-09-09 22:35 — augmentations (e42366f)
`lidar/augment.py` (cover, channel dropout, jitter, outliers,
miscalibration), `data/image_augment.py` (erasing, cover, colour, blur,
JPEG, bbox jitter); notebook section with the 4x3 resolution grid.

### 2026-09-09 17:42 — Ouster presets (70fdc89)
OS0/OS1/OS2 families by channels x steps per revolution; `azimuth_window`
maps the camera HFOV to scan columns.

### 2026-09-09 10:32 — LiDAR simulator (67b3a2f)
Ray marching against the BEDLAM depth, sensor offsets, noise, dropout,
occlusion augmentation. SMPL labels located on the project server
(42a917c).

### 2026-09-08 18:26 — data pipeline (2c8706c)
Loaders for SLOPER4D, LiDARHuman26M, Waymo (pose_complete_4) and BEDLAM
raw frames on one camera-frame schema; torch dataset; losses; metrics; 21
tests. BEDLAM v1 verified complete (390 archives, 6.54 TB, planar z-depth
in cm); BEDLAM 2.0 skipped.

## Model

### 2026-09-15 — point-anchored translation, surface and mesh losses
`model.transl_anchor: points`: pelvis = centroid of the valid input points
+ learned offset (LiDAR-HMR places this way; the scan alone gives 0.11 m
on Waymo). Chamfer (learned clothing offset), ICP (stop-gradient target
after a divergence) and Pose2Mesh vertex/normal/edge terms in `FusionLoss`.
Trainer: early stop (80k min, patience 5 evals), non-finite guard, eval
every 1k steps. Helma jobs: 128 CPUs, 32 workers (0.22 vs 0.42 s/step).

### 2026-09-14 — protocol
Placement NaN for unlabelled Waymo hips, absolute MPJPE, PVE, mesh-box
scale for Waymo; mirror test for LiDAR-HMR. Details under Experiments.

### 2026-09-12 — Waymo keypoints were supervising the wrong joints
`mix80-000` reached 57 mm MPJPE on SLOPER4D but 370 mm on Waymo while
`synth-only-000`, which never saw Waymo, got 103 mm there: the joint and
2D-keypoint losses compared Waymo's 15 keypoints index-by-index with the
24 SMPL joints. Fix: the batch carries `joint_convention_id`, the model
regresses COCO-17 joints from the mesh (`joints_coco`, `kp2d_coco` via
`J_regressor_coco.npy`), and the losses compare waymo15 rows to the 13
shared COCO joints (`WAYMO15_TO_COCO17` now in `data/schema.py`), as the
evaluation protocol already did. 1 test; 30-step smoke on the Waymo
mixture. `mix80` must be rerun; `synth-only-000` is unaffected.

### 2026-09-12 — epochs and live wandb mirror
`optim.max_epochs` (epoch = one pass over all training records, ~416k;
main runs 50, ablations 17; `max_steps` derived), `train/epoch` logged and
used as the x axis. Compute nodes have no proxy, and `wandb sync` cannot
read a live offline run, so the trainer now writes `metrics.jsonl` +
`config.json` per run and `lidar_bedlam/scripts/wandb_mirror.py` (login node, every 2
min) pushes new rows to wandb under the run's name; verified end to end
with `smoke-mirror-000`. The two runs started before this change
(`mix80-000`, `synth-only-000`, 12k steps = 59 epochs) sync at their end.

### 2026-09-12 — batch-size tests done, full runs at batch 2048
Round 2 (memmap loader, node-local staging, 4x H100, 600 steps each):

| batch | samples/s | peak GiB/GPU |
|---|---|---|
| 256 | 3,200 | 2.3 |
| 512 | 4,500 | 4.0 |
| 1024 | 5,200 | 7.2 |
| 2048 | 5,600 | 13.7 |

Throughput is flat above 1024 (loader-bound with 64 workers), memory far
from the 80 GiB. Chosen: batch 2048, lr 3e-4, 12k steps (24.6 M samples,
about 1.5 h per run) for `mix80` (80/10/10) and `synth_only` (100 %
BEDLAM); eval every 500 steps. Runs of the tests are on wandb
(`batchtest2-b<N>`).

### 2026-09-12 — loader was the bottleneck: memory-mapped shards (f83d95d)
First batch tests on Helma: 360-390 samples/s at batch 256 and 512 with
only 2.3 / 4.0 GiB peak per GPU, i.e. the GPUs waited for data. Cause:
`np.load` on an npz re-reads a whole member (96 MB image stack) on every
row access. `Shard` now maps the uncompressed members in place (zip local
headers parsed, `np.ndarray` views on one memmap), rows are copied only
where they become tensors; 28 ms/sample warm, 1 test. Jobs now use 64
CPUs, 16 workers per rank and stage only the needed shard directories to
node-local `/tmp` (`STAGE_DIRS`). Round 2 of the tests: `batchtest2-b<N>`.
First test also validated the whole chain: eval, `best.pt`, `DONE`, wandb
sync from the login node (run visible in wandb).

### 2026-09-12 — batch-size scaling tests submitted on Helma (1bba8d2, 7fdeb54)
`configs/batchtest.yaml` (80/10/10 mixture on group 01 + real shards,
600 steps) at batch 256 / 512 / 1024 / 2048, 1 h each, h100 partition,
run names `batchtest-b<N>`; trainer logs samples/s and peak GPU memory;
wandb login on the Helma login node with `lidar_bedlam/slurm/wandb_sync_loop.sh`
detached there. Requested runs afterwards: `configs/mix80.yaml` (80 %
BEDLAM, 10 % SLOPER4D, 10 % Waymo) and `configs/synth_only.yaml` (100 %
BEDLAM) at the largest batch that fits.

### 2026-09-10 09:41 — training stack (5e51d6f)
`train/config.py` (YAML + overrides, `${DATA_ROOT}`), `train/sampler.py`
(fixed per-batch mixture, DDP-sharded), `train/loop.py` (torchrun DDP,
AdamW + warm-up/cosine, bf16, eval every N steps, `last.pt`/`best.pt`,
auto-resume, SIGUSR1 checkpoint-and-exit, `DONE`, offline wandb),
`metrics/protocol.py`, gate modes `learned|none|hard|image_only|lidar_only`,
IoU-confidence head, `main.py`, `evaluate.py` (now `lidar_bedlam/scripts/lidar-bedlam-{main,eval}.py`), 16 configs,
`lidar_bedlam/slurm/train.sbatch` (self-resubmitting), `lidar_bedlam/slurm/wandb_sync.sh`. SMPL heads
forced to float32 under autocast. Smoke run: 40 steps on the 4090. 5 tests
(68 total). Helma: `/anvme` and `$WORK` not on compute nodes,
`/hnvme/workspace` is (`cluster.md`).

### 2026-09-09 11:13 — selective-attention fusion model (8846d44)
ViT with TokenHMR ViT-H weights (0 missing keys), point tokenizer, gated
dual cross-attention decoder with routing priors, SMPL heads (6D
rotations, translation via crop intrinsics), differentiable SMPL, 3D box;
`notebooks/capabilities.ipynb`.

## Experiments

### 2026-09-15 23:00 — main v2 with the pose prior submitted
`configs/a_full_main_v2_prior.yaml` (a-full-main-v2-prior, job 860568):
the v2 recipe (point anchor, 75/10/10/5) plus the BEDLAM pose prior on
the unobserved joints of Waymo rows. The running v2 without the prior
stays as the control; the short prior runs (2 seeds) decide within the
hour whether the prior costs anything on the metrics, the renders decide
whether it fixes the hands and feet. Headline row "main v2 + pose prior".

### 2026-09-15 22:40 — pose prior on the unobserved joints (rule 2) and frozen joints in the fit (rule 1)
Neither had been applied since the proposal. Now: `scripts/pose_prior_stats.py`
computes per-joint chordal mean rotations and the geodesic spread of the
BEDLAM poses (10,732 poses of synth/v1; hands have zero spread in BEDLAM,
floored at 3 deg); `losses/pose_prior.py` turns them into a per-joint
Gaussian energy on the geodesic angle for ankles, feet, head, wrists and
hands (joints 7, 8, 10, 11, 15, 20-23: no Waymo keypoint on the child
segment); `FusionLoss` applies it (`loss.pose_prior`, weight 0.05) to rows
without a mesh label; `pseudo_smpl.py` keeps those joints at their
initialisation during the fit (`FROZEN_JOINTS`). Test
`test_pose_prior_is_zero_at_the_mean_and_masked`. Runs:
a-abl-mixed-short-prior (2 seeds) and a-full-mix80-prior (860461-860463),
rows in the fusion axis and the headline table.

Open: scoring the local predictions of a-full-mix80's best.pt through
`run_model` + `score_baselines` gives Waymo 78.0 / 66.5 mm and mAP 0.44
while the trainer's own validation says 70.7 / 59.9 and 0.67; placement
matches (0.074 m). The mAP part is expected (the scorer builds boxes from
the mesh extent, the trainer scores the predicted box); the pose gap is
not understood yet. A cluster-side protocol evaluation of the same
checkpoint is queued to see which path is off. Figures for the run in
`outputs/figures/a-full-mix80/` (curves, crossing frame, placement by
distance).

### 2026-09-15 21:30 — anchored gate axis at the short schedule; first anchor run stopped at 80k
Fusion axis with the point anchor (2 seeds, 3,468 steps, training-time
validation; protocol evaluation queued): learned gates 76.6 / 64.9 mm,
0.075 m, mAP 0.66 on Waymo; gate none (plain sum) 71.8 / 59.5, 0.075 m,
0.67; gate hard (fixed priors) 73.6 / 61.3, 0.074 m, 0.67; LiDAR only
97.6 / 78.0, 0.079 m, 0.68. SLOPER4D: 45.7 / 43.6 / 42.1 / 54.5 mm at
0.027-0.030 m; 3DPW 59.5 / 58.5 / 59.7 / 95.9 mm. Placement is now
identical across the gate modes (the anchor sets it), and the learned
gate is the worst of the three fused variants on pose, plain sum the
best: the query-only gate adds nothing, as the gate-value analysis
predicted. full-anchor-points-000 (camera-anchor era config with the
anchor, 50/40/10) stopped at the 80k gate: 77.6 mm, 0.087 m, mAP 0.64
training-time; evaluation queued (859939) together with
a-abl-lidar-only-s1.

### 2026-09-15 19:40 — SAM 3D Body scored on the three splits
`results_sam3d.json` (stage 1 on the 4090, stage 2 on the CPU, ~2 h):
Waymo val 76.6 / 59.5, abs 633.4, 0.626 m, mAP 0.10; SLOPER4D 54.5 / 45.0 /
PVE 66.1, abs 123.2, 0.112 m, mAP 0.60; 3DPW 51.7 / 40.1 / PVE 75.2,
0.197 m, mAP 0.42. The strongest image-only row: on pose it edges CameraHMR
on Waymo (76.6 vs 76.9, PA 59.5 vs 60.0), on SLOPER4D (54.5 vs 51.6 MPJPE
but PVE 66.1 vs 64.1) they trade, on 3DPW it leads every image method
(51.7 / 40.1 vs CameraHMR 58.0 / 46.3). Its metric depth is the best of
the image rows on Waymo (0.63 m vs CameraHMR 0.73) and SLOPER4D (0.11 m vs
0.20), still 8x our anchored placement on Waymo. All seven compared
pipelines now have rows on all three splits.

### 2026-09-15 18:20 — what the gates do in the anchored model; anchored gate ablation queued
Gate values (weight of the image stream) of abl-anchor-points-000 over 512
records per validation set, last decoder layer: the semantic groups
(head, arms, hands, legs, feet) sit at 0.94-0.96 against a prior of 0.88,
i.e. nearly image-only; the 3D groups (root, torso, shape) sit at
0.31-0.41 against a prior of 0.12, i.e. the LiDAR stream carries ~65 %
of them, less than the prior asked for. Spread across records is small
(std 0.01 semantic, 0.02-0.08 3D groups), so the routing is a per-group
constant with little dependence on the input, as expected from a gate
computed from the query alone. Both "anchor · learned gates (reference)"
and "translation anchored to the point centroid" are that recipe
(learned gates, anchor, 50/40/10, short schedule); its full-schedule
version is a-full-main-mixed. Anchored gate ablation submitted:
a-abl-gate-none, a-abl-gate-hard, a-abl-lidar-only, 2 seeds each (jobs
858959-858965); rows in the fusion axis.

### 2026-09-15 17:40 — SAM 3D Body over the three splits (running)
`scripts/baselines/run_sam3d_body.py` (env `sam_3d_body`, 3.1 crops/s on the
4090: whole crop as the box, true crop intrinsics, MHR vertices + camera
translation per split) and `scripts/baselines/convert_sam3d_to_smpl.py`
(env `mhr_lod`, CPU: the MHR conversion tool in centimetres with the
camera translation added, 3.6 crops/s at batch 64) write the common npz.
64-crop smoke on Waymo val: 78.4 / 61.0 mm, placement 0.724 m, mAP 0.20,
CameraHMR territory. Chain `sam3d_full.sh` runs stage 1 over the three
splits with stage 2 pipelined behind it (~2 h), then scores into
`results_sam3d.json`; static row "SAM 3D Body (DINOv3-H+)" with a
footnote on the conversion.

### 2026-09-15 17:10 — main pipeline v2 = anchor on mix80 plus 3DPW
The anchored runs at 8k steps put mix80 (80/10/10) marginally ahead of the
50/40/10 mixture on every placement column (Waymo abs 98.0 vs 100.2 mm,
mAP 0.67 vs 0.66; SLOPER4D mAP 0.97 vs 0.95; 3DPW 56.9 vs 58.7 mm), as it
was at 150k with the camera anchor. User's call: mix80 is the base.
`configs/a_full_main_v2.yaml` (a-full-main-v2): point-anchored
translation, every batch 75 % BEDLAM (full pool), 10 % Waymo (4,591
records), 10 % SLOPER4D (21,062), 5 % 3DPW mesh-LiDAR (4,468, from the
synthetic share); full schedule with early stop, validated on the three
splits. A first v2 job with 50/25/15/10 was cancelled after 15 min and
its folder removed. Scaling the synthetic pool (data axis, 2x-32x at
fixed steps) stayed flat within noise, so v2 keeps the full pool.

### 2026-09-15 16:20 — LiDAR-HMR on 3DPW; anchored short reference; BEDLAM 2 images down
LiDAR-HMR mirrored on the 3DPW test crops (simulated main_0 scan):
117.6 / 80.7 / PVE 156.6, 0.131 m, mAP 0.47; pelvis on the centroid: 0.196 m,
0.36 (`results_threedpw_lidarhmr*.json`). Its pose is far behind the
image methods on this close-range split. a-abl-mixed-short (anchored
reference, seed 0, training-time val): Waymo 74.6 mm, 0.075 m, mAP 0.67;
SLOPER4D 43.7 mm, 0.027 m; 3DPW 59.8 mm, 0.066 m, as the earlier anchor
runs. BEDLAM 2: the eight street image folders are on the NAS (0.92 TB);
the depth repository answered 403 on the `.xxh128` checksum sidecars, so
the depth pass was restarted skipping them, with retries.

### 2026-09-15 15:25 — compared pipelines on 3DPW test
`results_threedpw.json` (4090 chain): HMR2 77.9 / 58.6 / PVE 95.5, 0.175 m,
mAP 0.39; TokenHMR 65.8 / 50.2 / 89.9, 0.185 m, 0.39; CameraHMR 58.0 /
46.3 / 82.2, 0.154 m, 0.48; Human3R 86.0 / 58.7 / 109.9 (placement
blanked, 225 crops without detection). PromptHMR from before: 61.7 / 47.8
/ 87.1, 0.348 m. Our short anchor runs: 60.2 / 50.4 / 90.8, 0.068 m,
0.79. On 3DPW the image methods place at 15-19 cm because the crops are
close (pedestrian at a few metres, true intrinsics); CameraHMR wins pose.
LiDAR-HMR failed: the 3DPW shards carry only simulated scan variants
(main_0, main_1, rigs); `baselines/common.load_shard` now falls back to
`main_0` when there is no `real` scan (our evaluation draws main_0 or
main_1 per shard). LiDAR-HMR rerun queued with the centroid variant.

### 2026-09-15 14:30 — point anchor becomes the model; 3DPW baselines; BEDLAM 2 street subset
Decision (user): the point-anchored translation is mandatory. Existing
configs stay untouched so finished runs re-evaluate with the camera
anchor they were trained with; every comparison config got a copy with
`transl_anchor: points` and the experiment prefixed `a-`
(`configs/a_full_*.yaml`, `configs/a_ablation_*.yaml`, 41 files; image-only
ones excluded because the anchor would leak the scan). `eval_runs.sbatch`
maps `a-abl-*` names. Core set submitted on Helma: a-full-{main-mixed,
mix80, real-only, synth-only, gate-none, lidar-only} and a-abl-mixed-short
(2 seeds); the rest of the axes wait for the go.

The five compared pipelines never ran on the 3DPW split (generated this
morning): chain `baselines_3dpw.sh` on the 4090 (HMR2 full, TokenHMR
tight, CameraHMR full, LiDAR-HMR mirrored, Human3R), scored into
`results_threedpw.json`; `build_scoreboard.static_rows` now merges per
split across the `results_*.json` files.

BEDLAM 2 (HF, licence accepted on the user's account): images 11.3 TB in
69 folders, depth 15.2 TB in 25 folders (two hdri folders alone 8.1 TB).
Street subset with depth chosen: rome tracking, rome vcam x3, rome dollyz
and dolly zoom, yakohama vcam-ego approach, middleeast vcam approach
(0.92 TB images + 2.84 TB depth); orbits, upper-body, hdri backdrops and
the chemical plant left out. Downloading on cluster2 straight to
`/mnt/nas/publicdatasets/bedlam2/{images,depth}` (`~/bedlam2/download_street.py`).

### 2026-09-15 13:20 — PromptHMR scored on the three splits
`run_prompthmr.py` over Waymo val, SLOPER4D test, 3DPW test (~17 crops/s on
the 4090; the crop as a 320 px window on a neutral 896 px canvas, box
around it, default camera, depth rescaled by the true focal, rotated into
the true camera; SMPL-X mapped to SMPL). `results_prompthmr.json`, static
row "PromptHMR, image only, box prompt, GT intrinsics": Waymo 82.4 / 64.8,
abs 1068.5, 1.040 m, mAP 0.03; SLOPER4D 58.4 / 49.7 / PVE 67.2, 0.228 m,
mAP 0.59; 3DPW 61.7 / 47.8 / PVE 87.1, 0.348 m, mAP 0.24. On pose it sits
between CameraHMR and HMR2 on Waymo and behind CameraHMR on SLOPER4D; its
metric depth is close to CameraHMR on SLOPER4D (0.23 vs 0.20 m) but not on
Waymo (1.04 m at 10-25 m). First compared pipeline with a 3DPW row.

### 2026-09-15 12:40 — anchor runs under the protocol, first 3DPW column
Forced re-evaluation of abl-anchor-points (2 seeds, `last.pt`): Waymo val
73.6 / 61.3 / abs 101.1 mm, 0.076 m, mAP 0.66; full SLOPER4D test 45.8 /
39.8 / PVE 52.1, 0.030 m, mAP 0.94; 3DPW test (6,617, first numbers of
that column) 60.2 / 50.4 / PVE 90.8, 0.068 m, mAP 0.79. The training-time
numbers were confirmed. Eval job 856989 (all runs, FORCE) still running.

### 2026-09-15 12:10 — interactive misplacement figure, PromptHMR camera fix
`showcase.comparison_figure_interactive` (plotly): image with labels and
the projected meshes, bird's-eye and side views; every method, the LiDAR
returns, the GT joints and mesh and the GT box footprint (new
`box_footprint`, `ValIndex.row` now carries `box3d`) are legend entries
that toggle in any combination. Notebook section 14 uses it (executed copy
spliced with the plotly outputs). The red dots on the crop are every 8th
vertex of our own mesh projected with the crop intrinsics; on the far
Waymo pick the mesh sits at the right depth (21.4-22.1 m vs 21.9 m
labelled) and is 1.72 m tall, the cloud only looks wide because vertices
of the arms and the outline are drawn as dots over a crowded crop.

PromptHMR: our crop intrinsics put the principal point hundreds of pixels
outside the crop, which the model never saw (first smoke: 3.6 m
placement). The runner now passes the true focal length with the
principal point at the crop centre and rotates the prediction back into
the true camera (exact rotation of e_z onto the ray through the crop
centre). A shadowed loop variable in the runner emptied the second batch;
fixed.

### 2026-09-15 11:30 — header figure: a whole Waymo sweep with our meshes
Frame 1557886649947221 (val, camera 2, segment 15224741240438106736_960_000_980_000)
has six pedestrians with 3D labels in view (the most in the val set).
`outputs/figures/waymo_header/waymo_sweep_dump_np.py` reads the v2 parquet
(pyarrow only) and ports the official range-image conversion to numpy
(azimuth from the column index with the extrinsic yaw correction, beam
inclinations reversed, per-pixel pose of the top laser, world-to-vehicle);
the waymo_open_dataset toolkit itself is unusable here (protobuf clash with
TensorFlow). Validation: every labelled keypoint is 3-5 cm from a return
and the six people coincide with our records. `waymo_header.py` runs
`abl-anchor-points-000/best.pt` on the six crops (pelvis 10-16 cm from the
labelled hip centres at 9-18 m) and draws the image overlay, the 3D sweep
view and a bird's-eye view. Notebook section 18 shows the first two.

### 2026-09-15 10:50 — point anchor: Waymo placement 0.222 -> 0.076 m at the short schedule
abl-anchor-points, two seeds, 3,468 steps (0.28 s/step with the new
loader defaults), training-time validation on the full Waymo val:
MPJPE 73.6, PA 61.3, abs 101.2 mm, placement 0.076 m, mAP 0.66 against
the learned-gate reference 108.4 / 90.3 / 244.9 / 0.222 / 0.47 and
LiDAR-HMR's 86.4 / 64.8 / 117.4 / 0.083 / 0.48. SLOPER4D subset: 42.3 /
37.7 / 51.2 mm, 0.027 m, mAP 0.96 (reference 81.9 / 58.6 / 155.5, 0.125,
0.66). Every column moves, pose included: with the placement solved by
the scan the queries stop spending capacity on depth. Full re-evaluation
queued (857100); full-anchor-points at 3k of 150k steps. 3DPW column was
NaN in training because `meshlidar/v1` was not in `STAGE_DIRS` (fixed,
a9084f4); the forced eval reads the workspace and is unaffected.

### 2026-09-15 10:40 — PromptHMR as a compared pipeline (in progress)
`third_party/PromptHMR` (submodule, Wang et al. CVPR 2025: promptable
SMPL-X regressor taking the whole image, boxes and intrinsics, metric
translation). Env `phmr_pt2.6` from its `scripts/install.sh`
(python 3.12, torch 2.6 cu126), checkpoints and the `smplx2smpl.pkl`
mapping from its fetch scripts (Google Drive), SMPL-X neutral copied from
the Human3R checkout, our SMPL files. Runner
`scripts/baselines/run_prompthmr.py`: crop as the image, one box over
the crop, our crop intrinsics as `cam_int`, SMPL-X vertices mapped to
SMPL, npz in the common format; static row `prompthmr` in the scoreboard.
`prompt_hmr` added to the mypy overrides.

### 2026-09-15 10:10 — SAM 3D Body to SMPL: the MHR conversion works
`third_party/MHR` (submodule) ships `tools/mhr_smpl_conversion` (optimisation
fit through a barycentric MHR-to-SMPL surface mapping). Test on 12 real
crops (6 Waymo val, 6 SLOPER4D test) with SAM 3D Body DINOv3-H+ (env
`sam_3d_body`, full crop as the box, our crop intrinsics) and the fit in
env `mhr_lod` (CPU torch; MHR assets downloaded to
`resources/pretrained-checkpoints/mhr/assets`, 4.7 GB, `lod1.fbx` needed
by `MHR.from_files`). Pitfall: the tool works in centimetres and expects
`pred_cam_t` added (its SAM3D helper does that; our SAM fork lacks the
`mhr_model_params` key, so `convert_mhr2smpl(mhr_vertices=100*(v+t))` is
the route). Result: MHR and SMPL vertices 0.65 / 0.61 cm apart (mean
nearest vertex, both directions), same extents; on the 6 SLOPER4D frames
the converted SMPL scores 32.1 mm MPJPE, 23.1 PA, placement 0.025 m
against the labels (one sequence at 2.3 m, not a benchmark). Scripts,
raw outputs and the overlay figure in `outputs/baselines/sam3d-body/smoke/`.
Notebook section 17 draws the overlay from the saved vertices (executed
copy spliced). Next: `scripts/baselines/run_sam3d_body.py` over the three
splits.

### 2026-09-15 09:20 — point-anchored translation head; TikZ architecture; LiDAR-HMR placed by the centroid alone
Why LiDAR-HMR places at 0.083 m: its loader subtracts the centroid of the
person's points, the network predicts the mesh in that local frame and the
centroid is added back (our baseline runner does the same with the box
centre of the points). Our head regresses (u, v, log z) in the camera frame
from an 8 m prior, nothing ties the depth to the scan; LiDAR-only reaches
0.233 m. New option `model.transl_anchor: points` (`ModelConfig.transl_anchor`,
`SmplHeads.offset_head` zero-initialised, `point_centroid` over the valid
input points; samples without points keep the camera parameterisation).
Configs `ablation_anchor_points.yaml` (abl-anchor-points, 2 seeds, jobs
856814/856815) and `full_anchor_points.yaml` (856816) on Helma; scoreboard
rows in the fusion axis and the headline table. Test
`test_point_anchored_translation_uses_valid_centroid`.

`score_baselines.py --place-at-centroid`: the pelvis of every prediction
moved onto the centroid of the record's input points (what a LiDAR method
gets from the scan alone). LiDAR-HMR mirrored on Waymo val: placement
0.106 m, abs 140.6 mm, mAP 0.45 (learned: 0.083 m, 117.4 mm, 0.48). The
scan alone gives 10 cm; their learned offset adds 2 cm. SLOPER4D test:
0.112 m, abs 140.3 mm, mAP 0.73 (learned 0.084 m, 117.3 mm, 0.76). Static row
`lidar-hmr-mirror-centroid` (`results_centroid.json`). Test
`test_place_at_centroid_moves_pelvis_onto_valid_points`.

Architecture figure redone in TikZ (`.docs/figures/architecture.{tex,pdf,png}`,
every number taken from the code; both translation anchors shown). Notebook
section 12 shows the PNG (builder updated, executed notebook spliced).

### 2026-09-15 08:45 — cluster2 as a second training host; pseudo-GT fit quality
cluster2 (ivlcluster02, `ssh cluster2`): 7x RTX 4090 24 GB, 96 CPUs, 251 GB
RAM, single-node Slurm (partition `rtx4090`, no time limit), internet
(wandb online, `~/.netrc`), NAS at `/mnt/nas` (74 TB free, 277 MB/s reads),
local disk nearly full (21 GB). Layout: repo `~/lidar-bedlam` (rsync, as
Helma), venv `~/venvs/lidar-bedlam` (uv 0.12, cache on the NAS at
`/mnt/nas/methods/max/uv-cache`), `resources/data` and `outputs` symlinked
to `/mnt/nas/methods/max/lidar-bedlam/{data,outputs}`. Shards copied from
the workstation at ~200 MB/s: real/v1, synth/v1, meshlidar/v1, body models
(~100 GB), then the full pool (v1_groups02-11, v1_group12, v1_pseudo,
~690 GB) chained behind it. NFS refuses chmod/chown: rsync with
`-rlt --no-perms --no-owner --no-group`. `lidar_bedlam/slurm/train_cluster2.sbatch`
(1 GPU default, `NPROC`/`--gres` for more, no staging, no resubmit chain).
Speed tests `c2-speed-{1,2,4}gpu` on `configs/loadertest.yaml` (512 per
GPU, 300 steps) submitted once the copy finished. cluster1 (6x A6000, 2
free, /mnt/md0 6.5 TB free, no NAS) kept as an option.

`scripts/compare_pseudo_gt.py`: LiDAR-HMR's published Waymo pseudo-GT
against ours, same measures (COCO limb joints regressed from the fitted
mesh vs the Waymo keypoints; median nearest-vertex distance of the
person's returns): theirs 27.0 / 27.1 mm and 3.2 / 3.1 cm (test / 2,000
train records), ours 22.1 mm and 2.7 cm (2,046 records), flat over the
distance bands for both. Ours is the tighter fit; the far-range placement
failure of `full-pseudo-waymo` (4.5 m at 20-30 m) is therefore not a
label-quality issue and stays open.

### 2026-09-15 08:30 — 3DPW test split in every val list, loader tests, LiDAR-HMR pseudo-GT
3DPW test shards generated (`meshlidar/v1/threedpw_test_w*`, 6,617 records,
16 shards); tokens being precomputed locally; every config with the
SLOPER4D val set (50) now also lists `threedpw_test`, so `eval_runs.sbatch`
(FORCE=1) scores all runs on it once the shards are on Helma; scoreboard
has a third column group (`SPLITS` P). Loader tests (`configs/loadertest.yaml`,
`train.sbatch` gained `NPROC` and `OMP_THREADS`): one GPU with 16 workers
runs 512/step at 0.26–0.33 s (1,600–2,000 samples/s), the 4-GPU jobs get
0.40–0.44 s per step for the same per-GPU batch, i.e. the ranks contend
for the 64 CPUs (64 workers x 8 OMP threads). Slurm caps CPUs at 32 per
GPU. Submitted 4-GPU tests `lt4-c128-w32-omp8`, `lt4-c128-w16-omp8`,
`lt4-c64-w16-omp1` (856791–856793). The single-GPU w32/w48 tests shared a
node and are not usable.

LiDAR-HMR published its Waymo pseudo-GT (Tsinghua cloud, `save_data/waymov2/
{train,test}.pkl`, 2.4 GB + 545 MB, downloaded to
`resources/data/external/lidar_hmr_waymov2/`): per record 14 keypoints,
a VPoser-regularised SMPL fit (betas, global_orient, transl, body_pose 63),
vertices, the person's points and segment/timestamp/object id, all in the
Waymo vehicle frame (z up). test.pkl: 1,873 records of the validation
split, 80 segments, 550 ids, median distance 13.7 m. Our own pseudo-GT
(`real/v1_pseudo`) has transl 13 cm from the hip centre (pelvis offset,
consistent); the full-pseudo-waymo run places well up to 10 m (0.21 m) but
4.5 m off at 20–30 m, so its far-range fits need a look before an overfit
test. LiDAR-only places better than main-mixed at every distance
(0.17/0.24/0.31 m vs 0.23/0.31/0.77 m): the image path hurts placement.

### 2026-09-15 08:10 — scoreboard: blue for values that beat every compared pipeline; batch-size headroom
`build_scoreboard.py` marks, in the headline table, every value of our runs
that beats the best static row of that column (baselines re-scored by us and
the quoted rows) in blue (`beats_static`, class `b`, wins over the rank
colours; legend chip). 75 blue cells at the time of writing, mostly
SLOPER4D columns; on Waymo only PA-MPJPE of a few runs (against the shown
CameraHMR 63.3, not the measured 60.0) and mAP of gate-none mix80.

Batch-size question (user asked for 2–4x larger batches for speed): the
model is 29.4 M parameters (decoder 21.3 M, point encoder 7.4 M), so
weights + grads + AdamW are 0.44 GiB per GPU; everything else is
activations. Allocator peak at 512 per GPU is 13.7–15.3 GiB (about 22–24
GiB in nvidia-smi with context and cache) of 96 GiB, so 2x (28 GiB), 3x
(42 GiB) and 4x (56 GiB) per GPU all fit. Throughput does not follow: the
12 Sep tests gave 5,200 samples/s at 1024 and 5,600 at 2048 (flat, loader
bound), and the current runs log 4,600–5,150 samples/s at 2048 with
0.40–0.44 s per step. A larger batch at the same step count therefore
costs proportionally more wall time; at the same sample budget it saves
nothing. The lever is the loader: the h100 nodes have 128 logical CPUs
and 754 GB RAM, the jobs request 64 CPUs and run 16 workers per rank.

### 2026-09-15 — ICP term diverged; fixed as a stop-gradient target

Every run with `lidar_icp` collapsed within the first evaluations
(identical 908 mm MPJPE / 7.6 m rows on both sets): the gradient through
the Kabsch SVD is unstable. The transform is now estimated under
`no_grad` and turned into a fixed per-vertex target; the loss value is
still the rigid displacement, its gradient pulls every facing vertex
towards its target, i.e. moves the body as a whole. 15 optimisation steps
on a Waymo batch: 0.21 -> 0.12 m, finite throughout. The trainer now
raises on a non-finite loss instead of training on. Two earlier crashes
at the first step were DDP refusing the unused clothing-offset parameter
in runs without the chamfer term (`find_unused_parameters` now also
when `lidar_chamfer` is 0) and strict checkpoint loading in the eval
script (non-strict now). Diverged / crashed run folders were removed and
the nine ICP runs, the six ICP/mesh runs and the evaluation resubmitted.
Ten comparison runs reached the 80k gate and stopped (gate none 77.0 /
60.6 mm Waymo, 0.294 m; SLOPER4D rig 46.1 / 38.7 mm on SLOPER4D).
Short chamfer-only screen: 109.6 vs 108.4 mm reference on Waymo, no gain.

### 2026-09-15 — LiDAR-to-surface losses, Pose2Mesh mesh terms, Human3R, 3DPW test

- **LiDAR-to-surface terms** (`losses/lidar_surface.py`). Correspondence
  rule as in the simulator: a mesh vertex counts when one of its faces
  has a normal pointing against the direction from the sensor origin
  (`sensor_origin` is now in every dataset item); every valid LiDAR
  return is matched to its nearest facing vertex (512 random facing
  vertices per sample, chunked cdist). `lidar_chamfer` = mean |distance -
  clothing offset|, the offset a learned model parameter (init 2 cm,
  clamped 0-10 cm, `clothing_offset` in the outputs). `lidar_icp` = three
  closest-point iterations solve the rigid Kabsch alignment of the facing
  vertices with the returns (differentiable through the SVD); the loss is
  the mean displacement of those vertices under that transform, so it
  moves the body as a whole. On the main-mixed checkpoint: Waymo 0.17 m
  chamfer residual, 0.21 m ICP displacement (the placement error it
  should remove), SLOPER4D 0.02 / 0.03 m; 0.3 s per 64-crop batch.
- **Pose2Mesh mesh terms** (`losses/mesh.py`), what LiDAR-HMR's paper
  cites as [3] for its final mesh loss: vertex coordinate L1
  (root-relative), joint coordinate loss (ours already), surface normal
  loss (|cos| between predicted face edges and the labelled face normal)
  and edge-length loss (|Delta length| per face edge); implemented from
  `third_party/LiDAR-HMR/models/pose2mesh/loss.py` and applied on records
  with SMPL labels (synthetic, SLOPER4D, 3DPW, pseudo-GT Waymo). Weights
  `vertex`, `normal`, `edge` in the loss section. Tests
  `tests/test_surface_losses.py`.
- Three loss-axis runs at the final schedule submitted on Helma:
  `full-chamfer` (5.0), `full-icp` (5.0), `full-meshloss` (5 / 0.5 / 5).
- LiDAR-HMR checkpoints: the README offers one Baidu-pan link
  ("Pretrained Models", not reachable from here); the copy on the NAS is
  the waymov2 mesh model only. No SLOPER4D-trained weights available.
- **Human3R** added as `third_party/Human3R`; runner
  `baselines/run_human3r.py` (env `human3r`, checkout `~/Documents/Human3R`).
  Detects 67 % of Waymo crops; depth on a tight crop is not metric (0.5x
  the true depth with the demo's 60 deg pseudo camera, 0.07x with the true
  K; the focal-normalised path needs a checkpoint variant we lack), so it
  enters the scoreboard as a pose-only row. 3DPW test split (24 sequences,
  4,483 frames, stride 5) is being generated with the fixed simulator from
  the NAS images for a third validation column.

### 2026-09-14 — PVE, box convention for mesh-only methods, LiDARCap quoted

- **Why LiDAR-HMR had a low Waymo mAP with the best placement.** Decomposing
  the box IoU on Waymo val: LiDAR-HMR centre error median 0.11 m (ours
  0.26 m), heading error median 10 deg (ours 7), but its box size is the
  tight mesh extent, 0.63 / 0.96 / 0.57 of the labelled box (x, up, z);
  CameraHMR the same 0.63 / 0.59. Waymo's labelled boxes are padded; our
  box head learns that from the box loss (size ratio 0.98 / 0.99 / 0.94),
  mesh-only methods cannot. The scorer now scales mesh-derived boxes on
  Waymo records by the median labelled-box / mesh-extent ratio measured on
  1,148 pseudo-GT fits of Waymo train, (1.39, 1.05, 1.45); SLOPER4D boxes
  are mesh extents already. abs / transl were never affected: abs is the
  mean joint distance without any centring, transl the pelvis (SLOPER4D)
  or hip-centre (Waymo) distance.
- **PVE** (root-relative per-vertex error, GT mesh from the record's SMPL
  parameters with the neutral model) added to the protocol, the trainer
  log, the eval script, the scorer and the scoreboard; Waymo has no meshes,
  so it is a SLOPER4D column. Re-evaluation job resubmitted on Helma with
  FORCE=1 for every finished run.
- **LiDARCap** (LiDARHuman26M's method, no code or weights we can run):
  quoted from the SLOPER4D paper, Table 4a, tested on SLOPER4D: trained on
  SLOPER4D 86.1 / 65.1 mm, trained on LH26M + SLOPER4D 79.2 / 60.1 mm
  (MPJPE / PA-MPJPE, their protocol). Shown as reported rows in the
  scoreboard and named as quoted in the paper.
- Paper: the metrics paragraph now says MPJPE / PA-MPJPE / PVE are
  root-relative and that the absolute joint error and the placement error
  are reported because image-only methods are depth-ambiguous.

### 2026-09-14 — LiDAR simulator: offset sensors were clipped, no facing test

Found through notebook section 5 (sensor 1 m to the right showed half a
person). Three defects in `lidar/simulate.py`, all fixed and unit-tested
on a synthetic cylinder (`tests/test_lidar.py`):

- The azimuth window of the rays was the camera's HFOV around the sensor's
  yaw, ignoring the translation: a person at 4 m is at -51 deg from a
  sensor 5 m to the right and got no rays at all; 1 m to the right the
  left part of the image was cut. Now the window comes from the image
  frustum's edges transformed into the sensor frame over the sensor's range
  (unwrapped around the image centre, so a LiDAR mounted looking backwards
  like the fusebike rig keeps one window).
- No facing test: a ray could "cross" a surface patch that faces away from
  the sensor (the far side of a limb as seen from a displaced sensor).
  Depth-map normals (one-sided at silhouettes, none where both neighbours
  jump > 0.5 m) now gate every hit with dot(ray, normal) < 0, which is what
  a mesh renderer's back-face test does.
- Hits were placed by linear interpolation between march samples 0.13 m
  apart (surface error up to +-6 cm); six bisection steps put them on the
  surface (cylinder radius 0.299-0.302 m for a 0.300 m cylinder). A ray
  entering a silhouette from the side lands on the silhouette line and is
  kept only within 0.25 m behind the front (a stand-in for the side surface
  the camera never saw); farther behind it is a wall and returns nothing.

Cylinder, OS1-128, no noise: sensor at the camera 1,906 returns over
-73..73 deg of the surface; 1 m right 1,844 (-51..113 deg); 5 m right 774;
10 m right 260, only the right-facing half. BEDLAM frame
`seq_000310/0145`, person returns old -> new: camera origin 699 -> 699,
10 cm above 685 -> 695, rig_waymo 1996 -> 2022, rig_sloper4d 708 -> 709,
rig_fusebike 1278 -> 1303, ball 1 m behind 578 -> 599, ball 1 m right
462 -> 721 (+56 %: the old window clipped it). So the rig variants and
small offsets in `synth/v1` are unchanged within 2 %; the laterally
displaced ball placements (the `main_*` and `target_waymo` variants draw
the sensor inside a 1 m ball) have too few returns and 6 cm surface
noise in v1. Regenerating the synthetic shards with the fixed simulator
(`synth/v2`) is the clean fix; it is a multi-hour job and every mixed run
would have to be retrained on it, so it is a decision, not done yet.
Notebook section 5 now shows 10 cm above, 1 m, 5 m and 10 m to the right.

### 2026-09-14 — capabilities notebook: showcase sections 12-16

`lidar_bedlam/utils/showcase.py` + six new sections in
`build_debug_notebook.py` (12 architecture block diagram from
`.docs/figures/architecture.html`; 13 scoreboard tables with the
green/yellow/red ranking, one cell per ablation axis + mirror test; 14
inference showcase: the headline checkpoint
`resources/pretrained-checkpoints/ours/full-main-mixed-last.pt` (pulled
from Helma at step 138k, re-pull when DONE) runs once over both validation
sets into `outputs/ours/full-main-mixed/` (baseline format, plus the
per-record gates), per-record protocol metrics for ours and the four
baselines, automatic picks (image methods misplace most / LiDAR-HMR pose
worst vs ours / farthest Waymo person) drawn as crop + bird's-eye + side
view with every method's mesh in the LiDAR returns, plus a plotly 3D view;
15 knowledge-transfer figure: real-only vs synth-only vs main-mixed vs
mix80 bars and the validation curves over training; 16 placement error by
distance bin for every method and the learned gates vs the priors).
Scoring the cached main-mixed predictions (step 138k) with the corrected
protocol: Waymo 84.1 / 66.5 mm, abs 358 mm, placement 0.347 m, mAP 0.30;
SLOPER4D 52.9 / 44.3 mm, abs 116 mm, placement 0.093 m, mAP 0.74. LiDAR-HMR
is at 0.081 m / 114 mm abs on Waymo: our placement there is the weak
column and the argument for the LiDAR-query redesign.

### 2026-09-14 — finals plateau by 60k: early stop for every queued run

Curves of the five running finals (eval every 2,000 steps): Waymo MPJPE
bottoms at 38-66k steps for every run (main-mixed 78.8 @ 50k, 3dpw 77.5 @
50k, mix80 84.5 @ 120k, synth-only 84.6 @ 66k) and then drifts up 3-4 mm
inside a +-2 mm evaluation noise band; real-only overfits monotonically
after 20k (Waymo 85.6 -> 90.0 mm, SLOPER4D 67 -> 79 mm, placement 0.18 ->
0.30 m). Simulating "min 80k steps, then stop after K evaluations without
a new best Waymo MPJPE" on the logged curves: K = 3, 5, 8 stop at 80k for
four of six runs; mix80 continues to 80/80/116k, synth-only to 80/80/82k.
Middle ground adopted: `optim.min_steps: 80000`, `optim.patience_evals: 5`
(= 10k steps; a new best MPJPE on any validation source resets it), cap
`max_steps: 150000` kept. Implemented in the trainer (rank 0 decides, the
flag is broadcast so all DDP ranks break; DONE is written; state survives
resume), unit-tested, set in all 16 `full_*.yaml`, synced to Helma so the
eleven queued runs (853646-853649, 853709-853712, 853714-853716) pick it
up; the five running finals finish their 150k. `best.pt` still tracks
Waymo mAP, which peaks as early as 8k on main-mixed, so tables use
`last.pt` of the early-stopped runs.

### 2026-09-14 — all pending comparison runs moved to the 150k schedule

Comparability over cost: the short pending jobs (`abl-rig-sloper4d` x2,
`abl-scale-long-*`, 853642-853645) were cancelled and replaced by full-
schedule twins of `full_main_mixed` that differ in one line each:
`full_rig_sloper4d`, `full_rig_waymo`, `full_ball025` (synthesis axis) and
`full_scale_2x` (data axis, 18 shards), jobs 853709-853712, single seed
like the finals. Together with `full-gate-none/hard`, `full-image-only`,
`full-lidar-only` (853646-853649) every row of the headline table then
has the same 150k steps; the 1/3-schedule ablation tables stay as the
cheap two-seed screen. `abl-scale-long-*` configs removed.

### 2026-09-14 — placement metric bug, absolute MPJPE, mirror test, comparison runs

**Which run is the main experiment.** The headline run is `full-main-mixed`
(`configs/full_main_mixed.yaml`: 50/40/10 synthetic / Waymo / SLOPER4D,
learned gates, random main scan variant, batch 2048, 150k steps). Every
ablation is cut from its one-third-schedule twin `abl-mixed-short`
(`configs/ablation_mixed_short.yaml`, 17 epochs = 3,468 steps, seeds 0 and
1): the fusion axis changes `gate_mode`, the synthesis axis changes the
synthetic scan `variant`, the label axis adds pseudo-GT Waymo or 3DPW rows,
the data axis caps the synthetic pool at fixed 3,468 steps.

**Bug (all Waymo placement numbers so far).** 39 of the 894 Waymo val
records have no labelled hip; unlabelled joints carry garbage coordinates
(z of -1 m), and the hip-centre placement error used them anyway, so those
39 records contributed errors of up to 12 m to every mean. LiDAR-HMR:
0.376 m over all, 0.081 m over the 855 records with labelled hips.
`metrics/protocol.py` now gives those records no placement error (nan,
skipped by the mean); mAP, MPJPE and PA were never affected (the GT box is
Waymo's own, joints use the validity mask). Every Waymo placement figure
logged before this entry is inflated, ours included (0.5-0.6 m in the
final runs will drop); the re-evaluation job below rewrites them.

**Absolute MPJPE** (`abs_mpjpe`: joints as predicted, no centring, valid
joints) added to the protocol, trainer log, eval script and scorer. It
separates the modalities at a glance: image-only 723-1044 mm on Waymo and
170-232 mm on SLOPER4D, LiDAR-HMR 114 / 121 mm.

Static baseline numbers (final; `outputs/baselines/results_static.json`):

| method | W MPJPE | W PA | W abs | W transl | W mAP | S MPJPE | S PA | S abs | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|---|---|
| HMR2.0 | 104.4 | 69.8 | 1044 | 1.048 m | 0.016 | 91.3 | 70.8 | 170 | 0.121 m | 0.40 |
| TokenHMR | 98.5 | 66.3 | 854 | 0.859 m | 0.024 | 74.5 | 54.8 | 232 | 0.204 m | 0.26 |
| CameraHMR | 76.9 | 60.0 | 723 | 0.731 m | 0.029 | 51.6 | 44.4 | 197 | 0.196 m | 0.40 |
| LiDAR-HMR | 84.7 | 62.9 | 114 | 0.081 m | 0.207 | 100.2 | 65.8 | 121 | 0.089 m | 0.76 |

**Mirror test (was LiDAR-HMR trained on the validation data?).** Inputs
mirrored left/right (points x -> -x; image flipped with the principal
point), predictions mirrored back through the SMPL left/right vertex map
(`--mirror` in both runners, `smpl_mirror_map` in `baselines/common.py`).
LiDAR-HMR: Waymo 84.7 -> 86.4 mm MPJPE, placement 0.081 -> 0.083 m;
SLOPER4D 100.2 -> 98.0 mm. CameraHMR (never saw either set): 76.9 -> 77.6
and 51.6 -> 51.8 mm. A memorised set would collapse under mirroring; a 2 %
change is generalisation. No evidence that LiDAR-HMR saw Waymo val; its
strong Waymo placement is simply LiDAR seeing the person.

**Comparison runs submitted on Helma** (jobs 853642-853649, all from the
repo root with `train.sbatch`): `abl-rig-sloper4d` seeds 0/1 (synthetic
LiDAR at the SLOPER4D rig pose, `configs/ablation_rig_sloper4d.yaml`);
`abl-scale-long-2x` / `-full` (data axis at 3x the steps, 10,404, to test
the "longer schedule may separate the pools" caveat); `full-gate-none`,
`full-gate-hard`, `full-image-only`, `full-lidar-only` (fusion axis at the
150k headline schedule: decides whether the two-stream design is leveraged
or a new query design is needed). Re-evaluation job 853665
(`lidar_bedlam/slurm/eval_runs.sbatch`) rescores every finished run's
`best.pt` and `last.pt` on the full sets with the corrected protocol and
the abs column into `outputs/eval/<run>-<ckpt>.json`;
`lidar_bedlam/scripts/pull_helma_results.sh` fetches them and
`lidar_bedlam/scripts/build_scoreboard.py` renders the one-page comparison
(published pipelines static, finals, fusion / synthesis / data / label
axes, mirror test; ranks per column within each table).

### 2026-09-14 — final runs, intermediate table after ~11 h (steps 92k-140k of 150k)

Latest validation line per run (Waymo val 894, SLOPER4D test full 9,904):

| run | step | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|---|
| full-synth-only | 98k | 86.3 | 64.5 | 1.62 m | 0.18 | 59.9 | 47.3 | 0.287 m | 0.32 |
| full-real-only | 106k | 87.8 | 69.0 | 0.78 m | 0.21 | 75.6 | 52.2 | 0.254 m | 0.34 |
| full-main-mixed | 96k | 81.0 | 64.2 | 0.60 m | 0.38 | 52.2 | 44.2 | 0.080 m | 0.79 |
| full-mix80 | 92k | 85.8 | 69.5 | 0.54 m | 0.46 | 47.1 | 41.0 | 0.072 m | 0.80 |
| full-pseudo-waymo | 140k | 87.5 | 65.5 | 1.28 m | 0.19 | 45.9 | 39.5 | 0.080 m | 0.79 |
| full-3dpw | 98k | 80.8 | 63.7 | 0.61 m | 0.38 | 51.6 | 43.0 | 0.079 m | 0.75 |

Waymo metrics have been flat since ~hour 5; the runs are ahead of the
20 h estimate (pseudo-waymo at 140k). Against the baselines: mixed runs
beat every image method on placement and mAP on both datasets and beat
CameraHMR on SLOPER4D pose (47.1 / 45.9 vs 51.6 mm); on Waymo pose
CameraHMR (76.9) still leads our best (80.8), and LiDAR-HMR's Waymo
placement (0.38 m) beats ours (0.54). The pseudo-Waymo run again loses
Waymo placement (1.28 m) while winning SLOPER4D pose.

All comparable pipelines on the same records and protocol (🟢 best, 🟡 second,
🔴 third per column; lower is better except mAP; SLOPER4D full test set):

| method | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| HMR2.0 (4D Humans) (image only) | 104.4 | 69.8 | 1.338 m | 0.02 | 91.3 | 70.8 | 0.121 m | 0.40 |
| TokenHMR (image only, tight crop) | 98.5 | 66.3 | 1.154 m | 0.02 | 74.5 | 54.8 | 0.204 m | 0.26 |
| CameraHMR (image only, GT intrinsics) | 🟢 76.9 | 🟢 60.0 | 1.023 m | 0.03 | 51.6 | 44.4 | 0.196 m | 0.40 |
| LiDAR-HMR (LiDAR only, Waymo weights) | 84.7 | 🟡 62.9 | 🟢 0.376 m | 0.21 | 100.2 | 65.8 | 0.089 m | 0.76 |
| ours · synth-only (step 98k) | 86.3 | 64.5 | 1.619 m | 0.18 | 59.9 | 47.4 | 0.287 m | 0.32 |
| ours · real-only (step 106k) | 87.8 | 69.0 | 0.781 m | 0.21 | 75.6 | 52.2 | 0.254 m | 0.34 |
| ours · main-mixed (step 96k) | 🔴 81.0 | 64.2 | 🔴 0.604 m | 🟡 0.38 | 52.2 | 44.2 | 0.081 m | 🟡 0.79 |
| ours · mix80 (step 92k) | 85.8 | 69.5 | 🟡 0.545 m | 🟢 0.46 | 🟡 47.1 | 🟡 41.0 | 🟢 0.072 m | 🟢 0.80 |
| ours · pseudo-waymo (step 140k) | 87.5 | 65.5 | 1.284 m | 0.19 | 🟢 45.9 | 🟢 39.5 | 🔴 0.080 m | 🔴 0.79 |
| ours · 3dpw (step 98k) | 🟡 80.8 | 🔴 63.7 | 0.614 m | 🔴 0.38 | 🔴 51.6 | 🔴 43.0 | 🟡 0.079 m | 0.75 |

### 2026-09-14 — image baselines scored with our protocol (TokenHMR, HMR2, CameraHMR)

Runners in `lidar_bedlam/scripts/baselines/` execute each published model
in its own conda env on exactly our evaluation records (the 256 px crops
of `real/v1`, `waymo_val` 894 and `sloper4d_test` 9,904) and write
camera-frame meshes; `lidar_bedlam/scripts/score_baselines.py` then derives
SMPL/COCO joints and the 3D box the same way as for our model and runs
`metrics/protocol.py` unchanged (translation = SMPL transl parameter, box
confidence 1). Weak-perspective cameras are converted to metric
translation with the crop's true intrinsics (HMR2's `cam_crop_to_full`
with the real principal point); CameraHMR gets the intrinsics as input
instead of estimating them.

- Environments: `4D-humans` conda env for all three image methods (numpy
  pinned < 2 so torch 2.2 initialises; `flatten_dict` added for TokenHMR);
  HMR2 from `~/Documents/4D-Humans` (`~/.cache/4DHumans` weights, epoch 35),
  TokenHMR from `~/Documents/tokenHMR` (that checkout hardcodes a NAS
  checkpoint path in `lib/utils/misc.py`, the runner swaps in the identical
  local file), CameraHMR from `third_party/CameraHMR` with `data/` symlinks
  to our SMPL, the mean params and `~/Downloads/camerahmr_checkpoint_cleaned.ckpt`.
- Inference cost is negligible: ~115 crops/s per ViT-H method on the 4090
  (8 s for Waymo val, ~90-170 s for SLOPER4D test); the whole pass over
  10.8k crops takes ~3 min per method, scoring ~2 min. The work was the
  adapters and environments (~4 h).
- Two input variants: `full` = our 256 px crop as is (same pixels our model
  sees), `tight` = HMR2-style square re-crop around the labelled joints.
  They agree within 1 mm and 0.02 mAP, so `full` is the protocol.

| method (full crop) | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| HMR2.0 (4D Humans) | 104.4 | 69.8 | 1.34 m | 0.016 | 91.3 | 70.8 | 0.121 m | 0.40 |
| TokenHMR (tight crop) | 98.5 | 66.3 | 1.15 m | 0.024 | 74.5 | 54.8 | 0.204 m | 0.26 |
| CameraHMR (GT intrinsics) | 76.9 | 60.0 | 1.02 m | 0.029 | 51.6 | 44.4 | 0.196 m | 0.40 |
| LiDAR-HMR (LiDAR only, Waymo weights) | 84.7 | 62.9 | 0.38 m | 0.21 | 100.2 | 65.8 | 0.089 m | 0.76 |
| ours main-mixed (10.2k steps, S on first 4,000) | 92.6 | 75.6 | 0.52 m | 0.44 | 55.7 | 45.5 | 0.08 m | 0.83 |

SLOPER4D on the full test set (9,904) vs the trainer's first 4,000: MPJPE
differs by up to 9 mm for HMR2/TokenHMR (82 vs 91, 66 vs 75), CameraHMR is
stable (51.7 vs 51.6); `outputs/baselines/results_4000.json` holds the
subset numbers, `results_full.json` the full ones.

Reading: HMR2 on Waymo reproduces the 106 mm reported for the same weights
in the LiDAR-HMR checkpoint notes, so the pipeline is sound. Image-only
methods are at 1.0-1.3 m placement error on Waymo (0.12-0.20 m on the
near-range SLOPER4D) and 0.02-0.03 box mAP; our fusion model is 2x closer
on Waymo, 2.5x on SLOPER4D and 0.44 / 0.83 mAP. On pose, CameraHMR is the
strongest baseline and beats our current checkpoints (Waymo 76.9 vs 92.6
mm, SLOPER4D 51.6 vs 55.7 mm); TokenHMR, whose frozen ViT-H tokens we use,
is close to us on Waymo (98.5 vs 92.6) and behind on SLOPER4D (74.5 vs
55.7). The paper claim therefore rests on placement and detection quality,
not on pose accuracy; the final 20 h runs decide the pose column.

LiDAR-HMR (release weights trained on Waymo, 1024 points centred on the
box centre in a z-up frame, mesh rotated back; `lidar-hmr` conda env:
torch 2.2 cu121, torch_geometric 2.4 + PyG wheels, `pointops` from
PointTransformerV2 and the vendored `pointnet2_ops` compiled with
`TORCH_CUDA_ARCH_LIST=8.9`, chumpy patched for numpy 1.26, a pytorch3d
shim; checkpoint copied to `resources/pretrained-checkpoints/lidar-hmr/`,
md5 f3f970fd…) runs at ~98 samples/s. On its own training domain (Waymo)
it is the best baseline for placement, 0.38 m (ours 0.52), and beats our
pose there too (84.7 vs 92.6 mm); on SLOPER4D its placement is as good as
ours (0.089 m, mAP 0.76 full / 0.83 on the first 4,000 = ours) but the
pose is far worse (100 vs 56 mm MPJPE). Reading: LiDAR alone gives the
placement, images alone give the pose, and our model is the only one
with both on both datasets, but the Waymo placement and pose columns are
not yet won, so the final runs matter. Local, untracked setup in the
submodules: `third_party/CameraHMR/data/` symlinks,
`third_party/LiDAR-HMR/smplx_models/smpl/SMPL_NEUTRAL.pkl` symlink and the
`pointnet2_ops_lib/setup.py` arch list. Prediction files are in
`outputs/baselines/<method>-<crop>/<split>.npz`, scores in
`outputs/baselines/results_*.json`.

### 2026-09-14 — final-schedule runs submitted (6 x ~20 h)
`configs/full_*.yaml`: 150,000 steps at batch 2048 (about 20 h incl.
evaluations every 2,000 steps on the full SLOPER4D test set of 9,904 and
Waymo val), checkpoints every 1,000 steps, the chain job resumes past the
24 h wall time; `best.pt` now = highest Waymo box mAP. Jobs 850403-850408:
`full-synth-only`, `full-real-only`, `full-main-mixed`, `full-mix80`,
`full-pseudo-waymo`, `full-3dpw`. These replace the 10.2k/12k-step main
table; the 3,468-step ablations stay as relative comparisons.

### 2026-09-13 — fusion ablations with two seeds; pseudo-GT and 3DPW ablations
50/40/10 mixture, 17 epochs = 3,468 steps; mean +- half-range over seeds
0 and 1 (Helma/wandb names: `abl-<variant>-000` and `abl-<variant>-s1`,
reference `abl-mixed-short-001` / `-s1`):

| variant | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| learned gates (reference) | 108.4+-3.4 | 90.3+-5.1 | 0.509 | 0.467 | 80.2+-1.1 | 59.9+-0.6 | 0.079 | 0.738+-0.015 |
| image only | 118.0+-0.3 | 94.6+-0.5 | 2.232 | 0.002 | 112.6+-6.8 | 78.5+-3.6 | 0.730 | 0.021 |
| LiDAR only | 218.3+-6.1 | 164.0+-1.3 | 0.589 | 0.331 | 144.6+-0.8 | 103.3+-0.5 | 0.068 | 0.635 |
| gate none (plain sum) | 115.7+-2.0 | 94.6+-1.0 | 0.511 | 0.444 | 101.5+-3.5 | 75.8+-1.3 | 0.096 | 0.555+-0.057 |
| gate hard (fixed priors) | 112.0+-1.6 | 93.7+-3.3 | 0.525 | 0.444 | 83.1+-4.3 | 62.2+-0.6 | 0.085 | 0.633+-0.045 |
| + pseudo-GT SMPL on Waymo (1 seed) | 108.5 | 73.4 | 1.443 | 0.167 | 69.5 | 48.4 | 0.065 | 0.717 |
| + 3DPW mesh-LiDAR 10 % (1 seed) | 104.1 | 84.6 | 0.514 | 0.456 | 80.4 | 60.9 | 0.072 | 0.737 |

Reading. With two seeds the fusion ordering is clear on SLOPER4D and on
mAP: learned gates 80 mm / 0.74 vs fixed priors 83 / 0.63 vs plain sum
102 / 0.56; on Waymo pose the three are within 4-8 mm (learned best).
Image only cannot place (2.2 m), LiDAR only cannot pose (218 mm).
3DPW records give a small Waymo pose gain (104 vs 108 mm) at equal
placement. Pseudo-GT Waymo labels give the best pose of every short run
(Waymo PA 73 vs 90 mm; SLOPER4D 69.5 / 48.4 mm) but Waymo placement is
far worse (1.44 m, mAP 0.17) and still falling at the end of the run;
on its own training shard the checkpoint is 0.85 m off the fitted
translations while the fitted translations themselves sit at the
keypoint hip centre, so the labels are consistent and the run is not
converged on placement (the direct translation loss on 40 % of every
batch starts at about 2.5 m). Candidates: longer schedule or a lower
translation weight for pseudo rows.

### 2026-09-13 — state at the machine switch
Helma queue (all pending, partition full): `ablation-{image-only,
lidar-only,gate-none,gate-hard}` seed 0 on 50/40/10, the same four plus
`ablation-mixed-short` as `-s1` (seed 1), `ablation-pseudo-waymo` (job
847660). The login-node mirror loop is running (`WANDB_MIRROR_DIR`).
On mbwm a detached script `outputs/logs/post_3dpw.sh` (log
`post_3dpw.log`) waits for the 3DPW generation, computes the tokens,
copies `meshlidar/v1` to Helma, submits `ablation-3dpw` and re-executes
the notebook. To follow from another machine: `ssh helma squeue -u
$USER`, wandb `erik_hm/lidar-bedlam`, and the Slurm logs in
`outputs/slurm-*.out` on Helma; results as `outputs/<run>/val_*.json`.

### 2026-09-13 — fusion ablations on 50/40/10 and a second seed (9 jobs)
Submitted: the four variants `abl-{image-only,lidar-only,gate-none,gate-hard}`
(seed 0, 50/40/10, vs `abl-mixed-short-001`; on Helma and wandb they carry
the reused name `-000`, since the 80/10/10 folders were deleted before) and seed 1 of the reference and the
four variants (`abl-*-s1`, `EXTRA_SET=seed=1`), jobs 847325-847333.

### 2026-09-13 — scaling curve at fixed compute: flat
50/40/10 mixture, 3,468 steps each, synthetic pool capped at k x the
Waymo train count (18 / 36 / 73 / 145 / 291 shards vs 806 for the
reference):

| synthetic pool | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| 2x (9k records) | 108.8 | 88.4 | 0.523 | 0.436 | 82.5 | 59.9 | 0.101 | 0.705 |
| 4x | 119.7 | 102.2 | 0.517 | 0.440 | 85.2 | 62.8 | 0.092 | 0.651 |
| 8x | 107.3 | 88.4 | 0.520 | 0.449 | 76.8 | 56.8 | 0.101 | 0.678 |
| 16x | 110.8 | 91.5 | 0.520 | 0.435 | 79.6 | 57.0 | 0.087 | 0.721 |
| 32x | 103.0 | 84.0 | 0.501 | 0.482 | 77.8 | 55.9 | 0.107 | 0.669 |
| full (85x, reference) | 111.8 | 95.4 | 0.507 | 0.465 | 81.3 | 59.2 | 0.079 | 0.723 |

Reading: within the noise (about +-5 mm, +-0.03 mAP) the size of the
synthetic pool does not matter at this budget; 9k synthetic records
already give the full effect. The gain comes from the presence and
variety of simulated LiDAR, not from volume, at least at 3.5 M synthetic
samples seen; a longer schedule may separate the curve.

### 2026-09-13 — synthesis axis done; scaling curve rerun at fixed steps
50/40/10 mixture, 17 epochs = 3,468 steps, reference `abl-mixed-short-001`:

| run | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| reference (ball 1 m, 12 resolutions) | 111.8 | 95.4 | 0.507 | 0.465 | 81.3 | 59.2 | 0.079 | 0.723 |
| ball 0.25 m | 105.0 | 85.8 | 0.514 | 0.458 | 72.5 | 52.6 | 0.062 | 0.736 |
| LiDAR at the Waymo rig pose | 103.0 | 84.8 | 0.515 | 0.458 | 79.7 | 59.5 | 0.078 | 0.671 |
| Waymo resolution only | 108.0 | 88.3 | 0.517 | 0.448 | 92.9 | 68.2 | 0.083 | 0.600 |

Reading: the rig-matched pose gives the best Waymo pose but loses on
SLOPER4D mAP (0.67 vs 0.72), the target-only resolution costs SLOPER4D
pose (93 vs 81 mm) without helping Waymo, and the small 0.25 m ball beats
the 1 m ball on both sets. Placement metrics are flat across the axis.
The first scaling runs were invalid: `max_epochs` over a capped pool gave
289-799 steps; the five configs now fix `max_steps: 3468` and were
resubmitted (jobs 845904-845908).

### 2026-09-13 — synthesis axis and scaling curve submitted (9 jobs)
Ablations rebased on the 50/40/10 mixture that won the main table
(`abl-mixed-short-001` as the new reference; the `-000` model-axis runs
were on 80/10/10). Submitted: `ablation-ball025`, `ablation-rig-waymo`,
`ablation-target-waymo`, `ablation-scale-{2,4,8,16,32}x` (jobs
844883-844891), 17 epochs each.

### 2026-09-12 — comparison batch, last validation of every run
Batch 2048, lr 3e-4; main table 10,200 steps (synth-only 12,000);
model-axis ablations 17 epochs = 3,468 steps on the 80/10/10 pool.
W = Waymo val (894), S = SLOPER4D test (4,000); MPJPE / PA in mm,
translation in m, box mAP.

| run | W MPJPE | W PA | W transl | W mAP | S MPJPE | S PA | S transl | S mAP |
|---|---|---|---|---|---|---|---|---|
| synth-only-000 | 99.3 | 72.7 | 1.389 | 0.186 | 72.4 | 51.0 | 0.250 | 0.357 |
| real-only-000 (80/20 W/S) | 90.8 | 71.5 | 0.692 | 0.301 | 66.4 | 53.0 | 0.104 | 0.693 |
| mix80-001 (80/10/10) | 102.4 | 84.9 | 0.520 | 0.476 | 56.4 | 45.0 | 0.068 | 0.785 |
| main-mixed-000 (50/40/10) | 92.6 | 75.6 | 0.518 | 0.436 | 55.7 | 45.5 | 0.057 | 0.828 |
| abl-mixed-short (learned gates) | 126.9 | 103.9 | 0.617 | 0.408 | 74.5 | 52.4 | 0.078 | 0.720 |
| abl-image-only | 153.0 | 117.0 | 2.135 | 0.007 | 103.7 | 66.3 | 0.692 | 0.035 |
| abl-lidar-only | 222.6 | 156.1 | 0.677 | 0.227 | 137.7 | 105.8 | 0.064 | 0.635 |
| abl-gate-none (plain sum) | 123.0 | 96.4 | 0.602 | 0.333 | 83.1 | 63.6 | 0.085 | 0.694 |
| abl-gate-hard (fixed priors) | 132.2 | 107.2 | 0.640 | 0.406 | 80.5 | 60.0 | 0.077 | 0.672 |

Reading: synthetic data buys placement (Waymo translation 0.69 -> 0.52 m,
mAP 0.30 -> 0.44-0.48) and SLOPER4D across the board (66 -> 56 mm, 10 ->
6 cm), but not Waymo pose, where real-only is best (90.8 mm) and the
50/40/10 mixture (92.6) beats 80/10/10 (102.4). Fusion: image-only cannot
place (2.1 m), LiDAR-only cannot pose (223 mm); both gated variants sit
between; plain sum vs learned gates is within noise on pose at 1/3
schedule, learned gates lead on mAP (0.41 vs 0.33).

### 2026-09-12 — mix80-001 (fixed Waymo supervision, corrected shards) finished
10,200 steps (50 epochs), last: Waymo MPJPE 102.4 / PA 84.9 mm, translation
0.52 m, box mAP 0.48; SLOPER4D MPJPE 56.4 / PA 45.0 mm, 6.8 cm, mAP 0.79.
Against `synth-only-000` (99.3 / 72.7 mm, 1.39 m, 0.19 on Waymo): adding
10 % real data cuts the Waymo placement error to 37 % and more than doubles
mAP at equal pose accuracy. Four gate ablations crashed on DDP's unused-
parameter check (gate MLP / one stream has no gradient in those modes);
fixed with `find_unused_parameters` for non-learned gates and resubmitted
(jobs 843895-843898).

### 2026-09-12 — comparison batch submitted (7 jobs)
Schedules made comparable: main-table rows count steps because their
pools differ (`real_only`, `main_mixed` = 10,200 steps at batch 2048, the
compute of `mix80-001`'s 50 epochs); ablations share the 80/10/10 pool and
count epochs (`max_epochs: 17`, one third). Submitted on Helma:
`real-only`, `main-mixed` (50/40/10), `ablation-mixed-short` (reference),
`ablation-image-only`, `ablation-lidar-only`, `ablation-gate-none`,
`ablation-gate-hard` (jobs 843794-843800), alongside `mix80-001`.

### 2026-09-12 — first full runs (12k steps, batch 2048)
`synth-only-000` (100 % BEDLAM, zero-shot on real data), last step:
Waymo MPJPE 99.3 / PA 72.7 mm, translation 1.39 m, box mAP 0.19;
SLOPER4D MPJPE 72.4 / PA 51.0 mm, translation 0.25 m, mAP 0.36.
`mix80-000` (80/10/10, trained with the Waymo keypoint bug): SLOPER4D
MPJPE 52-56 / PA 44 mm, translation 5-7 cm, mAP 0.78-0.89; Waymo
translation 0.73 m and mAP 0.47 but MPJPE 390-405 mm, the symptom that
exposed the bug. Both runs are on wandb. `mix80-001` resubmitted with the
fix and the corrected group-01 shards (job 843483).

### 2026-09-10 02:44 — plan and ablations decided (30c4b90, d46f545)
Waymo headline, SLOPER4D secondary, LiDARHuman26M and PedX dropped;
`plan.md`, `ablations.md` (ball-radius sweep, resolution ablation, rolling
shutter kept as a generation option only).

## Paper

### 2026-09-15 — architecture figure in TikZ
`.docs/figures/architecture.{tex,pdf,png}`, numbers from the code, both
translation anchors; notebook section 12 shows the PNG.

### 2026-09-14 — metrics paragraph
Root-relative MPJPE/PA/PVE, absolute error and placement, box convention,
quoted numbers (`main.tex`). Scoreboard HTML with rank colours and blue
for values beating every compared pipeline; footnotes for CameraHMR
(PA shown as 63.3 by request) and Human3R (placement blanked).

### 2026-09-11 13:38 — related work and bibliography (72af389, fc8943a, 0e3d746)
39 bib entries with venue macros from `latex-draft/bib/bib-short.def`;
related-work bullets; LIF-Net (IV 2026) and BikeActions (ICPR 2026)
official citations. Self-contained build: style files beside `main.tex`
and a `latexmkrc` (8445f2e).

### 2026-09-10 18:10 — contributions and story (3df2902, 41192fa, 09070d4, df39884)
`story.md` (claim, model / data / synthesis ladders, run order); three
contribution bullets; 6.3 extrinsics questions.

## Housekeeping

### 2026-09-15 — cluster2 as a second host
7x RTX 4090, single-node Slurm, NAS-backed data; NFS memmaps are
latency-bound, a sequential read into the page cache fixes it for sets
that fit in RAM (100 GB set: 0.62 s/step at 512 per GPU, H100 0.26 s).
`slurm/train_cluster2.sbatch`. Hourly Helma monitor rebuilds the
scoreboard artifact.

### 2026-09-13 — Helma file quota hit by wandb run folders
`rsync` to the workspace failed with "Disk quota exceeded": `outputs/`
held 50,127 files, 49,713 of them wandb run directories (about 700 per
mirrored or offline run; the workspace allows 61k files). Removed the
superseded run folders (batch tests, `mix80-000`, the 80/10/10 and the
invalid scaling ablations, all deleted from wandb by the user as well)
and every `outputs/*/wandb`; the mirror now writes its wandb files to
`$HOME/wandb-mirror` (`WANDB_MIRROR_DIR`). 268 files left in `outputs/`.

### 2026-09-12 — final layout, round 2 (top level minimal)
`scripts/` and `slurm/` moved inside the package; `data/` and
`checkpoints/` became `resources/data/` and `resources/pretrained-checkpoints/`
(gitignored, built by `lidar_bedlam/scripts/dm_link.py`); top level is now
`lidar_bedlam/ configs/ notebooks/ tests/ third_party/ .docs/ resources/
outputs/` plus the config files. The `python-project-init` skill was
rewritten to this contract (tree, folder table with tracked/gitignored and
use, steps). On Helma the data folder must be moved to `resources/data`
and the mirror loop restarted after the two running jobs finish.

### 2026-09-12 — preparation finished: final layout before moving to Helma
Flat package `lidar_bedlam/` (no `src/`), `utils/` for io and viz, entry
points `lidar_bedlam/scripts/lidar-bedlam-{main,eval}.py`, `notebooks/` at the top
level, job logs under `outputs/logs/`; ruff excludes `third_party/`,
uv default groups `dev` + `debug`. `HANDOFF.md` carries the `tree -L 2`
and the fixed package structure as the template for new projects. From
here on the work happens on Helma; the wandb hook must be extended to
match `*-main.py`.

### 2026-09-08 18:08 — scaffold (c9beeab)
uv, hatchling, ruff, mypy strict; five baseline submodules; BEDLAM audit;
dataset survey.

# Todos

Ordered by dependency: localisation first, the rest builds on a model
whose placement is competitive.

## Data

- [ ] Regenerate `synth/v2` with the fixed simulator (window unwrap, facing test); rerun the comparison set on it.
- [ ] Tokens per backbone next to each shard (`<shard>.tokens-{s,b,l}.npy`, ViT-H stays `.tokens.npy`) over real/v1, the synthetic pool, meshlidar/v1.
- [ ] 3DPW in the main mixture (currently only `full_3dpw`, 10 %) once localisation is fixed, then rerun the comparison set so every row saw the same data.
- [ ] Baselines on the 3DPW test split (crops exist; image baselines, LiDAR-HMR, Human3R).
- [ ] Dataset statistics for the paper; official xxh128 validation on the NAS; rolling shutter (default off); Waymo LiDAR-only records.
- [ ] cluster2: local disk or RAM-resident set; the full pool cannot be served over NFS.

## Model

- [ ] Localisation: read abl-anchor-points (2 seeds) and full-anchor-points; then decoupled heads (translation, orientation, shape from LiDAR; pose from the image), modality dropout, evidence-conditioned gate.
- [ ] Far-range placement failure of full-pseudo-waymo (4.5 m at 20-30 m): labels are fine at range, find the cause in the model or loader.
- [ ] Backbone ladder: `ViTConfig` presets S/B/L (384/12/6, 768/12/12, 1024/24/16; ViTPose weights, same class as ViT-H = TokenHMR), loader test per size; `full_backbone_{s,b,l}` + short versions with 2 seeds; scoreboard group "backbone axis". Optional extra point: DINOv3-B/16 (new backbone class: RoPE, registers).
- [ ] Inference time on one RTX 4090: `scripts/bench_inference.py`, batch 1 and 8, fp16, end to end (backbone on the crop, tokenizer, decoder, SMPL, box), median of 200 iterations, ms and Hz per backbone, plus LiDAR-only and image-only.
- [ ] Bike budget: target 10 Hz on the bike's compute (record platform, TDP, TensorRT); re-run the benchmark there or on the closest GPU; state which backbone meets 10 Hz and what it costs in accuracy.
- [ ] LiDAR-HMR retraining (their code and pickles, then our pseudo-GT converted) only if the anchor runs leave the gap unexplained.

## Experiments

- [ ] Loss axis: read the combined ICP/mesh runs (restarted after the divergence fix) and the full chamfer/ICP/mesh runs; keep what helps Waymo placement, abs, mAP.
- [ ] SAM 3D Body in the headline table: `scripts/baselines/run_sam3d_body.py` in its own env (checkpoints `facebook/sam-3d-body-dinov3`; DINOv3-H+ and ViT-H variants); MHR output mapped to our joints (or an SMPL fit), placement handled as for the image-only rows; footnote.
- [ ] Forced re-evaluation with the 3DPW column (queued on Helma, 856799); scoreboard already has the columns.
- [ ] After hand-in: realism ablations (no augmentation, no occlusion), LoRA on the backbone.

## Paper

- [ ] Data section (BEDLAM inputs, LiDAR simulation, realism, statistics).
- [ ] Method section (backbones, gates, translation anchors, losses); architecture figure is done.
- [ ] Results and ablation tables from the scoreboard; gate-map figure; accuracy-versus-speed curve from the backbone ladder.
- [ ] Anonymity pass: no rig, group or first-person names.

## Housekeeping

- [ ] Rotate the BEDLAM password and remove it from the shared NAS `be_download.sh`.
- [ ] Complete the Human-M3 download and the remaining 9 SLOPER4D sequences.
- [ ] Fix dangling symlinks in `publicdatasets/LiDARHuman26M`; consider deleting 1.4 TB of redundant Waymo tarballs.
