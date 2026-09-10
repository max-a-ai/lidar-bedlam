# Data pipeline, conventions, losses and metrics

## Where things live

| Concern | Module | Notes |
|---|---|---|
| File readers (PCD, PLY, EXR depth, images, masks) | `lidar_bedlam/io.py` | numpy only, no open3d |
| Camera model, rigid transforms, axis maps | `lidar_bedlam/geometry/camera.py` | OpenCV frame; Brown-Conrady distortion |
| Rotations | `lidar_bedlam/geometry/rotations.py` | axis-angle <-> matrix via scipy |
| Square person crops | `lidar_bedlam/geometry/crop.py` | crop spec, intrinsics of the crop |
| 3D boxes and exact oriented IoU | `lidar_bedlam/geometry/boxes.py` | 7-vector boxes |
| SMPL wrapper + frame change of SMPL params | `lidar_bedlam/body/smpl.py` | `transform_smpl_params` handles the pelvis offset |
| Legacy SMPL pkl conversion | `lidar_bedlam/body/convert_smpl.py` | run once, output under `data/generated/body_models` |
| Common sample schema | `lidar_bedlam/data/schema.py` | `Sample`, `CroppedSample`, `SampleMeta` |
| Dataset-agnostic pipeline | `lidar_bedlam/data/base.py` | `SampleSource`, SMPL-derived labels, crop, point sampling |
| Loaders | `lidar_bedlam/data/{sloper4d,lidarhuman26m,waymo,bedlam}.py` | one class per dataset |
| Torch dataset / collate-ready items | `lidar_bedlam/data/torch_dataset.py` | `HumanPoseDataset` |
| Training losses | `lidar_bedlam/losses/smpl.py` | `FusionLoss` and the individual terms |
| Evaluation metrics | `lidar_bedlam/metrics/pose.py`, `metrics/detection.py` | MPJPE, PA-MPJPE, PVE, translation error, box AP/mAP |

Losses sit next to the model side (they consume predictions and batches and
are only used in training). Metrics are numpy, independent of the model, and
are shared by our evaluation and by the baseline runners, so they live in
their own package. Evaluation scripts that glue a model to the metrics will go
to `lidar_bedlam/evaluation/`.

## Conventions (every loader must obey)

- **Frame:** OpenCV camera frame of the image: x right, y down, z forward.
  Metres. Up is -y, so 3D boxes yaw about the y axis (`CAMERA_UP_AXIS = 1`).
- **Pixels:** (u, v) with (0, 0) at the centre of the top-left pixel.
- **SMPL params:** axis-angle, neutral model, 10 betas. `transl` is the SMPL
  translation in the camera frame; converting from a world/LiDAR frame uses
  `transform_smpl_params`, which accounts for SMPL rotating about the shaped
  rest pelvis (verified numerically to 1e-6 m).
- **3D box:** `[cx, cy, cz, dx, dy, dz, yaw]`; `dx` is along the heading,
  `dy` the height, `dz` across. For SMPL samples the box tightly encloses the
  posed mesh with the heading of the pelvis (+z of the SMPL rest pose rotated
  by the global orientation). Waymo boxes come from the labelled 3D box.
- **Joints:** `joints3d` (J, 3) with `joints3d_valid` and a convention name
  (`smpl24` from the model, `waymo15` from Waymo labels). COCO-17 joints can
  be regressed from vertices with `SmplModel.coco_joints` for cross-dataset
  comparison (Waymo's 13 labelled body joints are a subset of COCO-17).
- **Point cloud:** the person's LiDAR points only (N, 3), camera frame. The
  torch dataset samples exactly `n_points` with a validity mask.
- **Crop:** square window of `padding * max(bbox side)` around the 2D box,
  resized to `out_size`; the crop has its own intrinsics (`CropSpec.camera`).

## Per-dataset facts

| Dataset | Native frame | SMPL GT | Points | 2D box | Calibration source |
|---|---|---|---|---|---|
| SLOPER4D | world (z up) | `RGB_frames.smpl_pose` = optimised `opt_pose` | `human_points` (world) | labelled bbox | `RGB_info.intrinsics`, `dist`, per-frame `cam_pose` (world->cam) |
| LiDARHuman26M | LiDAR (x fwd, y left, z up) | json `pose`/`beta`/`trans` | segment PLY | whole crop; principal point shifted by `top_left` | fixed LiDARCap rig (in code) |
| Waymo (pose_complete_4) | vehicle | none (15 keypoints) | `lidar` per object | `bb_2d` | record `intrinsic`, `extrinsic` (cam->vehicle, Waymo axes) |
| BEDLAM | camera | SMPL labels (`bedlam-labels-smpl`): `pose_cam`, `shape[:10]`, `trans_cam + cam_ext[:3,3]`, matched to mask persons by silhouette overlap | depth back-projection of body+clothing mask | mask extent | `cam_int` of the label row (cx = W/2) |

Smoke-test results on real samples (2026-09-08): person points reproject
inside the 2D box for 100 % (SLOPER4D), 100 % (LiDARHuman26M), 97 % (Waymo),
99 % (BEDLAM); SMPL joints project inside the box; the SMPL pelvis lies within
0.2 m of the point-cloud median.

## Batch item keys (`HumanPoseDataset`)

`image` (3,S,S) ImageNet-normalised, `points` (N,3), `points_valid` (N),
`intrinsics` (3,3) of the crop, `crop_origin` (x0,y0,side), `has_smpl`,
`global_orient` (3), `body_pose` (69), `betas` (10), `transl` (3),
`joints3d` (24,3), `joints3d_valid` (24), `joint_convention`, `kp2d` (24,3)
crop pixels + confidence, `has_kp2d`, `box3d` (7), `has_box3d`, `mask` (S,S),
`has_mask`, `key`, `dataset`.

## Detection-style evaluation

`metrics.detection.box_detection_metrics` computes the exact 3D IoU between
the predicted and ground-truth box of every sample and reports AP at IoU
0.25 / 0.5 / 0.7 plus their mean (mAP). With one prediction per crop and no
confidence, AP equals the fraction of boxes above the threshold; pass model
confidences to get a real precision-recall curve. `translation_error` and
`error_by_distance` report the 3D placement error, also binned by distance.

## LiDAR simulation (`lidar_bedlam/lidar/`)

`simulate.simulate(depth_m, camera, spec, camera_from_sensor)` casts a
spinning-LiDAR scan against a rendered depth map by ray marching: each beam
is a ray from the sensor origin, sampled at 192 log-spaced ranges; the first
sample behind the depth surface is refined by linear interpolation. Returns
carry the beam index, azimuth, noisy range and hit pixel, so
`select_mask(scan, person_mask)` yields the person's returns. A hit is only
accepted when the previous sample was inside the image and in front of the
surface, so a sensor offset from the camera never invents geometry the
camera did not see.

Sensors are described like Ouster products: a family sets the vertical FOV
(OS0 90 deg, OS1 45 deg, OS2 22.5 deg) and the channel count sets the vertical
resolution (32 / 64 / 128; 256 is a synthetic ablation). Azimuth is sampled on
the sensor's global grid of ``horizontal_steps`` per revolution (512 / 1024 /
2048), so a camera with HFOV ``h`` covers ``h / 360 * steps`` columns; for
example 90 deg at 1024 steps = 256 columns (`azimuth_window` reports this).

| Preset | Channels | Vertical FOV | Steps / rev | Person returns at 10 m (BEDLAM sample) |
|---|---|---|---|---|
| OS1-32 | 32 | 45 deg | 1024 | 31 |
| OS1-64 | 64 | 45 deg | 1024 | 67 |
| OS1-128 | 128 | 45 deg | 1024 | 131 |
| OS1-256 (synthetic) | 256 | 45 deg | 1024 | 260 |
| OS0-128 / OS2-128 | 128 | 90 / 22.5 deg | 1024 | - |
| waymo64 | 64 | -17.6..2.4 deg | 2650 | 363 |

`sensor_pose(translation, pitch, yaw, roll)` gives the camera-from-sensor
pose for the extrinsic-shift ablation (e.g. 0.5 m above the camera).
Range noise (2 cm std) and dropout are part of `LidarSpec`.

`augment.occlude(points, OcclusionConfig, rng)` returns a keep mask after a
random half-space cut, box cut-out, height cut (legs or head occluded) and
subsampling, never dropping below a minimum point count.

## Model (`lidar_bedlam/models/`)

| Module | Role |
|---|---|
| `vit.py` | ViT backbone with ViTPose parameter names; `load_backbone_weights` loads the ViT-H weights from the TokenHMR/HMR2 checkpoints (0 missing / 0 unexpected keys). Positional embeddings are interpolated to the 256x256 crop grid. |
| `point_encoder.py` | `PointTokenizer`: Fourier-encoded points, farthest-point-sampled centres with kNN pooling, small transformer. Output tokens carry absolute 3D positions (the spatial cue). |
| `selective_attention.py` | `SelectiveDecoder`: 12 joint-group queries (root, torso, head, arms, hands, legs, feet, shape). Each layer cross-attends to image tokens and LiDAR tokens separately and mixes them with a per-query gate initialised from a modality prior: LiDAR for the 3D cues (root = centre, placement and global orientation; torso; shape), camera for the semantic cues (head and head orientation, arms, hands, legs, feet). Gates are returned per layer for logging and ablation. |
| `fusion.py` | `SelectiveFusionModel`: heads for 6D rotations per group, betas, and the root translation as (pelvis pixel offset, log depth) resolved through the crop intrinsics; differentiable SMPL layer gives vertices, joints, projected 2D joints and the 7-vector 3D box so every term of `FusionLoss` applies. |

Sizes with ViT-H: 660 M parameters, 29 M trainable with the backbone frozen.

## Notebook

`debug/capabilities.ipynb` (generated by `scripts/build_debug_notebook.py`,
executed copy `debug/capabilities.executed.ipynb` is git-ignored) shows five
BEDLAM frames as RGB / depth / SMPL placeholder, every person as a rotatable
3D surface, the OS1-64 scan on the image with the column window of each
camera, per-person returns against the surface, the OS1 32/64/128/256-channel
scans from two sensor viewpoints, the real
datasets with LiDAR-to-mesh distances, a training batch with its losses, and
the model forward pass with the gate matrix.

LiDAR-to-mesh distance (median over points) on the shown samples: SLOPER4D
1.3 cm, LiDARHuman26M 7.8 cm (5.4 cm mean over 21 samples; the IMU-based
ground truth of that dataset is noisier, the SMPL convention was verified
against the alternatives, which give more than 12 cm).

## Augmentation (`lidar/augment.py`, `data/image_augment.py`)

Point cloud, applied in `HumanPoseDataset` when `point_augment` is set
(`PointAugmentConfig`), in this order:

| Effect | Function | Models |
|---|---|---|
| object occlusion: half-space, box cut-out, legs/head cut, subsampling | `occlude`, `height_cut_fraction`, `axis_cut_fraction` | other people, cars, furniture, partial views |
| sensor cover (azimuth / channel bands) | `cover_mask`, `SensorCover` | dirt or objects on the sensor, mounting obstructions |
| channel dropout | `drop_channels` | dead or weak channels, low-reflectivity returns |
| jitter (isotropic) and range jitter (along the ray) | `jitter`, `range_jitter` | range noise of the sensor |
| outliers | `add_outliers` | dust, edge returns, blooming |
| calibration error | `miscalibrated_pose` | LiDAR-camera extrinsic error |
| resolution (channels x steps per revolution) | `ouster(family, channels, steps)` | 32/64/128/256 x 512/1024/2048 = 12 settings |

Image (`ImageAugmentConfig`): random erasing inside the person box, cover of
one image side, brightness / contrast / saturation, Gaussian blur, JPEG
re-encoding, and bbox jitter (detector noise; the crop follows the jittered
box, the labels do not move).

Not implemented on purpose for now: horizontal flip (needs a consistent
mirror of SMPL parameters, points and joints), motion distortion of the
scan, and intensity, since the simulator has no reflectivity.

The interactive notebook section 5b exposes all of these per sample and
person with a 4 x 3 resolution checkbox grid.

## Sensor rigs (`lidar_bedlam/rigs/`)

`SensorRig` is a star graph: a base link and one `Sensor` per LiDAR / camera /
radar with `base_from_sensor` (4x4, sensor pose in the base frame), optional
intrinsics, and `optical_from_sensor` (the dataset's camera axes to OpenCV).
`rig.tree_text()` prints the connection tree with positions and rotation
angles; `rigs.plot.rigs_figure([...])` draws up to three rigs side by side
with per-sensor xyz triads (red, green, blue), dotted links to the base, and
orange camera frustums from the intrinsics. Loaders:

| Dataset | Mount | Base frame | Source |
|---|---|---|---|
| Waymo Open (v2 parquet) | car | vehicle: x fwd, y left, z up, rear axle | `camera_calibration` + `lidar_calibration` of one segment; camera axes x fwd (Waymo) |
| nuScenes | car | ego: x fwd, y left, z up, rear axle | `calibrated_sensor.json` (quaternion w,x,y,z + translation, camera intrinsics) |
| SLOPER4D | helmet | head LiDAR: x fwd, y left, z up | `dataset_params.json` (`lidar2cam`, intrinsics) |
| AVA (own car) | car | export ego at the roof LiDAR | nuScenes-format export `nas_drive2/car_data/dataset/nuscenes_sample` (7 cams 2200x1200 + LIDAR_TOP) |
| FUSE-Bike (own bicycle) | bicycle | export ego at the top LiDAR | nuScenes-format export `bike_data/.../fusebike` (1 cam + 2 LiDARs + concatenated) |

Every extrinsic is checked to be a proper rigid transform on load. The
notebook section 9 has a checkbox table to show up to three rigs.
`scripts/export_rigs.py` writes every rig to the Obsidian vault
(`.vault-lidar-bedlam/rigs/<slug>.{png,html,glb,txt}`) and the note
`rigs.md` that embeds them; the GLB is a coloured rod mesh for 3D viewers.

## Ego motion and LiDAR rolling shutter (`lidar/motion.py`)

A spinning LiDAR at 10 Hz sweeps the camera window column by column (a
52 deg window takes 14 ms, 90 deg takes 25 ms). On a moving platform every
column is measured from a different position; uncompensated datasets hand
such distorted clouds to the model. `apply_rolling_shutter(scan, EgoMotion,
window)` distorts a static-scene scan accordingly: `p_reported = p_true -
v * (t - t_ref)` per column, with an optional yaw rate about the up axis; the
reference time is the window centre (the image time) by default.

Ego speed is a `SpeedSetting(mean_kmh, std_kmh)` in 10 km/h steps, sampled
per sample and clipped to `[0, max]`. Presets: `STATIONARY` (0 ± 0) and
`WAYMO_URBAN` (20 ± 20, max 70). The latter follows the ego-speed
distribution measured on 40 Waymo validation segments (7,866 frames):

| statistic | km/h |
|---|---|
| mean | 25.2 |
| std | 22.5 |
| median | 20.6 |
| p90 | 63.3 |
| stopped (< 5 km/h) | 25 % |

Recommendation for the Waymo transfer: train with `WAYMO_URBAN`; the clipped
normal reproduces the stop fraction and the 0-60 km/h spread. At 20 km/h the
shift across a 52 deg window is 8 cm, at 60 km/h 24 cm, on the order of the
body width, so it matters for placement.

## Generated shards (`generate/`)

`Record` holds one person: clean and augmented 256 px crops, mask crop,
crop intrinsics and origin, SMPL (camera frame), 24 joints, 2D joints, 3D
box, `distance_scale`, and named LiDAR scans padded to 2048 points with
channel and column ids: `main_0`, `main_1` (random Ouster setting, ball
r <= 1 m), `ball025` (ball r <= 0.25 m), `target_waymo` (Waymo 64 ch /
2650 steps, ball), `check` (OS1-64 @ 1024 co-located; used for the 30-point
visibility rule), `rig_waymo`, `rig_sloper4d`, `rig_fusebike` (fixed
calibrated offsets). Shards are `np.savez` files of 512 records
(`w<worker>_<n>.npz`) with a `stats.json` (frames, rejects, per group).

Virtual distance (`lidar/distance.py`): on 30 % of frames the camera is
moved back by `d = (k - 1) z_median`, `k ~ U(1.5, 3)`; depth and masks are
re-rendered by forward splatting with a z-buffer, labels shift by
`(0, 0, d)`, the crop is taken from the original window and its detail
reduced by the shrink factor `z / (z + d)`. Rolling shutter is off in the
main data (Waymo points are motion-compensated); `--speed-mean/--speed-std`
enable it.
