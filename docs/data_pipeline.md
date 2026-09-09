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
| BEDLAM | camera (Unreal) | pending body-data download | depth back-projection of body+clothing mask | mask extent | `hfov` from camera CSV |

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

| Preset | Beams | Vertical FOV | Azimuth res | Person returns at 10 m (BEDLAM sample) |
|---|---|---|---|---|
| os32 | 32 | 45 deg | 0.35 deg | 31 |
| os64 | 64 | 45 deg | 0.35 deg | 67 |
| os128 | 128 | 45 deg | 0.35 deg | 131 |
| os256 | 256 | 45 deg | 0.175 deg | 517 |
| waymo64 | 64 | -17.6..2.4 deg | 0.14 deg | 363 |

`sensor_pose(translation, pitch, yaw, roll)` gives the camera-from-sensor
pose for the extrinsic-shift ablation (e.g. 0.5 m above the camera).
Range noise (2 cm std) and dropout are part of `LidarSpec`.

`augment.occlude(points, OcclusionConfig, rng)` returns a keep mask after a
random half-space cut, box cut-out, height cut (legs or head occluded) and
subsampling, never dropping below a minimum point count.
