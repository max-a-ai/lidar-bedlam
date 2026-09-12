# The story (red thread), decided 2026-09-10

**Claim.** Synthetic data, used correctly and enriched with LiDAR rendered
from BEDLAM's depth, improves in-the-wild SMPL pose, shape *and* 3D
placement. The model that turns that data into the gain is selective
attention fusion: the camera decides the semantics (head, arms, hands,
legs, feet), the LiDAR decides the metric quantities (body centre,
placement, global orientation, shape).

**Why this is a story and not three papers.** Real LiDAR + camera data
with SMPL labels barely exists (Waymo has keypoints only, SLOPER4D one
head rig). A camera-only estimator cannot recover absolute depth; a
LiDAR-only estimator gets sparse, semantics-free returns. So: C1 makes the
paired data (from depth), C2 is the model that needs it, C3 shows that
C1 + C2 beat what is available today, and every ablation removes exactly
one addition so the reader sees what each one buys.

## The ladder (each rung = one addition, read top to bottom)

| Rung | What is added | Config | Expect to see |
|---|---|---|---|
| 0 | camera only, same training data | `ablation_image_only` (+ TokenHMR / CameraHMR as external camera baselines) | good PA-MPJPE, poor translation and box AP |
| 1 | + LiDAR stream, no gate (uniform sum) | `ablation_gate_none` (+ LiDAR-HMR as external LiDAR baseline, `ablation_lidar_only` as the sensor-only bound) | placement improves, pose stalls or degrades |
| 2 | + selective routing with fixed priors | `ablation_gate_hard` | pose back to camera quality, placement kept |
| 3 | + learned gates (ours) | `ablation_mixed_short` / `main_mixed` | best on both axes; gate maps show the learned routing |

Rungs 0-3 are the **model axis** (Sec. 6.1 / 6.2). They all train on the
same mixture, so the only variable is fusion.

| Rung | What is added | Config | Expect to see |
|---|---|---|---|
| A | real data only (Waymo + SLOPER4D) | `real_only` | the "what you can do without us" row |
| B | + synthetic (our mixture) | `main_mixed` | gain on every metric, largest on placement by distance |
| C | synthetic only (zero-shot) | `synth_only` | transfer without any real label |
| D | scaling the synthetic pool 2x to 32x the Waymo count | `ablation_scale_{2,4,8,16,32}x` | monotone gain, the "used correctly" data curve |

Rungs A-D are the **data axis** (main table, Sec. 5). B is the headline.

| Rung | What "used correctly" means | Config | Expect to see |
|---|---|---|---|
| E | LiDAR placed in a 1 m ball around the camera (ours) vs 0.25 m vs fixed to the Waymo rig | `ablation_ball025`, `ablation_rig_waymo` | rig-fixed wins only on its own rig; the ball transfers to SLOPER4D too |
| F | all 12 resolutions mixed (ours) vs the target's only | `ablation_target_waymo` | mixing costs nothing on Waymo and transfers to SLOPER4D |
| G (after hand-in) | augmentations, occlusions, rolling shutter | `.docs/ablations.md` 6.5 | realism items |

Rungs E-G are the **synthesis axis** (Sec. 6.3 / 6.4): the choices that make
synthetic LiDAR transfer to real sensors.

## Metrics tie the rungs together

- pose and shape: MPJPE, PA-MPJPE (Waymo 13 shared COCO joints; SMPL-24 on
  SLOPER4D), PVE where a mesh exists
- positioning: translation error by distance bin, 3D box AP@0.25/0.5/0.7
  and mAP with the IoU-confidence head, i.e. the detector-style view
- gates: mean image gate per joint group per layer, one figure

## Order of runs on Helma

1. `main_mixed` (headline, needed by everything)
2. `real_only`, `synth_only` (main table)
3. `ablation_mixed_short`, `ablation_image_only`, `ablation_gate_none`,
   `ablation_gate_hard`, `ablation_lidar_only` (model axis)
4. `ablation_ball025`, `ablation_rig_waymo`, `ablation_target_waymo`
   (synthesis axis)
5. `ablation_scale_*` (data curve, cheapest to drop if time is short)
6. external baselines on the same protocol (no training)

## Paper skeleton mapped to the rungs

1 Introduction (claim) · 2 Related work · 3 Data: BEDLAM depth to LiDAR,
placement ball, resolutions, augmentations (C1) · 4 Model: selective
attention (C2) · 5 Experiments: main table A-C, baselines, per-distance
placement (C3) · 6 Ablations: model axis 0-3, synthesis axis E-F, data
curve D · 7 Limitations (Waymo keypoints not meshes, one rig on
SLOPER4D) · 8 Conclusion.
