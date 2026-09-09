# Experiment and ablation plan (decided 2026-09-10)

Hardware: one RTX 4090, frozen ViT-H backbone, 29 M trainable parameters.
Schedule definition: the **main** runs use the full schedule (all synthetic
samples, full epoch count); every **ablation** run uses one third of the
main schedule, identical for all variants of a study so they stay
comparable. Example: main = 20 epochs over the full synthetic set,
ablation = 7 epochs over the same set with the same learning-rate schedule.

## Main table (contribution 3)

| Training data | Eval | Metrics |
|---|---|---|
| synthetic only | SLOPER4D, LiDARHuman26M, Waymo | MPJPE, PA-MPJPE, PVE, translation error (by distance), 3D box AP@0.25/0.5/0.7, mAP |
| real only | same | same |
| synthetic + real (weighted) | same | same |

Baselines on the same protocol: CameraHMR, TokenHMR, LiDAR-HMR, SAM 3D Body
(joint-level via MHR), LIF-Net.

## Ablations before hand-in

| Study | Variants (main = ours) | Read as | Extra runs |
|---|---|---|---|
| 6.1 / 6.2 Selective attention | learned gates (ours), no gate (uniform sum), hard routing (priors frozen), image only, LiDAR only | 6.1 pose accuracy (MPJPE, PA-MPJPE); 6.2 LiDAR impact on placement (translation error, box AP / mAP) | 4 |
| 6.3 Sensor extrinsics | LiDAR fixed to the target rig (Waymo rig / SLOPER4D rig), random ball r = 0.25 m, random ball r = 1.0 m (camera as is) | transfer on Waymo and SLOPER4D; how much placement variance is needed | 2 |
| 6.4 LiDAR resolution | all 12 settings (4 channel counts x 3 step counts, ours) vs only the target's setting (Waymo 64 ch / 2650 cols; SLOPER4D OS1-128 / 1024) | transfer of resolution mixing | 1 |

Routing prior used by the gates: LiDAR for body centre, absolute placement,
global orientation and shape; camera for head and head orientation, arms,
hands, legs and feet.

## Ablations after hand-in

| Study | Variants | Extra runs |
|---|---|---|
| 6.5 Realism | no augmentation, no occlusion (point and image) | 2 |

Rolling shutter with ego speed stays a data-generation option
(`lidar/motion.py`, Waymo-measured preset 20 +- 20 km/h) but is not an
ablation: Waymo's top-LiDAR points are motion-compensated by the official
tooling, so the effect cannot show on the primary target.

## Run budget

3 main + 7 ablation runs before hand-in; with ablations at one third of the
main schedule this is roughly 3 + 7/3 = 5.3 main-run equivalents.
