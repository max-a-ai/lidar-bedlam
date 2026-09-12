# Plan to the paper (decided 2026-09-10, target Eurographics 2027)

Deadlines: abstract 25 Sep 2026, paper 1 Oct 2026 (full paper, 10 pages).
Headline: synthetic LiDAR-camera data from BEDLAM improves 3D placement and
pose of pedestrians on Waymo; SLOPER4D gives the additional full-SMPL proof.

The narrative and the run ladder live in `.docs/story.md`.

## 1. Decisions

| Topic | Decision |
|---|---|
| Primary benchmark | Waymo Open (5,559 object-frames with 3D+2D keypoints, LiDAR, camera): train 4,654, val 905 for selection and reporting |
| Secondary | SLOPER4D 4 sequences train, 2 test (full SMPL); LiDARHuman26M dropped (bad calibration), PedX skipped |
| Extra Waymo supervision | 5,006 LiDAR-only samples (no image) trained with the image branch masked, after the main pipeline works |
| Synthetic source | BEDLAM v1, 12 groups favouring outdoor scenes, 6 fps; pools of 2x, 4x, 8x, 16x, 32x the Waymo train count (9k to 150k persons); 32x is the main model |
| Person points | BEDLAM masks in synthesis; SAM 3 box-prompted masks on Waymo crops, LiDAR points kept inside the mask (box fallback); SLOPER4D ships segmented points; box-with-clutter stays an augmentation |
| Visibility | box at least 90 px high and 35 px wide (WayMoCo rule); synthetic persons also need at least 30 returns at 64 channels / 1024 steps; rejected fraction reported |
| LiDAR placement (main) | camera as is; LiDAR at a radius uniform in [0, 1] m, random direction, tilt up to 5 deg; ablation variants: fixed target rig (Waymo, SLOPER4D), ball 0.25 m |
| LiDAR resolution (main) | one of the 12 Ouster settings (32/64/128/256 channels x 512/1024/2048 steps) per scan; ablation: target setting only (Waymo 64 ch / 2650 steps, SLOPER4D OS1-128 / 1024) |
| Realism | point and image augmentation, occlusions, channel dropout, jitter, outliers, miscalibration; rolling shutter with ego speed is a generation option, not an ablation |
| Precompute | everything: crops, fp16 ViT-H tokens (256 x 1280 per crop, one clean and one augmented copy), masks, labels, and several LiDAR scans per sample (main x2, fixed-rig, ball-0.25, target-resolution, FUSE-Bike rig) |
| Backbone | ViT-H frozen (TokenHMR weights), so training touches only the 29 M-parameter fusion part; LoRA after hand-in |
| Model | selective decoder, gates with priors (LiDAR: root, torso, shape; camera: head, arms, hands, legs, feet), SMPL heads, IoU-confidence head for AP/mAP |
| Mixture | per-batch ratio 50 % real (Waymo 40 %, SLOPER4D 10 %), 50 % synthetic |
| Protocol | COCO-17 joints regressed from the mesh, 13 shared Waymo joints; MPJPE, PA-MPJPE, translation error by distance, 3D box AP@0.25/0.5/0.7 and mAP with the IoU confidence; PVE on SLOPER4D |
| Baselines | TokenHMR (on disk), SAM 3D Body (on disk), CameraHMR (download pending), LiDAR-HMR (checkpoint pending on the NAS); LIF-Net dropped |
| Cluster | Slurm, 4x H100 per node, 24 h wall time, up to ~10 nodes; torchrun DDP; auto checkpoint, auto resume, auto resubmit until max epochs; wandb entity `erik_hm`, project `lidar-bedlam`, enumerated run names |
| Small model | DINOv2 ViT-S/14 student at 224 px, distilled from the ViT-H model (outputs + joint-group features); ONNX -> TensorRT FP16 for the Jetson AGX Orin on FUSE-Bike; one qualitative figure from the bike |

## 2. Work packages

### WP1 Data (Sep 10-13)

1. `data/bedlam.py`: attach the SMPL labels (`bedlam-labels-smpl`, per-image `pose_cam`, `shape`, `trans_cam`, `cam_int`) to the extracted frames; match label rows to mask persons by projected pelvis inside the mask; verify by projecting joints onto the png (notebook SMPL column fills in).
2. Extract the 12 groups at 6 fps (`lidar_bedlam/scripts/extract_bedlam.py`), roughly 120 GB.
3. `generate.py`: per person sample -> shard record with crop, mask crop, K_crop, SMPL (camera frame), joints, box3d, and the LiDAR scan set listed above (points, channel, column, azimuth); visibility filter; statistics json. Sharded npz files of 512 samples.
4. Waymo: run SAM 3 over the 5,559 crops (GT box prompt), store masks and mask-selected points; build the same shard records; also the 5,006 LiDAR-only samples with `has_image = False`.
5. SLOPER4D: shard records from the loader (4 train / 2 test sequences).
6. `precompute_tokens.py`: frozen ViT-H tokens fp16 for every crop (clean + augmented copy), stored next to the shards; ~0.65 MB per crop.
7. Rsync shards to the cluster scratch.

### WP2 Training and evaluation (Sep 13-15)

1. `lidar_bedlam/scripts/lidar-bedlam-main.py` (torchrun): shard dataset, per-batch mixture sampler, AdamW, cosine schedule, AMP; wandb logging per epoch (`train/*`, `val/*`), checkpoint every epoch (`last.pt`, `best.pt`), resume from `last.pt`, unique run names `<experiment>-<NNN>`.
2. `lidar_bedlam/slurm/train.sbatch`: 4x H100, `--signal=B:USR1@900` trap that checkpoints and resubmits the same job with `--dependency=afterany` until `max_epochs` is reached; job chain id in the run name.
3. `lidar_bedlam/scripts/lidar-bedlam-eval.py`: metrics over Waymo val and SLOPER4D test from a checkpoint; writes json + LaTeX rows.
4. Baseline runners under `lidar_bedlam/baselines/`: TokenHMR, CameraHMR (image-only), LiDAR-HMR (LiDAR-only), SAM 3D Body (MHR joints mapped to COCO-17); same protocol.
5. Smoke: 1 k synthetic samples, 100 steps, on the 4090.

### WP3 Runs (Sep 15-28)

| Group | Runs | Steps | When |
|---|---|---|---|
| Main: synthetic only, real only, mixed | 3 | full | Sep 15-19 |
| Gates: no gate, hard routing, image only, LiDAR only | 4 | 1/3 | Sep 17-21 |
| Scaling: 2x, 4x, 8x, 16x (32x = main) | 4 | 1/3 | Sep 20-24 |
| Extrinsics: fixed rig, ball 0.25 | 2 | 1/3 | Sep 22-26 |
| Resolution: target setting only | 1 | 1/3 | Sep 24-26 |
| Baselines | 4 evals | - | Sep 16-20 |

After hand-in: realism (no augmentation, no occlusion), LoRA, Waymo-3DSkelMo
pseudo-labels vs synthetic, small-model distillation and Jetson benchmark.

### WP4 Paper (Sep 22 - Oct 1)

Tables and figures generated from the evaluation json by a script; method
and experiment sections drafted from the logs; introduction and conclusion
by the author; abstract 25 Sep.

## 3. Risks

- CameraHMR and LiDAR-HMR checkpoints arrive late: the table ships with
  TokenHMR and SAM 3D Body and the others are added in the rebuttal.
- Token shards: 32x pool x 2 copies is about 200 GB; the cluster copy must
  start as soon as WP1.6 finishes.
- SLOPER4D test is two sequences; the other nine are not on the NAS.
- Label-to-mask matching in BEDLAM must be verified on a sample of frames
  before generation starts.
