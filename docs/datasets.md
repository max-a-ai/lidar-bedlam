# Datasets

Survey date: 2026-09-08. Root of the NAS share: `/home/max/nas_drive/publicdatasets`
(sshfs, slow; do not `du` it). Local RAID: `/mnt/md0` (15 TB, 8.2 TB free).

## Synthetic training source

### BEDLAM v1 (on NAS, complete)

Path: `publicdatasets/bedlam`. 30 sequence groups, 390 archives, all names match
the official `b0_checksums_all.xxh128`; every tar ends with a valid EOF block and
every `*_gt.tar.gz` decompresses. Full xxh128 validation still has to be run on
the NAS itself (see `docs/bedlam_audit.md`).

| Modality | On NAS | Published |
|---|---|---|
| depth (EXR, one `Depth` FLOAT channel, cm) | 4.09 TB | 3.8 TB |
| png (1280x720 RGBA) | 2.39 TB | 2.2 TB |
| masks (per person: `_body.png`, `_clothing.png`, ...) | 33 GB | 30 GB |
| mp4 | 22 GB | 20 GB |
| gt (camera CSV + `be_seq.csv`) | 21 MB | 100 MB |

Missing on the NAS and needed for SMPL-X labels: the BEDLAM **body data**
download (SMPL-X animation files / neutral ground-truth motion info), which the
image `gt` tars only reference by body name (e.g. `rp_henry_posed_001_1084`).

### BEDLAM 2.0 (not on NAS, decision: skip)

29 TB in total (11 TB PNG, 15 TB 16-bit depth for only 44 % of frames, 164 GB
MP4+GT). Skipped for the conference submission on 2026-09-08.
Site: https://bedlam2.is.tuebingen.mpg.de/ , paper: https://arxiv.org/abs/2511.14394

## Real datasets with LiDAR + camera + human ground truth

Ranked by usefulness for validation / finetuning of this project.

| Rank | Dataset | NAS path | LiDAR | RGB | GT | LiDAR-cam calib | State |
|---|---|---|---|---|---|---|---|
| 1 | SLOPER4D | `SLOPER4D` | Ouster OS1-128 pcd | yes | SMPL per frame | `dataset_params.json` (`lidar2cam`, K) | only 6 of 15 sequences present |
| 2 | LiDARHuman26M | `LiDARHuman26M` | human segments (ply) | yes | SMPL (72 pose, betas, trans) | not on disk (toolkit) | extracted; several symlinks dangling |
| 3 | Waymo Open Perception | `WaymoPerception`, `waymo_extracted` | 5 LiDAR, parquet | 5 cams | 3D keypoints (14 kp) only | yes | complete; `smpl_pose`, `pkl-pseudoGT`, `lidar_hmr_results` are model pseudo-labels, not GT |
| 4 | FreeMotion (LiveHPS) | `FreeMotion` | 3 LiDAR | multi-cam | SMPL | unverified | raw archives (58 GB images tar) |
| 5 | RELI11D | `RELI11D_Dataset` | 512-pt human crops only | yes | SMPL | none on disk | pkl format tied to its own pipeline |
| 6 | Human-M3 | `human-m3` | yes | multi-cam | SMPL | unknown | INCOMPLETE: final zip part missing |
| - | HSC4D | `HSC4D` | raw pcap | no | bvh only | n/a | not usable (no images) |
| - | CODa | `CODa_full` | yes | yes | 3D boxes only | yes | no pose GT |
| - | LaserHuman | `laserhuman` | yes | unclear | SMPL | unknown | raw zips, text-to-motion oriented |

Camera-only sets present (no LiDAR): 3DPW, EMDB, AGORA, 4D-Humans-training, Human3.6M,
MPI-INF-3DHP, MMHU. Motion-only: AMASS, HumanML3D.

Not on the NAS but relevant: CIMI4D, HmPEAR (both lidarhumanmotion.net), Human-M3
final part.

### Waymo detail

- `WaymoPerception/{training,validation,testing}`: Waymo v2 parquet with
  `lidar`, `lidar_hkp` (3D keypoints), `camera_hkp`, `camera_image`,
  `camera_calibration`, `lidar_calibration`, `lidar_camera_projection`.
- `waymo_extracted/{training,validation}`: per-segment pkls with GT 3D
  keypoints + cropped pedestrian LiDAR (669 / 168 non-empty).
- `WaymoPerception/waymo_pose_complete_4`: crops used by the previous `lif`
  model (2D 15953, 3D 739, 3D+2D 527) plus HMR2 pseudo-GT pkls.
- A local copy of parts of WaymoPerception is at `/mnt/md0/WaymoPerception` (1.7 TB).
- `waymo_pose_complete.tar.xz` (967 GB) and `waymo_pose.tar.gz` (490 GB) duplicate
  extracted folders and could be deleted to free 1.4 TB on the NAS.

## Body models

No SMPL / SMPL-X model files exist under `publicdatasets`. Use
`/home/max/nas_drive/methods/max/data/body_models` (smpl, smplh, smplx,
`smpl_mean_params.npz`, `J_regressor_*`, `smplx2smpl.pkl`) or the local
copy in `~/Documents/ILfusion/data/body_models`. Never commit them.

## Baselines

| Method | Submodule | Input | Output | Weights |
|---|---|---|---|---|
| CameraHMR (3DV 2025) | `third_party/CameraHMR` | RGB | SMPL | registration at camerahmr.is.tue.mpg.de |
| TokenHMR (CVPR 2024) | `third_party/TokenHMR` | RGB | SMPL | registration at tokenhmr.is.tue.mpg.de; local copy in `~/nas_drive/methods/max/data/checkpoints/tokenhmr` |
| LiDAR-HMR (TMM 2025) | `third_party/LiDAR-HMR` | LiDAR | SMPL-X | Baidu pan only; local checkout `~/Documents/LiDAR-HMR` |
| SAM 3D Body (Meta 2026) | `third_party/sam-3d-body` | RGB | MHR (not SMPL) | HF gated, already in `~/.cache/huggingface/hub` |
| LIF-Net (IV 2025, ours) | `third_party/lif` | RGB + LiDAR | SMPL | `~/nas_drive/methods/lif/evals` |

Full web research notes with sources are in `docs/research_notes.md`.
