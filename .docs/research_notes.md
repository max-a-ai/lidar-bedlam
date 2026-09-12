# Research notes (web, 2026-09-08)

Collected by a background research pass; every number carries its source.

## BEDLAM v1 (Black et al., CVPR 2023)

Site https://bedlam.is.tuebingen.mpg.de/ · paper https://arxiv.org/abs/2306.16940 ·
code https://github.com/pixelite1201/BEDLAM · render tools
https://github.com/PerceivingSystems/bedlam_render

| Item | Value | Source |
|---|---|---|
| Sequences / fps / res | 10,450 sequences, 30 fps, 1280x720 | site, HF mirror |
| Images | 1.6 M PNG | https://huggingface.co/datasets/Intelligent-Systems/BEDLAM |
| Subjects | 271 bodies, 111 outfits, 1,691 textures, 27 hairstyles, 2,311 AMASS motions, 95 HDRI + 8 scenes, 1-10 people | paper |
| PNG / MP4 / masks / GT | 2.2 TB / 20 GB / 30 GB / 100 MB | HF |
| Depth | 3.8 TB (mirror 4.09 TB), 32-bit EXR | https://huggingface.co/datasets/Intelligent-Systems/BEDLAM-depth |
| Depth semantics | verified locally: planar z-depth, cm (see bedlam_audit.md) | this repo |
| Cameras | HFOV 52/65 or zoom 65->25; static + orbit | paper |
| Coordinates | Unreal, cm, left-handed, Z up | https://github.com/PerceivingSystems/bedlam_render/blob/main/unreal/render/unreal_coordinate_system.md |
| License | non-commercial research only, no redistribution | https://bedlam.is.tuebingen.mpg.de/license.html |

## BEDLAM 2.0 (Tesch et al., NeurIPS 2025 D&B) — skipped

Site https://bedlam2.is.tuebingen.mpg.de/ · paper https://arxiv.org/abs/2511.14394 ·
render tools https://github.com/PerceivingSystems/bedlam2_render ·
HF https://huggingface.co/datasets/Intelligent-Systems/BEDLAM2 and
https://huggingface.co/datasets/Intelligent-Systems/BEDLAM2-depth

| Item | Value |
|---|---|
| Sequences | 27,480, 74.5 h, 30 fps, 1280x720, 8.05 M frames, 13.3 M bboxes |
| New | 1,615 bodies (BMI 18-41), 4,643 motions, 187 outfits, 40 hairstyles, 15 3D envs, focal 14-400 mm, moving cameras incl. 13.6 % real captured trajectories, ARCTIC hand motions, no facial motion |
| PNG / MP4 / GT | 11 TB / 160 GB / 4 GB |
| Depth | 16-bit EXR, only 44 % of frames, 15 TB (mirror 16.7 TB); channel `FinalImageMovieRenderQueue_WorldDepth.R`; Cryptomatte masks + camera JSON in EXR metadata |
| Standalone masks | not found |
| Total | ~29 TB |
| License | same as v1 |

## Baselines

| Method | Repo | Paper | License | Weights | Deps | In | Out |
|---|---|---|---|---|---|---|---|
| CameraHMR | https://github.com/pixelite1201/CameraHMR | Patel & Black, 3DV 2025 | MPI non-commercial | fetch_demo_data.sh + registration | SMPL, Detectron2, ViTPose | RGB | SMPL |
| TokenHMR | https://github.com/saidwivedi/TokenHMR | Dwivedi et al., CVPR 2024 | non-commercial | registration | SMPL/SMPL-H, Detectron2, Python <=3.10 | RGB | SMPL |
| LiDAR-HMR | https://github.com/soullessrobot/LiDAR-HMR | Fan et al., arXiv 2311.11971, IEEE TMM 2025 | none stated | Baidu pan (pwd yfsg) | SMPL-X, Point Transformer V2 | LiDAR | SMPL-X; also ships Waymo-v2 mesh pseudo-GT |
| SAM 3D Body | https://github.com/facebookresearch/sam-3d-body | Yang et al., arXiv 2602.15989 | SAM license (permissive) | HF gated: facebook/sam-3d-body-dinov3, -vith | MHR body model, ViTDet or SAM3 | RGB | MHR mesh + keypoints (not SMPL-X) |
| LIF-Net | gitlab lrz iv/publications/lif | Buettner et al., IV 2025 | own | ~/nas_drive/methods/lif/evals | TokenHMR/HMR2 + PointNet++ | RGB + LiDAR | SMPL |

Other LiDAR+camera human methods: FusionPose/LiCamPose
(https://github.com/4DVLab/FusionPose, keypoints only), SMPLify-3D
(https://github.com/guidodumont/SMPLify-3D, image init + LiDAR ICP),
FreeCap (AAAI 2025, no code), Sen-Cap (ECCV 2026, no code), HUM3DIL
(CoRL 2022, no code). Survey: https://arxiv.org/abs/2509.12197 with
https://github.com/valeoai/3D-Human-Pose-Shape-Estimation-from-LiDAR.

## Real LiDAR + camera + human datasets

| Dataset | On NAS | Content | GT | Calib | Access |
|---|---|---|---|---|---|
| Waymo Open Perception | yes | 2,030 segments, 5 LiDAR + 5 cams, 10 Hz; 200 K 2D and 10 K 3D keypoint object-frames (v1.3.2+) | 14 keypoints; SMPL-X fits only via LiDAR-HMR pseudo-GT | yes | Google login, non-commercial |
| SLOPER4D | 6/15 seqs | 15 seqs, 12 subjects, 100 K+ LiDAR frames @20 Hz, OS1-128 + action cam | SMPL + trajectory + scene | yes | lidarhumanmotion.net, CC BY-NC-SA 4.0 |
| LiDARHuman26M | yes | 184 K frames, 13 subjects, 20 motions, 12-28 m | SMPL from IMU mocap | in toolkit | lidarhumanmotion.net |
| Human-M3 | incomplete | 3 outdoor scenes, 12.2 K frames, 4 cam-LiDAR pairs, multi-person | 3D pose / SMPL fits | multi-view | http://ivg.au.tsinghua.edu.cn/dataset/Human_M3/Human_M3.html |
| FreeMotion (LiveHPS) | yes (raw) | 578 K frames, 1-7 performers, 3 LiDAR + cams + IMU | SMPL | yes | https://github.com/4DVLab/LiveHPS |
| RELI11D | yes | 48 seqs, 3.3 h, LiDAR + IMU + RGB + event | SMPL | in code | lidarhumanmotion.net |
| HSC4D | yes | 3 scenes, LiDAR + IMU, no camera | SMPL/bvh | n/a | lidarhumanmotion.net |
| LaserHuman | yes (raw) | 11 scenes, text-to-motion | SMPL | ? | https://github.com/4DVLab/LaserHuman |
| CIMI4D | no | climbing, 180 K frames | SMPL | ? | lidarhumanmotion.net |
| HmPEAR | no | 300 K frames, 25 subjects, 40 actions | mocap 3D pose | ? | lidarhumanmotion.net |
| MMHU | yes | 1.73 M video frames, motion/intent labels | pseudo SMPL | n/a | HF, CC BY 4.0 |
