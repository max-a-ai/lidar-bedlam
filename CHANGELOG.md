# Change log (newest first)

Every implementation step, one entry each, so the history can be read
back to front. Dates are when the step was done; commit hashes refer to
`git log`.

## 2026-09-10

- WP1.4-1.6 infrastructure: `scripts/sam3_waymo_masks.py` (SAM 3 box-prompted person masks for the 5,560 Waymo crops, runs in the sam3 conda env, masks verified visually; full run started), `generate/real.py` + `scripts/generate_real.py` (Waymo and SLOPER4D records in the shard format, mask-selected points), `data/shards.py` (`ShardDataset`: scan-variant selection, optional precomputed tokens, on-the-fly point augmentation), `scripts/precompute_tokens.py` (frozen ViT-H tokens per shard, fp16). Background: synthetic v1 of the first group generating, SLOPER4D shards building. 1 test.
- WP1.3 done: `generate/records.py` (fixed-shape `Record`, npz shards with named scan variants, `Shard` reader), `lidar/placement.py` (ball placement with radius uniform in [0, r], rig-derived LiDAR poses in simulator axes), `lidar/distance.py` (virtual distance by moving the camera back with z-buffer re-rendering of depth and masks; the first attempt scaled the scene about the camera, which keeps angles and was wrong), `generate/synth.py` (per frame: 8 scan variants shared by all persons, visibility rule 90x35 px + 30 returns, augmented copy, distance augmentation on 30 % of frames) and `scripts/generate_synthetic.py` (multiprocess). Trial: 24 frames -> 115-190 records in 14 s on 4 workers; shard samples checked visually. 5 tests.
- WP1.1 done: `data/bedlam_labels.py` reads the BEDLAM SMPL labels (translation = `trans_cam + cam_ext[:3,3]`, verified 85 % of mesh vertices inside the masks vs 0 % for the alternatives) and matches label rows to mask persons by projected-silhouette overlap (98 % of labels matched, 96 % of mask persons labelled on the finished group); `BedlamFramesSource(labels_dir=..., smpl_model=...)` attaches them; 2 tests; overlays checked visually. Started the background extraction of the 11 remaining paper groups (`scripts/extract_bedlam_groups.sh`).
- Wrote `docs/plan.md` after the design interview and this change log; started WP1.1 (BEDLAM SMPL label attachment). (30c4b90)
- Added `scripts/verify_notebook.py` (executes the notebook, fails on cell or widget errors) and the mypy override for it. (a389203, 0d0ddf5)
- Added sensor rigs: `rigs/` schema, loaders for Waymo, nuScenes, SLOPER4D, AVA and FUSE-Bike (nuScenes exports), plotly figures, vault export (PNG, HTML, GLB, tree text) into `.vault-lidar-bedlam/rigs/` with `rigs.md`; `lidar/motion.py` with `SpeedSetting` (10 km/h steps, Waymo-measured preset 20 +- 20 km/h) and `apply_rolling_shutter`; `docs/ablations.md`; EG 2027 LaTeX skeleton in `paper/`; routing prior moved legs and arms to the camera. (d46f545)
- Added sensor and image augmentations (`lidar/augment.py`: cover, channel dropout, jitter, outliers, miscalibration; `data/image_augment.py`: erasing, cover, colour, blur, JPEG, bbox jitter), wired into `HumanPoseDataset`; interactive notebook section 5b with the 4 x 3 resolution grid. (e42366f)

## 2026-09-09

- LiDAR presets renamed to Ouster families (OS0/OS1/OS2, channels x steps per revolution); `azimuth_window` maps the camera HFOV to scan columns. (70fdc89)
- Selective-attention fusion model (`models/`: ViT with TokenHMR ViT-H weights, point tokenizer, gated decoder, SMPL heads, differentiable SMPL, 2D projection, 3D box), `viz.py`, and the generated `debug/capabilities.ipynb`. (8846d44)
- LiDAR simulator by ray marching against the BEDLAM depth, with sensor offsets, noise, dropout; occlusion augmentation. (67b3a2f)
- Located the BEDLAM SMPL / SMPL-X training labels (project server only) and added `scripts/fetch_bedlam_labels.sh`. (42a917c)

## 2026-09-08

- Data pipeline: loaders for SLOPER4D, LiDARHuman26M, Waymo (pose_complete_4) and BEDLAM raw frames sharing one camera-frame schema; torch dataset; losses; pose and 3D-box metrics; 21 tests. (2c8706c)
- Repository scaffold (uv, hatchling, ruff, mypy strict, vault), five baseline submodules, BEDLAM archive audit, dataset survey, project document. (c9beeab)
