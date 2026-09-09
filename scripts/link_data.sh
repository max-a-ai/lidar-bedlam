#!/bin/bash
# Creates the (git-ignored) data/ folder with symlinks to the datasets this
# project uses. Re-run anytime; existing links are replaced.
set -euo pipefail
cd "$(dirname "$0")/.."
NAS=/home/max/nas_drive/publicdatasets
mkdir -p data
link() { ln -sfn "$1" "data/$2"; }
link "$NAS/bedlam"                                   bedlam            # synthetic source (archives)
link "$NAS/SLOPER4D"                                 sloper4d          # real: LiDAR + RGB + SMPL + calib
link "$NAS/LiDARHuman26M"                            lidarhuman26m     # real: images + human LiDAR + SMPL
link "$NAS/WaymoPerception"                          waymo_perception  # real: v2 parquet (images, lidar, calib, keypoints)
link "$NAS/waymo_extracted"                          waymo_extracted   # real: per-object GT keypoints + pedestrian LiDAR
link "$NAS/nuscenes"                                  nuscenes          # AD: calibration tables + samples
link "$NAS/FreeMotion"                               freemotion        # real, raw archives
link "$NAS/RELI11D_Dataset"                          reli11d           # real, pkl format
link /home/max/nas_drive2/car_data/dataset/nuscenes_sample     ava               # own car AVA (nuScenes export, sample)
link /home/max/nas_drive/bike_data/v2b/2_processed_rosbags/nusc/elisabeth/fusebike fusebike  # own bicycle FUSE-Bike (nuScenes export)
link /home/max/nas_drive/methods/max/data/body_models body_models      # SMPL / SMPL-X model files (never commit)
link /mnt/md0/lidar-bedlam                           generated         # our generated synthetic dataset
mkdir -p /mnt/md0/lidar-bedlam
ls -la data
