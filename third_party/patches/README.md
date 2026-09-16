# Local patches to vendored submodules

The submodules point at their upstream repositories, so local edits are
kept here as patches instead of commits the upstream does not have.

- `lidar-hmr-pointnet2-setup.patch` — `third_party/LiDAR-HMR`: build fix
  for the pointnet2 CUDA extension (`models/LiDARCap/pointnet2_ops_lib/setup.py`)
  used by `lidar_bedlam/scripts/baselines/run_lidar_hmr.py`. Apply with
  `git -C third_party/LiDAR-HMR apply ../patches/lidar-hmr-pointnet2-setup.patch`.

`third_party/CameraHMR` carries no source change: its untracked content is
the downloaded checkpoints and data folder plus Python caches.
