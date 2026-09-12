# BEDLAM v1 audit (2026-09-08)

Location: `/home/max/nas_drive/publicdatasets/bedlam` (Synology, mounted via sshfs).

## Completeness

- `b0.txt` lists 30 sequence groups; `b0_checksums_all.xxh128` lists 390 archives.
- All 390 archives are present with exactly matching names (0 missing, 0 extra).
- Every `.tar` (300 depth/png, 30 masks, 30 mp4) ends with a 1024-byte zero
  EOF block and has a 512-byte aligned size, so none is truncated.
- Every `*_gt.tar.gz` (30) decompresses without error.
- Full xxh128 hashing was not done from this machine (6.5 TB over sshfs).
  Run `lidar_bedlam/scripts/validate_bedlam_on_nas.sh` on the NAS itself.

## Sizes (decimal)

| Modality | Archives | On NAS | Published |
|---|---|---|---|
| depth | 150 | 4.094 TB | 3.8 TB |
| png | 150 | 2.390 TB | 2.2 TB |
| masks | 30 | 33 GB | 30 GB |
| mp4 | 30 | 22 GB | 20 GB |
| gt | 30 | 21 MB | 100 MB |
| total | 390 | 6.54 TB | ~6 TB |

## Content (verified on `20221024_3-10_100_batch01handhair_static_highSchoolGym`)

- `png/seq_XXXXXX/seq_XXXXXX_FFFF.png`: 1280x720 RGBA uint8.
- `depth/seq_XXXXXX/seq_XXXXXX_FFFF_depth.exr`: single channel `Depth`, 32-bit
  float, ZIP compression, no camera metadata in the header.
  - Unit: centimetres (Unreal world units). Floor pixels of the sample lie at
    6.2 m to 14.6 m; sky is `1e8`.
  - Semantics: **planar z-depth** (distance along the optical axis), not ray
    length. Test: unprojecting the lower image half with z = Depth gives a
    dominant plane with 55 % inliers at 2 cm (RANSAC), versus 24 % under the
    ray-length hypothesis. Point cloud: `X = (u - cx) / fx * z`,
    `Y = (v - cy) / fy * z`, `Z = z`, with `fx = fy = W/2 / tan(hfov/2)`.
  - The depth is rendered from the **clothed** characters, so simulated LiDAR
    hits the clothing surface, as a real sensor would.
- `masks/seq_XXXXXX/seq_XXXXXX_FFFF_PP_{body,clothing,...}.png`: per person
  binary masks (0/255). Body and clothing are separate masks.
- `gt`: `be_seq.csv` (per sequence: camera root pose, per body: name,
  X, Y, Z, yaw, start_frame, textures, hair) and
  `ground_truth/camera/seq_XXXXXX_camera.csv` (per frame: x, y, z, yaw,
  pitch, roll in cm/deg, focal_length mm, sensor 36x20.25 mm, hfov).
- SMPL-X body parameters are **not** in these tars. They come from the separate
  BEDLAM body-data download (animation files), keyed by body name.

## Security note

`be_download.sh` on the share contains a plaintext login. Rotate it and strip
it from the script.
