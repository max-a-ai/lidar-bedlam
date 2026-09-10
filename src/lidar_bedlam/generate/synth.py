"""Synthetic sample generation from extracted BEDLAM frames.

Per frame: read image, depth, masks and labels once; simulate every scan
variant once (all persons of the frame share the scans); then cut one
:class:`Record` per labelled, visible person. Optional per-frame
virtual-distance augmentation moves the camera back by ``d`` (depth and
masks re-rendered, labels shifted, crop detail reduced), so persons get
the LiDAR density and pixel size they would have at the larger distance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.base import SMPL_FORWARD
from lidar_bedlam.data.bedlam import CM_TO_M, BedlamFramesSource
from lidar_bedlam.data.image_augment import ImageAugmentConfig, augment_image
from lidar_bedlam.generate.records import CROP, Record, Scan, write_shard
from lidar_bedlam.geometry.boxes import heading_yaw, oriented_box_from_points
from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS, PinholeCamera
from lidar_bedlam.geometry.crop import (
    crop_image,
    crop_mask,
    square_crop_from_bbox,
)
from lidar_bedlam.geometry.rotations import axis_angle_to_matrix
from lidar_bedlam.io import read_exr_depth, read_image
from lidar_bedlam.lidar.distance import move_camera_back, shrink_factor
from lidar_bedlam.lidar.motion import (
    EgoMotion,
    SpeedSetting,
    apply_rolling_shutter,
)
from lidar_bedlam.lidar.placement import (
    BallPlacement,
    lidar_to_simulator_axes,
    rig_lidar_pose,
)
from lidar_bedlam.lidar.simulate import (
    PRESETS,
    LidarScan,
    LidarSpec,
    azimuth_window,
    ouster,
    select_mask,
    simulate,
)
from lidar_bedlam.rigs.registry import load_all_rigs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SynthConfig:
    """Everything that defines one generated dataset version."""

    out_size: int = CROP
    padding: float = 1.2
    min_box_h_px: float = 90.0
    min_box_w_px: float = 35.0
    min_points: int = 30  # on the co-located OS1-64 @ 1024 check scan
    family: str = "OS1"
    channels: tuple[int, ...] = (32, 64, 128, 256)
    steps: tuple[int, ...] = (512, 1024, 2048)
    n_main: int = 2
    ball_r_max_m: float = 1.0
    ball_small_r_max_m: float = 0.25
    tilt_max_deg: float = 5.0
    speed_mean_kmh: float = 0.0
    speed_std_kmh: float = 0.0
    spin_hz: float = 10.0
    distance_aug_fraction: float = 0.3
    distance_scale: tuple[float, float] = (1.5, 3.0)
    shard_size: int = 512
    seed: int = 0


@dataclass
class FrameStats:
    """Counters for the dataset statistics."""

    frames: int = 0
    frames_distance_aug: int = 0
    persons_total: int = 0
    persons_unlabelled: int = 0
    rejected_box: int = 0
    rejected_points: int = 0
    records: int = 0
    per_group: dict[str, int] = field(default_factory=dict)


def _rig_variants(data_dir: Path) -> dict[str, tuple[LidarSpec, FloatArray]]:
    """Fixed LiDAR placements of the real rigs, in simulator axes."""
    rigs, _ = load_all_rigs(data_dir)
    out: dict[str, tuple[LidarSpec, FloatArray]] = {}
    wanted = {
        "waymo": ("CAM_FRONT", "LIDAR_TOP", PRESETS["waymo64"]),
        "sloper4d": ("CAM_HEAD", "LIDAR_HEAD", ouster("OS1", 128, 1024)),
        "fusebike": ("CAMERA_FRONT", "LIDAR_TOP", ouster("OS2", 128, 1024)),
    }
    for slug, (cam, lid, spec) in wanted.items():
        if slug in rigs:
            pose = lidar_to_simulator_axes(
                rig_lidar_pose(rigs[slug], cam, lid)
            )
            out[f"rig_{slug}"] = (spec, pose)
    return out


class SynthGenerator:
    """Builds records frame by frame from a :class:`BedlamFramesSource`."""

    def __init__(
        self,
        source: BedlamFramesSource,
        smpl: SmplModel,
        cfg: SynthConfig,
        data_dir: Path,
        seed_offset: int = 0,
    ) -> None:
        self.src = source
        self.smpl = smpl
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + seed_offset)
        self.rigs = _rig_variants(data_dir)
        self.ball = BallPlacement(cfg.ball_r_max_m, cfg.tilt_max_deg)
        self.ball_small = BallPlacement(
            cfg.ball_small_r_max_m, cfg.tilt_max_deg
        )
        self.speed = SpeedSetting(cfg.speed_mean_kmh, cfg.speed_std_kmh)
        self.image_aug = ImageAugmentConfig(
            bbox_scale_std=0.0, bbox_shift_std=0.0
        )
        self.stats = FrameStats()

    # -- scan plan --------------------------------------------------------

    def _random_spec(self) -> LidarSpec:
        c = int(self.rng.choice(self.cfg.channels))
        s = int(self.rng.choice(self.cfg.steps))
        return ouster(self.cfg.family, c, s)

    def scan_plan(self) -> dict[str, tuple[LidarSpec, FloatArray]]:
        """Variant name -> (spec, camera_from_sensor) for one frame."""
        plan: dict[str, tuple[LidarSpec, FloatArray]] = {}
        for i in range(self.cfg.n_main):
            plan[f"main_{i}"] = (
                self._random_spec(),
                self.ball.sample(self.rng),
            )
        plan["ball025"] = (
            self._random_spec(),
            self.ball_small.sample(self.rng),
        )
        plan["target_waymo"] = (PRESETS["waymo64"], self.ball.sample(self.rng))
        plan["check"] = (ouster("OS1", 64, 1024), np.eye(4))
        plan.update(self.rigs)
        return plan

    # -- frame processing -------------------------------------------------

    def frame_records(self, index: int) -> list[Record]:
        """All records of the frame that contains sample ``index``."""
        cfg = self.cfg
        group, seq, frame = self.src.frame_of(index)
        labels = self.src.matched_labels(group, seq, frame)
        masks = self.src.person_masks(group, seq, frame)
        self.stats.frames += 1
        self.stats.persons_total += len(masks)
        self.stats.persons_unlabelled += sum(
            1 for p in masks if p not in labels
        )
        if not labels:
            return []
        image = read_image(self.src.frame_image_path(index))
        depth_m = (
            read_exr_depth(self.src.depth_path(index)).astype(np.float64)
            * CM_TO_M
        )
        camera = next(iter(labels.values())).camera
        orig_masks = masks
        d = 0.0
        if self.rng.random() < cfg.distance_aug_fraction:
            k = float(self.rng.uniform(*cfg.distance_scale))
            z_med = float(
                np.median([lb.smpl.transl[2] for lb in labels.values()])
            )
            d = (k - 1.0) * z_med
            self.stats.frames_distance_aug += 1
            depth_m, masks = move_camera_back(depth_m, masks, camera, d)
        speed = self.speed.sample_mps(self.rng)
        scans: dict[str, tuple[LidarScan, LidarSpec, FloatArray]] = {}
        for name, (spec, pose) in self.scan_plan().items():
            scan = simulate(depth_m, camera, spec, pose, self.rng)
            if speed > 0:
                window = azimuth_window(camera, spec, pose)
                motion = EgoMotion(speed_mps=speed, spin_hz=cfg.spin_hz)
                scan = apply_rolling_shutter(scan, motion, window)
            scans[name] = (scan, spec, pose)
        records = []
        for pid, label in labels.items():
            if pid not in masks or not masks[pid].any():
                continue
            rec = self._person_record(
                group,
                seq,
                frame,
                pid,
                label.smpl,
                camera,
                image,
                orig_masks[pid],
                masks[pid],
                scans,
                d,
            )
            if rec is not None:
                records.append(rec)
        self.stats.records += len(records)
        self.stats.per_group[group] = self.stats.per_group.get(group, 0) + len(
            records
        )
        return records

    def _person_record(
        self,
        group: str,
        seq: str,
        frame: str,
        pid: str,
        smpl: SmplParams,
        camera: PinholeCamera,
        image: NDArray[np.uint8],
        orig_mask: NDArray[np.bool_],
        mask: NDArray[np.bool_],
        scans: dict[str, tuple[LidarScan, LidarSpec, FloatArray]],
        d: float,
    ) -> Record | None:
        """One record; ``mask`` is the (possibly moved-back) render mask."""
        cfg = self.cfg
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        bbox = np.array(
            [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64
        )
        if (
            bbox[3] - bbox[1] < cfg.min_box_h_px
            or bbox[2] - bbox[0] < cfg.min_box_w_px
        ):
            self.stats.rejected_box += 1
            return None
        person_scans: dict[str, Scan] = {}
        for name, (scan, spec, pose) in scans.items():
            sel = select_mask(scan, mask)
            person_scans[name] = Scan(
                points=sel.points,
                channel=sel.channel.astype(np.int16),
                column=sel.column.astype(np.int16),
                channels=spec.channels,
                steps=spec.horizontal_steps,
                sensor_pose=pose,
            )
        if len(person_scans["check"].points) < cfg.min_points:
            self.stats.rejected_points += 1
            return None
        # camera moved back by d: labels translate by (0, 0, d)
        params = SmplParams(
            smpl.global_orient,
            smpl.body_pose,
            smpl.betas,
            smpl.transl + np.array([0.0, 0.0, d]),
        )
        k = float((smpl.transl[2] + d) / smpl.transl[2])
        verts, joints = self.smpl.forward(params)
        forward = axis_angle_to_matrix(params.global_orient) @ SMPL_FORWARD
        box3d = oriented_box_from_points(
            verts, heading_yaw(forward, CAMERA_UP_AXIS), CAMERA_UP_AXIS
        )
        spec_crop = square_crop_from_bbox(bbox, cfg.out_size, cfg.padding)
        crop_cam = spec_crop.camera(camera)
        if d > 0:  # image content comes from the original (larger) window
            oy, ox = np.nonzero(orig_mask)
            obox = np.array(
                [ox.min(), oy.min(), ox.max(), oy.max()], dtype=np.float64
            )
            img_crop = crop_image(
                image, square_crop_from_bbox(obox, cfg.out_size, cfg.padding)
            )
            img_crop = _reduce_detail(
                img_crop, 1.0 / shrink_factor(float(smpl.transl[2]), d)
            )
        else:
            img_crop = crop_image(image, spec_crop)
        img_aug, _ = augment_image(
            img_crop,
            np.array([0, 0, cfg.out_size, cfg.out_size]),
            self.image_aug,
            self.rng,
        )
        kp = np.ones((24, 3))
        kp[:, :2] = crop_cam.project(joints)
        return Record(
            key=f"bedlam/{group}/{seq}/{frame}/{pid}",
            dataset="bedlam",
            image=img_crop,
            image_aug=img_aug,
            mask=crop_mask(mask, spec_crop),
            intrinsics=crop_cam.matrix,
            crop_origin=np.array([spec_crop.x0, spec_crop.y0, spec_crop.side]),
            has_image=True,
            has_smpl=True,
            global_orient=params.global_orient,
            body_pose=params.body_pose,
            betas=params.betas,
            transl=params.transl,
            joints3d=joints,
            joints3d_valid=np.ones(24, dtype=bool),
            joint_convention="smpl24",
            kp2d=kp,
            box3d=box3d,
            distance_scale=k,
            scans=person_scans,
        )


def _reduce_detail(image: NDArray[np.uint8], k: float) -> NDArray[np.uint8]:
    """Down- and up-sample by ``k`` to mimic the pixel size at k x distance."""
    h, w = image.shape[:2]
    small = Image.fromarray(image).resize(
        (max(1, int(round(w / k))), max(1, int(round(h / k)))),
        Image.Resampling.BOX,
    )
    return np.asarray(
        small.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8
    )


def frame_indices(source: BedlamFramesSource) -> list[int]:
    """One sample index per distinct frame (the first person of it)."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for i in range(len(source)):
        key = source.frame_of(i)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def generate(
    gen: SynthGenerator,
    indices: list[int],
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    """Process frames, writing shards of ``cfg.shard_size`` records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[Record] = []
    shard_id = 0
    for i in indices:
        buffer.extend(gen.frame_records(i))
        while len(buffer) >= gen.cfg.shard_size:
            write_shard(
                buffer[: gen.cfg.shard_size],
                out_dir / f"{prefix}_{shard_id:05d}.npz",
            )
            buffer = buffer[gen.cfg.shard_size :]
            shard_id += 1
    if buffer:
        write_shard(buffer, out_dir / f"{prefix}_{shard_id:05d}.npz")
        shard_id += 1
    stats = asdict(gen.stats)
    stats["shards"] = shard_id
    stats["config"] = asdict(gen.cfg)
    (out_dir / f"{prefix}_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
