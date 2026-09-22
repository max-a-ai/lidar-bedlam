"""Reader for datasets stored in the nuScenes metadata schema.

Covers both the ayadata car-data scenes and nuScenes itself: identical
JSON tables (``sample``, ``sample_data``, ``sample_annotation``,
``calibrated_sensor``, ``ego_pose``, ...) plus ``.pcd.bin`` scans of five
float32 per point.

Where the two differ, and it matters:

- **Version directory.** The car data keeps several ``v<X.Y>-mini``
  directories side by side and only the highest is fully annotated;
  nuScenes has one (``v1.0-mini``, ``v1.0-trainval``).
- **Scenes per metadata directory.** A car-data directory holds exactly
  one scene; a nuScenes one holds hundreds, so :class:`Scene` takes an
  optional ``scene_token`` and keeps only that scene's samples.
- **Distortion.** The car-data JPEGs are raw and carry
  ``camera_distortion``; nuScenes images are already rectified and the key
  is absent, which is read here as zero distortion.
- **Lidar extrinsic.** Both need it applied (the car rig's is a 180 degree
  yaw). The bike data was the exception and is not read by this module.

Images are rectified before cropping because
:meth:`CropSpec.camera` drops distortion coefficients, so a raw frame and
its crop intrinsics would disagree. See :func:`rectify_image`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.boxes import heading_yaw
from lidar_bedlam.geometry.camera import (
    CAMERA_UP_AXIS,
    PinholeCamera,
    invert_se3,
    se3,
    transform_points,
)
from lidar_bedlam.geometry.crop import bbox_from_points_2d
from lidar_bedlam.utils.io import read_image

FloatArray = NDArray[np.float64]

# nuScenes box corner ordering; BOX_EDGES indexes into box_corners() output.
BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_POINT_STRIDE = 5  # x, y, z, intensity, ring


def quat_to_rot(q: object) -> NDArray[np.float64]:
    """Rotation matrix from a nuScenes ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def box_corners(ann: dict[str, Any]) -> NDArray[np.float64]:
    """The 8 global-frame corners of an annotation box.

    ``size`` is nuScenes ``(width, length, height)``; the local axes are
    ``x`` along length, ``y`` along width, ``z`` up.
    """
    width, length, height = (float(v) for v in ann["size"])
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64) * length / 2
    y = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64) * width / 2
    z = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64) * height / 2
    local = np.stack([x, y, z], axis=1)
    return local @ quat_to_rot(ann["rotation"]).T + np.asarray(
        ann["translation"], dtype=np.float64
    )


def _version_key(name: str) -> list[int]:
    """Sort key for a ``v<X.Y>[-suffix]`` directory name."""
    digits = name[1:].split("-")[0]
    return [int(part) for part in digits.split(".") if part.isdigit()]


def latest_version(root: Path) -> str:
    """Highest ``v<X.Y>*`` metadata directory under a dataset root.

    The car data keeps every annotation round (``v1.0-mini`` through
    ``v2.0-mini``) and only the highest is complete, so sort numerically --
    lexically ``v1.9-mini`` beats ``v2.0-mini``.
    """
    versions = [
        p.name
        for p in root.glob("v*")
        if p.is_dir() and (p / "sample.json").exists()
    ]
    if not versions:
        raise FileNotFoundError(f"no v* metadata directory under {root}")
    return max(versions, key=_version_key)


class Scene:
    """One recorded scene: metadata, point clouds and camera frames."""

    def __init__(
        self,
        root: Path | str,
        version: str | None = None,
        scene_token: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.version = version or latest_version(self.root)
        self.scene_token = scene_token
        meta = self.root / self.version

        def load(name: str) -> list[dict[str, Any]]:
            with (meta / name).open() as fh:
                data: list[dict[str, Any]] = json.load(fh)
            return data

        self.sensor = {s["token"]: s for s in load("sensor.json")}
        self.calib = {c["token"]: c for c in load("calibrated_sensor.json")}
        for cal in self.calib.values():
            cal["channel"] = self.sensor[cal["sensor_token"]]["channel"]
        self.ego_pose = {e["token"]: e for e in load("ego_pose.json")}
        self.category = {c["token"]: c["name"] for c in load("category.json")}
        self.attribute = {
            a["token"]: a["name"] for a in load("attribute.json")
        }
        self.instance = {i["token"]: i for i in load("instance.json")}
        self.annotations = load("sample_annotation.json")

        samples = load("sample.json")
        if scene_token is not None:
            samples = [s for s in samples if s["scene_token"] == scene_token]
        # car-data sample.json has empty prev/next, so order by timestamp.
        self.samples = sorted(samples, key=lambda s: float(s["timestamp"]))
        self._keep = {s["token"] for s in self.samples}

        self.sample_data_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
        for rec in load("sample_data.json"):
            if not rec.get("is_key_frame"):
                continue
            if rec["sample_token"] not in self._keep:
                continue
            channel = self.calib[rec["calibrated_sensor_token"]]["channel"]
            self.sample_data_by_sample.setdefault(rec["sample_token"], {})[
                channel
            ] = rec

        self.annotations_by_sample: dict[str, list[dict[str, Any]]] = {}
        for ann in self.annotations:
            if ann["sample_token"] not in self._keep:
                continue
            self.annotations_by_sample.setdefault(
                ann["sample_token"], []
            ).append(ann)

    # -- lookups ---------------------------------------------------------

    @property
    def camera_channels(self) -> frozenset[str]:
        """Channels whose sensor modality is ``camera``.

        Keyed off the modality, not the name: the car rig calls them
        ``CAMERA_*`` and nuScenes ``CAM_*``, and nuScenes also has
        ``RADAR_*`` channels that must not be treated as cameras.
        """
        return frozenset(
            cal["channel"]
            for cal in self.calib.values()
            if self.sensor[cal["sensor_token"]]["modality"] == "camera"
        )

    @property
    def lidar_channel(self) -> str:
        """Channel of the lidar these annotations are counted against."""
        per_sample = self.sample_data_by_sample.values()
        for channel in ("LIDAR_TOP", "LIDAR_TOP_2"):
            if any(channel in chans for chans in per_sample):
                return channel
        raise KeyError("no LIDAR_TOP channel in this scene")

    def category_of(self, ann: dict[str, Any]) -> str:
        """Category name of an annotation, e.g. ``human.pedestrian.adult``."""
        name: str = self.category[
            self.instance[ann["instance_token"]]["category_token"]
        ]
        return name

    def pedestrians(self, sample_token: str) -> list[dict[str, Any]]:
        """Annotations in a sample whose category is under ``human.``."""
        return self.of_category(sample_token, ("human",))

    def of_category(
        self, sample_token: str, prefixes: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """Annotations of a sample whose category starts with a prefix."""
        return [
            a
            for a in self.annotations_by_sample.get(sample_token, [])
            if self.category_of(a).startswith(prefixes)
        ]

    def scene_tokens(self) -> list[str]:
        """Every scene token present in this metadata directory."""
        seen: dict[str, None] = {}
        for s in self.samples:
            seen.setdefault(s["scene_token"], None)
        return list(seen)

    def sensors(self, sample_token: str) -> dict[str, dict[str, Any]]:
        """Key-frame ``sample_data`` records of a sample, keyed by channel."""
        return self.sample_data_by_sample.get(sample_token, {})

    # -- geometry --------------------------------------------------------

    def load_points(
        self, sd_rec: dict[str, Any], frame: str = "global"
    ) -> NDArray[np.float64]:
        """``(N, 4)`` x, y, z, intensity in ``sensor``, ``ego`` or ``global``.

        The lidar extrinsic is applied for ``ego`` and ``global``; skipping it
        empties every box.
        """
        raw = np.fromfile(self.root / sd_rec["filename"], dtype=np.float32)
        points = raw.reshape(-1, _POINT_STRIDE)
        xyz = points[:, :3].astype(np.float64)
        intensity = points[:, 3:4].astype(np.float64)
        keep = np.isfinite(xyz).all(axis=1)
        xyz, intensity = xyz[keep], intensity[keep]
        if frame == "sensor":
            return np.hstack([xyz, intensity])

        cal = self.calib[sd_rec["calibrated_sensor_token"]]
        xyz = xyz @ quat_to_rot(cal["rotation"]).T + np.asarray(
            cal["translation"], dtype=np.float64
        )
        if frame == "ego":
            return np.hstack([xyz, intensity])
        if frame != "global":
            raise ValueError(f"unknown frame {frame!r}")

        ego = self.ego_pose[sd_rec["ego_pose_token"]]
        xyz = xyz @ quat_to_rot(ego["rotation"]).T + np.asarray(
            ego["translation"], dtype=np.float64
        )
        return np.hstack([xyz, intensity])

    def points_global(self, sd_rec: dict[str, Any]) -> NDArray[np.float64]:
        """:meth:`load_points` in the global frame, reusing a cached scan."""
        scan = _raw_scan(str(self.root / sd_rec["filename"]))
        cal = self.calib[sd_rec["calibrated_sensor_token"]]
        ego = self.ego_pose[sd_rec["ego_pose_token"]]
        xyz = scan[:, :3] @ quat_to_rot(cal["rotation"]).T + np.asarray(
            cal["translation"], dtype=np.float64
        )
        xyz = xyz @ quat_to_rot(ego["rotation"]).T + np.asarray(
            ego["translation"], dtype=np.float64
        )
        return np.hstack([xyz, scan[:, 3:4]])

    def in_box(
        self,
        points_global: NDArray[np.float64],
        ann: dict[str, Any],
        margin: float = 0.0,
    ) -> NDArray[np.bool_]:
        """Mask of global-frame points falling inside an annotation box."""
        rot = quat_to_rot(ann["rotation"])
        local = (
            points_global[:, :3]
            - np.asarray(ann["translation"], dtype=np.float64)
        ) @ rot
        width, length, height = (float(v) for v in ann["size"])
        half = np.array([length, width, height], dtype=np.float64) / 2 + margin
        inside: NDArray[np.bool_] = (np.abs(local) <= half).all(axis=1)
        return inside

    def to_camera(
        self, points_global: NDArray[np.float64], sd_rec: dict[str, Any]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Global points to the camera frame (z forward), plus calibration."""
        ego = self.ego_pose[sd_rec["ego_pose_token"]]
        cal = self.calib[sd_rec["calibrated_sensor_token"]]
        pts = (
            np.asarray(points_global, dtype=np.float64)
            - np.asarray(ego["translation"], dtype=np.float64)
        ) @ quat_to_rot(ego["rotation"])
        pts = (
            pts - np.asarray(cal["translation"], dtype=np.float64)
        ) @ quat_to_rot(cal["rotation"])
        return pts, cal

    def project(
        self,
        points_global: NDArray[np.float64],
        sd_rec: dict[str, Any],
        distort: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Global points to pixels, plus an in-front-of-camera mask.

        The stored JPEGs are raw, so ``distort`` must stay ``True`` to land on
        the image; pass ``False`` only against undistorted imagery.
        """
        pts, cal = self.to_camera(points_global, sd_rec)
        front: NDArray[np.bool_] = pts[:, 2] > 0.1
        depth = np.where(front, pts[:, 2], np.nan)
        xn, yn = pts[:, 0] / depth, pts[:, 1] / depth
        coeffs = cal.get("camera_distortion")
        if distort and coeffs:
            k1, k2, p1, p2, k3 = (float(v) for v in coeffs)
            r2 = xn * xn + yn * yn
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
            xn, yn = (
                xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn),
                yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn,
            )
        k = np.asarray(cal["camera_intrinsic"], dtype=np.float64)
        uv = np.stack([k[0, 0] * xn + k[0, 2], k[1, 1] * yn + k[1, 2]], axis=1)
        return uv, front

    def best_camera(
        self,
        ann: dict[str, Any],
        sample_token: str,
        distort: bool = False,
    ) -> tuple[str, NDArray[np.float64]] | None:
        """Channel whose frame holds the whole box, largest on-screen first.

        Returns ``(channel, box_corner_pixels)``, or ``None`` if no camera
        contains the entire box.

        ``distort`` must match the image the caller will use, or a box can
        pass here and fall outside the frame it is cropped from: rectifying
        moves a point away from the centre, so a box that fits the raw frame
        need not fit the rectified one. The default is the rectified
        (pinhole) convention that :class:`CarDataSource` produces; pass
        ``True`` to ask about the raw JPEGs instead.
        """
        corners = box_corners(ann)
        best: tuple[str, NDArray[np.float64]] | None = None
        best_height = 0.0
        cameras = self.camera_channels
        for channel, sd_rec in self.sensors(sample_token).items():
            if channel not in cameras:
                continue
            uv, front = self.project(corners, sd_rec, distort=distort)
            if not front.all():
                continue
            width, height = sd_rec["width"], sd_rec["height"]
            if not (
                (uv[:, 0] >= 0).all()
                and (uv[:, 0] < width).all()
                and (uv[:, 1] >= 0).all()
                and (uv[:, 1] < height).all()
            ):
                continue
            box_height = float(uv[:, 1].max() - uv[:, 1].min())
            if box_height > best_height:
                best, best_height = (channel, uv), box_height
        return best


def camera_of(scene: Scene, sd_rec: dict[str, Any]) -> PinholeCamera:
    """Raw-image camera of a ``sample_data`` record, distortion included."""
    cal = scene.calib[sd_rec["calibrated_sensor_token"]]
    k = np.asarray(cal["camera_intrinsic"], dtype=np.float64)
    # nuScenes images are already rectified and carry no distortion key.
    coeffs = cal.get("camera_distortion") or [0.0] * 5
    k1, k2, p1, p2, k3 = (float(v) for v in coeffs)
    return PinholeCamera(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=int(sd_rec["width"]),
        height=int(sd_rec["height"]),
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
    )


def rectify_image(
    image: NDArray[np.uint8], cam: PinholeCamera
) -> NDArray[np.uint8]:
    """Undistort a raw frame, keeping ``cam``'s intrinsic matrix.

    The result is a true pinhole image, so a plain ``K`` projection lands on
    it and :meth:`CropSpec.camera` (which drops distortion) stays valid.
    Field of view is preserved, so the frame edges gain black border.
    """
    if not cam.has_distortion:
        return image
    v, u = np.mgrid[0 : cam.height, 0 : cam.width]
    x = (u.ravel() - cam.cx) / cam.fx
    y = (v.ravel() - cam.cy) / cam.fy
    rays = np.stack([x, y, np.ones_like(x)], axis=1)
    src = cam.project(rays)  # where each output pixel reads from
    coords = np.stack([src[:, 1], src[:, 0]])
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(
            image[..., c], coords, order=1, mode="constant", cval=0
        ).reshape(cam.height, cam.width)
    return out


def undistorted(cam: PinholeCamera) -> PinholeCamera:
    """Same intrinsics with the distortion coefficients cleared."""
    return PinholeCamera(
        fx=cam.fx,
        fy=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
        width=cam.width,
        height=cam.height,
    )


def global_to_camera(scene: Scene, sd_rec: dict[str, Any]) -> FloatArray:
    """4x4 transform from the global frame to a camera's OpenCV frame."""
    ego = scene.ego_pose[sd_rec["ego_pose_token"]]
    cal = scene.calib[sd_rec["calibrated_sensor_token"]]
    ego_se3 = se3(quat_to_rot(ego["rotation"]), ego["translation"])
    cal_se3 = se3(quat_to_rot(cal["rotation"]), cal["translation"])
    return invert_se3(ego_se3 @ cal_se3)


def box_to_camera(
    ann: dict[str, Any], to_cam: NDArray[np.floating]
) -> FloatArray:
    """Annotation box -> the schema's camera-frame 7-box.

    ``size`` is ``(width, length, height)``; the schema orders the extents
    as length (along the heading), height (along up) and width (across).
    """
    rot = quat_to_rot(ann["rotation"])
    centre = transform_points(
        to_cam, np.asarray(ann["translation"], dtype=np.float64)[None]
    )[0]
    heading = np.asarray(to_cam, dtype=np.float64)[:3, :3] @ rot[:, 0]
    yaw = heading_yaw(heading, CAMERA_UP_AXIS)
    width, length, height = (float(v) for v in ann["size"])
    return np.concatenate([centre, [length, height, width], [yaw]])


@lru_cache(maxsize=12)
def rectified_frame(path: str, cam: PinholeCamera) -> NDArray[np.uint8]:
    """:func:`rectify_image` on a file, cached across the pedestrians in it.

    Several annotations share one frame, and rectifying a 2200x1200 image is
    the expensive part of loading a sample.
    """
    return rectify_image(read_image(Path(path)), cam)


@lru_cache(maxsize=6)
def _raw_scan(path: str) -> NDArray[np.float64]:
    """Sensor-frame ``(N, 4)`` scan of a file, cached; NaN rows dropped."""
    points = np.fromfile(Path(path), dtype=np.float32).reshape(
        -1, _POINT_STRIDE
    )
    xyz = points[:, :3].astype(np.float64)
    keep = np.isfinite(xyz).all(axis=1)
    return np.hstack([xyz[keep], points[keep, 3:4].astype(np.float64)])




class NuscSchemaSource(SampleSource):
    """Annotated objects from nuScenes-schema scenes, ready for the model.

    One sample per (annotation, best camera): the camera whose frame holds
    the whole 3D box, largest on screen first. Images are rectified, so the
    sample's camera is a plain pinhole.

    These datasets label boxes, not bodies, so there is no SMPL ground
    truth: ``smpl`` and ``joints3d`` stay ``None`` and ``box3d`` is the only
    label. Subclasses set ``name`` and the default categories.
    """

    name = "nusc_schema"
    #: category prefixes counted as a person, standing or riding
    PERSON = ("human",)
    #: nuScenes-style cyclists: the bike box also contains the rider
    CYCLIST = ("vehicle.bicycle", "vehicle.motorcycle")

    def __init__(
        self,
        scenes: Sequence[Scene],
        categories: tuple[str, ...] = ("human",),
        min_points: int = 50,
        min_box_height_px: float = 64.0,
        require_rider: bool = True,
    ) -> None:
        self.scenes = list(scenes)
        self.categories = categories
        self.min_points = min_points
        self.min_box_height_px = min_box_height_px
        self.require_rider = require_rider
        self.index: list[tuple[int, str, str, str]] = []
        for si, scene in enumerate(self.scenes):
            riders = self._rider_attributes(scene)
            for sample in scene.samples:
                token = sample["token"]
                for ann in scene.of_category(token, categories):
                    if not self._keep(scene, ann, riders):
                        continue
                    best = scene.best_camera(ann, token)
                    if best is None:
                        continue
                    channel, uv = best
                    if uv[:, 1].max() - uv[:, 1].min() < min_box_height_px:
                        continue
                    self.index.append((si, token, ann["token"], channel))

    @staticmethod
    def _rider_attributes(scene: Scene) -> set[str]:
        """Attribute tokens meaning "this cycle carries a rider"."""
        return {
            tok
            for tok, name in scene.attribute.items()
            if "with_rider" in name
        }

    def _keep(
        self, scene: Scene, ann: dict[str, Any], riders: set[str]
    ) -> bool:
        """Filter on returns, and on a rider for cycle categories."""
        if int(ann["num_lidar_pts"]) < self.min_points:
            return False
        category = scene.category_of(ann)
        if self.require_rider and category.startswith(self.CYCLIST):
            # an empty bike is not a person; only ridden ones carry a body
            return bool(riders & set(ann.get("attribute_tokens", [])))
        return True

    def __len__(self) -> int:
        return len(self.index)

    def meta(self, index: int) -> SampleMeta:
        si, sample_token, ann_token, channel = self.index[index]
        scene = self.scenes[si]
        return SampleMeta(
            dataset=self.name,
            sequence=f"{scene.root.name}/{channel}",
            frame=sample_token[:8],
            person=ann_token[:8],
        )

    def category(self, index: int) -> str:
        """Category name of a sample, e.g. ``human.pedestrian.adult``."""
        si, sample_token, ann_token, _ = self.index[index]
        scene = self.scenes[si]
        ann = self._annotation(scene, sample_token, ann_token)
        return scene.category_of(ann)

    @staticmethod
    def _annotation(
        scene: Scene, sample_token: str, ann_token: str
    ) -> dict[str, Any]:
        return next(
            a
            for a in scene.annotations_by_sample[sample_token]
            if a["token"] == ann_token
        )

    def load(self, index: int) -> Sample:
        si, sample_token, ann_token, channel = self.index[index]
        scene = self.scenes[si]
        sensors = scene.sensors(sample_token)
        sd_cam = sensors[channel]
        sd_lidar = sensors[scene.lidar_channel]
        ann = self._annotation(scene, sample_token, ann_token)

        raw_cam = camera_of(scene, sd_cam)
        camera = undistorted(raw_cam)
        image = rectified_frame(str(scene.root / sd_cam["filename"]), raw_cam)

        points_global = scene.points_global(sd_lidar)
        person = points_global[scene.in_box(points_global, ann)][:, :3]
        to_cam = global_to_camera(scene, sd_cam)
        points = transform_points(to_cam, person)

        corners = transform_points(to_cam, box_corners(ann))
        bbox = bbox_from_points_2d(
            camera.project(corners), camera.width, camera.height
        )
        if bbox is None:
            raise ValueError(
                f"box {ann_token} does not project into {channel}"
            )
        return Sample(
            meta=self.meta(index),
            image=image,
            camera=camera,
            bbox_xyxy=bbox,
            points=np.ascontiguousarray(points, dtype=np.float32),
            box3d=box_to_camera(ann, to_cam),
        )
