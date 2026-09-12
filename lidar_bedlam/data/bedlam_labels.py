"""BEDLAM SMPL training labels (``bedlam-labels-smpl``) per frame.

One ``.npz`` per sequence group with one row per person and frame:
``imgname`` (``seq_XXXXXX/seq_XXXXXX_FFFF.png``), ``pose_cam`` (72,
axis-angle, camera frame), ``shape`` (11; the 11th is BEDLAM's extra
shape coefficient, dropped here like CameraHMR does), ``trans_cam`` (3),
``cam_ext`` (4x4), ``cam_int`` (3x3), ``center`` / ``scale`` (bbox, side =
``200 * scale``), ``gtkps`` (44, 3) 2D keypoints, ``gender``.

The SMPL translation in the camera frame is ``trans_cam + cam_ext[:3, 3]``
(verified against the person masks: 85 % of projected vertices fall inside
the mask, the alternatives give 0 %). Label rows are matched to the mask
person ids of a frame by the overlap of the projected mesh silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import NUM_BETAS, SmplParams
from lidar_bedlam.geometry.camera import PinholeCamera

FloatArray = NDArray[np.float64]
IMAGE_SIZE = (1280, 720)


@dataclass(frozen=True)
class PersonLabel:
    """One labelled person in one frame."""

    smpl: SmplParams  # camera frame
    camera: PinholeCamera
    bbox_xyxy: FloatArray
    kp2d: FloatArray  # (44, 3) pixels + confidence
    gender: str
    row: int


class BedlamLabels:
    """Labels of one sequence group, indexed by ``seq/frame``."""

    def __init__(self, npz_path: Path) -> None:
        self.path = npz_path
        d = np.load(npz_path, allow_pickle=True)
        self.imgname: NDArray[np.str_] = d["imgname"]
        self.pose = np.asarray(d["pose_cam"], dtype=np.float64)
        self.shape = np.asarray(d["shape"], dtype=np.float64)[:, :NUM_BETAS]
        self.transl = (
            np.asarray(d["trans_cam"], dtype=np.float64)
            + np.asarray(d["cam_ext"], dtype=np.float64)[:, :3, 3]
        )
        self.cam_int = np.asarray(d["cam_int"], dtype=np.float64)
        self.center = np.asarray(d["center"], dtype=np.float64)
        self.scale = np.asarray(d["scale"], dtype=np.float64)
        self.kp2d = np.asarray(d["gtkps"], dtype=np.float64)
        self.gender = d["gender"] if "gender" in d.files else None
        self._rows: dict[str, list[int]] = {}
        for i, name in enumerate(self.imgname.tolist()):
            self._rows.setdefault(name[: -len(".png")], []).append(i)

    @staticmethod
    def find(labels_dir: Path, group: str) -> Path:
        """The npz of a group (``<group>_6fps.npz`` or ``_30fps.npz``)."""
        hits = sorted(labels_dir.glob(f"{group}_*fps.npz"))
        if not hits:
            msg = f"no label file for {group} under {labels_dir}"
            raise FileNotFoundError(msg)
        return hits[0]

    def frame_key(self, seq: str, frame: str) -> str:
        """Key of a frame as used by the label rows."""
        return f"{seq}/{seq}_{frame}"

    def has_frame(self, seq: str, frame: str) -> bool:
        """Whether any person is labelled in that frame."""
        return self.frame_key(seq, frame) in self._rows

    def persons(self, seq: str, frame: str) -> list[PersonLabel]:
        """All labelled persons of a frame."""
        out = []
        for i in self._rows.get(self.frame_key(seq, frame), []):
            k = self.cam_int[i]
            cam = PinholeCamera(
                k[0, 0], k[1, 1], k[0, 2], k[1, 2], *IMAGE_SIZE
            )
            half = self.scale[i] * 200.0 / 2.0
            cx, cy = self.center[i]
            out.append(
                PersonLabel(
                    smpl=SmplParams.from_pose72(
                        self.pose[i], self.shape[i], self.transl[i]
                    ),
                    camera=cam,
                    bbox_xyxy=np.array(
                        [cx - half, cy - half, cx + half, cy + half]
                    ),
                    kp2d=self.kp2d[i],
                    gender=str(self.gender[i])
                    if self.gender is not None
                    else "",
                    row=i,
                )
            )
        return out


def match_labels_to_masks(
    labels: list[PersonLabel],
    masks: dict[str, NDArray[np.bool_]],
    vertices_camera: list[FloatArray],
    min_overlap: float = 0.15,
) -> dict[str, PersonLabel]:
    """Assign mask person ids to labels by projected-silhouette overlap.

    ``vertices_camera[i]`` are camera-frame mesh vertices of ``labels[i]``.
    The score of a (label, mask) pair is the fraction of the label's
    projected vertices that fall inside the mask; pairs are assigned
    greedily by descending score, one-to-one, above ``min_overlap``.
    Occluded persons keep a lower but still distinctive score, unlabelled
    masks stay unmatched.
    """
    if not masks or not labels:
        return {}
    ids = list(masks)
    h, w = masks[ids[0]].shape
    scores = np.zeros((len(labels), len(ids)))
    for i, (label, verts) in enumerate(
        zip(labels, vertices_camera, strict=True)
    ):
        uv = label.camera.project(np.asarray(verts)[::4])
        ok = np.isfinite(uv).all(axis=1)
        u = np.rint(uv[ok, 0]).astype(int)
        v = np.rint(uv[ok, 1]).astype(int)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inb.any():
            continue
        for j, pid in enumerate(ids):
            scores[i, j] = masks[pid][v[inb], u[inb]].sum() / len(u)
    assigned: dict[str, PersonLabel] = {}
    used_labels: set[int] = set()
    for flat in np.argsort(-scores, axis=None):
        i, j = divmod(int(flat), len(ids))
        if scores[i, j] < min_overlap:
            break
        if i in used_labels or ids[j] in assigned:
            continue
        assigned[ids[j]] = labels[i]
        used_labels.add(i)
    return assigned
