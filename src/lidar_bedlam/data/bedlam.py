"""BEDLAM raw-frame loader (extracted from the official tars).

Expected layout under ``root`` (one directory per sequence group)::

    <group>/png/seq_XXXXXX/seq_XXXXXX_FFFF.png
    <group>/depth/seq_XXXXXX/seq_XXXXXX_FFFF_depth.exr
    <group>/masks/seq_XXXXXX/seq_XXXXXX_FFFF_PP_{body,clothing,...}.png
    <group>/ground_truth/camera/seq_XXXXXX_camera.csv

Depth is planar z-depth in centimetres (sky = 1e8), rendered from clothed
characters. The camera CSV gives the horizontal FOV per frame. When a
``labels_dir`` (BEDLAM SMPL training labels) is given, SMPL parameters in
the camera frame are attached per person by matching label rows to mask
person ids (see :mod:`bedlam_labels`); the person point cloud is the dense
back-projection of the body+clothing mask.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.bedlam_labels import (
    BedlamLabels,
    PersonLabel,
    match_labels_to_masks,
)
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.io import read_exr_depth, read_image, read_mask

CM_TO_M = 0.01
SKY_DEPTH_CM = 1e7  # anything beyond this is background
PERSON_MASK_PARTS = ("body", "clothing")


class BedlamFramesSource(SampleSource):
    """One sample per (frame, person) with a body mask."""

    name = "bedlam"

    def __init__(
        self,
        root: Path,
        groups: list[str] | None = None,
        frame_stride: int = 1,
        labels_dir: Path | None = None,
        smpl_model: SmplModel | None = None,
    ) -> None:
        self.root = root
        self.labels_dir = labels_dir
        self.smpl_model = smpl_model
        self._labels: dict[str, BedlamLabels] = {}
        self._matches: dict[tuple[str, str, str], dict[str, PersonLabel]] = {}
        self.groups = groups or sorted(
            p.name for p in root.iterdir() if p.is_dir()
        )
        self._hfov: dict[tuple[str, str], dict[str, float]] = {}
        self._index: list[tuple[str, str, str, str]] = []
        for group in self.groups:
            for seq_dir in sorted((root / group / "masks").glob("seq_*")):
                seq = seq_dir.name
                persons: dict[str, set[str]] = {}
                for m in seq_dir.glob(f"{seq}_*_body.png"):
                    frame, person = m.name[len(seq) + 1 :].split("_")[:2]
                    persons.setdefault(frame, set()).add(person)
                for frame in sorted(persons):
                    if int(frame) % frame_stride:
                        continue
                    for person in sorted(persons[frame]):
                        self._index.append((group, seq, frame, person))

    def __len__(self) -> int:
        return len(self._index)

    def meta(self, index: int) -> SampleMeta:
        group, seq, frame, person = self._index[index]
        return SampleMeta(self.name, f"{group}/{seq}", frame, person)

    def frame_persons(self, index: int) -> list[int]:
        """Indices of all persons in the same frame as ``index``."""
        group, seq, frame, _ = self._index[index]
        return [
            i
            for i, (g, s, f, _p) in enumerate(self._index)
            if (g, s, f) == (group, seq, frame)
        ]

    def labels_for(self, group: str) -> BedlamLabels | None:
        """Label table of a group (cached), or None without labels."""
        if self.labels_dir is None:
            return None
        if group not in self._labels:
            self._labels[group] = BedlamLabels(
                BedlamLabels.find(self.labels_dir, group)
            )
        return self._labels[group]

    def frame_of(self, index: int) -> tuple[str, str, str]:
        """(group, seq, frame) of a sample."""
        group, seq, frame, _ = self._index[index]
        return group, seq, frame

    def frame_image_path(self, index: int) -> Path:
        """Path of the png of the sample's frame."""
        group, seq, frame, _ = self._index[index]
        return self.root / group / "png" / seq / f"{seq}_{frame}.png"

    def person_masks(
        self, group: str, seq: str, frame: str
    ) -> dict[str, NDArray[np.bool_]]:
        """Body+clothing mask per person id of a frame."""
        return self._person_masks(group, seq, frame)

    def _person_masks(
        self, group: str, seq: str, frame: str
    ) -> dict[str, NDArray[np.bool_]]:
        masks: dict[str, NDArray[np.bool_]] = {}
        for _g, s, f, pid in self._index:
            if (s, f) != (seq, frame):
                continue
            masks[pid] = self._read_mask(group, seq, frame, pid)
        return masks

    def _read_mask(
        self, group: str, seq: str, frame: str, person: str
    ) -> NDArray[np.bool_]:
        base = self.root / group / "masks" / seq
        mask: NDArray[np.bool_] | None = None
        for part in PERSON_MASK_PARTS:
            path = base / f"{seq}_{frame}_{person}_{part}.png"
            if path.exists():
                m = read_mask(path)
                mask = m if mask is None else (mask | m)
        if mask is None:
            msg = f"no mask for {group}/{seq}/{frame}/{person}"
            raise FileNotFoundError(msg)
        return mask

    def matched_labels(
        self, group: str, seq: str, frame: str
    ) -> dict[str, PersonLabel]:
        """Label per mask person id of a frame (cached per frame)."""
        key = (group, seq, frame)
        if key in self._matches:
            return self._matches[key]
        labels = self.labels_for(group)
        result: dict[str, PersonLabel] = {}
        if labels is not None and self.smpl_model is not None:
            persons = labels.persons(seq, frame)
            verts = [self.smpl_model.forward(p.smpl)[0] for p in persons]
            result = match_labels_to_masks(
                persons, self._person_masks(group, seq, frame), verts
            )
        self._matches[key] = result
        return result

    def depth_path(self, index: int) -> Path:
        """Path of the depth EXR of the sample's frame."""
        group, seq, frame, _ = self._index[index]
        return self.root / group / "depth" / seq / f"{seq}_{frame}_depth.exr"

    def _camera_row(
        self, group: str, seq: str, frame: str
    ) -> dict[str, float]:
        key = (group, seq)
        if key not in self._hfov:
            path = (
                self.root
                / group
                / "ground_truth"
                / "camera"
                / f"{seq}_camera.csv"
            )
            with open(path, newline="") as fh:
                rows = {r["name"]: r for r in csv.DictReader(fh)}
            self._hfov[key] = {
                name: float(r["hfov"]) for name, r in rows.items()
            }
        return {"hfov": self._hfov[key][f"{seq}_{frame}.png"]}

    def load(self, index: int) -> Sample:
        group, seq, frame, person = self._index[index]
        base = self.root / group
        image = read_image(base / "png" / seq / f"{seq}_{frame}.png")
        h, w = image.shape[:2]
        camera = PinholeCamera.from_hfov(
            self._camera_row(group, seq, frame)["hfov"], w, h
        )
        depth_cm = read_exr_depth(
            base / "depth" / seq / f"{seq}_{frame}_depth.exr"
        )
        mask = np.zeros((h, w), dtype=bool)
        for part in PERSON_MASK_PARTS:
            path = base / "masks" / seq / f"{seq}_{frame}_{person}_{part}.png"
            if path.exists():
                mask |= read_mask(path)
        mask &= depth_cm < SKY_DEPTH_CM
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            msg = f"empty person mask for {self.meta(index).key}"
            raise ValueError(msg)
        bbox = np.array(
            [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64
        )
        points = camera.unproject_depth(depth_cm * CM_TO_M, mask)
        label = self.matched_labels(group, seq, frame).get(person)
        if label is not None:
            camera = label.camera  # the labels' own intrinsics (cx = W/2)
        return Sample(
            meta=self.meta(index),
            image=image,
            camera=camera,
            bbox_xyxy=bbox,
            points=np.ascontiguousarray(points, dtype=np.float32),
            smpl=label.smpl if label is not None else None,
            mask=mask,
        )
