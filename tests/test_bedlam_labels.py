"""BEDLAM label table parsing and label-to-mask matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.data.bedlam_labels import (
    BedlamLabels,
    PersonLabel,
    match_labels_to_masks,
)
from lidar_bedlam.geometry.camera import PinholeCamera


def _write_npz(path: Path) -> None:
    n = 3
    cam_ext = np.tile(np.eye(4), (n, 1, 1))
    cam_ext[:, :3, 3] = [[0, 0, 5], [1, 0, 6], [0, 0, 7]]
    np.savez(
        path,
        imgname=np.array(
            ["seq_000001/seq_000001_0005.png"] * 2
            + ["seq_000001/seq_000001_0010.png"]
        ),
        pose_cam=np.zeros((n, 72)),
        shape=np.zeros((n, 11)),
        trans_cam=np.tile([0.1, 0.2, 0.3], (n, 1)),
        cam_ext=cam_ext,
        cam_int=np.tile(
            [[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]], (n, 1, 1)
        ),
        center=np.tile([640.0, 360.0], (n, 1)),
        scale=np.full(n, 1.0),
        gtkps=np.zeros((n, 44, 3)),
        gender=np.array(["male", "female", "neutral"]),
    )


def test_labels_table(tmp_path: Path) -> None:
    group = "20221010_3_1000_batch01hand"
    path = tmp_path / f"{group}_6fps.npz"
    _write_npz(path)
    assert BedlamLabels.find(tmp_path, group) == path
    labels = BedlamLabels(path)
    assert labels.has_frame("seq_000001", "0005")
    persons = labels.persons("seq_000001", "0005")
    assert len(persons) == 2
    assert np.allclose(persons[0].smpl.transl, [0.1, 0.2, 5.3])
    assert np.allclose(persons[1].smpl.transl, [1.1, 0.2, 6.3])
    assert persons[0].smpl.betas.shape == (10,)
    assert np.allclose(persons[0].bbox_xyxy, [540, 260, 740, 460])
    assert persons[0].camera.fx == 1000 and persons[0].gender == "male"
    assert len(labels.persons("seq_000001", "0010")) == 1


def test_matching_by_silhouette_overlap() -> None:
    cam = PinholeCamera(1000, 1000, 640, 360, 1280, 720)
    masks = {
        "00": np.zeros((720, 1280), bool),
        "01": np.zeros((720, 1280), bool),
    }
    masks["00"][300:420, 600:680] = True  # around (640, 360)
    masks["01"][300:420, 900:980] = True  # around (940, 360)
    from lidar_bedlam.body.smpl import SmplParams

    def label() -> PersonLabel:
        return PersonLabel(
            SmplParams(np.zeros(3), np.zeros(69), np.zeros(10), np.zeros(3)),
            cam, np.zeros(4), np.zeros((44, 3)), "", 0,
        )  # fmt: skip

    rng = np.random.default_rng(0)
    body = rng.uniform(-0.2, 0.2, size=(400, 3)) * [1, 2, 1]
    labels = [label(), label(), label()]
    verts = [
        body + [1.5, 0.0, 5.0],  # projects onto mask 01
        body + [0.0, 0.0, 5.0],  # projects onto mask 00
        body + [-3.0, 0.0, 5.0],  # outside every mask
    ]
    got = match_labels_to_masks(labels, masks, verts)
    assert got["01"] is labels[0] and got["00"] is labels[1]
    assert len(got) == 2
