"""Records rendered from an SMPL mesh: mask, LiDAR offset, labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.threedpw import ActorLabel, Frame
from lidar_bedlam.generate.mesh_synth import (
    MeshLidarGenerator,
    MeshSynthConfig,
)
from lidar_bedlam.generate.synth import SynthConfig
from lidar_bedlam.geometry.camera import PinholeCamera

SMPL_DIR = Path("resources/data/generated/body_models")


@pytest.mark.skipif(not SMPL_DIR.exists(), reason="SMPL model files absent")
def test_mesh_records_have_offset_returns_and_labels() -> None:
    model = SmplModel(SMPL_DIR)
    models = {"neutral": model, "male": model, "female": model}
    smpl = SmplParams(
        np.array([np.pi, 0.0, 0.0]),  # upright in the OpenCV camera frame
        np.zeros(69),
        np.zeros(10),
        np.array([0.0, 0.3, 4.0]),
    )
    camera = PinholeCamera.from_hfov(70.0, 640, 480)
    frame = Frame(
        "seq", 7, Path("none.jpg"), camera, [ActorLabel(0, "neutral", smpl)]
    )
    cfg = SynthConfig(seed=1)
    gen = MeshLidarGenerator(
        None, models, cfg, MeshSynthConfig(0.03, 0.03), Path("resources/data")
    )
    image = np.zeros((480, 640, 3), np.uint8)
    recs = gen.records_from_frame(frame, image)
    assert len(recs) == 1
    r = recs[0]
    assert r.dataset == "3dpw" and r.key == "3dpw/train/seq/00007/00"
    assert r.has_smpl and r.joint_convention == "smpl24"
    assert r.mask.sum() > 500
    verts, _ = model.forward(smpl)
    pts = r.scans["check"].points
    assert len(pts) >= cfg.min_points
    d = np.linalg.norm(pts[:, None] - verts[None], axis=-1).min(1)
    assert 0.015 < np.median(d) < 0.045  # returns sit on the clothing layer
    assert np.allclose(r.transl, smpl.transl)
