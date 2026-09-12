"""File readers and point sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.data.base import sample_points
from lidar_bedlam.utils.io import read_pcd, read_ply_xyz


def test_read_binary_pcd_with_extra_field(tmp_path: Path) -> None:
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rec = np.zeros(
        2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    )
    rec["x"], rec["y"], rec["z"] = pts.T
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n"
    )
    path = tmp_path / "a.pcd"
    path.write_bytes(header.encode() + rec.tobytes())
    assert np.array_equal(read_pcd(path), pts)


def test_read_binary_ply(tmp_path: Path) -> None:
    pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    path = tmp_path / "a.ply"
    path.write_bytes(header.encode() + pts.astype("<f4").tobytes())
    assert np.array_equal(read_ply_xyz(path), pts)


def test_sample_points_pads_and_subsamples() -> None:
    rng = np.random.default_rng(0)
    pts = np.arange(15, dtype=np.float32).reshape(5, 3)
    sub, valid = sample_points(pts, 3, rng)
    assert sub.shape == (3, 3) and valid.all()
    pad, valid = sample_points(pts, 8, rng)
    assert pad.shape == (8, 3) and valid[:5].all() and not valid[5:].any()
    empty, valid = sample_points(np.zeros((0, 3), np.float32), 4, rng)
    assert empty.shape == (4, 3) and not valid.any()
