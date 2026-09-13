"""Mesh rasterisation: depth values, ids and the clothing offset."""

from __future__ import annotations

import numpy as np

from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.lidar.mesh_depth import (
    SKY_M,
    offset_mesh,
    render_depth,
    vertex_normals,
)


def _quad(z: float, size: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    s = size
    verts = np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return verts, faces


def test_render_depth_two_quads_nearest_wins() -> None:
    cam = PinholeCamera.from_hfov(60.0, 64, 48)
    far, near = _quad(4.0, 1.0), _quad(2.0, 0.3)
    depth, ids = render_depth([far, near], cam)
    cy, cx = 24, 32
    assert np.isclose(depth[cy, cx], 2.0, atol=1e-3) and ids[cy, cx] == 1
    assert depth[0, 0] == SKY_M and ids[0, 0] == -1
    # the big far quad shows around the small near one
    assert np.isclose(depth[cy, 20], 4.0, atol=1e-3) and ids[cy, 20] == 0


def test_offset_moves_vertices_outwards() -> None:
    # a tetrahedron around the origin: every normal points away from it
    verts = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    n = vertex_normals(verts, faces)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    moved = offset_mesh(verts, faces, 0.03)
    assert np.all(
        np.linalg.norm(moved, axis=1) > np.linalg.norm(verts, axis=1)
    )
    assert np.allclose(np.linalg.norm(moved - verts, axis=1), 0.03)
