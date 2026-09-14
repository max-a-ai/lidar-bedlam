"""LiDAR-to-surface and mesh losses on a small closed mesh."""

from __future__ import annotations

import numpy as np
import torch

from lidar_bedlam.losses.lidar_surface import (
    chamfer_loss,
    facing_vertex_mask,
    icp_loss,
)
from lidar_bedlam.losses.mesh import edge_loss, normal_loss, vertex_loss


def _sphere(n: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-sphere mesh (lat/long grid) with outward faces."""
    lat = np.linspace(-np.pi / 2 + 0.2, np.pi / 2 - 0.2, n)
    lon = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    verts, faces = [], []
    for a in lat.tolist():
        for b in lon.tolist():
            verts.append(
                [np.cos(a) * np.cos(b), np.sin(a), np.cos(a) * np.sin(b)]
            )
    m = len(lon)
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            faces += [[a, c, b], [b, c, d]]
    v = torch.tensor(np.array(verts), dtype=torch.float32)
    f = torch.tensor(np.array(faces), dtype=torch.int64)
    # orient outwards
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = torch.cross(v1 - v0, v2 - v0, dim=-1)
    flip = (nrm * (v0 + v1 + v2)).sum(-1) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return v, f


def test_only_the_near_hemisphere_faces_the_sensor() -> None:
    v, f = _sphere()
    origin = torch.tensor([[0.0, 0.0, -5.0]])  # sensor in front (negative z)
    mask = facing_vertex_mask(
        v[None] + torch.tensor([0.0, 0.0, 4.0]), f, origin
    )
    z = v[:, 2]
    assert mask[0][z < -0.3].all() and not mask[0][z > 0.3].any()


def test_chamfer_is_the_clothing_offset_residual() -> None:
    v, f = _sphere()
    verts = v[None] + torch.tensor([0.0, 0.0, 4.0])
    origin = torch.zeros(1, 3)
    # returns 3 cm outside the near hemisphere
    near = v[v[:, 2] < -0.5] * 1.03 + torch.tensor([0.0, 0.0, 4.0])
    pts = near[None]
    valid = torch.ones(1, len(near), dtype=torch.bool)
    exact = chamfer_loss(
        verts, f, origin, pts, valid, torch.tensor(0.03), n_sub=2000
    )
    wrong = chamfer_loss(
        verts, f, origin, pts, valid, torch.tensor(0.0), n_sub=2000
    )
    assert exact < 0.01 and wrong > exact
    # the far hemisphere never explains a return: distances stay large
    far = v[v[:, 2] > 0.5] * 1.03 + torch.tensor([0.0, 0.0, 4.0])
    loss_far = chamfer_loss(
        verts,
        f,
        origin,
        far[None],
        torch.ones(1, len(far), dtype=torch.bool),
        torch.tensor(0.03),
        n_sub=2000,
    )
    assert loss_far > 0.3


def test_icp_recovers_a_shift() -> None:
    v, f = _sphere()
    origin = torch.zeros(1, 3)
    verts = v[None] + torch.tensor([0.0, 0.0, 4.0])
    near = v[v[:, 2] < -0.3] + torch.tensor([0.0, 0.0, 4.0])
    shifted = near[None] + torch.tensor([0.15, 0.0, 0.0])
    valid = torch.ones(1, len(near), dtype=torch.bool)
    loss = icp_loss(verts, f, origin, shifted, valid, iterations=4, n_sub=2000)
    assert 0.08 < float(loss) < 0.30
    aligned = icp_loss(
        verts, f, origin, near[None], valid, iterations=4, n_sub=2000
    )
    assert float(aligned) < 0.03


def test_mesh_terms_vanish_on_the_label() -> None:
    v, f = _sphere()
    verts = v[None]
    root = torch.zeros(1, 3)
    assert float(vertex_loss(verts, verts, root, root)) == 0.0
    assert float(normal_loss(verts, verts, f)) < 1e-4
    assert float(edge_loss(verts, verts, f)) == 0.0
    assert float(edge_loss(verts * 1.1, verts, f)) > 0.0
