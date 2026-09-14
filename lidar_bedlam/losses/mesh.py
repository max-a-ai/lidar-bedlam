"""Mesh supervision in the style of Pose2Mesh (Choi et al., ECCV 2020),
which LiDAR-HMR reuses for its final mesh: vertex coordinates, surface
normals and edge lengths against the labelled SMPL mesh. Records without
SMPL labels (Waymo keypoints) contribute nothing.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def vertex_loss(
    pred: Tensor, gt: Tensor, root_pred: Tensor, root_gt: Tensor
) -> Tensor:
    """Mean L1 over root-relative vertex coordinates (metres), per sample."""
    return (
        ((pred - root_pred[:, None]) - (gt - root_gt[:, None]))
        .abs()
        .mean((1, 2))
    )


def normal_loss(pred: Tensor, gt: Tensor, faces: Tensor) -> Tensor:
    """|cos| between the predicted face edges and the labelled face normal.

    Zero when every predicted edge lies in the labelled face plane.
    """
    e1 = functional.normalize(
        pred[:, faces[:, 1]] - pred[:, faces[:, 0]], dim=-1
    )
    e2 = functional.normalize(
        pred[:, faces[:, 2]] - pred[:, faces[:, 0]], dim=-1
    )
    e3 = functional.normalize(
        pred[:, faces[:, 2]] - pred[:, faces[:, 1]], dim=-1
    )
    g1 = gt[:, faces[:, 1]] - gt[:, faces[:, 0]]
    g2 = gt[:, faces[:, 2]] - gt[:, faces[:, 0]]
    n = functional.normalize(torch.cross(g1, g2, dim=-1), dim=-1)
    cos = torch.stack([(e * n).sum(-1).abs() for e in (e1, e2, e3)], -1)
    return cos.mean((1, 2))


def edge_loss(pred: Tensor, gt: Tensor, faces: Tensor) -> Tensor:
    """Mean |edge length difference| over the three edges of every face."""

    def lengths(v: Tensor) -> Tensor:
        a, b, c = v[:, faces[:, 0]], v[:, faces[:, 1]], v[:, faces[:, 2]]
        return torch.stack(
            [(a - b).norm(dim=-1), (a - c).norm(dim=-1), (b - c).norm(dim=-1)],
            -1,
        )

    return (lengths(pred) - lengths(gt)).abs().mean((1, 2))
