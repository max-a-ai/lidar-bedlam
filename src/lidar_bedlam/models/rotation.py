"""Differentiable rotation representations (torch)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def rot6d_to_matrix(x: Tensor) -> Tensor:
    """6D rotation (Zhou et al. 2019) (..., 6) to matrices (..., 3, 3)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2).transpose(-1, -2)


def matrix_to_rot6d(mat: Tensor) -> Tensor:
    """Matrices (..., 3, 3) to the 6D representation (first two columns)."""
    return mat[..., :, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)


IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
