"""Point-cloud tokenizer: sparse person points -> a set of LiDAR tokens.

Per-point features (Fourier-encoded xyz) are grouped around farthest-point
sampled centres (kNN), pooled, given a positional encoding of the centre,
and refined by a small transformer. Output: (B, M, d) tokens plus their
3D centres, which carry the *spatial* cue for the fusion stage.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def fourier_features(xyz: Tensor, num_bands: int) -> Tensor:
    """Sin/cos encoding of coordinates (..., 3) -> (..., 3 + 6 * bands)."""
    freqs = 2.0 ** torch.arange(num_bands, device=xyz.device, dtype=xyz.dtype)
    ang = xyz[..., None] * freqs * math.pi  # (..., 3, bands)
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return torch.cat([xyz, enc.flatten(-2)], dim=-1)


def farthest_point_sample(xyz: Tensor, m: int) -> Tensor:
    """Indices (B, m) of farthest-point samples of xyz (B, N, 3)."""
    b, n, _ = xyz.shape
    idx = torch.zeros(b, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), float("inf"), device=xyz.device, dtype=xyz.dtype)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(m):
        idx[:, i] = farthest
        centre = xyz[batch, farthest][:, None]
        dist = torch.minimum(dist, ((xyz - centre) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return idx


def knn(query: Tensor, points: Tensor, k: int) -> Tensor:
    """Indices (B, M, k) of the k nearest points for each query."""
    d = torch.cdist(query, points)
    return d.topk(k, dim=-1, largest=False).indices


def gather(points: Tensor, idx: Tensor) -> Tensor:
    """Gather (B, N, C) by indices (B, ...) -> (B, ..., C)."""
    b = points.shape[0]
    flat = idx.reshape(b, -1)
    out = torch.gather(
        points, 1, flat[..., None].expand(-1, -1, points.shape[-1])
    )
    return out.reshape(*idx.shape, points.shape[-1])


class PointTokenizer(nn.Module):
    """Group points into ``num_tokens`` local tokens of width ``dim``."""

    def __init__(
        self,
        dim: int,
        num_tokens: int = 128,
        k: int = 16,
        num_bands: int = 6,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.k = k
        self.num_bands = num_bands
        in_dim = 3 + 6 * num_bands
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.group_mlp = nn.Sequential(
            nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.centre_pos = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, dim * 4, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu",
        )  # fmt: skip
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        """Tokens (B, M, dim) and their centres (B, M, 3) for xyz (B, N, 3)."""
        # normalise by the cloud centroid for the per-point features; the
        # absolute centre position is added back through ``centre_pos``
        centroid = xyz.mean(1, keepdim=True)
        local = xyz - centroid
        feats = self.point_mlp(fourier_features(local, self.num_bands))
        idx = farthest_point_sample(local, min(self.num_tokens, xyz.shape[1]))
        centres = gather(local, idx)  # (B, M, 3)
        nbr = knn(centres, local, min(self.k, xyz.shape[1]))  # (B, M, k)
        nbr_xyz = gather(local, nbr) - centres[:, :, None]
        nbr_feat = gather(feats, nbr)
        grouped = self.group_mlp(torch.cat([nbr_feat, nbr_xyz], -1)).amax(2)
        abs_centres = centres + centroid
        tokens = grouped + self.centre_pos(
            fourier_features(abs_centres, self.num_bands)
        )
        return self.norm(self.encoder(tokens)), abs_centres
