"""Selective cross-attention: joint-group queries route to image or LiDAR.

Each query stands for a body part (or shape / root). It attends to the
image tokens and to the LiDAR tokens separately; a per-query gate mixes the
two. The gate starts from a prior (image for hands, feet/ankles and head
orientation; LiDAR for torso, legs, root placement and shape) and is
learned, so the routing is a soft, inspectable bias rather than a hard
mask. Gate values are returned for logging and ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class JointGroup:
    """A query: which SMPL joints it drives and its modality prior."""

    name: str
    joints: tuple[int, ...]
    image_prior: float  # logit; > 0 prefers image, < 0 prefers LiDAR


# SMPL joint indices, see body.smpl.SMPL_JOINT_NAMES
JOINT_GROUPS: tuple[JointGroup, ...] = (
    JointGroup("root", (0,), -2.0),
    JointGroup("torso", (3, 6, 9, 12, 13, 14), -2.0),
    JointGroup("head", (15,), 2.0),
    JointGroup("left_arm", (16, 18), 1.0),
    JointGroup("right_arm", (17, 19), 1.0),
    JointGroup("left_hand", (20, 22), 2.0),
    JointGroup("right_hand", (21, 23), 2.0),
    JointGroup("left_leg", (1, 4), -2.0),
    JointGroup("right_leg", (2, 5), -2.0),
    JointGroup("left_foot", (7, 10), 2.0),
    JointGroup("right_foot", (8, 11), 2.0),
    JointGroup("shape", (), -2.0),
)


class SelectiveLayer(nn.Module):
    """One decoder layer: gated dual cross-attention + self-attention + FFN."""

    def __init__(self, dim: int, num_heads: int, priors: Tensor) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.attn_img = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_pts = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate_bias = nn.Parameter(priors.clone())
        self.gate_hidden = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU())
        self.gate_out = nn.Linear(dim // 4, 1)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        self.norm_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(
        self, q: Tensor, img: Tensor, pts: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update queries; return them and the image gates (B, G)."""
        qn = self.norm_q(q)
        from_img, _ = self.attn_img(qn, img, img, need_weights=False)
        from_pts, _ = self.attn_pts(qn, pts, pts, need_weights=False)
        gate = torch.sigmoid(
            self.gate_bias + self.gate_out(self.gate_hidden(qn)).squeeze(-1)
        )
        q = q + gate[..., None] * from_img + (1 - gate[..., None]) * from_pts
        qs = self.norm_s(q)
        q = q + self.self_attn(qs, qs, qs, need_weights=False)[0]
        q = q + self.ffn(self.norm_f(q))
        return q, gate


class SelectiveDecoder(nn.Module):
    """Stack of selective layers over learnable joint-group queries."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        groups: tuple[JointGroup, ...] = JOINT_GROUPS,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.queries = nn.Parameter(torch.randn(len(groups), dim) * 0.02)
        priors = torch.tensor([g.image_prior for g in groups])
        self.layers = nn.ModuleList(
            [SelectiveLayer(dim, num_heads, priors) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, img: Tensor, pts: Tensor) -> tuple[Tensor, Tensor]:
        """Group features (B, G, dim) and gates per layer (L, B, G)."""
        q = self.queries[None].expand(img.shape[0], -1, -1)
        gates = []
        for layer in self.layers:
            q, gate = layer(q, img, pts)
            gates.append(gate)
        return self.norm(q), torch.stack(gates)
