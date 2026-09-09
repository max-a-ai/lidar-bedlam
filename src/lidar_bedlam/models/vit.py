"""Plain ViT backbone compatible with ViTPose / HMR2 / TokenHMR weights.

Parameter names follow the ViTPose implementation (``patch_embed.proj``,
``pos_embed`` with a leading class slot, ``blocks.N.{norm1,attn,norm2,mlp}``,
``last_norm``), so the ViT-H weights shipped with TokenHMR/HMR2 load
directly. The positional embedding is interpolated to the input grid, so
any input size that is a multiple of the patch size works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


@dataclass(frozen=True)
class ViTConfig:
    """Architecture hyper-parameters."""

    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16
    patch_size: int = 16
    mlp_ratio: float = 4.0
    pretrain_grid: tuple[int, int] = (16, 12)  # HMR2 trains on 256x192


VIT_H = ViTConfig()
VIT_TINY = ViTConfig(embed_dim=64, depth=2, num_heads=2, pretrain_grid=(4, 4))


class Mlp(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block."""
        out: Tensor = self.fc2(self.act(self.fc1(x)))
        return out


class Attention(nn.Module):
    """Multi-head self-attention with fused qkv projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        proj: Tensor = self.proj(out.transpose(1, 2).reshape(b, n, c))
        return proj


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block."""
        x = x + self.attn(self.norm1(x))
        out: Tensor = x + self.mlp(self.norm2(x))
        return out


class PatchEmbed(nn.Module):
    """Conv patchifier (ViTPose uses padding 4 with a 16x16 stride-16 conv)."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=4
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Return tokens (B, N, C) and the token grid (H', W')."""
        x = self.proj(x)
        hp, wp = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (hp, wp)


class ViT(nn.Module):
    """ViT returning patch tokens (B, N, C) and their grid shape."""

    def __init__(self, cfg: ViTConfig = VIT_H) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.embed_dim)
        gh, gw = cfg.pretrain_grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, gh * gw + 1, cfg.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.last_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def _pos_embed_for(self, grid: tuple[int, int]) -> Tensor:
        gh, gw = self.cfg.pretrain_grid
        cls, patches = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        if grid == (gh, gw):
            return patches + cls
        p = patches.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=grid, mode="bicubic", align_corners=False)
        return p.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1) + cls

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Tokens (B, H'*W', C) for images (B, 3, H, W)."""
        tokens, grid = self.patch_embed(x)
        tokens = tokens + self._pos_embed_for(grid)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.last_norm(tokens), grid


def load_backbone_weights(
    vit: ViT, checkpoint: Path, prefix: str = "backbone."
) -> tuple[list[str], list[str]]:
    """Load ViTPose-style weights from an HMR2/TokenHMR checkpoint.

    Returns the missing and unexpected keys for inspection.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    result = vit.load_state_dict(sub, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)
