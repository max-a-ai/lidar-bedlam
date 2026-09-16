"""TokenHMR's pose tokenizer (Dwivedi et al. 2024) as a plain torch module.

A VQ-VAE over the 21 SMPL body joints (hands excluded) trained on AMASS
and MOYO poses: the encoder maps a pose to a latent sequence of 160 codes
of 256 dimensions, the decoder maps that sequence back to a pose. Every
latent the decoder sees at training time is a quantised code, so a pose
decoded from a nearby latent is a plausible human pose by construction.
The pseudo-GT fitter optimises in this latent space instead of over the
joint rotations, which keeps every candidate label realistic.

The architecture is ported layer by layer from
``third_party/TokenHMR/tokenization/models`` so the released weights
(``resources/pretrained-checkpoints/tokenhmr_tokenizer.pt``, the
``tokenizer.pth`` of the TokenHMR release with its yacs config flattened
to a dict) load without the vendored package, which builds an SMPL-H body
model at import. Rotations use the pytorch3d 6D convention (first two
matrix rows), as the tokenizer was trained with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

NUM_JOINTS = 21  # SMPL body joints 1..21: hands (22, 23) are not tokenised


def matrix_to_rot6d_rows(mat: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6): the first two rows (pytorch3d convention)."""
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def rot6d_rows_to_matrix(d6: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt on the two rows (pytorch3d)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class _ResBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.conv1 = nn.Conv1d(width, width, 3, 1, dilation, dilation)
        self.conv2 = nn.Conv1d(width, width, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.activation1(self.norm1(x)))
        h = self.conv2(self.activation2(self.norm2(h)))
        out: Tensor = x + h
        return out


class _Resnet1D(nn.Module):
    """``depth`` residual blocks with dilations 3^k, deepest first."""

    def __init__(self, width: int, depth: int, growth: int) -> None:
        super().__init__()
        blocks = [_ResBlock(width, growth**k) for k in range(depth)][::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.model(x)
        return out


class _Encoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        width: int,
        depth: int,
        growth: int,
        token_size_mul: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(6, width, 3, 1, 1), nn.ReLU()]
        layers += [
            nn.Upsample(((NUM_JOINTS * 2) // 10) * 10),
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
        ]
        for _ in range(token_size_mul - 1):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, width, 3, 1, 1),
                nn.ReLU(),
            ]
        layers.append(
            nn.Sequential(
                nn.Conv1d(width, width, 4, 2, 1),
                _Resnet1D(width, depth, growth),
            )
        )
        layers.append(nn.Conv1d(width, code_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)
        self.num_tokens = num_tokens

    def forward(self, rot6d: Tensor) -> Tensor:
        """(B, 21, 6) -> latent (B, code_dim, num_tokens)."""
        x = self.encoder(rot6d.permute(0, 2, 1))
        if x.shape[-1] != self.num_tokens:
            x = F.interpolate(
                x, size=self.num_tokens, mode="linear", align_corners=False
            )
        out: Tensor = x
        return out


class _Decoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        width: int,
        depth: int,
        growth: int,
        token_size_div: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(code_dim, width, 3, 1, 1),
            nn.ReLU(),
        ]
        sizes = [
            int(NUM_JOINTS + (num_tokens - NUM_JOINTS) * k / token_size_div)
            for k in range(token_size_div)
        ][::-1]
        for size in sizes:
            layers += [
                nn.Upsample(size),
                nn.Conv1d(width, width, 3, 1, 1),
                nn.ReLU(),
            ]
        layers.append(
            nn.Sequential(
                _Resnet1D(width, depth, growth),
                nn.Conv1d(width, width, 3, 1, 1),
            )
        )
        layers.append(nn.Conv1d(width, 6, 3, 1, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        """latent (B, code_dim, T) -> rot6d (B, 21, 6)."""
        out: Tensor = self.decoder(latent).permute(0, 2, 1)
        return out


class PoseTokenizer(nn.Module):
    """Encoder, codebook and decoder of the released tokenizer."""

    def __init__(self, arch: dict[str, Any]) -> None:
        super().__init__()
        self.code_dim = int(arch["CODE_DIM"])
        # the released weights were trained without the later NUM_TOKENS
        # override: the latent keeps the encoder's natural length, 160
        self.num_tokens = (
            ((NUM_JOINTS // 10) * 10)
            * 2 ** int(arch["TOKEN_SIZE_MUL"])
            // 2 ** int(arch["DOWN_T"])
        )
        self.encoder = _Encoder(
            self.code_dim,
            int(arch["WIDTH"]),
            int(arch["DEPTH"]),
            int(arch["DILATION_RATE"]),
            int(arch["TOKEN_SIZE_MUL"]),
            self.num_tokens,
        )
        self.decoder = _Decoder(
            self.code_dim,
            int(arch["WIDTH"]),
            int(arch["DEPTH"]),
            int(arch["DILATION_RATE"]),
            int(arch["TOKEN_SIZE_DIV"]),
            self.num_tokens,
        )
        self.codebook: Tensor
        self.register_buffer(
            "codebook", torch.zeros(int(arch["NB_CODE"]), self.code_dim)
        )

    @classmethod
    def load(cls, path: Path) -> PoseTokenizer:
        """The released weights; the module is frozen and in eval mode."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tok = cls(ck["arch"])
        state = {
            ("codebook" if k == "quantizer.codebook" else k): v
            for k, v in ck["net"].items()
        }
        tok.load_state_dict(state, strict=True)
        tok.eval()
        for p in tok.parameters():
            p.requires_grad_(False)
        return tok

    def encode(self, body_rotmats: Tensor) -> Tensor:
        """Body rotations (B, 21, 3, 3) -> continuous latent (B, C, T)."""
        out: Tensor = self.encoder(matrix_to_rot6d_rows(body_rotmats))
        return out

    def decode(self, latent: Tensor) -> Tensor:
        """Latent (B, C, T) -> body rotations (B, 21, 3, 3)."""
        return rot6d_rows_to_matrix(self.decoder(latent))

    def quantize(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Nearest codebook entries: quantised latent, code indices (B, T)."""
        b, c, t = latent.shape
        flat = latent.permute(0, 2, 1).reshape(-1, c)
        dist = (
            (flat**2).sum(-1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + (self.codebook**2).sum(-1)[None]
        )
        idx = dist.argmin(-1)
        q = self.codebook[idx].reshape(b, t, c).permute(0, 2, 1)
        return q, idx.reshape(b, t)
