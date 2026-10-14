"""PointPillars BEV encoder, ported from HEAL / OpenCOOD (OpenPCDet credits).

The LiDAR branch of HEAL's ``HeterBaseline_opv2v_lidar_attfuse`` checkpoint:
pillar feature net (VFE) -> scatter to a 0.4 m pseudo-image -> BEV
backbone (three ResNet-style stages, upsampled and concatenated to 384
channels at stride 2) -> shrink conv (256 channels). Weights are loaded
from the published state dict by prefix (``encoder_m1``, ``backbone_m1``,
``shrinker_m1``); nothing of HEAL is imported.

Pillars are computed on HEAL's global grid (+-102.4 m, 512 x 512 cells,
z in [-3, 1] m) so the absolute-coordinate features match the pretrained
weights, but only a window of cells around the person is scattered and
run through the backbone: a person covers a few cells, the backbone's
receptive field a few dozen.

Frame: HEAL works in a LiDAR frame (x forward, y left, z up); our records
are in the camera frame (x right, y down, z forward). ``camera_to_lidar``
converts, and the sample's point centroid is placed at z = -0.9 m, where
a pedestrian's centre sits for a roof LiDAR 1.9 m above ground, so
cameras at any height (BEDLAM) stay inside the pretrained z range.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

PERSON_Z_M = -0.9  # where the point centroid is placed in the LiDAR frame


@dataclass(frozen=True)
class PillarGrid:
    """HEAL's pillar grid; the defaults are the OPV2V checkpoint's."""

    x_min: float = -102.4
    y_min: float = -102.4
    z_min: float = -3.0
    x_max: float = 102.4
    y_max: float = 102.4
    z_max: float = 1.0
    voxel: float = 0.4
    z_size: float = 4.0
    max_points: int = 32
    num_point_features: int = 4  # x, y, z, intensity

    @property
    def nx(self) -> int:
        return round((self.x_max - self.x_min) / self.voxel)

    @property
    def ny(self) -> int:
        return round((self.y_max - self.y_min) / self.voxel)


def camera_to_lidar(points: Tensor, valid: Tensor) -> Tensor:
    """Camera frame (x right, y down, z fwd) -> LiDAR (x fwd, y left, z up).

    The valid points' centroid is moved to ``PERSON_Z_M`` in height.
    """
    x = points[..., 2]
    y = -points[..., 0]
    z = -points[..., 1]
    w = valid.to(z.dtype)
    n = w.sum(1).clamp(min=1.0)
    z_mean = (z * w).sum(1) / n
    z = z - z_mean[:, None] + PERSON_Z_M
    return torch.stack([x, y, z], -1)


def voxelize(
    points: Tensor, valid: Tensor, grid: PillarGrid
) -> tuple[Tensor, Tensor, Tensor]:
    """Group points into pillars on the global grid.

    ``points`` (B, N, 3) in the LiDAR frame, ``valid`` (B, N). Returns the
    pillar points (P, max_points, 4) zero-padded with intensity 0, the
    number of points per pillar (P,) and the coords (P, 4) as
    ``(batch, z=0, iy, ix)`` in HEAL's order. Points outside the x/y
    range are dropped; z is clamped into range.
    """
    b_n, n = points.shape[:2]
    pts = points.float()
    x, y, z = pts.unbind(-1)
    inside = (
        valid
        & (x >= grid.x_min)
        & (x < grid.x_max)
        & (y >= grid.y_min)
        & (y < grid.y_max)
    )
    z = z.clamp(grid.z_min, grid.z_max - 1e-3)
    ix = ((x - grid.x_min) / grid.voxel).floor().long()
    iy = ((y - grid.y_min) / grid.voxel).floor().long()
    ix = ix.clamp(0, grid.nx - 1)
    iy = iy.clamp(0, grid.ny - 1)
    bidx = torch.arange(b_n, device=pts.device)[:, None].expand(b_n, n)
    key = (bidx * grid.ny + iy) * grid.nx + ix
    feats4 = torch.stack([x, y, z, torch.zeros_like(x)], -1)
    key = key[inside]
    feats4 = feats4[inside]
    order = torch.argsort(key, stable=True)
    key = key[order]
    feats4 = feats4[order]
    uniq, inverse, counts = torch.unique_consecutive(
        key, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(key), device=pts.device) - starts[inverse]
    keep = rank < grid.max_points
    out = pts.new_zeros(len(uniq), grid.max_points, grid.num_point_features)
    out[inverse[keep], rank[keep]] = feats4[keep]
    num = counts.clamp(max=grid.max_points)
    cells = grid.nx * grid.ny
    coords = torch.stack(
        [
            uniq // cells,
            torch.zeros_like(uniq),
            (uniq % cells) // grid.nx,
            uniq % grid.nx,
        ],
        -1,
    )
    return out, num, coords


class PFNLayer(nn.Module):
    """One pillar feature net layer: linear, BN, ReLU, max over points."""

    def __init__(self, cin: int, cout: int, last: bool) -> None:
        super().__init__()
        self.last = last
        if not last:
            cout //= 2
        self.linear = nn.Linear(cin, cout, bias=False)
        self.norm = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.relu(x)
        x_max = x.max(dim=1, keepdim=True)[0]
        if self.last:
            return x_max
        return torch.cat([x, x_max.expand(-1, x.shape[1], -1)], 2)


class PillarVFE(nn.Module):
    """Pillar points (P, M, 4) -> pillar features (P, 64)."""

    def __init__(self, grid: PillarGrid, num_filters: int = 64) -> None:
        super().__init__()
        self.grid = grid
        # xyz + intensity, cluster offsets (3), centre offsets (3)
        self.pfn_layers = nn.ModuleList(
            [PFNLayer(grid.num_point_features + 6, num_filters, last=True)]
        )
        self.x_offset = grid.voxel / 2 + grid.x_min
        self.y_offset = grid.voxel / 2 + grid.y_min
        self.z_offset = grid.z_size / 2 + grid.z_min

    def forward(self, feats: Tensor, num: Tensor, coords: Tensor) -> Tensor:
        g = self.grid
        mean = feats[:, :, :3].sum(1, keepdim=True) / num.to(feats.dtype).view(
            -1, 1, 1
        )
        f_cluster = feats[:, :, :3] - mean
        cx = coords[:, 3].to(feats.dtype)[:, None] * g.voxel + self.x_offset
        cy = coords[:, 2].to(feats.dtype)[:, None] * g.voxel + self.y_offset
        cz = coords[:, 1].to(feats.dtype)[:, None] * g.z_size + self.z_offset
        f_center = torch.stack(
            [feats[:, :, 0] - cx, feats[:, :, 1] - cy, feats[:, :, 2] - cz],
            -1,
        )
        x = torch.cat([feats, f_cluster, f_center], -1)
        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :] < num[:, None]
        )
        x = x * mask.to(x.dtype)[..., None]
        for pfn in self.pfn_layers:
            x = pfn(x)
        return x.squeeze(1)


def scatter_window(
    pillars: Tensor,
    coords: Tensor,
    origin: Tensor,
    batch_size: int,
    window: int,
) -> Tensor:
    """Pillar features (P, C) -> pseudo-image (B, C, window, window).

    ``origin`` (B, 2) is the ``(iy, ix)`` grid cell of each sample's window
    corner; pillars outside their sample's window are dropped.
    """
    b = coords[:, 0]
    ly = coords[:, 2] - origin[b, 0]
    lx = coords[:, 3] - origin[b, 1]
    keep = (ly >= 0) & (ly < window) & (lx >= 0) & (lx < window)
    canvas = pillars.new_zeros(batch_size, pillars.shape[1], window, window)
    canvas[b[keep], :, ly[keep], lx[keep]] = pillars[keep]
    return canvas


class BevBackbone(nn.Module):
    """HEAL's BaseBEVBackbone: three stages, upsampled and concatenated."""

    def __init__(
        self,
        cin: int = 64,
        layer_nums: tuple[int, ...] = (3, 5, 8),
        strides: tuple[int, ...] = (2, 2, 2),
        filters: tuple[int, ...] = (64, 128, 256),
        up_strides: tuple[int, ...] = (1, 2, 4),
        up_filters: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        c_in = [cin, *filters[:-1]]
        for i, (n_layers, stride, c) in enumerate(
            zip(layer_nums, strides, filters, strict=True)
        ):
            layers: list[nn.Module] = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in[i], c, 3, stride=stride, bias=False),
                nn.BatchNorm2d(c, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(n_layers):
                layers += [
                    nn.Conv2d(c, c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ]
            self.blocks.append(nn.Sequential(*layers))
            up = up_strides[i]
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c, up_filters[i], up, stride=up, bias=False
                    ),
                    nn.BatchNorm2d(up_filters[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )
        self.out_channels = sum(up_filters)

    def forward(self, x: Tensor) -> Tensor:
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks, strict=True):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, 1)


class ShrinkConv(nn.Module):
    """HEAL's DownsampleConv with one DoubleConv: 384 -> 256 channels."""

    def __init__(self, cin: int = 384, cout: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# HEAL state-dict prefix -> our attribute
PREFIXES = {
    "encoder_m1.pillar_vfe.": "vfe.",
    "backbone_m1.": "backbone.",
    "shrinker_m1.layers.0.double_conv.": "shrink.layers.0.",
}


class PointPillars(nn.Module):
    """Points in the LiDAR frame -> 256-channel BEV features at 0.8 m.

    ``forward`` returns the feature window (B, 256, W/2, W/2) and the
    stride-2 cell of each sample's point centroid inside it (B, 2).
    """

    out_channels = 256
    stride = 2

    def __init__(self, window: int = 32, grid: PillarGrid | None = None):
        super().__init__()
        if window % 8:
            msg = f"window must be a multiple of 8, got {window}"
            raise ValueError(msg)
        self.grid = grid or PillarGrid()
        self.window = window
        self.vfe = PillarVFE(self.grid)
        self.backbone = BevBackbone()
        self.shrink = ShrinkConv(self.backbone.out_channels, self.out_channels)

    def load_heal(self, path: Path) -> None:
        """Load the LiDAR branch of a HEAL checkpoint, strictly."""
        raw: dict[str, Tensor] = torch.load(
            path, map_location="cpu", weights_only=True
        )
        state: dict[str, Tensor] = {}
        for key, value in raw.items():
            for src, dst in PREFIXES.items():
                if key.startswith(src):
                    state[dst + key[len(src) :]] = value
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            msg = (
                f"HEAL weights do not match: missing {sorted(missing)[:5]}, "
                f"unexpected {sorted(unexpected)[:5]}"
            )
            raise RuntimeError(msg)

    def window_origin(self, points: Tensor, valid: Tensor) -> Tensor:
        """Grid cell (iy, ix) of each sample's window corner."""
        g = self.grid
        w = valid.to(points.dtype)[..., None]
        c = (points * w).sum(1) / w.sum(1).clamp(min=1.0)
        ix = ((c[:, 0] - g.x_min) / g.voxel).floor().long()
        iy = ((c[:, 1] - g.y_min) / g.voxel).floor().long()
        half = self.window // 2
        oy = (iy - half).clamp(0, g.ny - self.window)
        ox = (ix - half).clamp(0, g.nx - self.window)
        return torch.stack([oy, ox], -1)

    def forward(self, points: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        feats, num, coords = voxelize(points, valid, self.grid)
        pillars = self.vfe(feats, num, coords)
        origin = self.window_origin(points, valid)
        canvas = scatter_window(
            pillars, coords, origin, points.shape[0], self.window
        )
        bev: Tensor = self.shrink(self.backbone(canvas))
        g = self.grid
        w = valid.to(points.dtype)[..., None]
        c = (points * w).sum(1) / w.sum(1).clamp(min=1.0)
        ix = ((c[:, 0] - g.x_min) / g.voxel).floor().long()
        iy = ((c[:, 1] - g.y_min) / g.voxel).floor().long()
        centre = torch.stack(
            [
                (iy - origin[:, 0]) // self.stride,
                (ix - origin[:, 1]) // self.stride,
            ],
            -1,
        )
        return bev, centre


def roi_features(bev: Tensor, centre: Tensor, roi: int) -> Tensor:
    """The ``roi x roi`` window of cells around ``centre`` (B, 2) as
    (B, C, roi, roi); windows are clamped into the map."""
    b, _, h, w = bev.shape
    offs = torch.arange(roi, device=bev.device) - roi // 2
    yy = (centre[:, 0:1] + offs[None]).clamp(0, h - 1)  # (B, roi)
    xx = (centre[:, 1:2] + offs[None]).clamp(0, w - 1)
    bidx = torch.arange(b, device=bev.device)[:, None, None]
    patch = bev.permute(0, 2, 3, 1)[bidx, yy[:, :, None], xx[:, None, :]]
    return patch.permute(0, 3, 1, 2)


class BevTrunk(nn.Module):
    """Camera-frame points -> per-joint-group features (B, G, dim).

    A frozen HEAL PointPillars gives the BEV map; the ``roi x roi`` cells
    around the person are flattened and mapped to the joint-group features
    that ``SmplHeads`` consume. ``finetune`` unfreezes the pillars.
    """

    def __init__(
        self,
        dim: int,
        num_groups: int,
        weights: Path | None,
        window: int = 32,
        roi: int = 4,
        finetune: bool = False,
    ) -> None:
        super().__init__()
        self.pillars = PointPillars(window)
        if weights is not None:
            if not weights.exists():
                msg = f"HEAL pillar weights not found: {weights}"
                raise FileNotFoundError(msg)
            self.pillars.load_heal(weights)
        self.finetune = finetune
        if not finetune:
            for p in self.pillars.parameters():
                p.requires_grad_(False)
            self.pillars.eval()
        self.roi = roi
        self.num_groups = num_groups
        self.dim = dim
        cin = PointPillars.out_channels * roi * roi
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(cin),
            nn.Linear(cin, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, num_groups * dim),
        )
        self.group_embed = nn.Parameter(torch.zeros(1, num_groups, dim))

    def train(self, mode: bool = True) -> BevTrunk:
        """Frozen pillars keep their batch-norm statistics."""
        super().train(mode)
        if not self.finetune:
            self.pillars.eval()
        return self

    def forward(self, points: Tensor, valid: Tensor | None) -> Tensor:
        if valid is None:
            valid = torch.ones(
                points.shape[:2], dtype=torch.bool, device=points.device
            )
        pts = camera_to_lidar(points.float(), valid)
        if self.finetune:
            bev, centre = self.pillars(pts, valid)
        else:
            with torch.no_grad():
                bev, centre = self.pillars(pts, valid)
        patch = roi_features(bev, centre, self.roi)
        feats: Tensor = self.head(patch.float())
        return feats.view(-1, self.num_groups, self.dim) + self.group_embed


def heal_config_summary(path: Path) -> dict[str, Any]:
    """Shapes of the HEAL tensors the trunk loads (for diagnostics)."""
    raw: dict[str, Tensor] = torch.load(
        path, map_location="cpu", weights_only=True
    )
    return {
        k: tuple(v.shape)
        for k, v in raw.items()
        if any(k.startswith(p) for p in PREFIXES)
    }
