"""BEV trunk: pillar voxelisation, HEAL weights, model plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lidar_bedlam.models.bev_pillars import (
    PERSON_Z_M,
    BevTrunk,
    PillarGrid,
    PointPillars,
    camera_to_lidar,
    roi_features,
    scatter_window,
    voxelize,
)
from lidar_bedlam.models.selective_attention import JOINT_GROUPS

BODY_MODELS = Path("resources/data/generated/body_models")
HEAL = BODY_MODELS / "heal_pointpillars.pth"


def test_voxelize_groups_points_into_pillars() -> None:
    grid = PillarGrid(max_points=2)
    # sample 0: three points in one cell (one dropped by max_points) and
    # one in another; sample 1: one point outside the x range
    a = [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0], [5.0, 5.0, 0.0]]
    b = [[200.0, 0.0, 0.0], [1.0, 1.0, 9.0], [1.0, 1.0, 9.0], [0.0, 0.0, 0.0]]
    points = torch.tensor([a, b])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    feats, num, coords = voxelize(points, valid, grid)
    assert feats.shape == (3, 2, 4)
    assert num.tolist() == [2, 1, 2]
    assert coords[:, 0].tolist() == [0, 0, 1]
    assert (coords[:, 1] == 0).all()
    cell = round((0.1 - grid.x_min) / grid.voxel)
    assert coords[0, 2].item() == cell and coords[0, 3].item() == cell
    # z is clamped into the grid, intensity is zero
    assert feats[2, :, 2].max() < grid.z_max
    assert (feats[..., 3] == 0).all()


def test_camera_to_lidar_axes_and_height() -> None:
    pts = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    out = camera_to_lidar(pts, valid)
    assert torch.allclose(out[0, :, 0], torch.tensor([3.0, 3.0]))  # fwd
    assert torch.allclose(out[0, :, 1], torch.tensor([-1.0, -1.0]))  # left
    assert out[0, :, 2].mean().item() == pytest.approx(PERSON_Z_M)
    assert (out[0, 0, 2] - out[0, 1, 2]).item() == pytest.approx(2.0)


def test_scatter_and_roi_window() -> None:
    pillars = torch.arange(1.0, 4.0)[:, None].expand(3, 2).clone()
    coords = torch.tensor([[0, 0, 10, 10], [0, 0, 11, 12], [0, 0, 50, 50]])
    origin = torch.tensor([[8, 8]])
    canvas = scatter_window(pillars, coords, origin, 1, 8)
    assert canvas.shape == (1, 2, 8, 8)
    assert canvas[0, 0, 2, 2] == 1.0 and canvas[0, 0, 3, 4] == 2.0
    assert canvas.sum() == 2 * (1.0 + 2.0)  # the far pillar is dropped
    bev = torch.arange(64.0).view(1, 1, 8, 8)
    patch = roi_features(bev, torch.tensor([[0, 7]]), roi=4)
    assert patch.shape == (1, 1, 4, 4)
    # rows clamped at the top edge (y = 0), columns at the right (x = 7)
    assert patch[0, 0, 0, 0] == 5.0
    assert patch[0, 0, -1, -1] == 1 * 8 + 7


def test_pointpillars_shapes_without_weights() -> None:
    model = PointPillars(window=16).eval()
    pts = torch.randn(2, 100, 3) * 0.3 + torch.tensor([20.0, -5.0, -0.9])
    valid = torch.ones(2, 100, dtype=torch.bool)
    bev, centre = model(pts, valid)
    assert bev.shape == (2, 256, 8, 8)
    assert (centre == 4).all()  # the person sits at the window centre


@pytest.mark.skipif(not HEAL.exists(), reason="HEAL checkpoint not linked")
def test_heal_weights_load_strictly_and_respond_to_points() -> None:
    model = PointPillars(window=32)
    model.load_heal(HEAL)
    model.eval()
    torch.manual_seed(0)
    pts = torch.randn(1, 300, 3) * 0.3 + torch.tensor([15.0, 2.0, -0.9])
    valid = torch.ones(1, 300, dtype=torch.bool)
    with torch.no_grad():
        bev, centre = model(pts, valid)
        empty, _ = model(pts, torch.zeros_like(valid))
    assert torch.isfinite(bev).all()
    patch = roi_features(bev, centre, 4)
    patch_empty = roi_features(empty, centre, 4)
    assert (patch - patch_empty).abs().mean() > 1e-3


def test_bev_trunk_group_features_and_frozen_pillars() -> None:
    trunk = BevTrunk(32, len(JOINT_GROUPS), None, window=16, roi=2)
    pts = torch.randn(3, 50, 3) + torch.tensor([0.0, 0.0, 8.0])
    feats = trunk(pts, None)
    assert feats.shape == (3, len(JOINT_GROUPS), 32)
    assert all(not p.requires_grad for p in trunk.pillars.parameters())
    assert not trunk.pillars.training  # frozen BN stays in eval mode
    feats.sum().backward()
    assert trunk.head[2].weight.grad is not None


@pytest.mark.skipif(
    not (BODY_MODELS / "smpl").exists(), reason="body models not linked"
)
def test_fusion_model_with_bev_trunk_trains() -> None:
    from lidar_bedlam.losses.smpl import FusionLoss
    from lidar_bedlam.models.fusion import ModelConfig, SelectiveFusionModel
    from lidar_bedlam.models.vit import VIT_TINY

    model = SelectiveFusionModel(
        ModelConfig(
            vit=VIT_TINY,
            dim=64,
            num_heads=4,
            num_layers=1,
            smpl_model_dir=BODY_MODELS,
            use_backbone=False,
            transl_anchor="points",
            trunk="bev",
            bev_weights=HEAL,
            bev_window=16,
            bev_roi=2,
        )
    )
    assert model.decoder is None and model.bev is not None
    b = 2
    batch = {
        "points": torch.randn(b, 40, 3) * 0.3 + torch.tensor([0.0, 0.0, 8.0]),
        "points_valid": torch.ones(b, 40, dtype=torch.bool),
        "intrinsics": torch.tensor(
            [[[500.0, 0, 128], [0, 500, 128], [0, 0, 1]]]
        ).expand(b, 3, 3),
        "has_smpl": torch.tensor([True, True]),
        "global_orient": torch.zeros(b, 3),
        "body_pose": torch.zeros(b, 69),
        "betas": torch.zeros(b, 10),
        "transl": torch.tensor([[0.0, 0.0, 8.0]] * b),
        "joints3d": torch.zeros(b, 24, 3),
        "joints3d_valid": torch.ones(b, 24, dtype=torch.bool),
        "kp2d": torch.zeros(b, 24, 3),
        "has_kp2d": torch.tensor([False, False]),
        "box3d": torch.tensor([[0.0, 0.0, 8.0, 0.9, 1.8, 1.0, 0.0]] * b),
        "has_box3d": torch.tensor([False, False]),
        "joint_convention_id": torch.tensor([0, 0]),
    }
    out = model(batch)
    assert out["gates"].shape == (1, b, len(JOINT_GROUPS))
    assert out["vertices"].shape == (b, 6890, 3)
    total, _ = FusionLoss()(out, batch)
    total.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(total)
    assert model.bev.head[2].weight.grad is not None
    g = model.heads.transl_head.weight.grad  # unselected where-branch
    assert g is None or not g.any()
    assert model.heads.offset_head.weight.grad is not None
