"""Model building blocks and a full forward/backward pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lidar_bedlam.losses.smpl import FusionLoss
from lidar_bedlam.models.fusion import (
    ModelConfig,
    SelectiveFusionModel,
    box_from_vertices,
    translation_from_raw,
)
from lidar_bedlam.models.point_encoder import (
    PointTokenizer,
    farthest_point_sample,
)
from lidar_bedlam.models.rotation import matrix_to_rot6d, rot6d_to_matrix
from lidar_bedlam.models.selective_attention import JOINT_GROUPS
from lidar_bedlam.models.vit import VIT_TINY, ViT, ViTConfig

SMPL_DIR = Path("resources/data/generated/body_models")


def test_rot6d_roundtrip_and_orthonormal() -> None:
    x = torch.randn(5, 6)
    r = rot6d_to_matrix(x)
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(r @ r.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.det(r), torch.ones(5), atol=1e-5)
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(r)), r, atol=1e-5)


def test_joint_groups_cover_all_smpl_joints_once() -> None:
    joints = sorted(j for g in JOINT_GROUPS for j in g.joints)
    assert joints == list(range(24))


def test_vit_tokens_and_pos_embed_interpolation() -> None:
    vit = ViT(VIT_TINY)
    tokens, grid = vit(torch.randn(2, 3, 64, 96))
    assert grid == (4, 6) and tokens.shape == (2, 24, 64)


def test_vit_loads_vitpose_style_state_dict() -> None:
    from lidar_bedlam.models.vit import load_backbone_weights

    src = ViT(
        ViTConfig(embed_dim=64, depth=2, num_heads=2, pretrain_grid=(4, 4))
    )
    sd = {"backbone." + k: v for k, v in src.state_dict().items()}
    path = Path("/tmp/claude_test_vit.ckpt")
    torch.save({"state_dict": sd}, path)
    dst = ViT(
        ViTConfig(embed_dim=64, depth=2, num_heads=2, pretrain_grid=(4, 4))
    )
    missing, unexpected = load_backbone_weights(dst, path)
    assert not missing and not unexpected
    assert torch.equal(dst.pos_embed, src.pos_embed)


def test_farthest_point_sample_spreads() -> None:
    xyz = torch.tensor([[[0.0, 0, 0], [10, 0, 0], [0.1, 0, 0], [0, 10, 0]]])
    idx = farthest_point_sample(xyz, 3)[0].tolist()
    assert idx[0] == 0 and set(idx[1:]) == {1, 3}


def test_point_tokenizer_shapes() -> None:
    tok = PointTokenizer(dim=32, num_tokens=8, k=4, num_heads=4)
    tokens, centres = tok(torch.randn(2, 50, 3))
    assert tokens.shape == (2, 8, 32) and centres.shape == (2, 8, 3)


def test_translation_from_raw_centre_is_optical_axis() -> None:
    k = torch.tensor([[[500.0, 0, 128], [0, 500, 128], [0, 0, 1]]])
    raw = torch.tensor([[0.0, 0.0, float(np.log(6.0))]])
    t = translation_from_raw(raw, k, 256)
    assert torch.allclose(t, torch.tensor([[0.0, 0.0, 6.0]]))


def test_box_from_vertices_matches_numpy() -> None:
    from lidar_bedlam.geometry.boxes import oriented_box_from_points

    rng = np.random.default_rng(0)
    verts = rng.normal(size=(1, 200, 3))
    yaw = 0.4
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # fwd = R @ z
    box = box_from_vertices(torch.tensor(verts), torch.tensor(rot)[None])
    ref = oriented_box_from_points(
        verts[0], np.arctan2(rot[2, 2], rot[0, 2]), 1
    )
    assert np.allclose(box[0].numpy(), ref, atol=1e-6)


def _tiny_model(smpl: bool) -> SelectiveFusionModel:
    cfg = ModelConfig(
        vit=VIT_TINY, dim=64, num_heads=4, num_layers=2, point_tokens=8,
        point_knn=4, smpl_model_dir=SMPL_DIR if smpl else None,
        freeze_backbone=False,
    )  # fmt: skip
    return SelectiveFusionModel(cfg)


def test_forward_without_smpl() -> None:
    model = _tiny_model(smpl=False)
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "points": torch.randn(2, 40, 3) + torch.tensor([0.0, 0.0, 8.0]),
        "intrinsics": torch.tensor(
            [[[500.0, 0, 32], [0, 500, 32], [0, 0, 1]]]
        ).expand(2, 3, 3),
    }
    out = model(batch)
    assert out["global_orient"].shape == (2, 1, 3, 3)
    assert out["body_pose"].shape == (2, 23, 3, 3)
    assert out["betas"].shape == (2, 10) and out["transl"].shape == (2, 3)
    assert out["gates"].shape == (2, 2, len(JOINT_GROUPS))
    assert out["box_conf"].shape == (2,) and (out["box_conf"] <= 1).all()
    assert ((out["gates"] >= 0) & (out["gates"] <= 1)).all()
    # routing prior: 3D groups start LiDAR-heavy, semantic groups image-heavy
    from lidar_bedlam.models.selective_attention import LIDAR_GROUPS

    g0 = out["gates"][0, 0]
    for i, g in enumerate(JOINT_GROUPS):
        if g.name in LIDAR_GROUPS:
            assert g0[i] < 0.2
        else:
            assert g0[i] > 0.8


@pytest.mark.skipif(not SMPL_DIR.exists(), reason="SMPL model files absent")
def test_forward_backward_with_smpl_and_loss() -> None:
    model = _tiny_model(smpl=True)
    b = 2
    batch = {
        "image": torch.randn(b, 3, 64, 64),
        "points": torch.randn(b, 40, 3) + torch.tensor([0.0, 0.0, 8.0]),
        "intrinsics": torch.tensor(
            [[[500.0, 0, 32], [0, 500, 32], [0, 0, 1]]]
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
        "box3d": torch.zeros(b, 7),
        "has_box3d": torch.tensor([True, True]),
    }
    out = model(batch)
    assert out["vertices"].shape == (b, 6890, 3)
    assert out["joints3d"].shape == (b, 24, 3) and out["kp2d"].shape == (
        b,
        24,
        2,
    )
    assert out["box3d"].shape == (b, 7)
    total, parts = FusionLoss()(out, batch)
    total.backward()  # type: ignore[no-untyped-call]
    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(total)


def test_forward_from_precomputed_tokens_without_backbone() -> None:
    cfg = ModelConfig(
        vit=VIT_TINY, dim=64, num_heads=4, num_layers=2, point_tokens=8,
        point_knn=4, use_backbone=False, crop_size=64,
    )  # fmt: skip
    model = SelectiveFusionModel(cfg)
    assert model.backbone is None
    batch = {
        "tokens": torch.randn(2, 16, VIT_TINY.embed_dim).half(),
        "has_image": torch.tensor([True, False]),
        "points": torch.randn(2, 40, 3) + torch.tensor([0.0, 0.0, 8.0]),
        "intrinsics": torch.tensor(
            [[[500.0, 0, 32], [0, 500, 32], [0, 0, 1]]]
        ).expand(2, 3, 3),
    }
    out = model(batch)
    assert out["transl"].shape == (2, 3)
    # the LiDAR-only sample used the learned no-image token
    img = model.image_tokens(batch)
    assert torch.allclose(img[1], model.no_image.expand_as(img)[1])
