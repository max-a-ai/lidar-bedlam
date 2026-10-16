"""Sensor-modality dropout and the forced single-sensor evaluation."""

from __future__ import annotations

import torch

from lidar_bedlam.train.modality import drop_modalities, force_modality


def _batch(n: int = 8) -> dict[str, torch.Tensor]:
    valid = torch.ones(n, 5, dtype=torch.bool)
    valid[0] = False  # an empty cloud
    has_image = torch.ones(n, dtype=torch.bool)
    has_image[1] = False  # a LiDAR-only record
    return {
        "points": torch.randn(n, 5, 3),
        "points_valid": valid,
        "has_image": has_image,
    }


def test_dropout_never_removes_both_and_respects_absent_sensors() -> None:
    g = torch.Generator().manual_seed(0)
    seen_img_only = seen_pts_only = False
    for _ in range(50):
        b = _batch()
        out = drop_modalities(b, 0.5, 0.5, g)
        has_img, has_pts = out["has_image"], out["has_points"]
        assert (has_img | has_pts).all()
        assert not has_pts[0] and has_img[0]  # empty cloud keeps its image
        assert not has_img[1] and has_pts[1]  # LiDAR-only keeps its points
        dropped = b["points_valid"].any(1) & ~out["points_valid"].any(1)
        assert (out["points"][dropped] == 0).all()
        seen_img_only |= bool((has_img & ~has_pts)[2:].any())
        seen_pts_only |= bool((~has_img & has_pts)[2:].any())
    assert seen_img_only and seen_pts_only


def test_dropout_off_is_identity() -> None:
    b = _batch()
    assert drop_modalities(b, 0.0, 0.0) is b


def test_force_modality_variants() -> None:
    b = _batch()
    img = force_modality(b, "image")
    assert img["has_image"][0] and img["has_image"][2:].all()
    assert not img["has_image"][1]  # the LiDAR-only record has no image
    assert not img["has_points"][2:].any()
    assert not img["has_points"][0] and img["has_points"][1]  # LiDAR-only
    pts = force_modality(b, "points")
    assert pts["has_points"][1:].all()
    assert not pts["has_image"][2:].any()
    assert pts["has_image"][0]  # the empty cloud keeps its image
    assert force_modality(b, "both") is b


def test_fusion_model_substitutes_no_points_token() -> None:
    from lidar_bedlam.models.fusion import ModelConfig, SelectiveFusionModel
    from lidar_bedlam.models.vit import VIT_TINY

    model = SelectiveFusionModel(
        ModelConfig(
            vit=VIT_TINY, dim=32, num_heads=4, num_layers=1, point_tokens=4,
            point_knn=4, use_backbone=False, transl_anchor="points",
        )
    ).eval()  # fmt: skip
    with torch.no_grad():
        model.no_points.fill_(3.0)
    b = 2
    batch = {
        "tokens": torch.randn(b, 4, VIT_TINY.embed_dim),
        "points": torch.randn(b, 16, 3) + torch.tensor([0.0, 0.0, 8.0]),
        "points_valid": torch.ones(b, 16, dtype=torch.bool),
        "intrinsics": torch.tensor(
            [[[500.0, 0, 128], [0, 500, 128], [0, 0, 1]]]
        ).expand(b, 3, 3),
        "has_image": torch.tensor([True, True]),
    }
    with torch.no_grad():
        full = model(batch)
        only_img = model(force_modality(batch, "image"))
        only_pts = model(force_modality(batch, "points"))
    for out in (full, only_img, only_pts):
        assert torch.isfinite(out["transl"]).all()
    assert not torch.allclose(full["transl"], only_img["transl"])
    # without points the translation comes from the camera parameterisation
    assert not torch.allclose(only_img["transl"], only_pts["transl"])
