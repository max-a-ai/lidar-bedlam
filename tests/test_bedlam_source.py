"""BedlamFramesSource indexes only frames whose png and depth exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.data.bedlam import BedlamFramesSource


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_incomplete_frames_are_skipped(tmp_path: Path) -> None:
    g, seq = "20990101_group", "seq_000001"
    base = tmp_path / g
    for frame, persons in (("0000", ("00", "01")), ("0005", ("00",))):
        for p in persons:
            _touch(base / "masks" / seq / f"{seq}_{frame}_{p}_body.png")
    # only frame 0000 is fully extracted
    _touch(base / "png" / seq / f"{seq}_0000.png")
    _touch(base / "depth" / seq / f"{seq}_0000_depth.exr")
    _touch(base / "png" / seq / f"{seq}_0005.png")  # depth still missing
    src = BedlamFramesSource(tmp_path, frame_stride=5)
    assert len(src) == 2 and src.incomplete == 1
    assert {src.meta(i).person for i in range(2)} == {"00", "01"}
    assert src.meta(0).frame == "0000"


def test_person_mask_unions_body_clothing_and_hair(tmp_path: Path) -> None:
    from PIL import Image

    g, seq, frame = "20990101_hair", "seq_000001", "0000"
    base = tmp_path / g / "masks" / seq
    base.mkdir(parents=True)
    parts = (
        ("body", slice(4, 8)),
        ("clothing", slice(6, 10)),
        ("hair", slice(0, 3)),
    )
    for part, rows in parts:
        m = np.zeros((12, 12), np.uint8)
        m[rows, 2:6] = 255
        Image.fromarray(m).save(base / f"{seq}_{frame}_00_{part}.png")
    _touch(tmp_path / g / "png" / seq / f"{seq}_{frame}.png")
    _touch(tmp_path / g / "depth" / seq / f"{seq}_{frame}_depth.exr")
    src = BedlamFramesSource(tmp_path, frame_stride=1)
    mask = src.person_masks(g, seq, frame)["00"]
    assert mask.shape == (12, 12)
    assert mask[0:3, 2:6].all()  # hair
    assert mask[4:10, 2:6].all()  # body + clothing
    assert int(mask.sum()) == (3 + 6) * 4
