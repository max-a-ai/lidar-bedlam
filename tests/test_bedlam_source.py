"""BedlamFramesSource indexes only frames whose png and depth exist."""

from __future__ import annotations

from pathlib import Path

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
