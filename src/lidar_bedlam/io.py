"""Readers for the raw file formats used by the datasets.

All readers return plain numpy arrays. Point clouds are float32 (N, 3) in
the file's native frame; depth maps are float32 (H, W).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("U", 4): "u4", ("I", 4): "i4"}


def read_image(path: Path) -> UInt8Array:
    """Read an image as RGB uint8 (H, W, 3); alpha channels are dropped."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> NDArray[np.bool_]:
    """Read a binary mask image as bool (H, W); any non-zero pixel is True."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 0


def read_exr_depth(path: Path, channel: str = "Depth") -> Float32Array:
    """Read a single-channel float EXR (BEDLAM depth) as float32 (H, W)."""
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        arr = f.channels()[channel].pixels
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_pcd(path: Path) -> Float32Array:
    """Read a PCD v0.7 file (ascii or binary) and return xyz float32 (N, 3).

    Only the x, y, z fields are returned; extra fields such as rgb are
    skipped. ``binary_compressed`` is not supported.
    """
    with open(path, "rb") as fh:
        header: dict[str, list[str]] = {}
        while True:
            line = fh.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            header[key] = vals
            if key == "DATA":
                break
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n = int(header["POINTS"][0])
        data_kind = header["DATA"][0]
        dtype = np.dtype(
            [
                (f, _PCD_TYPES[(t, s)], (c,) if c > 1 else ())
                for f, t, s, c in zip(
                    fields, types, sizes, counts, strict=True
                )
            ]
        )
        if data_kind == "binary":
            rec = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype)
        elif data_kind == "ascii":
            rec = np.loadtxt(fh, dtype=dtype, max_rows=n)
        else:
            msg = f"unsupported PCD DATA kind {data_kind!r} in {path}"
            raise ValueError(msg)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=-1)
    return np.ascontiguousarray(xyz, dtype=np.float32)


def read_ply_xyz(path: Path) -> Float32Array:
    """Read a PLY file with float x/y/z vertex properties as float32 (N, 3).

    Supports ``binary_little_endian`` and ``ascii`` vertex-only files.
    """
    with open(path, "rb") as fh:
        fmt = ""
        n = 0
        props: list[str] = []
        while True:
            line = fh.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("property"):
                props.append(line.split()[2])
            elif line == "end_header":
                break
        if fmt == "binary_little_endian":
            dtype = np.dtype([(p, "<f4") for p in props])
            rec = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype)
        elif fmt == "ascii":
            dtype = np.dtype([(p, "f4") for p in props])
            rec = np.loadtxt(fh, dtype=dtype, max_rows=n)
        else:
            msg = f"unsupported PLY format {fmt!r} in {path}"
            raise ValueError(msg)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=-1)
    return np.ascontiguousarray(xyz, dtype=np.float32)
