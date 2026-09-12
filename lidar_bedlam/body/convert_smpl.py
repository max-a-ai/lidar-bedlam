"""Convert legacy chumpy-based SMPL ``.pkl`` files to chumpy-free ``.pkl``.

The original SMPL release pickles ``chumpy`` objects, which no longer load
on modern numpy. This module registers a minimal stub so the arrays can be
extracted, then re-pickles plain numpy arrays, which ``smplx.SMPL`` loads
directly (it only reads ``.pkl`` for SMPL).

Usage::

    uv run python -m lidar_bedlam.body.convert_smpl \
        resources/data/body_models/smpl/SMPL_NEUTRAL.pkl \
        resources/data/generated/body_models/smpl/SMPL_NEUTRAL.pkl
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

_KEYS = (
    "v_template",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "kintree_table",
    "weights",
    "f",
)


class _Ch:
    """Stand-in for ``chumpy.Ch`` that keeps only the array payload."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        payload = self.__dict__.get("x", self.__dict__.get("_x"))
        arr = np.asarray(payload)
        return arr.astype(dtype) if dtype is not None else arr


def _install_chumpy_stub() -> None:
    ch = types.ModuleType("chumpy")
    ch.Ch = _Ch  # type: ignore[attr-defined]
    chch = types.ModuleType("chumpy.ch")
    chch.Ch = _Ch  # type: ignore[attr-defined]
    reorder = types.ModuleType("chumpy.reordering")
    reorder.Select = _Ch  # type: ignore[attr-defined]
    reorder.Transpose = _Ch  # type: ignore[attr-defined]
    sys.modules.setdefault("chumpy", ch)
    sys.modules.setdefault("chumpy.ch", chch)
    sys.modules.setdefault("chumpy.reordering", reorder)


def _to_array(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "toarray"):  # scipy sparse matrix
        return np.asarray(value.toarray())
    return np.asarray(value)


def convert(src: Path, dst: Path) -> None:
    """Write the SMPL arrays of ``src`` into a chumpy-free ``dst`` pickle."""
    _install_chumpy_stub()
    with open(src, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    arrays = {k: _to_array(data[k]) for k in _KEYS}
    arrays["shapedirs"] = arrays["shapedirs"][:, :, :10]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        pickle.dump(arrays, fh, protocol=4)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``convert_smpl SRC.pkl DST.npz``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("usage: convert_smpl SRC.pkl DST.pkl\n")
        return 2
    convert(Path(args[0]), Path(args[1]))
    sys.stdout.write(f"wrote {args[1]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
