"""Stage 2 of the SAM 3D Body row: SMPL fitted to the MHR meshes.

Uses the conversion tool of the MHR repository (optimisation through a
barycentric MHR-to-SMPL surface mapping). The tool works in centimetres
and expects the camera translation added, so the vertices are handed over
as ``100 * (v + cam_t)``; the fitted SMPL comes back in metres in the
camera frame. Writes the npz the scorer reads (SMPL vertices, translation,
global orientation, betas).

Runs in the ``mhr_lod`` conda environment (CPU torch) from the tool's
folder, which loads its assets by relative path::

    cd <repo>/third_party/MHR/tools/mhr_smpl_conversion && PYTHONPATH=. \\
        ~/miniconda3/envs/mhr_lod/bin/python \\
        <repo>/lidar_bedlam/scripts/baselines/convert_sam3d_to_smpl.py \\
        --mhr <repo>/outputs/baselines/sam3d-body/waymo_val.mhr.npz \\
        --out <repo>/outputs/baselines/sam3d-body/waymo_val.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MHR_ASSETS = REPO / "resources/pretrained-checkpoints/mhr/assets"
SMPL_NEUTRAL = (
    REPO / "resources/data/generated/body_models/smpl/SMPL_NEUTRAL.pkl"
)


def run(args: argparse.Namespace) -> None:
    """Fit SMPL to every MHR mesh of the file, in batches."""
    import smplx
    import torch
    from conversion import Conversion
    from mhr.mhr import MHR

    z = np.load(args.mhr)
    keys = [str(k) for k in z["key"]]
    verts = z["mhr_vertices"].astype(np.float32)
    cam_t = z["cam_t"].astype(np.float32)
    if args.limit:
        keys, verts, cam_t = (
            keys[: args.limit],
            verts[: args.limit],
            cam_t[: args.limit],
        )
    mhr = MHR.from_files(
        folder=MHR_ASSETS, device=torch.device("cpu"), lod=1,
        wants_pose_correctives=False,
    )  # fmt: skip
    smpl = smplx.SMPL(
        model_path=str(SMPL_NEUTRAL), gender="neutral", num_betas=10
    )
    conv = Conversion(mhr, smpl, method="pytorch", batch_size=args.batch_size)
    t0 = time.time()
    out_v, out_t, out_r, out_b = [], [], [], []
    for start in range(0, len(keys), args.batch_size):
        sl = slice(start, min(len(keys), start + args.batch_size))
        cm = 100.0 * (verts[sl] + cam_t[sl][:, None, :])
        res = conv.convert_mhr2smpl(
            mhr_vertices=torch.from_numpy(cm),
            return_smpl_vertices=True,
            return_smpl_parameters=True,
            return_fitting_errors=False,
            batch_size=args.batch_size,
        )
        out_v.append(_np(res.result_vertices).astype(np.float16))
        params = {k: _np(v) for k, v in res.result_parameters.items()}
        out_t.append(params["transl"].astype(np.float64))
        out_r.append(params["global_orient"].astype(np.float64))
        out_b.append(params["betas"].astype(np.float64))
        sys.stdout.write(
            f"{sl.stop}/{len(keys)} fitted, "
            f"{sl.stop / (time.time() - t0):.2f} samples/s\n"
        )
        sys.stdout.flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "key": np.array(keys),
        "vertices": np.concatenate(out_v),
        "transl": np.concatenate(out_t),
        "global_orient": np.concatenate(out_r),
        "betas": np.concatenate(out_b),
        "meta/method": np.array("sam3d-body"),
        "meta/conversion": np.array("MHR tools/mhr_smpl_conversion, pytorch"),
        "meta/seconds_fit": np.array(time.time() - t0),
        "meta/samples": np.array(len(keys)),
    }
    for k in z.files:
        if k.startswith("meta/") and k not in arrays:
            arrays[k] = z[k]
    np.savez(args.out, **arrays)
    sys.stdout.write(f"wrote {args.out} ({len(keys)} records)\n")


def _np(v: Any) -> Any:
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mhr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
