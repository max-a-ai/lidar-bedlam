"""Run LiDAR-HMR (release weights trained on Waymo) on our person point
clouds and write camera-frame meshes.

Runs in the ``lidar-hmr`` conda environment (torch 2.2 + pointops +
pointnet2_ops); the vendored repo is made the cwd for its relative asset
paths::

    ~/miniconda3/envs/lidar-hmr/bin/python \\
        lidar_bedlam/scripts/baselines/run_lidar_hmr.py \\
        --shards resources/data/generated/real/v1 \\
        --pattern "waymo_val_*.npz" \\
        --out outputs/baselines/lidar-hmr/waymo_val.npz

The method expects the person's points in a z-up frame centred on the
bounding-box centre with exactly 1024 points, as in its own data loaders;
our OpenCV camera-frame points are rotated to that frame and the mesh is
rotated back. Records without LiDAR returns are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lidar_bedlam.scripts.baselines.common import (  # noqa: E402
    Predictions,
    Timer,
    list_shards,
    load_shard,
    mirror_back,
    smpl_mirror_map,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO = REPO_ROOT / "third_party/LiDAR-HMR"
CHECKPOINT = (
    REPO_ROOT / "resources/pretrained-checkpoints/lidar-hmr/lidar_hmr_mesh.pth"
)  # local copy of nas_drive/methods/max/LiDAR-HMR/lidar_hmr_mesh.pth
CONFIG = "configs/mesh/waymo.yaml"
SMPL_DIR = REPO_ROOT / "resources/data/generated/body_models"
N_POINTS = 1024
# z-up frame from the OpenCV camera frame: forward = z, left = -x, up = -y
CAM_TO_ZUP = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _shim_pytorch3d() -> None:
    """``matrix_to_axis_angle`` is the only pytorch3d symbol the model uses."""
    try:
        import pytorch3d  # noqa: F401

        return
    except ImportError:
        pass
    import torch

    def matrix_to_axis_angle(mat: Any) -> Any:
        r = mat.reshape(-1, 3, 3)
        cos = ((r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2] - 1.0) / 2.0).clamp(-1, 1)
        angle = torch.acos(cos)
        axis = torch.stack(
            [
                r[:, 2, 1] - r[:, 1, 2],
                r[:, 0, 2] - r[:, 2, 0],
                r[:, 1, 0] - r[:, 0, 1],
            ],
            dim=-1,
        )
        sin = torch.sin(angle)
        small = sin.abs() < 1e-6
        scale = torch.where(small, 0.5, angle / (2.0 * sin.clamp_min(1e-12)))
        out = axis * scale[:, None]
        return out.reshape(*mat.shape[:-2], 3)

    pkg = types.ModuleType("pytorch3d")
    transforms = types.ModuleType("pytorch3d.transforms")
    conv = types.ModuleType("pytorch3d.transforms.rotation_conversions")
    conv.matrix_to_axis_angle = (  # type: ignore[attr-defined]
        matrix_to_axis_angle
    )
    transforms.rotation_conversions = conv  # type: ignore[attr-defined]
    pkg.transforms = transforms  # type: ignore[attr-defined]
    sys.modules["pytorch3d"] = pkg
    sys.modules["pytorch3d.transforms"] = transforms
    sys.modules["pytorch3d.transforms.rotation_conversions"] = conv


def load_model() -> Any:
    """LiDAR_HMR with the Waymo config and the release weights."""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    _shim_pytorch3d()
    for name, alias in (("bool", bool), ("int", int), ("float", float)):
        if not hasattr(np, name):  # the code base predates numpy 1.24
            setattr(np, name, alias)
    import torch
    from models.pmg_config import (
        config,
        update_config,
    )
    from models.pose_mesh_net import (
        LiDAR_HMR,
    )

    update_config(CONFIG)
    model = LiDAR_HMR(pmg_cfg=config, train_pmg=True).cuda()
    state = torch.load(CHECKPOINT, map_location="cuda")
    model.load_state_dict(state["net"])
    return model.eval()


def fix_points(points: np.ndarray[Any, Any], rng: np.random.Generator) -> Any:
    """Exactly ``N_POINTS`` rows: random subset or repeats (their recipe)."""
    n = len(points)
    if n >= N_POINTS:
        return points[rng.choice(n, N_POINTS, replace=False)]
    extra = rng.choice(n, N_POINTS - n, replace=True)
    return np.concatenate([points, points[extra]])


def run(args: argparse.Namespace) -> None:
    """Predict every record with LiDAR returns of the matching shards."""
    import smplx
    import torch
    from scipy.spatial.transform import Rotation

    shards = list_shards(args.shards, args.pattern)
    if not shards:
        msg = f"no shards for {args.pattern} in {args.shards}"
        raise SystemExit(msg)
    model = load_model()
    smpl = smplx.create(
        str(SMPL_DIR), model_type="smpl", gender="neutral", num_betas=10
    ).cuda()
    regressor = smpl.J_regressor.double().cpu().numpy()
    sym = smpl_mirror_map(smpl.v_template.double().cpu().numpy())
    v_template = smpl.v_template.double().cpu().numpy()
    shapedirs = smpl.shapedirs.double().cpu().numpy()[:, :, :10]
    rng = np.random.default_rng(args.seed)
    preds = Predictions()
    timer = Timer()
    skipped = 0
    for shard in shards:
        data = load_shard(shard, points=True)
        keep = [i for i, p in enumerate(data["points"]) if len(p) > 0]
        skipped += len(data["key"]) - len(keep)
        for start in range(0, len(keep), args.batch_size):
            rows = keep[start : start + args.batch_size]
            roots, clouds = [], []
            for i in rows:
                p = data["points"][i].astype(np.float64)
                if args.mirror:  # left/right mirror in the camera frame
                    p = p * np.array([-1.0, 1.0, 1.0])
                p = p @ CAM_TO_ZUP.T
                root = (p.max(0) + p.min(0)) / 2.0
                roots.append(root)
                clouds.append(fix_points(p - root, rng))
            pcd = torch.from_numpy(np.stack(clouds)).float().cuda()
            timer.start()
            with torch.no_grad():
                ret = model(pcd)
            torch.cuda.synchronize()
            timer.stop(len(rows))
            root_arr = np.stack(roots)
            mesh = ret["mesh_refine"].double().cpu().numpy()
            verts_zup = mesh + root_arr[:, None]
            verts = verts_zup @ CAM_TO_ZUP  # back to the camera frame
            betas = ret["pose_beta"].double().cpu().numpy()
            theta = ret["pose_theta"].double().cpu().numpy().reshape(-1, 24, 3)
            r_zup = Rotation.from_rotvec(theta[:, 0]).as_matrix()
            r_cam = CAM_TO_ZUP.T @ r_zup
            aa = Rotation.from_matrix(r_cam).as_rotvec()
            if args.mirror:
                zeros = np.zeros((len(rows), 3))
                verts, _, aa = mirror_back(verts, zeros, aa, sym)
            # rest pelvis of the predicted shape: regressor on the shaped
            # template (transl = posed root joint - rest pelvis)
            v_shaped = v_template + np.einsum("bl,vkl->bvk", betas, shapedirs)
            rest_pelvis = np.einsum("v,bvk->bk", regressor[0], v_shaped)
            joints = np.einsum("jv,nvk->njk", regressor, verts)
            transl = joints[:, 0] - rest_pelvis
            preds.add(
                data["key"][rows],
                verts,
                transl,
                aa,
                betas,
                n_points=np.array([len(data["points"][i]) for i in rows]),
            )
        sys.stdout.write(
            f"{shard.name}: {timer.samples} done, {timer.rate:.0f} samples/s\n"
        )
        sys.stdout.flush()
    preds.save(
        args.out,
        method="lidar-hmr",
        mirror=int(args.mirror),
        seconds=timer.seconds,
        samples=timer.samples,
        skipped_no_points=skipped,
    )
    sys.stdout.write(
        f"wrote {args.out}: {timer.samples} samples ({skipped} without "
        f"points skipped), {timer.seconds:.1f} s model time "
        f"({timer.rate:.0f}/s)\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="left/right mirror the input points and un-mirror the mesh",
    )
    args = ap.parse_args(argv)
    args.shards = args.shards.resolve()
    args.out = args.out.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
