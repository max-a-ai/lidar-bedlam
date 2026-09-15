"""Small evaluation figures written during training.

Per evaluation and validation source two PNGs. ``image/val_<source>``: a
row of crops with the ground-truth keypoints (green) and the predicted
keypoints (red). ``image/val_plot_<source>``: three samples beneath each
other, left the crop with the predicted mesh projected onto it, right the
input points with the predicted mesh (red) and the label (green mesh, or
the labelled joints where the record has no mesh). The trainer references
the files from ``metrics.jsonl`` and the wandb mirror uploads them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD

_WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
_CSEL = WAYMO15_TO_COCO17[_WSEL]


def _denormalise(image: Tensor) -> np.ndarray[Any, Any]:
    img = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    out: np.ndarray[Any, Any] = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return out


def _keypoints(
    i: int, batch: dict[str, Any], pred: dict[str, Tensor]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """(GT (K, 3) with confidence, predicted (K, 2)) for sample ``i``."""
    conv = batch.get("joint_convention_id")
    waymo = conv is not None and int(conv[i]) == 1 and "kp2d_coco" in pred
    gt = batch["kp2d"][i].detach().cpu().numpy()
    if waymo:
        return gt[_WSEL], pred["kp2d_coco"][i, _CSEL].detach().cpu().numpy()
    return gt[:24], pred["kp2d"][i, :24].detach().cpu().numpy()


def render_eval_grid(
    batch: dict[str, Any],
    pred: dict[str, Tensor],
    path: Path,
    n: int = 6,
) -> Path:
    """Write the figure for the first ``n`` samples of a batch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, int(batch["image"].shape[0]))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    axes = np.atleast_1d(axes)
    transl_err = (
        (pred["transl"] - batch["transl"]).norm(dim=-1).detach().cpu().numpy()
    )
    for i, ax in enumerate(axes):
        ax.imshow(_denormalise(batch["image"][i]))
        gt, pr = _keypoints(i, batch, pred)
        ok = gt[:, 2] > 0
        ax.scatter(gt[ok, 0], gt[ok, 1], s=10, c="lime", label="gt")
        ax.scatter(pr[:, 0], pr[:, 1], s=8, c="red", marker="x", label="pred")
        ax.set_title(
            f"{batch['dataset'][i]} | dz {transl_err[i]:.2f} m", fontsize=9
        )
        ax.set_xlim(0, batch["image"].shape[-1])
        ax.set_ylim(batch["image"].shape[-2], 0)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def _project(
    k: np.ndarray[Any, Any], pts: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    z = np.clip(pts[:, 2], 1e-3, None)
    out: np.ndarray[Any, Any] = np.stack(
        [k[0, 0] * pts[:, 0] / z + k[0, 2], k[1, 1] * pts[:, 1] / z + k[1, 2]],
        -1,
    )
    return out


def _wire(
    ax: Any,
    verts2d: np.ndarray[Any, Any],
    faces: np.ndarray[Any, Any],
    colour: str,
) -> None:
    from matplotlib.collections import PolyCollection

    ax.add_collection(
        PolyCollection(
            verts2d[faces],
            facecolors="none",
            edgecolors=colour,
            linewidths=0.25,
            alpha=0.35,
        )  # fmt: skip
    )


def _set_equal(ax: Any, centre: np.ndarray[Any, Any], radius: float) -> None:
    for name, c in zip("xyz", centre, strict=True):
        getattr(ax, f"set_{name}lim")(c - radius, c + radius)


def _placement_error(
    i: int, batch: dict[str, Any], pred: dict[str, Tensor]
) -> float:
    """Metres between the predicted and labelled root: SMPL translation on
    mesh-labelled records, the hip centre (protocol) on keypoint records."""
    if bool(batch["has_smpl"][i]):
        return float((pred["transl"][i] - batch["transl"][i]).norm())
    valid = batch["joints3d_valid"][i]
    if "joints_coco" not in pred or not (bool(valid[4]) and bool(valid[10])):
        return float("nan")
    gt = batch["joints3d"][i]
    g_root = (gt[4] + gt[10]) / 2.0
    p_root = (pred["joints_coco"][i, 11] + pred["joints_coco"][i, 12]) / 2.0
    return float((p_root - g_root).norm())


def render_val_plot(
    batch: dict[str, Any],
    pred: dict[str, Tensor],
    faces: np.ndarray[Any, Any],
    smpl: Any,
    path: Path,
    n: int = 3,
) -> Path:
    """Three samples (rows): projection on the crop, 3D estimate vs label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, int(batch["image"].shape[0]))
    fig = plt.figure(figsize=(8.4, 3.9 * n))
    verts = pred["vertices"].detach().cpu().float().numpy()
    sub = np.arange(0, verts.shape[1], 4)  # every 4th vertex in 3D
    for i in range(n):
        k = batch["intrinsics"][i].detach().cpu().numpy()
        img = _denormalise(batch["image"][i])
        ax = fig.add_subplot(n, 2, 2 * i + 1)
        ax.imshow(img)
        _wire(ax, _project(k, verts[i]), faces, "red")
        gt, _ = _keypoints(i, batch, pred)
        ok = gt[:, 2] > 0
        ax.scatter(gt[ok, 0], gt[ok, 1], s=9, c="lime")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{batch['dataset'][i]} | placement error "
            f"{_placement_error(i, batch, pred):.2f} m",
            fontsize=9,
        )
        ax3 = fig.add_subplot(n, 2, 2 * i + 2, projection="3d")
        pts = batch["points"][i].detach().cpu().numpy()
        valid = batch["points_valid"][i].detach().cpu().numpy().astype(bool)
        pts = pts[valid]
        if len(pts):
            ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.5, c="0.45")
        pv = verts[i][sub]
        ax3.scatter(pv[:, 0], pv[:, 2], -pv[:, 1], s=0.6, c="red", alpha=0.5)
        if bool(batch["has_smpl"][i]):
            gv = _gt_vertices(batch, i, smpl)[sub]
            ax3.scatter(
                gv[:, 0], gv[:, 2], -gv[:, 1], s=0.6, c="lime", alpha=0.5
            )
        else:
            j = batch["joints3d"][i].detach().cpu().numpy()
            jv = batch["joints3d_valid"][i].detach().cpu().numpy().astype(bool)
            j = j[jv]
            if len(j):
                ax3.scatter(j[:, 0], j[:, 2], -j[:, 1], s=14, c="lime")
        centre = verts[i].mean(0)
        _set_equal(ax3, np.array([centre[0], centre[2], -centre[1]]), 1.1)
        ax3.view_init(elev=12, azim=-60)
        ax3.locator_params(nbins=4)
        ax3.set_xlabel("x", fontsize=7)
        ax3.set_ylabel("z", fontsize=7)
        ax3.set_zlabel("-y", fontsize=7)
        ax3.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def _gt_vertices(
    batch: dict[str, Any], i: int, smpl: Any
) -> np.ndarray[Any, Any]:
    """Labelled mesh of sample ``i`` in the camera frame."""
    from lidar_bedlam.body.smpl import SmplParams

    def a(key: str) -> np.ndarray[Any, Any]:
        out: np.ndarray[Any, Any] = (
            batch[key][i].detach().cpu().double().numpy()
        )
        return out

    verts, _ = smpl.forward(
        SmplParams(a("global_orient"), a("body_pose"), a("betas"), a("transl"))
    )
    return np.asarray(verts, dtype=np.float64)


def spread_batch(dataset: Any, n: int) -> dict[str, Any]:
    """Batch of ``n`` records at evenly spaced positions of ``dataset``."""
    from torch.utils.data import default_collate

    size = len(dataset)
    idx = sorted({(j * size) // n for j in range(min(n, size))})
    return dict(default_collate([dataset[j] for j in idx]))


@torch.no_grad()
def eval_figures(
    model: torch.nn.Module,
    loaders: dict[str, Any],
    run_dir: Path,
    step: int,
    device: torch.device,
    n: int = 6,
    smpl: Any = None,
    faces: np.ndarray[Any, Any] | None = None,
) -> dict[str, str]:
    """Figures per source (keypoint row, and the val plot when the SMPL
    model and faces are given); returns log keys -> relative path."""
    from lidar_bedlam.train.loop import tensors_only, to_device

    out: dict[str, str] = {}
    model.eval()
    for name, loader in loaders.items():
        batch = to_device(next(iter(loader)), device)
        pred = model(tensors_only(batch))
        rel = Path("vis") / f"{name}_{step:07d}.png"
        render_eval_grid(batch, pred, run_dir / rel, n)
        out[f"image/val_{name}"] = str(rel)
        if smpl is not None and faces is not None and "vertices" in pred:
            # three records spread over the source, not three neighbouring
            # frames of one person
            batch = to_device(spread_batch(loader.dataset, 3), device)
            pred = model(tensors_only(batch))
            rel = Path("vis") / f"plot_{name}_{step:07d}.png"
            render_val_plot(batch, pred, faces, smpl, run_dir / rel)
            out[f"image/val_plot_{name}"] = str(rel)
    model.train()
    return out
