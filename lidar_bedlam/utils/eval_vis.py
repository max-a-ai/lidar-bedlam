"""Small evaluation figures written during training.

One PNG per evaluation and validation source: a row of crops with the
ground-truth keypoints (green) and the predicted keypoints (red), the
dataset name and the translation error. The trainer references the file
from ``metrics.jsonl`` and the wandb mirror uploads it as an image.
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


@torch.no_grad()
def eval_figures(
    model: torch.nn.Module,
    loaders: dict[str, Any],
    run_dir: Path,
    step: int,
    device: torch.device,
    n: int = 6,
) -> dict[str, str]:
    """One figure per validation source; returns log keys -> relative path."""
    from lidar_bedlam.train.loop import tensors_only, to_device

    out: dict[str, str] = {}
    model.eval()
    for name, loader in loaders.items():
        batch = to_device(next(iter(loader)), device)
        pred = model(tensors_only(batch))
        rel = Path("vis") / f"{name}_{step:07d}.png"
        render_eval_grid(batch, pred, run_dir / rel, n)
        out[f"image/val_{name}"] = str(rel)
    model.train()
    return out
