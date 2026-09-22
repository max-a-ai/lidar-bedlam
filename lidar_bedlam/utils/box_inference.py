"""Run the trained model on a 3D-box dataset and look at one sample.

The box datasets (car data, nuScenes, Argoverse 2, DAIR-V2X) label boxes,
not bodies: there is no SMPL and no keypoints, so nothing here computes an
accuracy number. It exists to put a prediction next to the returns it came
from and let a human page through samples.

The headline runs are trained without an image backbone, on precomputed
ViT-H tokens, so :func:`load_model` also loads the frozen ViT and
:func:`predict` feeds its tokens into the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.torch_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    HumanPoseDataset,
)
from lidar_bedlam.models.vit import VIT_H, ViT, load_backbone_weights
from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import build_model, tensors_only, to_device

REPO = Path(__file__).resolve().parents[2]
#: pseudo-GT v2 with the Waymo box loss off -- the run that fixed the hands
NOBOX_CHECKPOINT = REPO / "outputs/helma/ckpt/t-short-nobox-000/best.pt"
NOBOX_CONFIG = REPO / "configs/t_short_real.yaml"
VIT_CHECKPOINT = REPO / "resources/pretrained-checkpoints/tokenhmr_vith"


@dataclass
class Predictor:
    """A loaded model plus the frozen ViT that supplies its image tokens."""

    model: Any
    vit: ViT
    device: torch.device
    step: int

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Model output for one collated batch."""
        moved = to_device(batch, self.device)
        with torch.autocast(
            device_type="cuda",
            enabled=self.device.type == "cuda",
            dtype=torch.float16,
        ):
            tokens, _ = self.vit(moved["image"])
        moved["tokens"] = tokens.float()
        out: dict[str, torch.Tensor] = self.model(tensors_only(moved))
        return out


def load_model(
    config: Path = NOBOX_CONFIG,
    checkpoint: Path = NOBOX_CHECKPOINT,
    device: str | None = None,
) -> Predictor:
    """Build the model, load the checkpoint and the frozen ViT."""
    dev = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = load_config(config, [])
    model = build_model(cfg).to(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    vit = ViT(VIT_H).to(dev).eval()
    load_backbone_weights(vit, VIT_CHECKPOINT)
    return Predictor(model, vit, dev, int(state.get("step", -1)))


def batch_of(
    source: SampleSource, indices: list[int], n_points: int = 1024
) -> dict[str, Any]:
    """Collate chosen samples of a source into one batch."""
    dataset = HumanPoseDataset([source], smpl_model=None, n_points=n_points)
    items = [dataset[i] for i in indices]
    collated: dict[str, Any] = torch.utils.data.default_collate(items)
    return collated


def denormalise(image: torch.Tensor) -> NDArray[np.float64]:
    """A normalised CHW crop back to an RGB array for display."""
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def show(
    source: SampleSource,
    index: int,
    predictor: Predictor,
    axes: Any = None,
) -> Any:
    """Draw one sample: crop with the mesh on it, and mesh among returns.

    The right panel's vertical axis is image up (camera ``-y``), which is
    world up only for a level camera -- several of these rigs are pitched.
    """
    import matplotlib.pyplot as plt

    batch = batch_of(source, [index])
    pred = predictor(batch)
    verts = pred["vertices"].float().cpu().numpy()[0]
    image = denormalise(batch["image"][0])
    k = batch["intrinsics"][0].numpy()
    points = batch["points"][0].numpy()[
        batch["points_valid"][0].numpy().astype(bool)
    ]

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
    uv = (verts / verts[:, 2:3]) @ k.T

    axes[0].imshow(image)
    axes[0].scatter(uv[::4, 0], uv[::4, 1], s=0.7, c="crimson", alpha=0.5)
    axes[0].set_xlim(0, image.shape[1])
    axes[0].set_ylim(image.shape[0], 0)
    axes[0].axis("off")
    label = getattr(source, "category", lambda _i: source.name)(index)
    axes[0].set_title(
        f"{label}  |  {len(points)} returns  |  "
        f"{float(verts[:, 2].mean()):.1f} m",
        fontsize=9,
    )

    axes[1].scatter(
        points[:, 2], -points[:, 1], s=7, c="black", alpha=0.6, label="LiDAR"
    )
    axes[1].scatter(
        verts[::8, 2],
        -verts[::8, 1],
        s=1.4,
        c="crimson",
        alpha=0.5,
        label="mesh",
    )
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("z, depth (m)")
    axes[1].set_ylabel("image up, -y (m)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(f"{source.name}  sample {index}", fontsize=9)
    return axes


def browse(
    source: SampleSource,
    predictor: Predictor,
    start: int = 0,
    step: int = 1,
) -> Any:
    """Prev/next buttons over a source, falling back to a static figure.

    Without a live ipywidgets front end (a plain nbconvert run, or a viewer
    with no kernel) the widget would render as nothing, so in that case one
    sample is drawn normally instead.
    """
    import matplotlib.pyplot as plt

    if len(source) == 0:
        raise ValueError(f"{source.name}: no samples matched the filter")
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        show(source, min(start, len(source) - 1), predictor)
        return None

    out = widgets.Output()
    state = {"i": min(start, len(source) - 1)}

    def draw() -> None:
        out.clear_output(wait=True)
        with out:
            fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
            show(source, state["i"], predictor, axes)
            fig.suptitle(
                f"sample {state['i'] + 1} of {len(source)}", fontsize=10
            )
            fig.tight_layout()
            plt.show()

    def move(delta: int) -> Any:
        def handler(_button: Any) -> None:
            state["i"] = (state["i"] + delta) % len(source)
            draw()

        return handler

    back = widgets.Button(description="< previous")
    forward = widgets.Button(description="next >")
    back.on_click(move(-step))
    forward.on_click(move(step))
    display(widgets.HBox([back, forward]), out)  # type: ignore[no-untyped-call]
    draw()
    return out
