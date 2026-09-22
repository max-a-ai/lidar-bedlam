"""Gallery of a trained checkpoint on a 3D-box dataset (nuScenes or TUMTraf).

Picks pedestrians spread over the source, runs the model once on the
batch, and draws one row per sample: the plain rectified crop, the crop
with the predicted mesh wireframe projected onto it, and the mesh among
the LiDAR returns of the labelled box in 3D. Box datasets carry no SMPL or
keypoint labels, so the figure is qualitative; the title carries the
return count and depth.

    uv run python lidar_bedlam/scripts/box_dataset_gallery.py \\
        --out outputs/gallery/nuscenes_mini.png --n 6
    uv run python lidar_bedlam/scripts/box_dataset_gallery.py \\
        --checkpoint outputs/helma/ckpt/m-boxhead-000/last.pt \\
        --config configs/m_boxhead.yaml --out outputs/gallery/boxhead.png
    uv run python lidar_bedlam/scripts/box_dataset_gallery.py \\
        --dataset tumtraf --out outputs/gallery/tumtraf_r02_s01.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lidar_bedlam.utils import eval_vis
from lidar_bedlam.utils.box_inference import (
    NOBOX_CHECKPOINT,
    NOBOX_CONFIG,
    Predictor,
    batch_of,
    denormalise,
    load_model,
)


def draw(
    source: Any, indices: list[int], predictor: Predictor, path: Path
) -> Path:
    """Render the rows for ``indices`` of ``source`` into ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faces = np.asarray(predictor.model.smpl.faces, dtype=np.int64)
    batch = batch_of(source, indices)
    pred = predictor(batch)
    verts_all = pred["vertices"].float().cpu().numpy()
    fig = plt.figure(figsize=(11.4, 3.6 * len(indices)))
    for row, i in enumerate(indices):
        verts = verts_all[row]
        img = (denormalise(batch["image"][row]) * 255).astype(np.uint8)
        k = batch["intrinsics"][row].numpy()
        valid = batch["points_valid"][row].numpy().astype(bool)
        pts = batch["points"][row].numpy()[valid]
        label = getattr(source, "category", lambda _i: source.name)(i)
        ax0 = fig.add_subplot(len(indices), 3, 3 * row + 1)
        ax0.imshow(img)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(f"{source.name} {label} #{i}", fontsize=8)
        ax = fig.add_subplot(len(indices), 3, 3 * row + 2)
        ax.imshow(img)
        eval_vis._wire(ax, eval_vis._project(k, verts), faces, "red")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{len(pts)} returns | depth {float(verts[:, 2].mean()):.1f} m",
            fontsize=8,
        )
        ax3 = fig.add_subplot(len(indices), 3, 3 * row + 3, projection="3d")
        if len(pts):
            ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.5, c="0.45")
        sub = verts[::4]
        ax3.scatter(
            sub[:, 0], sub[:, 2], -sub[:, 1], s=0.6, c="red", alpha=0.5
        )
        c = verts.mean(0)
        eval_vis._set_equal(ax3, np.array([c[0], c[2], -c[1]]), 1.1)
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


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="samples, spread evenly")
    ap.add_argument("--checkpoint", type=Path, default=NOBOX_CHECKPOINT)
    ap.add_argument("--config", type=Path, default=NOBOX_CONFIG)
    ap.add_argument(
        "--dataset", choices=("nuscenes", "tumtraf"), default="nuscenes"
    )
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument(
        "--categories",
        default="human",
        help="comma-separated nuScenes prefixes",
    )
    ap.add_argument("--min-points", type=int, default=50)
    ap.add_argument("--min-box-height-px", type=int, default=64)
    args = ap.parse_args(argv)

    source: Any
    if args.dataset == "tumtraf":
        from lidar_bedlam.data.tumtraf import TumTrafSource

        source = TumTrafSource(
            min_points=args.min_points,
            min_box_height_px=args.min_box_height_px,
        )
    else:
        from lidar_bedlam.data.nuscenes_boxes import (
            NuScenesSource,
            nuscenes_scenes,
        )

        source = NuScenesSource(
            nuscenes_scenes(version=args.version),
            categories=tuple(args.categories.split(",")),
            min_points=args.min_points,
            min_box_height_px=args.min_box_height_px,
        )
    n = min(args.n, len(source))
    if n == 0:
        sys.stdout.write("no samples pass the filters\n")
        return 1
    indices = sorted({(j * len(source)) // n for j in range(n)})
    predictor = load_model(args.config, args.checkpoint)
    sys.stdout.write(
        f"{len(source)} samples, checkpoint step {predictor.step}, "
        f"drawing {indices}\n"
    )
    draw(source, indices, predictor, args.out)
    sys.stdout.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
