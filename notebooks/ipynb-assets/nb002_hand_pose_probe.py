"""Cell bodies of ``002-hand-pose-probe.ipynb``: one function per cell.

Reproduce the trainer's Waymo validation plot for a checkpoint and re-pose
the wrists and hands of a prediction interactively.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, display
from scipy.spatial.transform import Rotation

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import (
    build_dataset,
    build_model,
    tensors_only,
    to_device,
)
from lidar_bedlam.utils import eval_vis
from lidar_bedlam.utils.eval_vis import spread_batch

ROOT = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
JOINTS = {"L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23}


@dataclass
class Probe:
    """A loaded checkpoint with the three records the trainer plots."""

    run: str
    source: str
    model: Any
    faces: np.ndarray[Any, Any]
    smpl_eval: SmplModel
    batch: dict[str, Any]
    pred: dict[str, Any]
    device: torch.device
    step: int


def load_probe(run: str, config: str, source: str = "waymo_val") -> Probe:
    """Load ``outputs/helma/ckpt/<run>/last.pt`` and predict the three spread

    records of ``source``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(ROOT / config)
    model = build_model(cfg)
    state = torch.load(
        ROOT / "outputs/helma/ckpt" / run / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in state["model"].items()}
    )
    model.to(device).eval()
    assert model.smpl is not None
    src = next(s for s in cfg.data.val if s.name == source)
    dataset = build_dataset(src, cfg, train=False)
    batch = to_device(spread_batch(dataset, 3), device)
    with torch.no_grad():
        pred = model(tensors_only(batch))
    print(f"{run} step {state['step']}, {len(dataset)} {source} records")
    for i, key in enumerate(batch["key"]):
        print(f"  sample {i}: {key}")
    return Probe(
        run,
        source,
        model,
        np.asarray(model.smpl.faces, dtype=np.int64),
        SmplModel(Path(cfg.body_models)),
        batch,
        pred,
        device,
        int(state["step"]),
    )


def reproduce_val_plot(probe: Probe) -> None:
    """The trainer's own renderer on the same three records: identical to the

    wandb panel.
    """
    out = ROOT / "outputs" / "hand_probe" / f"{probe.run}_{probe.source}.png"
    path = eval_vis.render_val_plot(
        probe.batch, probe.pred, probe.faces, probe.smpl_eval, out
    )
    display(Image(filename=str(path)))


def _euler_to_matrix(
    x: float, y: float, z: float, device: torch.device
) -> torch.Tensor:
    m = Rotation.from_euler("XYZ", [x, y, z], degrees=True).as_matrix()
    return torch.as_tensor(m, dtype=torch.float32, device=device)


def _repose(
    probe: Probe, i: int, rots: dict[str, torch.Tensor | None]
) -> dict[str, Any]:
    body_pose = probe.pred["body_pose"][i : i + 1].clone()
    for name, m in rots.items():
        if m is not None:
            body_pose[0, JOINTS[name] - 1] = m
    with torch.no_grad():
        out = probe.model.smpl(
            betas=probe.pred["betas"][i : i + 1],
            global_orient=probe.pred["global_orient"][i : i + 1],
            body_pose=body_pose,
            transl=probe.pred["transl"][i : i + 1],
            pose2rot=False,
        )
    pred = {k: v[i : i + 1] for k, v in probe.pred.items() if v.ndim > 0}
    pred["body_pose"] = body_pose
    pred["vertices"] = out.vertices
    return pred


def _angles(body_pose: torch.Tensor) -> str:
    rows = []
    for name, j in JOINTS.items():
        e = Rotation.from_matrix(body_pose[0, j - 1].cpu().numpy()).as_euler(
            "XYZ", degrees=True
        )
        rows.append(f"{name}: x {e[0]:6.1f}  y {e[1]:6.1f}  z {e[2]:6.1f}")
    return "\n".join(rows)


def _draw(probe: Probe, i: int, pred: dict[str, Any]) -> None:
    batch = {
        k: (v[i : i + 1] if hasattr(v, "__getitem__") else v)
        for k, v in probe.batch.items()
    }
    verts = pred["vertices"].detach().cpu().float().numpy()[0]
    k = batch["intrinsics"][0].detach().cpu().numpy()
    img = eval_vis._denormalise(batch["image"][0])
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    eval_vis._wire(ax, eval_vis._project(k, verts), probe.faces, "red")
    gt, _ = eval_vis._keypoints(0, batch, pred)
    ok = gt[:, 2] > 0
    ax.scatter(gt[ok, 0], gt[ok, 1], s=9, c="lime")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{batch['dataset'][0]} | placement error "
        f"{eval_vis._placement_error(0, batch, pred):.2f} m"
    )
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    pts = (
        batch["points"][0]
        .detach()
        .cpu()
        .numpy()[batch["points_valid"][0].detach().cpu().numpy().astype(bool)]
    )
    if len(pts):
        ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.5, c="0.45")
    sub = verts[::4]
    ax3.scatter(sub[:, 0], sub[:, 2], -sub[:, 1], s=0.6, c="red", alpha=0.5)
    j = batch["joints3d"][0].detach().cpu().numpy()
    jv = batch["joints3d_valid"][0].detach().cpu().numpy().astype(bool)
    if jv.any():
        ax3.scatter(j[jv, 0], j[jv, 2], -j[jv, 1], s=14, c="lime")
    c = verts.mean(0)
    eval_vis._set_equal(ax3, np.array([c[0], c[2], -c[1]]), 1.1)
    ax3.view_init(elev=12, azim=-60)
    ax3.locator_params(nbins=4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)  # bytes: survives the Agg backend
    plt.close(fig)
    display(Image(data=buf.getvalue()))
    print(_angles(pred["body_pose"]))


def panel(probe: Probe, i: int) -> None:
    """Interactive panel for record ``i``: wrists and hands as predicted, zero,

    or from Euler sliders.
    """
    import ipywidgets as w

    modes = ["zero", "predicted", "custom"]

    def slider(desc: str) -> Any:
        return w.FloatSlider(
            value=0.0,
            min=-90.0,
            max=90.0,
            step=1.0,
            description=desc,
            continuous_update=False,
            layout=w.Layout(width="260px"),
        )

    def group(title: str) -> dict[str, Any]:
        return {
            "mode": w.Dropdown(options=modes, value="zero", description=title),
            "mirror": w.Checkbox(
                value=True, description="mirror right = left"
            ),
            "L": [slider(f"L {a}") for a in "xyz"],
            "R": [slider(f"R {a}") for a in "xyz"],
        }

    wrist, hand = group("wrists"), group("hands")
    out = w.Output()

    def rotation(g: dict[str, Any], side: str) -> torch.Tensor | None:
        if g["mode"].value == "predicted":
            return None
        if g["mode"].value == "zero":
            return torch.eye(3, device=probe.device)
        x, y, z = (s.value for s in g["L"])
        if side == "R" and not g["mirror"].value:
            x, y, z = (s.value for s in g["R"])
        elif side == "R":
            y, z = -y, -z
        return _euler_to_matrix(x, y, z, probe.device)

    def update(*_: object) -> None:
        rots = {
            "L_Wrist": rotation(wrist, "L"),
            "R_Wrist": rotation(wrist, "R"),
            "L_Hand": rotation(hand, "L"),
            "R_Hand": rotation(hand, "R"),
        }
        with out:
            out.clear_output(wait=True)
            _draw(probe, i, _repose(probe, i, rots))

    def box(g: dict[str, Any]) -> Any:
        return w.VBox(
            [
                w.HBox([g["mode"], g["mirror"]]),
                w.HBox([w.VBox(g["L"]), w.VBox(g["R"])]),
            ]
        )

    for g in (wrist, hand):
        for widget in [g["mode"], g["mirror"], *g["L"], *g["R"]]:
            widget.observe(update, names="value")
    update()
    display(w.VBox([w.HBox([box(wrist), box(hand)]), out]))
