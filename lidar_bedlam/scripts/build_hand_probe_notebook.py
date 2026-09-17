"""Generate ``notebooks/hand_pose_probe.ipynb`` (wrist and hand pose probe).

Reproduces the Waymo validation plot of a trained run (the three records
``spread_batch`` picks, exactly as the trainer draws them) and then, one
cell per record, lets the wrist (SMPL joints 20, 21) and hand (22, 23)
rotations of the prediction be replaced interactively: keep the
prediction, zero the joint, or set Euler angles with sliders. The mesh is
re-posed by the model's own SMPL layer and drawn the way the trainer
draws it, so what the sliders show is what a label or output fix would
give.

The notebook is generated from code so it stays in sync with the package;
edit this script, not the notebook::

    uv run python lidar_bedlam/scripts/build_hand_probe_notebook.py
    uv run python lidar_bedlam/scripts/verify_notebook.py \\
        --notebook notebooks/hand_pose_probe.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("notebooks/hand_pose_probe.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Add a markdown cell."""
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", text.strip()))


md("""
# Wrist and hand pose probe on the Waymo validation plot

The trainer's `image/val_plot_waymo_val` figure shows three Waymo
validation records: the crop with the predicted mesh projected onto it,
and the input points with the predicted mesh (red) and the labelled
joints (green). The predicted hands bend at the wrist although Waymo
labels only the wrist position, so nothing in the data asks for a wrist
rotation.

This notebook reproduces that figure from a checkpoint and then, one cell
per record, replaces the wrist (SMPL joints 20 `L_Wrist`, 21 `R_Wrist`)
and hand (22 `L_Hand`, 23 `R_Hand`) rotations of the prediction. Every
other joint, the shape and the placement stay as predicted; only the
posed mesh is recomputed, by the model's own SMPL layer.

Per joint group the mode is
* **predicted** — the network output,
* **zero** — the rest rotation (identity), i.e. a straight hand,
* **custom** — Euler angles (degrees, intrinsic x-y-z) from the sliders;
  the mirror box applies the left values to the right hand with the y and
  z signs flipped, which is the SMPL left/right symmetry.

The checkpoint comes from Helma:

    rsync -4 -az helma:/hnvme/workspace/v103fe17-lidar-bedlam/outputs/<run>/last.pt outputs/helma/ckpt/<run>/
""")

code("""
import io
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, display
from scipy.spatial.transform import Rotation as R

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
RUN = "b-full-mix80-000"
CONFIG = ROOT / "configs" / "b_full_mix80.yaml"
CHECKPOINT = ROOT / "outputs" / "helma" / "ckpt" / RUN / "last.pt"
SOURCE = "waymo_val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = load_config(CONFIG)
model = build_model(cfg)
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(
    {k.removeprefix("module."): v for k, v in state["model"].items()}
)
model.to(DEVICE).eval()
assert model.smpl is not None
FACES = np.asarray(model.smpl.faces, dtype=np.int64)
SMPL_EVAL = SmplModel(Path(cfg.body_models))

src = next(s for s in cfg.data.val if s.name == SOURCE)
dataset = build_dataset(src, cfg, train=False)
# the three records the trainer draws: evenly spread over the source
BATCH = to_device(spread_batch(dataset, 3), DEVICE)
with torch.no_grad():
    PRED = model(tensors_only(BATCH))
print(f"{RUN} step {state['step']}, {len(dataset)} {SOURCE} records")
for i, key in enumerate(BATCH["key"]):
    print(f"  sample {i}: {key}")
""")

md("""
## The trainer's figure, reproduced

Same renderer, same three records, same checkpoint as the wandb panel
`image/val_plot_waymo_val` at the final step.
""")

code("""
OUT_DIR = ROOT / "outputs" / "hand_probe"
path = eval_vis.render_val_plot(
    BATCH, PRED, FACES, SMPL_EVAL, OUT_DIR / f"{RUN}_{SOURCE}.png"
)
display(Image(filename=str(path)))
""")

md("""
## Re-posing one record

`repose` swaps the wrist and hand rotations of one prediction and runs
the SMPL layer again; `draw` renders the record like the trainer does
(projection on the crop, 3D against the points and the labelled joints)
and prints the joint angles in use.
""")

code("""
# SMPL joint index -> body_pose index (joint - 1)
JOINTS = {"L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23}


def euler_to_matrix(x: float, y: float, z: float) -> torch.Tensor:
    m = R.from_euler("XYZ", [x, y, z], degrees=True).as_matrix()
    return torch.as_tensor(m, dtype=torch.float32, device=DEVICE)


def repose(i: int, rots: dict[str, torch.Tensor | None]) -> dict:
    \"\"\"Prediction ``i`` with the given joint rotations replaced
    (``None`` keeps the predicted one); vertices recomputed.\"\"\"
    body_pose = PRED["body_pose"][i : i + 1].clone()
    for name, m in rots.items():
        if m is not None:
            body_pose[0, JOINTS[name] - 1] = m
    with torch.no_grad():
        out = model.smpl(
            betas=PRED["betas"][i : i + 1],
            global_orient=PRED["global_orient"][i : i + 1],
            body_pose=body_pose,
            transl=PRED["transl"][i : i + 1],
            pose2rot=False,
        )
    pred = {k: v[i : i + 1] for k, v in PRED.items() if v.ndim > 0}
    pred["body_pose"] = body_pose
    pred["vertices"] = out.vertices
    return pred


def angles(body_pose: torch.Tensor) -> str:
    rows = []
    for name, j in JOINTS.items():
        e = R.from_matrix(body_pose[0, j - 1].cpu().numpy()).as_euler(
            "XYZ", degrees=True
        )
        rows.append(f"{name}: x {e[0]:6.1f}  y {e[1]:6.1f}  z {e[2]:6.1f}")
    return "\\n".join(rows)


def draw(i: int, pred: dict) -> None:
    \"\"\"One record the way ``render_val_plot`` draws a row.\"\"\"
    batch = {
        k: (v[i : i + 1] if hasattr(v, "__getitem__") else v)
        for k, v in BATCH.items()
    }
    verts = pred["vertices"].detach().cpu().float().numpy()[0]
    k = batch["intrinsics"][0].detach().cpu().numpy()
    img = eval_vis._denormalise(batch["image"][0])
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    eval_vis._wire(ax, eval_vis._project(k, verts), FACES, "red")
    gt, _ = eval_vis._keypoints(0, batch, pred)
    ok = gt[:, 2] > 0
    ax.scatter(gt[ok, 0], gt[ok, 1], s=9, c="lime")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    err = eval_vis._placement_error(0, batch, pred)
    ax.set_title(f"{batch['dataset'][0]} | placement error {err:.2f} m")
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    pts = batch["points"][0].detach().cpu().numpy()
    valid = batch["points_valid"][0].detach().cpu().numpy().astype(bool)
    pts = pts[valid]
    if len(pts):
        ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.5, c="0.45")
    sub = verts[::4]
    ax3.scatter(sub[:, 0], sub[:, 2], -sub[:, 1], s=0.6, c="red", alpha=0.5)
    j = batch["joints3d"][0].detach().cpu().numpy()
    jv = batch["joints3d_valid"][0].detach().cpu().numpy().astype(bool)
    if jv.any():
        ax3.scatter(j[jv, 0], j[jv, 2], -j[jv, 1], s=14, c="lime")
    centre = verts.mean(0)
    eval_vis._set_equal(
        ax3, np.array([centre[0], centre[2], -centre[1]]), 1.1
    )
    ax3.view_init(elev=12, azim=-60)
    ax3.locator_params(nbins=4)
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_zlabel("-y")
    fig.tight_layout()
    # PNG bytes, not the figure object: the trainer's renderer above put
    # matplotlib on the Agg backend, which leaves widget outputs without an
    # inline figure formatter
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    display(Image(data=buf.getvalue()))
    print(angles(pred["body_pose"]))
""")

md("""
## Interactive panel

One panel per record. The wrist and hand groups each have a mode; the
sliders apply in **custom** mode. The angle print-out below the figure
gives the values in use, so a setting that looks right can be copied
into the label fitter (`FitV2Config.neutral_joints`) or an output
post-process.
""")

code("""
def panel(i: int) -> W.Widget:
    \"\"\"Widgets for record ``i``; the figure redraws on every change.\"\"\"
    modes = ["zero", "predicted", "custom"]

    def slider(desc: str) -> W.FloatSlider:
        return W.FloatSlider(
            value=0.0, min=-90.0, max=90.0, step=1.0, description=desc,
            continuous_update=False, layout=W.Layout(width="260px"),
        )

    def group(title: str) -> dict:
        return {
            "mode": W.Dropdown(options=modes, value="zero", description=title),
            "mirror": W.Checkbox(value=True, description="mirror right = left"),
            "L": [slider(f"L {a}") for a in "xyz"],
            "R": [slider(f"R {a}") for a in "xyz"],
        }

    wrist = group("wrists")
    hand = group("hands")
    out = W.Output()

    def rotation(g: dict, side: str) -> torch.Tensor | None:
        if g["mode"].value == "predicted":
            return None
        if g["mode"].value == "zero":
            return torch.eye(3, device=DEVICE)
        x, y, z = (s.value for s in g["L"])
        if side == "R" and not g["mirror"].value:
            x, y, z = (s.value for s in g["R"])
        elif side == "R":
            y, z = -y, -z
        return euler_to_matrix(x, y, z)

    def update(*_: object) -> None:
        rots = {
            "L_Wrist": rotation(wrist, "L"),
            "R_Wrist": rotation(wrist, "R"),
            "L_Hand": rotation(hand, "L"),
            "R_Hand": rotation(hand, "R"),
        }
        with out:
            out.clear_output(wait=True)
            draw(i, repose(i, rots))

    def box(g: dict) -> W.Widget:
        return W.VBox(
            [
                W.HBox([g["mode"], g["mirror"]]),
                W.HBox([W.VBox(g["L"]), W.VBox(g["R"])]),
            ]
        )

    for g in (wrist, hand):
        for w in [g["mode"], g["mirror"], *g["L"], *g["R"]]:
            w.observe(update, names="value")
    update()
    return W.VBox([W.HBox([box(wrist), box(hand)]), out])
""")

for i, title in enumerate(
    (
        "Record 0: walking past the pickup (placement error about 0.3 m)",
        "Record 1: standing at the road-work sign",
        "Record 2: standing at the kerb, arms hanging",
    )
):
    md(f"## {title}\n\nSample {i} of the figure above.")
    code(f"display(panel({i}))")


def main() -> int:
    """Write the notebook."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, text in CELLS:
        cell = (
            nbformat.v4.new_markdown_cell(text)
            if kind == "markdown"
            else nbformat.v4.new_code_cell(text)
        )
        nb.cells.append(cell)
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} with {len(nb.cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
