"""Generate ``debug/capabilities.ipynb`` (what the repository can do).

The notebook is generated from code so it stays in sync with the package;
edit this script, not the notebook. Execute it to verify::

    uv run python scripts/build_debug_notebook.py
    uv run python scripts/verify_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("debug/capabilities.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Add a markdown cell."""
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", text.strip()))


md("""
# lidar-bedlam — capabilities and data visualisation

What the repository can do right now, on real files:

1. BEDLAM frames (image, depth, masks) and the simulated LiDAR on top of them
2. Every person as a dense 3D surface (and later the SMPL mesh) with the simulated LiDAR returns
3. LiDAR resolution (Ouster OS1 with 32 / 64 / 128 / 256 channels) from two sensor viewpoints
4. Real datasets (SLOPER4D, LiDARHuman26M, Waymo) with LiDAR-to-mesh distances
5. A training batch, the losses, and the selective-attention model forward pass

Run from the repo (`uv run jupyter lab`) after `python3 scripts/dm_link.py`.
Cells that need the BEDLAM SMPL labels say so and stay empty until
`bash scripts/fetch_bedlam_labels.sh` has been run and the label loader exists.
""")

code("""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.base import add_smpl_derived, crop_sample
from lidar_bedlam.data.bedlam import CM_TO_M, BedlamFramesSource
from lidar_bedlam.io import read_exr_depth
from lidar_bedlam.lidar.simulate import PRESETS, azimuth_window, camera_hfov_deg, select_mask, sensor_pose, simulate
from lidar_bedlam.viz import (
    box_trace, draw_mask_outline, draw_points, figure_3d, mesh_trace,
    point_to_surface_distance, points_trace,
)

ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA = ROOT / "data"
BEDLAM_RAW = DATA / "generated" / "bedlam_raw"
LABELS = DATA / "generated" / "bedlam_labels"
HAVE_LABELS = LABELS.exists() and any(LABELS.rglob("*.npz"))
smpl = SmplModel(DATA / "generated" / "body_models")
rng = np.random.default_rng(0)
print("repo:", ROOT, "| BEDLAM labels downloaded:", HAVE_LABELS)
""")

md("## 1. BEDLAM ground truth: image, depth, (SMPL)")

code("""
src = BedlamFramesSource(BEDLAM_RAW, frame_stride=5)
metas = [src.meta(i) for i in range(len(src))]
sequences = sorted({m.sequence for m in metas})
picked = [sequences[i] for i in np.linspace(0, len(sequences) - 1, 5).astype(int)]
sample_ids = []
for seq in picked:
    idx = [i for i, m in enumerate(metas) if m.sequence == seq and m.person == "00"]
    sample_ids.append(idx[len(idx) // 2])
samples = [src.load(i) for i in sample_ids]
depths = [read_exr_depth(src.depth_path(i)).astype(np.float64) * CM_TO_M for i in sample_ids]
persons = [[src.load(j) for j in src.frame_persons(i)] for i in sample_ids]
print(f"{len(src)} person-frames in {len(sequences)} sequences; showing", [s.meta.key for s in samples])
""")

code("""
fig, axes = plt.subplots(5, 3, figsize=(18, 17))
for row, (s, depth, ppl) in enumerate(zip(samples, depths, persons)):
    ax = axes[row, 0]; ax.imshow(s.image); ax.set_title(f"{s.meta.sequence.split('/')[-1]} frame {s.meta.frame}: RGB, {len(ppl)} persons")
    for p in ppl:
        x0, y0, x1, y1 = p.bbox_xyxy
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="yellow", lw=1))
    ax = axes[row, 1]
    d = np.where(depth < 1e4, depth, np.nan)
    im = ax.imshow(d, cmap="turbo", vmin=0, vmax=np.nanpercentile(d, 99)); ax.set_title("depth (planar z)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="depth [m]")
    ax = axes[row, 2]
    if HAVE_LABELS:
        ax.imshow(s.image); ax.set_title("SMPL overlay: labels downloaded, loader pending")
    else:
        ax.set_title("SMPL overlay (empty: run scripts/fetch_bedlam_labels.sh)")
        ax.text(0.5, 0.5, "no SMPL labels yet", ha="center", va="center", transform=ax.transAxes)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")

md("""
## 2. Every person in 3D (rotatable)

Dense surface points come from the depth map inside each person's body+clothing mask,
i.e. exactly the clothed surface a LiDAR would see. The SMPL body mesh is added once
the labels are attached (the gap between these points and the mesh is the clothing offset).
""")

code("""
def person_surface(p, n=4000):
    pts = p.points.astype(np.float64)
    return pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]

for s, ppl in zip(samples, persons):
    traces = [points_trace(person_surface(p), f"person {p.meta.person} surface", color=i, size=1.5)
              for i, p in enumerate(ppl)]
    fig = figure_3d(traces, f"{s.meta.sequence.split('/')[-1]} frame {s.meta.frame}: dense surfaces of all persons"
                    + ("" if HAVE_LABELS else " (SMPL meshes pending labels)"))
    fig.show()
""")

md("""
## 3. Simulated LiDAR on the image: background + person masks + returns coloured by channel

Sensor: Ouster OS1-64 (64 channels over 45 deg vertical FOV, 1024 azimuth steps per
revolution). Only the azimuth window inside the camera's horizontal FOV is simulated;
the table below gives that window in columns for the three Ouster horizontal modes.
""")

code("""
print(f"{'sample':60s} {'HFOV':>7s} {'512':>5s} {'1024':>5s} {'2048':>5s}")
for s in samples:
    cols = [azimuth_window(s.camera, PRESETS["OS1-64"]).columns * m // 1024 for m in (512, 1024, 2048)]
    print(f"{s.meta.key:60s} {camera_hfov_deg(s.camera):6.1f}° {cols[0]:5d} {cols[1]:5d} {cols[2]:5d}")
""")

code("""
fig, axes = plt.subplots(5, 1, figsize=(16, 45))
scans = []
for ax, s, depth, ppl in zip(axes, samples, depths, persons):
    scan = simulate(depth, s.camera, PRESETS["OS1-64"], rng=rng)
    scans.append(scan)
    ax.imshow(s.image)
    for p in ppl:
        draw_mask_outline(ax, p.mask, "yellow")
    hit = np.zeros(s.mask.shape, bool)
    for p in ppl:
        hit |= p.mask
    person_scan = select_mask(scan, hit)
    sc = draw_points(ax, scan.pixel, scan.channel, size=2, cmap="turbo")
    draw_points(ax, person_scan.pixel, size=10, label=f"on persons: {len(person_scan.points)}")
    ax.legend(loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
    w = azimuth_window(s.camera, PRESETS["OS1-64"])
    ax.set_title(f"{s.meta.key}: OS1-64, {w.columns} columns in {w.hfov_deg:.1f}° HFOV, scene returns {len(scan.points)}, on persons {len(person_scan.points)}")
    fig.colorbar(sc, ax=ax, fraction=0.02, label="channel (0 = top)")
plt.tight_layout(); plt.show()
""")

md("""
## 4. Per person: simulated LiDAR returns against the surface (and the SMPL mesh once available)

Red = OS1-64 returns on this person, grey = dense clothed surface. When the SMPL labels are attached,
the body mesh will be drawn too, so the LiDAR-to-body distance (the clothing offset that also exists
on real data) can be inspected directly.
""")

code("""
for s, scan, ppl in zip(samples, scans, persons):
    p = ppl[0]
    sel = select_mask(scan, p.mask)
    traces = [points_trace(person_surface(p), "dense clothed surface", color="lightgrey", size=1.2),
              points_trace(sel.points, f"OS1-64 returns ({len(sel.points)})", color="red", size=3)]
    figure_3d(traces, f"{p.meta.key}: LiDAR returns vs surface", height=550).show()
""")

md("""
## 5. Resolution and viewpoint: OS1 with 32 / 64 / 128 / 256 channels (1024 steps/rev), sensor 10 cm above the camera, and 1 m to the right
""")

code("""
s, depth, ppl = samples[0], depths[0], persons[0]
person_mask = np.zeros(s.mask.shape, bool)
for p in ppl:
    person_mask |= p.mask
viewpoints = {"10 cm above camera": sensor_pose(np.array([0.0, -0.10, 0.0])),
              "1 m right of camera": sensor_pose(np.array([1.0, 0.0, 0.0]))}
beams = ["OS1-32", "OS1-64", "OS1-128", "OS1-256"]
x0, y0, x1, y1 = s.bbox_xyxy
pad = 60
fig, axes = plt.subplots(len(beams), len(viewpoints), figsize=(14, 22))
grid_scans = {}
for r, name in enumerate(beams):
    for c, (vname, pose) in enumerate(viewpoints.items()):
        scan = simulate(depth, s.camera, PRESETS[name], pose, rng)
        sel = select_mask(scan, person_mask)
        grid_scans[(name, vname)] = sel
        ax = axes[r, c]
        ax.imshow(s.image)
        draw_points(ax, sel.pixel, sel.channel, size=12, cmap="turbo")
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y1 + pad, y0 - pad)
        ax.set_title(f"{name}, {vname}: {len(sel.points)} returns on person(s)")
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")

code("""
for vname in viewpoints:
    traces = [points_trace(person_surface(ppl[0]), "dense surface", color="lightgrey", size=1.0)]
    for name in beams:
        sel = grid_scans[(name, vname)]
        pm = ppl[0].mask[sel.pixel[:, 1], sel.pixel[:, 0]]
        traces.append(points_trace(sel.points[pm], f"{name} ({int(pm.sum())})", size=3))
    figure_3d(traces, f"{ppl[0].meta.key}: channel counts from {vname} (toggle in legend)", height=600).show()
""")

md("""
## 5b. Interactive: sample, person, the 12 resolutions, occlusions and jitter

Pick a sample and a person, tick any of the 4 x 3 resolutions (channels x azimuth steps per
revolution), choose the viewpoint, cover part of the camera image and part of the LiDAR
window, add object occlusion, jitter, outliers, channel dropout and a calibration error.
Ego speed (expected value ± standard deviation, 10 km/h steps; Waymo urban ≈ 20 ± 20 km/h)
drives the LiDAR rolling shutter: the sweep crosses the camera window column by column at
10 Hz, so a moving platform reports each column from a different position.
Every change re-renders the 2D overlay (one panel per ticked resolution) and the 3D plot.
""")

code("""
import ipywidgets as W
from IPython.display import display, clear_output
from lidar_bedlam.lidar.simulate import ouster
from lidar_bedlam.lidar.augment import (
    SensorCover, add_outliers, axis_cut_fraction, cover_mask, drop_channels,
    height_cut_fraction, jitter, miscalibrated_pose,
)
from lidar_bedlam.data.image_augment import cover_side
from lidar_bedlam.lidar.motion import EgoMotion, SpeedSetting, apply_rolling_shutter, sweep_duration_s

CHANNELS, STEPS = [32, 64, 128, 256], [512, 1024, 2048]
sample_dd = W.Dropdown(options=[(f"{s.meta.sequence.split('/')[-1]} f{s.meta.frame}", i) for i, s in enumerate(samples)], description="sample")
person_dd = W.Dropdown(description="person")
def _refresh_persons(*_):
    person_dd.options = [(f"person {p.meta.person}", j) for j, p in enumerate(persons[sample_dd.value])]
    person_dd.value = 0
sample_dd.observe(_refresh_persons, "value"); _refresh_persons()
grid_boxes = {(c, st): W.Checkbox(value=(c == 64 and st == 1024), indent=False, layout=W.Layout(width="70px")) for c in CHANNELS for st in STEPS}
cells = [W.Label("channels \\\\ steps")] + [W.Label(str(st), layout=W.Layout(width="70px")) for st in STEPS]
for c in CHANNELS:
    cells += [W.Label(f"{c} channels", layout=W.Layout(width="110px"))] + [grid_boxes[(c, st)] for st in STEPS]
grid = W.GridBox(cells, layout=W.Layout(grid_template_columns="110px 70px 70px 70px"))
family_dd = W.Dropdown(options=["OS0", "OS1", "OS2"], value="OS1", description="family")
viewpoint_dd = W.Dropdown(options=[("camera", 0), ("10 cm above", 1), ("1 m right", 2)], description="viewpoint")
cam_side = W.Dropdown(options=["none", "left", "right", "top", "bottom"], description="camera cover")
cam_frac = W.FloatSlider(0.3, min=0.0, max=0.8, step=0.05, description="cover frac", continuous_update=False)
lidar_left = W.FloatSlider(0.0, min=0.0, max=0.9, step=0.05, description="LiDAR left", continuous_update=False)
lidar_right = W.FloatSlider(0.0, min=0.0, max=0.9, step=0.05, description="LiDAR right", continuous_update=False)
lidar_top = W.FloatSlider(0.0, min=0.0, max=0.9, step=0.05, description="LiDAR top", continuous_update=False)
lidar_bottom = W.FloatSlider(0.0, min=0.0, max=0.9, step=0.05, description="LiDAR bottom", continuous_update=False)
obj_occ = W.Dropdown(options=["none", "legs", "head", "left side", "right side", "front"], description="object occl.")
obj_frac = W.FloatSlider(0.4, min=0.05, max=0.9, step=0.05, description="occl. frac", continuous_update=False)
jitter_s = W.FloatSlider(0.0, min=0.0, max=0.1, step=0.005, description="jitter [m]", continuous_update=False, readout_format=".3f")
outlier_s = W.FloatSlider(0.0, min=0.0, max=0.3, step=0.01, description="outliers", continuous_update=False)
chdrop_s = W.FloatSlider(0.0, min=0.0, max=0.8, step=0.05, description="chan. dropout", continuous_update=False)
miscal_s = W.FloatSlider(0.0, min=0.0, max=3.0, step=0.1, description="miscalib [deg]", continuous_update=False)
speed_mean = W.IntSlider(0, min=0, max=130, step=10, description="speed E [km/h]", continuous_update=False)
speed_std = W.IntSlider(0, min=0, max=60, step=10, description="speed SD [km/h]", continuous_update=False)
spin_hz = W.Dropdown(options=[10, 20], value=10, description="spin [Hz]")
out = W.Output()

def render(*_):
    with out:
        clear_output(wait=True)
        if sample_dd.value is None or person_dd.value is None:
            return
        s, depth = samples[sample_dd.value], depths[sample_dd.value]
        p = persons[sample_dd.value][person_dd.value]
        pose = [np.eye(4), sensor_pose(np.array([0.0, -0.1, 0.0])), sensor_pose(np.array([1.0, 0.0, 0.0]))][viewpoint_dd.value]
        rng_i = np.random.default_rng(0)
        if miscal_s.value > 0:
            pose = miscalibrated_pose(pose, miscal_s.value, 0.01 * miscal_s.value, rng_i)
        image = s.image if cam_side.value == "none" else cover_side(s.image, cam_side.value, cam_frac.value)
        cover = SensorCover(lidar_left.value, lidar_right.value, lidar_top.value, lidar_bottom.value)
        results = {}
        for (c, st), cb in grid_boxes.items():
            if not cb.value:
                continue
            spec = ouster(family_dd.value, c, st)
            scan = select_mask(simulate(depth, s.camera, spec, pose, rng_i), p.mask)
            w = azimuth_window(s.camera, spec, pose)
            speed = SpeedSetting(float(speed_mean.value), float(speed_std.value)).sample_mps(np.random.default_rng(1))
            scan = apply_rolling_shutter(scan, EgoMotion(speed_mps=speed, spin_hz=float(spin_hz.value)), w)
            keep = cover_mask(scan.channel, scan.azimuth_deg, c, w.az_min_deg, w.az_max_deg, cover)
            if chdrop_s.value > 0:
                keep &= drop_channels(scan.channel, c, chdrop_s.value, rng_i)
            pts, pix, ch = scan.points[keep].astype(np.float64), scan.pixel[keep], scan.channel[keep]
            if obj_occ.value != "none" and len(pts):
                k = {"legs": lambda: height_cut_fraction(pts, obj_frac.value, True),
                     "head": lambda: height_cut_fraction(pts, obj_frac.value, False),
                     "left side": lambda: axis_cut_fraction(pts, 0, obj_frac.value, True),
                     "right side": lambda: axis_cut_fraction(pts, 0, obj_frac.value, False),
                     "front": lambda: axis_cut_fraction(pts, 2, obj_frac.value, True)}[obj_occ.value]()
                pts, pix, ch = pts[k], pix[k], ch[k]
            pts = jitter(pts, jitter_s.value, rng_i)
            if outlier_s.value > 0:
                pts = add_outliers(pts, outlier_s.value, 0.3, rng_i)
            results[(c, st)] = (pts, pix, ch, w.columns)
        if speed_mean.value or speed_std.value:
            dur = sweep_duration_s(w, spec, float(spin_hz.value))
            print(f"rolling shutter: ego {speed * 3.6:.0f} km/h sampled from {speed_mean.value} ± {speed_std.value} km/h, sweep over the window {dur * 1000:.1f} ms -> up to {speed * dur * 100:.1f} cm shift across the window")
        n = max(len(results), 1); ncols = min(n, 3); nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 7 * nrows), squeeze=False)
        x0, y0, x1, y1 = p.bbox_xyxy; pad = 60
        for ax, ((c, st), (pts, pix, ch, ncol)) in zip(axes.flat, results.items()):
            ax.imshow(image); draw_mask_outline(ax, p.mask, "yellow")
            if len(pix):
                draw_points(ax, pix, ch, size=14, cmap="turbo")
            ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y1 + pad, y0 - pad); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{family_dd.value}-{c} @ {st} steps ({ncol} cols): {len(pts)} pts")
        for ax in list(axes.flat)[len(results):]:
            ax.axis("off")
        plt.tight_layout(); plt.show()
        traces = [points_trace(person_surface(p), "dense surface", color="lightgrey", size=1.0)]
        for (c, st), (pts, _, _, _) in results.items():
            if len(pts):
                traces.append(points_trace(pts, f"{family_dd.value}-{c} @ {st} ({len(pts)})", size=3))
        figure_3d(traces, f"{p.meta.key}: selected resolutions (toggle in legend)", height=650).show()

for w in [sample_dd, person_dd, family_dd, viewpoint_dd, cam_side, cam_frac, lidar_left, lidar_right,
          lidar_top, lidar_bottom, obj_occ, obj_frac, jitter_s, outlier_s, chdrop_s, miscal_s,
          speed_mean, speed_std, spin_hz, *grid_boxes.values()]:
    w.observe(render, "value")
ui = W.VBox([W.HBox([sample_dd, person_dd, family_dd, viewpoint_dd]), grid,
             W.HBox([cam_side, cam_frac]), W.HBox([lidar_left, lidar_right]), W.HBox([lidar_top, lidar_bottom]),
             W.HBox([obj_occ, obj_frac]), W.HBox([jitter_s, outlier_s]), W.HBox([chdrop_s, miscal_s]),
             W.HBox([speed_mean, speed_std, spin_hz])])
display(ui, out); render()
""")

md("""
## 6. Real datasets: LiDAR points vs the ground-truth SMPL mesh

This is the distance that transfers to real data: points lie on clothing and sensor noise,
the mesh is the body. Colour = distance of each LiDAR point to the nearest mesh vertex.
""")

code("""
from lidar_bedlam.data.sloper4d import Sloper4dSource
from lidar_bedlam.data.lidarhuman26m import LidarHuman26mSource
from lidar_bedlam.data.waymo import WaymoSource

real = {
    "sloper4d": (Sloper4dSource(DATA / "sloper4d", smpl, ["seq008_running_001"]), 100),
    "lidarhuman26m": (LidarHuman26mSource(DATA / "lidarhuman26m", smpl, "test"), 500),
    "waymo": (WaymoSource(DATA / "waymo_perception" / "waymo_pose_complete_4", "val"), 3),
}
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
real_samples = {}
for ax, (name, (source, idx)) in zip(axes, real.items()):
    rs = source.load(idx)
    add_smpl_derived(rs, smpl)
    real_samples[name] = rs
    c = crop_sample(rs, 512)
    ax.imshow(c.image)
    if "vertices" in rs.extra:
        draw_points(ax, c.camera.project(rs.extra["vertices"]), size=1, label="SMPL vertices")
    if rs.joints3d is not None:
        uv = c.camera.project(rs.joints3d[rs.joints3d_valid])
        ax.scatter(uv[:, 0], uv[:, 1], s=40, c="lime", marker="x", label=f"joints ({rs.joint_convention})")
    pv = c.camera.project(rs.points.astype(np.float64))
    ax.scatter(pv[:, 0], pv[:, 1], s=6, c="cyan", label=f"LiDAR ({len(rs.points)})")
    ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.legend(loc="lower right"); ax.set_title(f"{name}: {rs.meta.key}")
plt.tight_layout(); plt.show()
""")

code("""
for name, rs in real_samples.items():
    traces = []
    if "vertices" in rs.extra:
        v = rs.extra["vertices"]
        traces.append(mesh_trace(v, smpl.faces, "GT SMPL mesh", opacity=0.5))
        d = point_to_surface_distance(rs.points, v)
        traces.append(points_trace(rs.points, "LiDAR (colour = dist to mesh)", color=d, size=3,
                                   colorscale="Turbo", colorbar_title="m"))
        print(f"{name}: LiDAR-to-mesh distance median {np.median(d):.3f} m, mean {d.mean():.3f} m, p90 {np.percentile(d, 90):.3f} m")
    else:
        traces.append(points_trace(rs.points, "LiDAR", color="red", size=3))
        traces.append(points_trace(rs.joints3d[rs.joints3d_valid], "GT keypoints", color="lime", size=5))
    if rs.box3d is not None:
        traces.append(box_trace(rs.box3d, "3D box"))
    figure_3d(traces, f"{name}: {rs.meta.key}", height=600).show()
""")

md("## 7. A training batch and the losses")

code("""
from torch.utils.data import DataLoader, Subset
from lidar_bedlam.data.torch_dataset import HumanPoseDataset
from lidar_bedlam.losses.smpl import FusionLoss

sources = [real["sloper4d"][0], real["lidarhuman26m"][0], real["waymo"][0]]
ds = HumanPoseDataset(sources, smpl, out_size=256, n_points=1024)
ids = [100, len(sources[0]) + 500, len(sources[0]) + len(sources[1]) + 3, 101]
batch = next(iter(DataLoader(Subset(ds, ids), batch_size=len(ids))))
for k, v in batch.items():
    print(f"{k:18s}", tuple(v.shape) if hasattr(v, "shape") else v)
pred_gt = {k: batch[k] for k in ["global_orient", "body_pose", "betas", "transl", "joints3d", "box3d"]}
pred_gt["kp2d"] = batch["kp2d"][..., :2]
total, parts = FusionLoss()(pred_gt, batch)
print("loss on ground truth:", float(total), {k: round(float(v), 5) for k, v in parts.items()})
""")

md("""
## 8. The selective-attention fusion model

ViT-H image tokens (TokenHMR weights) and LiDAR point tokens feed joint-group queries; each query
mixes image and LiDAR cross-attention through a learned gate that starts from a prior
(image for head / hands / feet, LiDAR for torso / legs / root / shape).
""")

code("""
from lidar_bedlam.models.fusion import ModelConfig, SelectiveFusionModel
from lidar_bedlam.models.selective_attention import JOINT_GROUPS
from lidar_bedlam.models.vit import VIT_H, load_backbone_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SelectiveFusionModel(ModelConfig(vit=VIT_H, smpl_model_dir=DATA / "generated" / "body_models"))
ckpt = Path("/home/max/nas_drive/methods/max/data/checkpoints/tokenhmr/tokenhmr_model_latest/data/checkpoints/tokenhmr_model_latest.ckpt")
if ckpt.exists():
    missing, unexpected = load_backbone_weights(model.backbone, ckpt)
    print("ViT-H weights from TokenHMR: missing", len(missing), "unexpected", len(unexpected))
model = model.to(device).eval()
n_all = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters: {n_all/1e6:.1f} M total, {n_train/1e6:.1f} M trainable (backbone frozen)")
with torch.no_grad():
    out = model({k: batch[k].to(device) for k in ["image", "points", "intrinsics"]})
for k, v in out.items():
    print(f"{k:14s}", tuple(v.shape))
loss_total, loss_parts = FusionLoss()({k: v.cpu() for k, v in out.items()}, batch)
print("untrained loss:", float(loss_total), {k: round(float(v), 4) for k, v in loss_parts.items()})
""")

code("""
gates = out["gates"].mean(1).cpu().numpy()  # (layers, groups), averaged over the batch
fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(gates, vmin=0, vmax=1, cmap="coolwarm", aspect="auto")
ax.set_xticks(range(len(JOINT_GROUPS))); ax.set_xticklabels([g.name for g in JOINT_GROUPS], rotation=45, ha="right")
ax.set_yticks(range(gates.shape[0])); ax.set_ylabel("decoder layer")
fig.colorbar(im, ax=ax, label="image gate (1 = image, 0 = LiDAR)")
ax.set_title("selective-attention gates at initialisation (priors)"); plt.tight_layout(); plt.show()
""")

md("""
## 9. Sensor rigs: calibration trees of the datasets

Every dataset's calibration as a star graph around its base link: each sensor is drawn with
its own x (red), y (green), z (blue) axes at its calibrated pose, cameras with an orange
frustum computed from their intrinsics (through the dataset's camera axis convention).
Tick up to three rigs in the table; each figure is titled with the dataset and its mount.
The same rigs are exported to `.docs/figures` by `scripts/export_rigs.py` (PNG, HTML, GLB).
""")

code("""
from lidar_bedlam.rigs.registry import load_all_rigs
from lidar_bedlam.rigs.plot import rigs_figure

loaded, skipped = load_all_rigs(DATA)
rigs = {rig.title: rig for rig in loaded.values()}
for slug, why in skipped.items():
    print("skipped:", slug, why)
for rig in rigs.values():
    print(rig.tree_text()); print()
""")

code("""
rig_boxes = {title: W.Checkbox(value=(i < 3), indent=False, layout=W.Layout(width="40px")) for i, title in enumerate(rigs)}
rows = [W.HBox([W.Label("show", layout=W.Layout(width="40px")), W.Label("dataset (mount)", layout=W.Layout(width="220px")),
                W.Label("lidars", layout=W.Layout(width="60px")), W.Label("cameras", layout=W.Layout(width="70px")), W.Label("base frame")])]
for title, rig in rigs.items():
    rows.append(W.HBox([rig_boxes[title], W.Label(title, layout=W.Layout(width="220px")),
                        W.Label(str(len(rig.by_kind("lidar"))), layout=W.Layout(width="60px")),
                        W.Label(str(len(rig.by_kind("camera"))), layout=W.Layout(width="70px")), W.Label(rig.base_frame)]))
rig_out = W.Output()

def render_rigs(*_):
    with rig_out:
        clear_output(wait=True)
        chosen = [rigs[t] for t, cb in rig_boxes.items() if cb.value]
        if not chosen:
            print("tick at least one rig"); return
        if len(chosen) > 3:
            print("showing the first three of", len(chosen), "ticked rigs")
        rigs_figure(chosen[:3]).show()

for cb in rig_boxes.values():
    cb.observe(render_rigs, "value")
display(W.VBox(rows), rig_out); render_rigs()
""")


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
