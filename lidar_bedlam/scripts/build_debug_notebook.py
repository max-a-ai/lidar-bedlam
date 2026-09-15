"""Generate ``notebooks/capabilities.ipynb`` (what the repository can do).

The notebook is generated from code so it stays in sync with the package;
edit this script, not the notebook. Execute it to verify::

    uv run python lidar_bedlam/scripts/build_debug_notebook.py
    uv run python lidar_bedlam/scripts/verify_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("notebooks/capabilities.ipynb")

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

Run from the repo (`uv run jupyter lab`) after `python3 lidar_bedlam/scripts/dm_link.py`.
Cells that need the BEDLAM SMPL labels say so and stay empty until
`bash lidar_bedlam/scripts/fetch_bedlam_labels.sh` has been run and the label loader exists.
""")

code("""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.base import add_smpl_derived, crop_sample
from lidar_bedlam.data.bedlam import CM_TO_M, BedlamFramesSource
from lidar_bedlam.utils.io import read_exr_depth, read_mask
from lidar_bedlam.lidar.simulate import PRESETS, azimuth_window, camera_hfov_deg, select_mask, sensor_pose, simulate
from lidar_bedlam.utils.viz import (
    box_trace, draw_mask_outline, draw_points, figure_3d, mesh_trace,
    point_to_surface_distance, points_trace,
)

ROOT = Path.cwd() if (Path.cwd() / "resources").exists() else Path.cwd().parent
DATA = ROOT / "resources" / "data"
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
        ax.set_title("SMPL overlay (empty: run lidar_bedlam/scripts/fetch_bedlam_labels.sh)")
        ax.text(0.5, 0.5, "no SMPL labels yet", ha="center", va="center", transform=ax.transAxes)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")

md("""## 1b. Person masks include the hair

BEDLAM renders one mask per part (`body`, `clothing`, and `hair` in the
`*handhair*` groups). The person mask is the union of every part that
exists; without `hair` the head of a long-haired character lost its top
and the simulated LiDAR returns on the hair were dropped. The sample
below is the one where this was noticed.""")

code("""
HAIR_KEY = ("20221024_3-10_100_batch01handhair_static_highSchoolGym", "seq_000099", "0075", "00")
hair_idx = next(i for i, m in enumerate(metas)
                if m.sequence == f"{HAIR_KEY[0]}/{HAIR_KEY[1]}" and m.frame == HAIR_KEY[2] and m.person == HAIR_KEY[3])
hs = src.load(hair_idx)
hdepth = read_exr_depth(src.depth_path(hair_idx)).astype(np.float64) * CM_TO_M
mask_dir = BEDLAM_RAW / HAIR_KEY[0] / "masks" / HAIR_KEY[1]
parts = {}
for part in ("body", "clothing", "hair"):
    path = mask_dir / f"{HAIR_KEY[1]}_{HAIR_KEY[2]}_{HAIR_KEY[3]}_{part}.png"
    if path.exists():
        parts[part] = read_mask(path)
without_hair = parts["body"] | parts["clothing"]
with_hair = without_hair | parts.get("hair", np.zeros_like(without_hair))
hscan = simulate(hdepth, hs.camera, PRESETS["OS1-128"], rng=np.random.default_rng(0))
n_before, n_after = len(select_mask(hscan, without_hair).points), len(select_mask(hscan, with_hair).points)
x0, y0, x1, y1 = hs.bbox_xyxy.astype(int); pad = 40
crop = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes[0].imshow(hs.image[crop]); axes[0].set_title("RGB crop")
for ax, (title, m) in zip(axes[1:3], (("body | clothing (old)", without_hair), ("body | clothing | hair (now)", with_hair))):
    ax.imshow(hs.image[crop]); ax.imshow(np.ma.masked_where(~m[crop], m[crop]), alpha=0.5, cmap="autumn"); ax.set_title(title)
axes[3].imshow(hs.image[crop])
pc = select_mask(hscan, with_hair); ph = select_mask(hscan, parts.get("hair", np.zeros_like(with_hair)))
axes[3].scatter(pc.pixel[:, 0] - crop[1].start, pc.pixel[:, 1] - crop[0].start, s=6, c="cyan", label=f"person returns: {n_after} (was {n_before})")
axes[3].scatter(ph.pixel[:, 0] - crop[1].start, ph.pixel[:, 1] - crop[0].start, s=10, c="red", label=f"on hair: {len(ph.points)}")
axes[3].legend(loc="lower right"); axes[3].set_title("OS1-128 returns on the person")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
print(f"hair adds {int(parts['hair'].sum())} mask pixels and {n_after - n_before} LiDAR returns for {'/'.join(HAIR_KEY)}")
""")

md("""
## 2. Every person in 3D (rotatable)

Dense surface points come from the depth map inside each person's body+clothing+hair mask,
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
## 5. Resolution and viewpoint: OS1 with 32 / 64 / 128 / 256 channels (1024 steps/rev), sensor 10 cm above the camera, 1 m, 5 m and 10 m to the right

The simulator marches every beam through the camera's depth map and keeps
the first crossing with a surface that faces the beam (depth-map normals:
dot(beam, normal) < 0). From the camera's own position every visible
surface faces the beams and the whole front of the person returns; the
farther the sensor moves to the right, the more of the person's left-facing
surface turns away from it and returns nothing, while the right-facing
surface (and only that) keeps returning. The depth map does not know the
person's right side, so a far-offset sensor sees fewer, not wrong, points.
""")

code("""
s, depth, ppl = samples[0], depths[0], persons[0]
person_mask = np.zeros(s.mask.shape, bool)
for p in ppl:
    person_mask |= p.mask
viewpoints = {"10 cm above camera": sensor_pose(np.array([0.0, -0.10, 0.0])),
              "1 m right of camera": sensor_pose(np.array([1.0, 0.0, 0.0])),
              "5 m right of camera": sensor_pose(np.array([5.0, 0.0, 0.0])),
              "10 m right of camera": sensor_pose(np.array([10.0, 0.0, 0.0]))}
beams = ["OS1-32", "OS1-64", "OS1-128", "OS1-256"]
x0, y0, x1, y1 = s.bbox_xyxy
pad = 60
fig, axes = plt.subplots(len(beams), len(viewpoints), figsize=(26, 22))
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
The same rigs are exported to `.docs/figures` by `lidar_bedlam/scripts/export_rigs.py` (PNG, HTML, GLB).
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

md("""## 10. 3DPW: real image + LiDAR simulated on the clothing-offset mesh

3DPW has real images and SMPL labels but no depth. The labelled mesh is
pushed outwards along its normals by 1-4 cm (the clothing layer a LiDAR
actually hits), rasterised into a depth map and scanned with the same
sensor plan as BEDLAM. Shards live in `resources/data/generated/meshlidar/v1`.
Executing the cell shows set 0; **re-roll** shows set 1, 2, ... (seeded:
the same sets every session).""")

code("""
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17

def load_shards(pattern):
    paths = sorted(p for p in (DATA / "generated").glob(pattern) if ".fit." not in p.name and not p.name.endswith("stats.npz"))
    return [Shard(p) for p in paths]

def project_crop(K, pts):
    z = np.maximum(pts[:, 2], 1e-3)
    return np.stack([K[0, 0] * pts[:, 0] / z + K[0, 2], K[1, 1] * pts[:, 1] / z + K[1, 2]], 1)

pw_shards = load_shards("meshlidar/v1/threedpw_*.npz")
pw_total = sum(len(s) for s in pw_shards)
pw_roll = 0
pw_btn = W.Button(description="re-roll", icon="refresh"); pw_out = W.Output()

def pw_render(*_):
    global pw_roll
    with pw_out:
        clear_output(wait=True)
        if not pw_shards:
            print("no 3DPW shards yet: run lidar_bedlam/scripts/generate_3dpw.py"); return
        rng = np.random.default_rng(1000 + pw_roll)
        picks = rng.integers(0, pw_total, 4)
        print(f"set {pw_roll}: {pw_total} records, showing", list(picks))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        traces = []
        for j, (ax, g) in enumerate(zip(axes, picks)):
            si, i = 0, int(g)
            while i >= len(pw_shards[si]):
                i -= len(pw_shards[si]); si += 1
            sh = pw_shards[si]; K = sh.array("intrinsics")[i]
            ax.imshow(sh.array("image")[i]); draw_mask_outline(ax, sh.array("mask")[i], "yellow")
            sc = sh.scan(i, "main_0"); uv = project_crop(K, sc.points)
            sp = ax.scatter(uv[:, 0], uv[:, 1], c=sc.points[:, 2], s=8, cmap="turbo")
            ax.set_title(f"{sh.array('key')[i].split('/')[2]} f{sh.array('key')[i].split('/')[3]} p{sh.array('key')[i].split('/')[4]}\\n{sc.channels}ch x {sc.steps}: {len(sc.points)} returns", fontsize=9)
            ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                params = SmplParams(sh.row("global_orient", i), sh.row("body_pose", i), sh.row("betas", i), sh.row("transl", i))
                v, _ = smpl.forward(params)
                traces += [mesh_trace(v, smpl.faces, "SMPL label (skin)", opacity=0.45),
                           points_trace(sc.points, "LiDAR returns (clothing offset)", color=sc.points[:, 2], size=3, colorscale="Turbo")]
        fig.colorbar(sp, ax=axes[-1], fraction=0.03, label="depth [m]"); plt.tight_layout(); plt.show()
        figure_3d(traces, "first sample: label mesh and simulated returns").show()
    pw_roll += 1

pw_btn.on_click(pw_render); display(pw_btn, pw_out); pw_render()
""")

md("""## 11. Waymo pseudo ground truth: fitted SMPL on the point cloud

Waymo has 3D keypoints, no meshes. `lidar_bedlam/scripts/pseudo_smpl_waymo.py`
initialises SMPL from the trained model and fits pose, shape and
translation to the keypoints, the 2D keypoints and the LiDAR returns
(pulled to 3 cm outside the mesh). Shards: `real/v1_pseudo`. Green =
Waymo keypoints, red = COCO joints of the fitted mesh; the 3D view shows
the fitted mesh inside the point cloud. Re-roll as above.""")

code("""
ps_shards = load_shards("real/v1_pseudo/waymo_train_*.npz")
ps_fit = {p.name.replace(".fit.npz", ""): np.load(p) for p in sorted((DATA / "generated" / "real" / "v1_pseudo").glob("*.fit.npz"))}
ps_total = sum(len(s) for s in ps_shards)
ps_roll = 0
ps_btn = W.Button(description="re-roll", icon="refresh"); ps_out = W.Output()
WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]; CSEL = WAYMO15_TO_COCO17[WSEL]
J_COCO = np.load(DATA / "generated" / "body_models" / "J_regressor_coco.npy")

def ps_render(*_):
    global ps_roll
    with ps_out:
        clear_output(wait=True)
        if not ps_shards:
            print("no pseudo-GT shards yet: run lidar_bedlam/scripts/pseudo_smpl_waymo.py"); return
        rng = np.random.default_rng(2000 + ps_roll)
        picks = rng.integers(0, ps_total, 3)
        print(f"set {ps_roll}: {ps_total} records, showing", list(picks))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        traces = []
        for j, (ax, g) in enumerate(zip(axes, picks)):
            si, i = 0, int(g)
            while i >= len(ps_shards[si]):
                i -= len(ps_shards[si]); si += 1
            sh = ps_shards[si]; K = sh.array("intrinsics")[i]
            params = SmplParams(sh.row("global_orient", i), sh.row("body_pose", i), sh.row("betas", i), sh.row("transl", i))
            v, _ = smpl.forward(params); coco = J_COCO @ v
            kp = sh.array("kp2d")[i]; ok = kp[:15, 2] > 0
            err = ps_fit[sh.path.stem]["error_m"][i] * 1000 if sh.path.stem in ps_fit else float("nan")
            ax.imshow(sh.array("image")[i])
            uvv = project_crop(K, v[::10]); ax.scatter(uvv[:, 0], uvv[:, 1], s=1, c="red", alpha=0.4)
            ax.scatter(kp[:15][ok, 0], kp[:15][ok, 1], s=25, c="lime", label="Waymo kp")
            uvc = project_crop(K, coco[CSEL]); ax.scatter(uvc[:, 0], uvc[:, 1], s=25, c="red", marker="x", label="fitted COCO")
            ax.set_title(f"{sh.array('key')[i].split('/')[-1][:24]}\\naccepted={bool(sh.array('has_smpl')[i])}, kp err {err:.0f} mm", fontsize=9)
            ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.legend(loc="lower left", fontsize=8)
                sc = sh.scan(i, "real")
                traces += [mesh_trace(v, smpl.faces, "pseudo-GT SMPL", opacity=0.5),
                           points_trace(sc.points, "LiDAR returns", color="black", size=3),
                           points_trace(sh.array("joints3d")[i][:15][sh.array("joints3d_valid")[i][:15]], "Waymo keypoints", color="lime", size=6)]
        plt.tight_layout(); plt.show()
        figure_3d(traces, "first sample: fitted mesh in the point cloud").show()
    ps_roll += 1

ps_btn.on_click(ps_render); display(ps_btn, ps_out); ps_render()
""")


md("""## 12. Architecture: selective-attention fusion

Block diagram of the model (`.docs/figures/architecture.tex`, TikZ; the PDF
goes into the paper draft, the PNG is shown here). Image tokens come from a
frozen ViT-H (TokenHMR weights, precomputed per shard); LiDAR returns go
through a trained point tokenizer; 12 joint-group queries attend to both
streams and mix them with a learned gate that starts from a routing prior.
The translation head has two anchors: the camera parameterisation
(u, v, log z) used so far, and the point anchor (centroid of the input
points plus a learned offset, `model.transl_anchor: points`) added on
2026-09-15 after the LiDAR-HMR comparison.""")

code("""
from IPython.display import Image
display(Image(filename=str(ROOT / ".docs" / "figures" / "architecture.png"), width=1400))
""")

md("""## 13. Scoreboard: our runs against the published pipelines

Every row is scored on the same records with the same protocol
(`metrics/protocol.py`): Waymo val (894 crops) and the full SLOPER4D test
split (9,904). Baseline meshes come from `lidar_bedlam/scripts/baselines`
(each model in its own env), our runs from Helma (`pull_helma_results.sh`).
Per column within a table: green best, yellow second, red third; lower is
better except mAP; `abs` is the joint error without any centring, i.e.
pose and placement together.""")

code("""
import subprocess
from lidar_bedlam.scripts import build_scoreboard as sb

subprocess.run(["bash", str(ROOT / "lidar_bedlam/scripts/pull_helma_results.sh"), str(ROOT / "outputs/helma")], capture_output=True)
HELMA, BASE = ROOT / "outputs" / "helma", ROOT / "outputs" / "baselines"
display(HTML(f"<style>{sb.STYLE}</style>"))
title, desc, spec = sb.GROUPS[0]
rows = sb.static_rows(BASE / "results_static.json", sb.STATIC_ROWS) + sb.group_rows(HELMA, spec)
display(HTML(f"<h3>{title}</h3><p>{desc}</p>" + sb.table_html(rows)))
""")

md("""### 13b. Ablation axes (one cell each)

The one-third-schedule screen (17 epochs = 3,468 steps, reference
`abl-mixed-short`, two seeds where available); every axis changes one thing.""")

code("""
title, desc, spec = sb.GROUPS[1]  # fusion axis
display(HTML(f"<h3>{title}</h3><p>{desc}</p>" + sb.table_html(sb.group_rows(HELMA, spec))))
""")

code("""
title, desc, spec = sb.GROUPS[2]  # synthesis axis
display(HTML(f"<h3>{title}</h3><p>{desc}</p>" + sb.table_html(sb.group_rows(HELMA, spec))))
""")

code("""
title, desc, spec = sb.GROUPS[3]  # data axis
display(HTML(f"<h3>{title}</h3><p>{desc}</p>" + sb.table_html(sb.group_rows(HELMA, spec))))
""")

code("""
title, desc, spec = sb.GROUPS[4]  # label sources
display(HTML(f"<h3>{title}</h3><p>{desc}</p>" + sb.table_html(sb.group_rows(HELMA, spec))))
mirror = sb.static_rows(BASE / "results_mirror.json", sb.MIRROR_ROWS)
if mirror:
    display(HTML("<h3>Mirror test (memorisation check)</h3><p>Inputs mirrored left/right, meshes mirrored back through the SMPL symmetry map; a memorised validation set would collapse.</p>" + sb.table_html(mirror)))
""")

md("""## 14. Inference: where the mesh lands in 3D

Our headline checkpoint (`resources/pretrained-checkpoints/ours/full-main-mixed-last.pt`,
pulled from Helma) is run once over both validation sets and cached in
`outputs/ours/full-main-mixed/`; the baselines' meshes are read from
`outputs/baselines/`. For every method the paper protocol is evaluated per
record, then three records per dataset are picked automatically: the one
where the image methods misplace the person most while we do not, the one
where LiDAR-HMR's pose is worst relative to ours, and (Waymo) the farthest
person we still place within 0.3 m. Bird's-eye view: depth along the ray is
where the image methods fail; side view: pose.""")

code("""
from lidar_bedlam.utils import showcase as sc

CFG = ROOT / "configs" / "full_main_mixed.yaml"
CKPT = ROOT / "resources" / "pretrained-checkpoints" / "ours" / "full-main-mixed-last.pt"
OURS = ROOT / "outputs" / "ours" / "full-main-mixed"
smpl_eval = SmplModel(DATA / "generated" / "body_models")
if CKPT.exists():
    ours_paths = sc.run_model(CFG, CKPT, OURS, device=device)
    print("our predictions:", {k: str(p.relative_to(ROOT)) for k, p in ours_paths.items()})
else:
    ours_paths = {}
    print("checkpoint missing:", CKPT)
""")

code("""
showcase = {}
for split in ["waymo_val", "sloper4d_test"]:
    if split not in ours_paths:
        continue
    tables = sc.load_tables(ours_paths[split], ROOT / "outputs" / "baselines", split)
    per_record = sc.per_record_metrics(tables, CFG, split, smpl_eval)
    index = sc.ValIndex(CFG, split)
    picks = sc.pick_records(per_record, split)
    showcase[split] = (tables, per_record, index, picks)
    print(split, "records with predictions per method:", {n: len(m) for n, m in per_record.items()})
    for p in picks:
        print("  pick:", p.key, "—", p.reason)
""")

code("""
for split, (tables, per_record, index, picks) in showcase.items():
    for p in picks:
        fig = sc.comparison_figure(index.row(p.key), tables, per_record, smpl_eval, title=f"{split}: {p.reason}")
        plt.show()
""")

code("""
# first pick of each set in 3D: every method's mesh inside the LiDAR returns
for split, (tables, per_record, index, picks) in showcase.items():
    if not picks:
        continue
    rec = index.row(picks[0].key)
    traces = [points_trace(rec["points"], "LiDAR returns", color="black", size=2)]
    if rec["has_smpl"]:
        v_gt, _ = smpl_eval.forward(rec["smpl"])
        traces.append(mesh_trace(v_gt, smpl_eval.faces, "GT SMPL", color="limegreen", opacity=0.35))
    for name, t in tables.items():
        if picks[0].key in t.index:
            traces.append(mesh_trace(t.vertices[t.index[picks[0].key]].astype(np.float64), smpl_eval.faces, name, color=sc.COLORS.get(name, "grey"), opacity=0.45))
    figure_3d(traces, f"{split}: {picks[0].reason}").show()
""")

md("""## 15. Knowledge transfer: cheap synthetic data on top of real data

Same architecture, same schedule; the only difference is the training
mixture. `real-only` sees Waymo + SLOPER4D train records only,
`synth-only` sees BEDLAM with simulated LiDAR only, `main-mixed` adds the
synthetic pool at 50 % of every batch. If the synthetic data transfers, the
mixed run must beat real-only on both real validation sets, and the curves
should show real-only overfitting its small pool.""")

code("""
fig = sc.transfer_figure(HELMA, {
    "real-only": "full-real-only-000",
    "synth-only": "full-synth-only-000",
    "main-mixed (real + synth)": "full-main-mixed-000",
    "mix80 (real + 80 % synth)": "full-mix80-000",
})
plt.show()
""")

md("""## 16. Placement error by distance and the learned gates

Left: Waymo placement error per ground-truth distance bin (image methods
degrade with distance, LiDAR-based methods do not). Right: the gates of the
trained model averaged over the validation records, compared with the
priors they started from (section 8): the decoder keeps 3D cues on the
LiDAR stream and semantic cues on the image stream, and learns per-layer
deviations from the prior.""")

code("""
res_ours = OURS / "results.json"
if not res_ours.exists() and ours_paths:
    subprocess.run(["uv", "run", "python", str(ROOT / "lidar_bedlam/scripts/score_baselines.py"), "--config", str(CFG),
                    "--pred", *[str(p) for p in ours_paths.values()], "--out", str(res_ours)], cwd=ROOT, capture_output=True)
fig = sc.distance_figure(ROOT / "outputs" / "baselines" / "results_static.json", res_ours if res_ours.exists() else None)
plt.show()
""")

code("""
for split, path in ours_paths.items():
    g = np.load(path)["gates"].astype(np.float64)  # (N, layers, groups)
    fig = sc.gates_figure(g, [gr.name for gr in JOINT_GROUPS], f"learned gates of full-main-mixed on {split} (mean over {len(g)} records)")
    plt.show()
""")

md("""## 17. SAM 3D Body as a compared pipeline: MHR mesh to SMPL

SAM 3D Body predicts the MHR body model, not SMPL. The MHR repository ships
an optimisation-based conversion (barycentric surface mapping, then a fit
of SMPL pose, shape and translation), which we run in its own environment
after inference. The check below overlays the MHR mesh and the converted
SMPL for 12 real crops (6 Waymo val, 6 SLOPER4D test; the inference used
the whole crop as the person box and our crop intrinsics). The two meshes
are 0.65 / 0.61 cm apart (mean nearest-vertex distance, both directions)
with identical extents, so the converted SMPL can be scored with the same
protocol as every other row. On the six SLOPER4D frames the converted SMPL
scores 32 mm MPJPE, 23 mm PA-MPJPE and 2.5 cm placement against the
labels (one sequence at 2.3 m, a smoke test, not a benchmark). The vertices
come from `outputs/baselines/sam3d-body/smoke/mhr_vs_smpl.npz`; pitfalls
and the two-stage scripts are recorded in progress.md (2026-09-15).""")

code("""
import numpy as np
from scipy.spatial import cKDTree

z = np.load(ROOT / "outputs" / "baselines" / "sam3d-body" / "smoke" / "mhr_vs_smpl.npz")
keys, mhr, smpl = [str(k) for k in z["keys"]], z["mhr_vertices"].astype(np.float64), z["smpl_vertices"]
d_ms = [cKDTree(s).query(m)[0].mean() for m, s in zip(mhr, smpl)]
d_sm = [cKDTree(m).query(s)[0].mean() for m, s in zip(mhr, smpl)]
print(f"{len(keys)} crops: MHR -> SMPL {np.mean(d_ms) * 100:.2f} cm, SMPL -> MHR {np.mean(d_sm) * 100:.2f} cm (mean nearest vertex)")
pick = [0, 1, len(keys) - 1]
fig = plt.figure(figsize=(12, 8))
for c, i in enumerate(pick):
    m, s = mhr[i], smpl[i]
    for r, (a, b, view) in enumerate(((0, 1, "front (x, -y)"), (2, 1, "side (z, -y)"))):
        ax = fig.add_subplot(2, 3, r * 3 + c + 1)
        ax.scatter(m[:, a], -m[:, b], s=0.4, c="0.55", label="MHR (SAM 3D Body)")
        ax.scatter(s[:, a], -s[:, b], s=0.4, c="#0E6B64", label="SMPL (converted)")
        ax.set_aspect("equal")
        ax.set_title(f"{keys[i].split('/')[0]} #{i}, {view}", fontsize=9)
        ax.tick_params(labelsize=7)
        if r == 0 and c == 0:
            ax.legend(markerscale=12, fontsize=8, loc="lower right")
fig.suptitle("SAM 3D Body MHR mesh vs SMPL fitted with the MHR conversion tool (camera frame, metres)")
fig.tight_layout()
plt.show()
""")

md("""## 18. In the wild: a whole Waymo sweep with our meshes

Waymo val frame 1557886649947221 (segment 15224741240438106736_960_000_980_000,
camera 2): six pedestrians with 3D keypoint labels are in the camera's
view. The full sweep of all five lasers (169,518 points, numpy port of the
official range-image conversion incl. the top laser's per-pixel pose; the
labelled keypoints lie 3-5 cm from the nearest return, which validates the
port) is transformed into the camera frame exactly as the training records
are. The six crops go through the point-anchored model (short schedule,
`abl-anchor-points-000/best.pt`); the predicted pelvises land 10-16 cm from
the labelled hip centres at 9-18 m. Left: camera image with the sweep
(colour = depth) and the meshes projected. Right: the sweep around the
crossing in 3D with the meshes on the road returns; the two unlabelled
pedestrians stay raw points. Scripts and the frame dump are in
`outputs/figures/waymo_header/`.""")

code("""
from IPython.display import Image, display
for name in ("header_image.png", "header_sweep.png"):
    display(Image(filename=str(ROOT / "outputs" / "figures" / "waymo_header" / name), width=1400))
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
