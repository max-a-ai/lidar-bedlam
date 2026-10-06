"""Generate ``notebooks/bedlam_v1_vs_v2.ipynb`` (BEDLAM 1.0 against 2.0).

What is on disk of each release, the camera statistics (height, pitch,
field of view, motion) against the rigs this project targets, example
frames side by side, the body and asset inventories, whether an evaluation
on BEDLAM 2.0 is possible with what was downloaded, and what LiDAR
simulation each release allows. The last section is the recommendation
per target domain.

Everything is read from the NAS copies and the local BEDLAM 1.0 extraction;
the small BEDLAM 2.0 ground-truth archives are unpacked into a cache under
``outputs/bedlam2_cache`` on first run (about 1 GB, a few minutes).

The notebook is generated from code so it stays in sync with the package;
edit this script, not the notebook::

    uv run python lidar_bedlam/scripts/build_bedlam_compare_notebook.py
    uv run python lidar_bedlam/scripts/verify_notebook.py \\
        --notebook notebooks/bedlam_v1_vs_v2.ipynb --timeout 1800
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("notebooks/bedlam_v1_vs_v2.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Add a markdown cell."""
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", text.strip()))


md("""
# BEDLAM 1.0 against BEDLAM 2.0 for this project

We train on synthetic LiDAR simulated from BEDLAM 1.0 (clothed depth maps,
SMPL labels, 391k person records). BEDLAM 2.0 (arXiv 2511.14394,
November 2025) is on the NAS as eight scene groups, 923 GB. This notebook
answers, with the data on disk: what 2.0 adds, whether its camera
viewpoints cover rigs that 1.0 does not (head-mounted, pole, drone),
whether we can evaluate on it, how we could simulate LiDAR from it, and
where 1.0 is already sufficient.

## What BEDLAM 2.0 improves over 1.0 (paper and measured here)

* **Moving, realistic cameras**: dolly, tracking, follow, approach, ego view, with shake; 1.0 is mostly static.
* **Wide lenses**: median horizontal FOV 63 to 106 degrees against 37 to 65 in 1.0.
* **Richer bodies**: shoes, strand hair, more clothing and body-shape diversity, more environments.
* **World-coordinate ground truth** emphasised; made for methods that estimate humans in world frame.
* **Scale**: 27,480 sequences, 8 M frames at 1280 x 720, 30 fps; 1.0 has 380k frames at 6 fps labels.
""")

code("""
from __future__ import annotations

import csv
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

ROOT = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
V1_NAS = Path("/home/max/nas_drive/publicdatasets/bedlam")
V1_LOCAL = ROOT / "resources/data/generated/bedlam_raw"
V1_LABELS = ROOT / "resources/data/generated/bedlam_labels/smpl/bedlam-labels"
V2_NAS = Path("/home/max/nas_drive/publicdatasets/bedlam2")
CACHE = ROOT / "outputs/bedlam2_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def table(rows, header):
    h = "".join(f"<th style='text-align:left;padding:2px 10px'>{c}</th>" for c in header)
    b = "".join(
        "<tr>" + "".join(f"<td style='padding:2px 10px'>{c}</td>" for c in r) + "</tr>"
        for r in rows
    )
    display(HTML(f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"))


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


# BEDLAM 2.0 scenes: unpack the small ground-truth archives (camera CSVs,
# placement files, animation configs) and the overview contact sheets
v2_scenes = sorted(
    p.name for p in (V2_NAS / "images").iterdir()
    if p.is_dir() and p.name[0].isdigit() and (p / "ground_truth").is_dir()
)
for name in v2_scenes:
    if not (CACHE / name / "be_seq.csv").exists():
        gt = next((V2_NAS / "images" / name / "ground_truth").glob("*_exr_meta_csv.tar.gz"))
        with tarfile.open(gt) as tf:
            tf.extractall(CACHE, filter="data")
    if not (CACHE / name / "overview").exists():
        ov = next((V2_NAS / "images" / name / "overview").glob("*_overview.tar"))
        with tarfile.open(ov) as tf:
            tf.extractall(CACHE, filter="data")
v1_local = sorted(p.name for p in V1_LOCAL.iterdir() if p.is_dir())
v1_nas_groups = sorted({
    re.sub(r"_(depth|gt|masks|mp4|png)\\..*$", "", p.name)
    for p in V1_NAS.iterdir() if p.suffix in (".tar", ".gz") and "_" in p.name
})
print(f"BEDLAM 1.0: {len(v1_nas_groups)} sequence groups on the NAS, {len(v1_local)} extracted locally")
print(f"BEDLAM 2.0: {len(v2_scenes)} scene groups on the NAS: " + ", ".join(v2_scenes))
""")

md("""
## 1. What is on disk

BEDLAM 1.0 is complete: PNG, 16-bit depth of the clothed characters,
per-part masks, camera CSVs and the processed SMPL label tables. The
BEDLAM 2.0 download holds the PNG archives, the camera ground truth and the
overview sheets. Its `depth/` folders are empty (the depth release sits in
a gated HuggingFace repo we have no access to), there are **no masks**, and
the per-frame `meta.json` files are Unreal camera metadata, **not body
parameters**. The body ground truth of 2.0 (SMPL-X animations composed with
the placement files) is a separate download that was not made.
""")

code("""
rows = []
for name in v2_scenes:
    base = V2_NAS / "images" / name
    png = sum(p.stat().st_size for p in (base / "png").glob("*.tar"))
    gt = sum(p.stat().st_size for p in (base / "ground_truth").glob("*.tar.gz"))
    depth = sum(p.stat().st_size for p in (V2_NAS / "depth" / name).rglob("*") if p.is_file())
    seqs = len(list((CACHE / name / "ground_truth/meta_exr_csv").glob("*.csv")))
    frames = sum(1 for f in (CACHE / name / "ground_truth/meta_exr_csv").glob("*.csv") for _ in open(f)) - seqs
    rows.append([name, seqs, f"{frames:,}", human(png), human(gt), "none" if depth == 0 else human(depth), "none", "camera only"])
table(rows, ["BEDLAM 2.0 scene", "sequences", "frames", "PNG", "ground truth", "depth", "masks", "body GT"])

rows = []
for name in v1_local:
    base = V1_LOCAL / name
    seqs = len(list((base / "png").glob("seq_*")))
    frames = sum(1 for _ in (base / "depth").rglob("*_depth.exr"))
    masks = sum(1 for _ in (base / "masks").rglob("*_body.png"))
    lab = sorted(V1_LABELS.glob(f"{name}_*fps.npz"))
    rows.append([name, seqs, f"{frames:,}", "yes", "clothed, EXR", f"{masks:,} person masks", "SMPL table" if lab else "missing"])
table(rows, ["BEDLAM 1.0 group (local)", "sequences", "frames", "PNG", "depth", "masks", "body GT"])
print("BEDLAM 1.0 groups on the NAS but not extracted locally:", len(set(v1_nas_groups) - set(v1_local)))
""")

md("""
## 2. Cameras: height, pitch, field of view

The question behind this section: does 2.0 give us viewpoints our target
rigs have and 1.0 lacks? The target rigs are drawn into the scatter as
markers: Waymo and nuScenes (car roof, 1.7 to 2.1 m, level), SLOPER4D
(head-mounted, 1.6 m, tilted 12 degrees down), TUMTraf S110 (pole, about
7 m, 30 to 50 degrees down) and a drone band (10 to 30 m, 45 to 90 degrees
down). Heights are camera z in metres above the ground plane of the scene;
pitch is negative when looking down.
""")

code("""
def camera_rows(csv_files):
    z, pitch, hfov, speed = [], [], [], []
    for f in csv_files:
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]) / 100.0
        z.append(xyz[:, 2]); pitch.append([float(r["pitch"]) for r in rows]); hfov.append([float(r["hfov"]) for r in rows])
        if len(xyz) > 1:
            speed.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 30.0)  # m/s at 30 fps
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in
            (("z", z), ("pitch", pitch), ("hfov", hfov), ("speed", speed))}

CAMS = {}
for name in v1_local:
    CAMS[("1.0", name)] = camera_rows(sorted((V1_LOCAL / name / "ground_truth/camera").glob("*.csv")))
for name in v2_scenes:
    CAMS[("2.0", name)] = camera_rows(sorted((CACHE / name / "ground_truth/meta_exr_csv").glob("*.csv")))

rows = []
for (rel, name), c in CAMS.items():
    rows.append([rel, name, f"{len(c['z']):,}", f"{np.median(c['z']):.1f} ({np.percentile(c['z'],10):.1f}..{np.percentile(c['z'],90):.1f})",
                 f"{np.median(c['pitch']):.0f} ({np.percentile(c['pitch'],10):.0f}..{np.percentile(c['pitch'],90):.0f})",
                 f"{np.median(c['hfov']):.0f}", f"{np.median(c['speed']):.2f}" if len(c['speed']) else "-"])
table(rows, ["release", "group", "frames", "height m (p10..p90)", "pitch deg (p10..p90)", "HFOV deg", "camera speed m/s (median)"])
""")

code("""
TARGETS = {
    "Waymo / nuScenes (car roof)": (1.9, -1.0, "tab:blue"),
    "SLOPER4D (head-mounted)": (1.6, -12.0, "tab:green"),
    "TUMTraf S110 (pole)": (7.0, -40.0, "tab:orange"),
    "drone band": (20.0, -70.0, "tab:red"),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for rel, color in (("1.0", "0.35"), ("2.0", "crimson")):
    z = np.concatenate([c["z"] for (r, _), c in CAMS.items() if r == rel])
    p = np.concatenate([c["pitch"] for (r, _), c in CAMS.items() if r == rel])
    h = np.concatenate([c["hfov"] for (r, _), c in CAMS.items() if r == rel])
    axes[0].hist(z, bins=np.linspace(0, 8, 41), histtype="step", lw=2, color=color, label=f"BEDLAM {rel}", density=True)
    axes[1].hist(p, bins=np.linspace(-60, 20, 41), histtype="step", lw=2, color=color, label=f"BEDLAM {rel}", density=True)
    axes[2].hist(h, bins=np.linspace(20, 130, 45), histtype="step", lw=2, color=color, label=f"BEDLAM {rel}", density=True)
for name, (hz, pt, col) in TARGETS.items():
    axes[0].axvline(hz, color=col, ls="--", lw=1, label=name)
    axes[1].axvline(pt, color=col, ls="--", lw=1)
axes[0].set_xlabel("camera height above ground (m)"); axes[1].set_xlabel("pitch (deg, negative = looking down)"); axes[2].set_xlabel("horizontal FOV (deg)")
for ax in axes: ax.set_yticks([]); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="upper right")
axes[2].legend(fontsize=8)
fig.suptitle("Camera distributions, frame-weighted; dashed lines = target rigs", fontsize=10)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8.5, 5.5))
rng = np.random.default_rng(0)
for (rel, name), c in CAMS.items():
    n = min(600, len(c["z"])); idx = rng.choice(len(c["z"]), n, replace=False)
    ax.scatter(c["pitch"][idx], c["z"][idx], s=4, alpha=0.35, color="0.4" if rel == "1.0" else "crimson")
for name, (hz, pt, col) in TARGETS.items():
    ax.scatter([pt], [hz], marker="*", s=260, color=col, edgecolor="k", zorder=5, label=name)
ax.axhspan(10, 30, xmin=0.0, xmax=0.45, color="tab:red", alpha=0.06)
ax.set_xlabel("pitch (deg)"); ax.set_ylabel("camera height (m)"); ax.set_yscale("log"); ax.set_ylim(0.4, 40)
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("grey: BEDLAM 1.0 frames, red: BEDLAM 2.0 frames, stars: target rigs", fontsize=10)
plt.show()
""")

md("""
### Reading the camera plots

* Both releases live between 1 and 6 m of camera height. 2.0's cameras are
  configured for 0.5 to 2.5 m above ground; only the `dollyz_zoom` scene
  rises to 5.6 m while pitching 35 degrees down.
* 1.0's `closeup` and `stadium` groups already give 5 to 6 m at 15 to 37
  degrees down, which is the closest either release gets to a pole camera.
* **Neither release contains drone views.** No frame sits above 6 m, none
  pitches below 40 degrees. The "hovering, coming down from the top"
  impression comes from 2.0's wide lenses and camera shake, not from
  altitude. For a drone domain the options are rendering our own cameras
  with the released 2.0 assets, or simulating LiDAR only (view-free) from
  body meshes and accepting that the image branch sees no drone imagery.
* 2.0's field of view (63 to 106 degrees) is far wider than any target rig
  here (Waymo 50, SLOPER4D 75, TUMTraf 75 degrees). Wide-angle crops look
  different after our square crop and would need matching intrinsics
  augmentation; 1.0's 37 to 65 degrees already brackets the real cameras.
* Where 2.0 is new: **camera motion**. Median camera speed is 0.5 to 2 m/s
  (follow, approach, ego) against a static 1.0. For the head-mounted
  SLOPER4D rig and for any vehicle-ego setting that is the one property 1.0
  cannot offer, though our model is single-frame and sees motion only as
  blur.
""")

code("""
rows = []
for name in v2_scenes:
    anim = json.load(open(CACHE / name / "be_camera_animations.json"))
    info = anim.get("info", {}); cfg = info.get("config", {})
    rows.append(["2.0", name, info.get("config_type", "?"), "yes" if cfg.get("follow") else "no",
                 cfg.get("camera_shake", "none") or "none",
                 f"{cfg.get('camera_height_min', '?')}..{cfg.get('camera_height_max', '?')} cm",
                 ", ".join(f"{a:.0f}-{b:.0f}" for a, b in cfg.get("hfov_ranges", []))])
for name in v1_local:
    comment = next(csv.DictReader(open(V1_LOCAL / name / "be_seq.csv")))["Comment"]
    kv = dict(p.split("=", 1) for p in comment.split(";") if "=" in p)
    rows.append(["1.0", name, kv.get("cam_config", "?"), "no", "none",
                 f"pitch {kv.get('cam_pitch_min')}..{kv.get('cam_pitch_max')} deg, offsets {kv.get('cam_x_offset')}/{kv.get('cam_y_offset')}/{kv.get('cam_z_offset')}", "fixed per group"])
table(rows, ["release", "group", "camera config", "follows a body", "shake", "height / pitch range", "HFOV range"])
""")

md("""
## 3. Example frames

Left blocks: BEDLAM 1.0 frames from the local extraction, right blocks:
BEDLAM 2.0 frames pulled from the PNG archives where the cache has them,
otherwise the official overview sheet of the scene (30 frames of one
sequence). Titles carry the camera height, pitch and FOV of that frame.
""")

code("""
from PIL import Image

def cam_of(csv_path, frame_png):
    for r in csv.DictReader(open(csv_path)):
        if r["name"] == frame_png:
            return float(r["z"]) / 100, float(r["pitch"]), float(r["hfov"])
    return None

V1_SHOW = [
    ("20221010_3_1000_batch01hand", "seq_000000", "0000"),
    ("20221018_3-8_250_batch01hand_pitchDown52_stadium", "seq_000000", "0000"),
    ("20221011_1_250_batch01hand_closeup_suburb_c", "seq_000000", "0005"),
    ("20221019_3-8_250_highbmihand_orbit_stadium", "seq_000000", "0000"),
]
fig, axes = plt.subplots(1, len(V1_SHOW), figsize=(4.4 * len(V1_SHOW), 2.9))
for ax, (g, s, f) in zip(axes, V1_SHOW):
    p = V1_LOCAL / g / "png" / s / f"{s}_{f}.png"
    if not p.exists():
        p = next((V1_LOCAL / g / "png" / s).glob("*.png"))
    ax.imshow(Image.open(p)); ax.axis("off")
    c = cam_of(V1_LOCAL / g / "ground_truth/camera" / f"{s}_camera.csv", p.name)
    ax.set_title(f"1.0 {g.split('_', 3)[-1]}\\n" + (f"h {c[0]:.1f} m, pitch {c[1]:.0f}, hfov {c[2]:.0f}" if c else ""), fontsize=8)
plt.tight_layout(); plt.show()

v2_pngs = sorted(CACHE.rglob("png/seq_*/*.png"))
if v2_pngs:
    by_scene = {}
    for p in v2_pngs:
        by_scene.setdefault(p.parts[-4], []).append(p)
    picks = [ps[len(ps) // 2] for ps in by_scene.values()][:4]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 2.9))
    for ax, p in zip(np.atleast_1d(axes), picks):
        scene = p.parts[-4]; seq = p.parts[-2]
        ax.imshow(Image.open(p)); ax.axis("off")
        c = cam_of(CACHE / scene / "ground_truth/meta_exr_csv" / f"{seq}_camera.csv", p.name)
        ax.set_title(f"2.0 {scene.split('_', 3)[-1]}\\n" + (f"h {c[0]:.1f} m, pitch {c[1]:.0f}, hfov {c[2]:.0f}" if c else ""), fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("no 2.0 frames in the cache: showing the overview sheets instead (extract a few with "
          "tar --wildcards from <scene>/png/*_png.0.tar into outputs/bedlam2_cache to see full frames)")

for name in v2_scenes:
    sheets = sorted((CACHE / name / "overview/images").glob("*.png"))
    if not sheets:
        continue
    im = Image.open(sheets[len(sheets) // 2])
    fig, ax = plt.subplots(figsize=(14, 14 * im.size[1] / im.size[0] * 0.5))
    ax.imshow(im.crop((0, 0, im.size[0], im.size[1] // 2))); ax.axis("off")
    ax.set_title(f"2.0 {name}: official overview sheet (one sequence, first half)", fontsize=9)
    plt.show()
""")

md("""
## 4. Bodies, assets and labels

Counts from the placement files (`be_seq.csv`): bodies per sequence, how
many distinct subjects, clothing textures, hair grooms and shoes a scene
draws from. 1.0 numbers come from the local groups, 2.0 from all eight
scenes. The second table is what the 1.0 SMPL label tables give us per
record, the quantity 2.0 does not have on disk.
""")

code("""
def placement(f):
    rows = list(csv.DictReader(open(f)))
    per = Counter(); g = None; subj, cloth, hair, shoe = set(), set(), set(), set()
    for r in rows:
        if r["Type"] == "Group":
            g = r["Comment"].split(";")[0]
        elif r["Type"] == "Body":
            per[g] += 1; subj.add(r["Body"])
            kv = dict(p.split("=", 1) for p in r["Comment"].split(";") if "=" in p)
            cloth.add(kv.get("texture_clothing")); hair.add(kv.get("hair")); shoe.add(kv.get("shoe", ""))
    shoe.discard("")
    return len(per), sum(per.values()), np.median(list(per.values())) if per else 0, len(subj), len(cloth), len(hair), len(shoe)

rows = []
for name in v2_scenes:
    n, b, med, s, c, h, sh = placement(CACHE / name / "be_seq.csv")
    rows.append(["2.0", name, n, b, f"{med:.0f}", s, c, h, sh])
for name in v1_local:
    n, b, med, s, c, h, sh = placement(V1_LOCAL / name / "be_seq.csv")
    rows.append(["1.0", name, n, b, f"{med:.0f}", s, c, h, sh if sh else "none (barefoot)"])
table(rows, ["release", "group", "sequences", "bodies", "bodies / seq", "subjects", "clothing textures", "hair grooms", "shoes"])

rows = []
for name in v1_local:
    lab = sorted(V1_LABELS.glob(f"{name}_*fps.npz"))
    if not lab:
        continue
    d = np.load(lab[0], allow_pickle=True)
    depth = d["trans_cam"][:, 2] + d["cam_ext"][:, 2, 3]
    rows.append([name, f"{len(d['imgname']):,}", f"{np.median(depth):.1f} ({np.percentile(depth, 10):.1f}..{np.percentile(depth, 90):.1f})",
                 f"{np.median(200 * d['scale']):.0f}", f"{np.std(d['shape'][:, 0]):.2f}"])
table(rows, ["BEDLAM 1.0 group", "label rows (6 fps)", "person depth m (p10..p90)", "bbox side px (median)", "beta_0 std"])
print("BEDLAM 2.0: no body parameters on disk; bodies per sequence and asset variety are the only body facts available.")
""")

md("""
## 5. Can we evaluate on BEDLAM 2.0?

Not with the current download. An evaluation needs per-frame SMPL
parameters in the camera frame (for MPJPE, PVE and placement) and either
depth or masks to build the person point cloud and the crop. 2.0 on disk
has neither body parameters, depth nor masks. What it would take:

1. Download the 2.0 body ground truth (SMPL-X animations plus the
   composition tool, or the processed per-frame tables if the release has
   them by now) from the BEDLAM 2.0 site; the camera CSVs are already here.
2. Convert SMPL-X to SMPL: pose joints 1 to 21 share the kinematic tree,
   the hands collapse to SMPL's two hand joints, shape needs a fit (the
   `smplx` transfer tool or a joint-based least squares, 1 to 2 cm error).
3. Person points: no depth, so simulate LiDAR on the SMPL mesh exactly as
   for 3DPW (`generate/mesh_synth.py`), which yields clothing-free points;
   crops from the projected mesh box since there are no masks.
4. Then `score_baselines`-style scoring on a held-out scene.

Until step 1 is done, 2.0 is image-only material for us. SLOPER4D and 3DPW
stay the evaluation sets; 2.0's value would be a wide-FOV, moving-camera
test of the image branch, not a LiDAR benchmark.
""")

md("""
## 6. LiDAR simulation: clothed depth (1.0) against body mesh (2.0)

Our 1.0 pipeline casts LiDAR rays against the rendered depth of the clothed
character, so the returns carry clothing offsets, hair and the exact
silhouette; Waymo returns measured 2.5 cm median point-to-mesh distance,
1.0 synth 1.8 cm. Without 2.0 depth the only route is the 3DPW mesh path:
returns on the bare SMPL surface plus a random 1 to 4 cm offset. The cell
below shows both on one 1.0 frame so the difference is visible: the clothed
returns (what we train on) against mesh-simulated returns on the same body.
""")

code("""
from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.bedlam import BedlamFramesSource
from lidar_bedlam.lidar.mesh_depth import offset_mesh, render_depth
from lidar_bedlam.lidar.simulate import ouster, simulate
from lidar_bedlam.lidar.placement import BallPlacement

smpl = SmplModel(ROOT / "resources/data/generated/body_models")
group = "20221018_3-8_250_batch01hand_pitchDown52_stadium"
src = BedlamFramesSource(V1_LOCAL, groups=[group], frame_stride=50, labels_dir=V1_LABELS, smpl_model=smpl)
sample = next(src.load(i) for i in range(len(src)) if src.load(i).smpl is not None)
cam = sample.camera
rng = np.random.default_rng(0)
spec = ouster("OS1", 128, 1024)
pose = BallPlacement().sample(rng)
# (a) the 1.0 route: rays against the clothed depth map, person mask selects the body
depth_cm = __import__("lidar_bedlam.utils.io", fromlist=["read_exr_depth"]).read_exr_depth(
    V1_LOCAL / group / "depth" / sample.meta.sequence.split("/")[1] / f"{sample.meta.sequence.split('/')[1]}_{sample.meta.frame}_depth.exr")
depth_m = np.where(depth_cm < 1e7, depth_cm * 0.01, 1e4).astype(np.float64)
scan_a = simulate(depth_m, cam, spec, pose, rng)
uv = scan_a.pixel
keep = sample.mask[np.clip(uv[:, 1], 0, cam.height - 1), np.clip(uv[:, 0], 0, cam.width - 1)]
pts_a = scan_a.points[keep]
# (b) the mesh route a 2.0 pipeline would have to use: rays against the SMPL body only
verts, _ = smpl.forward(sample.smpl)
mesh = offset_mesh(verts, smpl.faces, 0.02)
depth_b, _ = render_depth([(mesh, smpl.faces)], cam)
scan_b = simulate(depth_b.astype(np.float64), cam, spec, pose, np.random.default_rng(0))
pts_b = scan_b.points

from scipy.spatial import cKDTree
d_a = cKDTree(verts).query(pts_a)[0] * 100; d_b = cKDTree(verts).query(pts_b)[0] * 100
fig = plt.figure(figsize=(15, 4.6))
ax = fig.add_subplot(1, 3, 1); ax.imshow(sample.image); ax.axis("off")
x0, y0, x1, y1 = sample.bbox_xyxy; ax.set_xlim(x0 - 40, x1 + 40); ax.set_ylim(y1 + 40, y0 - 40)
ax.set_title(f"1.0 frame, person {sample.meta.person}", fontsize=9)
for k, (pts, dd, title) in enumerate(((pts_a, d_a, "clothed depth (our 1.0 route)"), (pts_b, d_b, "SMPL mesh + 2 cm (the 2.0 route)"))):
    ax = fig.add_subplot(1, 3, 2 + k, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=3, c="0.3")
    ax.scatter(verts[::6, 0], verts[::6, 2], -verts[::6, 1], s=0.4, c="red", alpha=0.4)
    c = verts.mean(0); r = 1.1
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(-c[1] - r, -c[1] + r)
    ax.view_init(elev=12, azim=-60); ax.set_title(f"{title}\\n{len(pts)} returns, point-to-body median {np.median(dd):.1f} cm, p90 {np.percentile(dd, 90):.1f} cm", fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
## 7. Verdict: 1.0, 2.0, or a mixture

| target domain | what the rig looks like | BEDLAM 1.0 | BEDLAM 2.0 | recommendation |
|---|---|---|---|---|
| car roof (Waymo, nuScenes) | 1.7 to 2.1 m, level, 50 degree lens, static relative to the person | 1.0's default groups match height, pitch and FOV | wide lenses and moving cameras do not match a car rig | **1.0 is sufficient**; our Waymo results are limited by labels and the box loss, not by the synthetic imagery |
| head-mounted (SLOPER4D) | 1.6 m, 12 degrees down, 75 degree lens, walking camera | height and pitch covered; no camera motion | `follow`, `vcam`, `approach` scenes move like a walking person, with shake | **mixture**: 2.0 motion scenes would add realistic blur and framing; worth a small image-only ablation once body GT exists |
| infrastructure pole (TUMTraf) | 7 m, 30 to 50 degrees down | `closeup` and `stadium` groups reach 5.6 m at 15 to 37 degrees | `dollyz_zoom` reaches 5.6 m at 35 degrees | both fall short of 7 m and 45 degrees; **neither helps more**; render custom cameras if the pole domain matters |
| drone | 10 to 30 m, 45 to 90 degrees down | nothing above 6 m | nothing above 6 m | **neither**; needs own rendering with the released 2.0 assets, or LiDAR-only training from meshes |

What 2.0 would cost us: a separate body-GT download and an SMPL-X to SMPL
conversion, LiDAR simulated on bare meshes instead of clothed depth (our
measured clothing offsets would be lost), crops without masks, and 8 to 15
GB per PNG archive to extract. What it would give: moving wide-angle
cameras, shoes, strand hair and more varied bodies in the image branch.

For the current paper the synthetic data is not the bottleneck: the Waymo
gains of the last two weeks came from the box loss and the pseudo-GT
refit, and SLOPER4D and 3DPW sit at 44 to 48 mm with 1.0 alone. Keep 1.0
as the training source. Revisit 2.0 as an image-branch mixture (not a
replacement) once its body ground truth is downloaded, and only for the
head-mounted and ego-motion domains where its cameras add something.
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
