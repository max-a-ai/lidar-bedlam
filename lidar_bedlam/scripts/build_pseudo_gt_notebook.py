"""Generate ``notebooks/waymo_pseudo_gt.ipynb`` (pseudo-GT SMPL viewer).

Three rotatable 3D panels per page: the fitted SMPL mesh of a Waymo
training record inside its LiDAR returns, with the labelled 3D keypoints.
The buttons above the panels page through the records; the order is a
fixed permutation, so the first page is always the same three samples.

The notebook is generated from code so it stays in sync with the package;
edit this script, not the notebook. Execute it to verify::

    uv run python lidar_bedlam/scripts/build_pseudo_gt_notebook.py
    uv run python lidar_bedlam/scripts/verify_notebook.py \\
        --notebook notebooks/waymo_pseudo_gt.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("notebooks/waymo_pseudo_gt.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Add a markdown cell."""
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", text.strip()))


md("""
# Waymo pseudo ground truth — the fitted SMPL meshes in 3D

Waymo labels pedestrians with 3D keypoints and boxes, not with meshes.
`lidar_bedlam/scripts/pseudo_smpl_waymo.py` initialises SMPL from the
trained fusion model and optimises pose, shape and translation against
the 3D keypoints, the 2D keypoints and the LiDAR returns (pulled to 3 cm
outside the mesh). Records whose fitted keypoint error stays below
8 cm are accepted and carry `has_smpl=True` in the shards under
`resources/data/generated/real/v1_pseudo`.

This notebook shows nothing but that pseudo ground truth: each page is
three rotatable 3D views of an accepted record. The order is a fixed
permutation, so the first page is always the same three samples. Figures
are plain plotly outputs (they render in any viewer, also without a
kernel); set `INTERACTIVE = True` in the setup cell for paging buttons
instead, which needs a live ipywidgets front end.
""")

code("""
from pathlib import Path

import ipywidgets as W
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.utils.viz import mesh_trace, points_trace

ROOT = Path.cwd() if (Path.cwd() / "resources").exists() else Path.cwd().parent
DATA = ROOT / "resources" / "data" / "generated"
PSEUDO = DATA / "real" / "v1_pseudo"

smpl = SmplModel(DATA / "body_models")
shards = [Shard(p) for p in sorted(PSEUDO.glob("waymo_train_*.npz")) if ".fit." not in p.name]
fits = {p.name[: -len(".fit.npz")]: dict(np.load(p)) for p in sorted(PSEUDO.glob("*.fit.npz"))}
# only accepted fits are pseudo ground truth
samples = [(si, int(i)) for si, sh in enumerate(shards) for i in np.nonzero(sh.array("has_smpl"))[0]]
ORDER = np.random.default_rng(0).permutation(len(samples))  # fixed: page 1 never changes
PER_PAGE = 3
N_PAGES = (len(samples) + PER_PAGE - 1) // PER_PAGE
INTERACTIVE = False  # True: paging buttons (needs a live ipywidgets front end)
SHOW_PAGES = 2  # static mode: how many pages to draw per section
print(f"{len(shards)} shards, {sum(len(s) for s in shards)} records, {len(samples)} with pseudo-GT SMPL -> {N_PAGES} pages")
""")

md("""
## What one panel shows

- **light blue mesh** — the fitted SMPL body, i.e. the pseudo ground truth
- **black points** — the person's Waymo LiDAR returns (the fit pulls them
  3 cm outside the mesh, which is roughly the clothing offset)
- **green points** — the labelled Waymo 3D keypoints the fit was driven by

Every panel is centred on its own mesh and keeps metric scale (axes in
metres, `x` right, `z` forward, up is up in the camera frame of the
record). Drag to rotate, scroll to zoom, and use the shared legend to
switch the meshes, returns or keypoints off. The panel title gives the
record, its distance from the sensor and the keypoint error after the fit.
""")

code("""
def record(si: int, i: int) -> dict:
    \"\"\"Everything one panel draws, in the record's camera frame.\"\"\"
    sh = shards[si]
    params = SmplParams(
        sh.row("global_orient", i).astype(np.float64),
        sh.row("body_pose", i).astype(np.float64),
        sh.row("betas", i).astype(np.float64),
        sh.row("transl", i).astype(np.float64),
    )
    verts, _ = smpl.forward(params)
    scan = sh.scan(i, "real")
    valid = sh.array("joints3d_valid")[i][:15]
    fit = fits.get(sh.path.stem)
    return {
        "key": str(sh.array("key")[i]),
        "verts": verts,
        "points": scan.points.astype(np.float64),
        "keypoints": sh.array("joints3d")[i][:15][valid].astype(np.float64),
        "range_m": float(np.linalg.norm(verts.mean(0))),
        "error_mm": float(fit["error_m"][i]) * 1000 if fit is not None else float("nan"),
    }


def panel_title(rec: dict) -> str:
    name = rec["key"].split("/")[-2]
    return (f"{name}<br><sub>{rec['range_m']:.1f} m away, {len(rec['points'])} returns, "
            f"keypoint error {rec['error_mm']:.0f} mm</sub>")


def add_panel(fig, rec: dict, col: int) -> None:
    \"\"\"Draw one record into column ``col``, centred on its mesh.\"\"\"
    centre = rec["verts"].mean(0)
    first = col == 1  # one shared legend, driven by the first panel
    traces = [
        mesh_trace(rec["verts"] - centre, smpl.faces, "pseudo-GT SMPL mesh", color="lightblue", opacity=0.55),
        points_trace(rec["points"] - centre, "LiDAR returns", color="black", size=2.5),
        points_trace(rec["keypoints"] - centre, "Waymo 3D keypoints", color="limegreen", size=5),
    ]
    for t in traces:
        fig.add_trace(t.update(legendgroup=t.name, showlegend=first), row=1, col=col)


def page_figure(page: int):
    \"\"\"The three 3D views of one page.\"\"\"
    picks = ORDER[page * PER_PAGE : (page + 1) * PER_PAGE]
    recs = [record(*samples[int(g)]) for g in picks]
    fig = make_subplots(
        rows=1, cols=len(recs), specs=[[{"type": "scene"}] * len(recs)],
        subplot_titles=[panel_title(r) for r in recs], horizontal_spacing=0.02,
    )
    for col, rec in enumerate(recs, start=1):
        add_panel(fig, rec, col)
    scene = {
        "aspectmode": "data",
        "xaxis_title": "x right [m]", "yaxis_title": "z forward [m]", "zaxis_title": "up [m]",
        "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 0.8}},
    }
    fig.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"orientation": "h", "x": 0, "y": 0, "itemsizing": "constant"},
        **{f"scene{'' if c == 1 else c}": scene for c in range(1, len(recs) + 1)},
    )
    fig.update_annotations(font_size=11)
    return fig


def pager(figure_fn, n_pages: int, label_fn, unit: str = "3"):
    \"\"\"Paging buttons above one figure that is updated in place.

    The figure is a widget canvas, so a click redraws the same figure.
    Rendering into an ``Output`` widget instead appends a new figure per
    click in several frontends, which is how this notebook ended up
    showing four copies of the same three panels.
    \"\"\"
    state = {"page": 0}
    first = W.Button(description=f"back to the first {unit}", icon="undo")
    prev = W.Button(description=f"previous {unit}", icon="chevron-left")
    nxt = W.Button(description=f"next {unit}", icon="chevron-right", button_style="primary")
    status = W.HTML()
    bar = W.HBox([first, prev, nxt, status])
    try:
        canvas = go.FigureWidget(figure_fn(0))
        box = W.VBox([bar, canvas])
        out = None
    except ImportError:  # no anywidget: fall back to an Output widget
        canvas, out = None, W.Output()
        box = W.VBox([bar, out])

    def draw() -> None:
        page = state["page"]
        status.value = label_fn(page)
        first.disabled = prev.disabled = page == 0
        nxt.disabled = page >= n_pages - 1
        fig = figure_fn(page)
        if canvas is None:
            out.outputs = ()
            with out:
                display(fig)
            return
        with canvas.batch_update():
            for k, dst in enumerate(canvas.data):
                src = fig.data[k] if k < len(fig.data) else None
                dst.visible = src is not None  # short last page: hide the spare panel
                if src is None:
                    continue
                for attr in ("x", "y", "z", "i", "j", "k"):  # points, meshes, images
                    if hasattr(dst, attr):
                        setattr(dst, attr, getattr(src, attr))
            notes = fig.layout.annotations
            for k, note in enumerate(canvas.layout.annotations):
                note.text = notes[k].text if k < len(notes) else ""

    def step(delta: int):
        def handler(*_) -> None:
            state["page"] = 0 if delta == 0 else min(max(state["page"] + delta, 0), n_pages - 1)
            draw()
        return handler

    first.on_click(step(0))
    prev.on_click(step(-1))
    nxt.on_click(step(+1))
    display(box)
    draw()
    return state


def show(figure_fn, n_pages: int, label_fn, unit: str = "3", pages=None):
    \"\"\"Paging buttons when INTERACTIVE, else the first pages as plain figures.

    A plain figure is a plotly output that every viewer renders; the widget
    pager only works with a live kernel and a widget front end, which is
    why the executed notebook showed empty cells.
    \"\"\"
    if INTERACTIVE:
        return pager(figure_fn, n_pages, label_fn, unit)
    for page in (range(min(SHOW_PAGES, n_pages)) if pages is None else pages):
        display(HTML(label_fn(page)))  # plain HTML: a widget label is dropped by static viewers
        figure_fn(page).show()
    return None
""")

md("""
## The samples

The first `SHOW_PAGES` pages of the fixed order; change `SHOW_PAGES` or
pass `pages=[...]` to `show` for other samples. With `INTERACTIVE = True`
the buttons `next 3`, `previous 3` and `back to the first 3` page instead.
""")

code("""
if samples:
    show(page_figure, N_PAGES,
         lambda p: f"&nbsp;&nbsp;<b>samples {p * PER_PAGE + 1}&ndash;{min((p + 1) * PER_PAGE, len(samples))}"
                   f" of {len(samples)}</b> &nbsp;(page {p + 1} of {N_PAGES})")
else:
    print("no pseudo-GT shards yet: run lidar_bedlam/scripts/pseudo_smpl_waymo.py")
""")

md("""
## One person, all three pseudo ground truths

Every page below is a single Waymo record shown three times: our fit, the
pedestrian-generation project's, and LiDAR-HMR's, on the same LiDAR
returns and the same labelled keypoints, with the camera crop underneath.

None of the three label sets shares an identifier with the others, so each
is matched geometrically in the Waymo frames they have in common.
`match_ped_gen_records.py` pairs a record with the pedestrian instance
nearest to it in world coordinates (median 0.0 cm), and
`match_lidar_hmr_records.py` pairs it with the fit in the same frame whose
mesh centre is nearest on the ground plane (median 7.9 cm; all 4,586 of our
records match, since their `train.pkl` covers every frame we use). 1,489
records have all three.

The pedestrian-generation mesh arrives about 29 cm too low, because a
track is stored relative to the ego vehicle at the origin, so its height
is measured from the vehicle origin rather than from the road. It is shown
twice, faint grey as delivered and salmon after being lifted by the
vertical part of the translation that best fits its joints to the Waymo
keypoints. Nothing else about it is changed, and `VERTICAL_ONLY = False`
switches that correction to a full three-axis fit.

Each panel title carries the same two measures: the median distance from
the LiDAR returns to that mesh, and the mean distance from the labelled
keypoints to the joints regressed from it.
""")

code("""
import pickle

from scipy.spatial import cKDTree

from lidar_bedlam.data.schema import WAYMO15_TO_COCO17

MATCHED = DATA / "ped_gen" / "waymo_smpl_matched.npz"
HMR_MATCH = DATA / "lidar_hmr" / "waymo_match.npz"
EGO_TOL = 0.05  # m; a ped_gen track whose ego-frame check misses by more is skipped
VERTICAL_ONLY = True  # their frame is ego-relative: the height is the part to fix
raw = dict(np.load(MATCHED)) if MATCHED.exists() else {}
n_raw = len(raw.get("key", []))
keep = raw["ego_residual_m"] < EGO_TOL if n_raw else None
pg = {k: (v[keep] if getattr(v, "shape", ()) and v.shape[0] == n_raw else v) for k, v in raw.items()}
hmr = dict(np.load(HMR_MATCH)) if HMR_MATCH.exists() else {}
match = dict(np.load(DATA / "ped_gen" / "waymo_match.npz"))
UP = {str(k): np.asarray(mat, np.float64)[:3, :3] @ np.array([0.0, 0.0, 1.0])
      for k, mat in zip(match["key"], match["world_to_camera"])}  # world up per record
by_key = {str(k): (si, i) for si, sh in enumerate(shards) for i, k in enumerate(sh.array("key"))}
PG_AT = {str(k): i for i, k in enumerate(pg.get("key", []))}
HMR_AT = {str(k): i for i, k in enumerate(hmr.get("key", []))}
TRIPLES = sorted(set(PG_AT) & set(HMR_AT))
TRI_N = len(TRIPLES)
TRI_ORDER = np.random.default_rng(0).permutation(TRI_N)  # fixed: record 1 never changes
WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]  # the 13 Waymo joints SMPL shares
CSEL = WAYMO15_TO_COCO17[WSEL]


def joint_shift(verts: np.ndarray, kp15: np.ndarray, valid15: np.ndarray) -> np.ndarray:
    \"\"\"Translation putting the mesh's joints closest to the Waymo keypoints.\"\"\"
    ok = valid15[WSEL]
    if not ok.any():
        return np.zeros(3)
    return np.asarray((kp15[WSEL][ok] - (smpl.coco_regressor @ verts)[CSEL][ok]).mean(0), dtype=np.float64)


def joint_error(verts: np.ndarray, kp15: np.ndarray, valid15: np.ndarray) -> float:
    \"\"\"Mean distance from the mesh's joints to the Waymo keypoints.\"\"\"
    ok = valid15[WSEL]
    if not ok.any():
        return float("nan")
    return float(np.linalg.norm((smpl.coco_regressor @ verts)[CSEL][ok] - kp15[WSEL][ok], axis=1).mean())


def triple_record(n: int) -> dict:
    \"\"\"One record with all three pseudo ground truths in its camera frame.\"\"\"
    key = TRIPLES[n]
    si, j = by_key[key]
    sh = shards[si]
    kp15 = sh.array("joints3d")[j][:15].astype(np.float64)
    valid15 = sh.array("joints3d_valid")[j][:15]
    ours, _ = smpl.forward(SmplParams(sh.row("global_orient", j).astype(np.float64),
                                      sh.row("body_pose", j).astype(np.float64),
                                      sh.row("betas", j).astype(np.float64),
                                      sh.row("transl", j).astype(np.float64)))
    # the pedestrian-generation fit, rebuilt and lifted onto the keypoints
    i = PG_AT[key]
    rest, _ = smpl.forward(SmplParams(np.zeros(3), pg["body_pose"][i].astype(np.float64),
                                      pg["betas"][i].astype(np.float64), np.zeros(3)))
    delivered = (rest - rest.mean(0)) @ pg["orient"][i].astype(np.float64).T + pg["centroid"][i].astype(np.float64)
    full = joint_shift(delivered, kp15, valid15)
    lift = float(full @ UP[key])
    lifted = delivered + (lift * UP[key] if VERTICAL_ONLY else full)
    # LiDAR-HMR's fit, rebuilt from its parameters and put in our camera frame
    h = HMR_AT[key]
    hmr_mesh, _ = smpl.forward(SmplParams(hmr["global_orient"][h].astype(np.float64),
                                          hmr["body_pose"][h].astype(np.float64),
                                          hmr["betas"][h].astype(np.float64),
                                          hmr["transl"][h].astype(np.float64)))
    to_cam = np.asarray(hmr["vehicle_to_camera"][h], dtype=np.float64)
    hmr_mesh = hmr_mesh @ to_cam[:3, :3].T + to_cam[:3, 3]
    points = sh.scan(j, "real").points.astype(np.float64)
    p2m = lambda v: float(np.median(cKDTree(v).query(points)[0]))  # noqa: E731
    return {
        "key": key,
        "image": sh.array("image")[j],
        "points": points,
        "keypoints": kp15[valid15],
        "ours": ours, "delivered": delivered, "lifted": lifted, "hmr": hmr_mesh,
        "lift_cm": lift * 100,
        "hmr_match_cm": float(hmr["distance_m"][h]) * 100,
        "p2m": {"ours": p2m(ours), "ped_gen": p2m(lifted), "lidar_hmr": p2m(hmr_mesh)},
        "kp": {"ours": joint_error(ours, kp15, valid15),
               "ped_gen": joint_error(lifted, kp15, valid15),
               "lidar_hmr": joint_error(hmr_mesh, kp15, valid15)},
        "range_m": float(np.linalg.norm(ours.mean(0))),
    }


if TRI_N:
    look = [triple_record(int(g)) for g in TRI_ORDER[: min(150, TRI_N)]]
    print(f"{TRI_N} records have all three pseudo ground truths")
    print(f"medians over {len(look)} of them, in cm:")
    for name, label in (("ours", "ours"), ("ped_gen", "ped_gen, lifted"), ("lidar_hmr", "LiDAR-HMR")):
        print(f"  {label:16s} point-to-mesh {np.median([r['p2m'][name] for r in look]) * 100:5.1f}"
              f"   keypoints {np.median([r['kp'][name] for r in look]) * 100:5.1f}")
else:
    print("no three-way matches yet: run match_ped_gen_records.py, cache_ped_gen_smpl.py --match, match_lidar_hmr_records.py")
""")

code("""
SOURCES = (("ours", "our pseudo-GT SMPL", "lightblue"),
           ("ped_gen", "ped_gen SMPL, lifted", "lightsalmon"),
           ("lidar_hmr", "LiDAR-HMR SMPL", "plum"))


def triple_page(n: int):
    \"\"\"One record: the three fits side by side, its camera crop below.\"\"\"
    rec = triple_record(int(TRI_ORDER[n]))
    centre = rec["ours"].mean(0)
    titles = []
    for name, label, _ in SOURCES:
        lift = f", lifted {rec['lift_cm']:.0f} cm" if name == "ped_gen" else ""
        titles.append(f"{label.split(',')[0]}<br><sub>point-to-mesh {rec['p2m'][name] * 100:.1f} cm, "
                      f"keypoints {rec['kp'][name] * 100:.1f} cm{lift}</sub>")
    fig = make_subplots(
        rows=2, cols=3, row_heights=[0.66, 0.34], vertical_spacing=0.05,
        specs=[[{"type": "scene"}] * 3, [None, {"type": "xy"}, None]],
        subplot_titles=[*titles, "camera crop"], horizontal_spacing=0.02,
    )
    for col, (name, label, colour) in enumerate(SOURCES, start=1):
        traces = [mesh_trace(rec[{"ours": "ours", "ped_gen": "lifted", "lidar_hmr": "hmr"}[name]] - centre,
                             smpl.faces, label, color=colour, opacity=0.5)]
        if name == "ped_gen":
            traces.append(mesh_trace(rec["delivered"] - centre, smpl.faces,
                                     "ped_gen SMPL, as delivered", color="lightgrey", opacity=0.25))
        traces += [points_trace(rec["points"] - centre, "LiDAR returns", color="black", size=2.5),
                   points_trace(rec["keypoints"] - centre, "Waymo 3D keypoints", color="limegreen", size=5)]
        for t in traces:
            fig.add_trace(t.update(legendgroup=t.name, showlegend=col == 1 or t.name.endswith(("lifted", "HMR SMPL", "delivered"))),
                          row=1, col=col)
    fig.add_trace(go.Image(z=rec["image"], hoverinfo="skip"), row=2, col=2)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x", row=2, col=2)
    scene = {
        "aspectmode": "data",
        "xaxis_title": "x right [m]", "yaxis_title": "z forward [m]", "zaxis_title": "up [m]",
        "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 0.8}},
    }
    fig.update_layout(
        height=860, margin={"l": 0, "r": 0, "t": 90, "b": 0},
        title={"text": f"{rec['key']}<br><sub>{rec['range_m']:.1f} m away, {len(rec['points'])} returns, "
                       f"LiDAR-HMR matched to {rec['hmr_match_cm']:.0f} cm</sub>", "x": 0.02, "font": {"size": 13}},
        legend={"orientation": "h", "x": 0, "y": 0, "itemsizing": "constant"},
        **{f"scene{'' if c == 1 else c}": scene for c in range(1, 4)},
    )
    fig.update_annotations(font_size=11, selector={"yref": "paper"})
    return fig


TRI_LABEL = lambda p: f"&nbsp;&nbsp;<b>record {p + 1} of {TRI_N}</b> &nbsp;{TRIPLES[int(TRI_ORDER[p])]}"  # noqa: E731
if TRI_N:
    show(triple_page, TRI_N, TRI_LABEL, unit="record", pages=range(min(4, TRI_N)))
else:
    print("no three-way matches yet: see the cell above")
""")

md("""
Any other record of the fixed order: set `RECORDS` below and run the cell.
""")

code("""
RECORDS = [4, 5]  # positions in the fixed order (0-based); TRI_N records exist
if TRI_N:
    show(triple_page, TRI_N, TRI_LABEL, unit="record", pages=[r for r in RECORDS if r < TRI_N])
""")

md("""
## A third pseudo ground truth: the labels LiDAR-HMR trained on

LiDAR-HMR published its own SMPL fits to Waymo, the ones its Waymo
experiments use as ground truth. They live in
`resources/data/external/lidar_hmr_waymov2/` as `train.pkl` (2.5 GB) and
`test.pkl` (571 MB, 1,873 records, loaded here). A record carries the
fitted mesh, its SMPL parameters, the 14 keypoints the fit was driven by,
the person's LiDAR points, and a `location_dict` naming the Waymo segment,
the frame timestamp and the laser object id.

Everything in a record shares the Waymo vehicle frame, so this section
needs no matching: each panel is one of their records with its own returns
and its own keypoints, drawn exactly like the sections above. The panel
title gives the same two measures, the median distance from their returns
to their mesh and the mean distance from the labelled keypoints to the
joints regressed from it. `lidar_bedlam/scripts/compare_pseudo_gt.py`
reports those measures over the whole file, split by range.

Their keypoints are a 14-joint set with no nose, so the head is
unconstrained; the 12 limb joints are the ones compared.
""")

code("""
import pickle

from lidar_bedlam.data.schema import LIDARHMR14_TO_COCO17

HMR_DIR = DATA.parent / "external" / "lidar_hmr_waymov2"
HMR_SPLIT = "test"  # "train" works too: 2.5 GB, a few seconds and some GB of RAM
hmr_path = HMR_DIR / f"{HMR_SPLIT}.pkl"
hmr = pickle.load(hmr_path.open("rb")) if hmr_path.exists() else []
HMR_N = len(hmr)
HMR_ORDER = np.random.default_rng(0).permutation(HMR_N)  # fixed: page 1 never changes
HMR_PAGES = (HMR_N + PER_PAGE - 1) // PER_PAGE
HSEL = np.nonzero(LIDARHMR14_TO_COCO17 >= 0)[0]  # their 12 limb joints
HCSEL = LIDARHMR14_TO_COCO17[HSEL]
# Waymo vehicle frame (x forward, y left, z up) -> our camera frame (x right, y down, z forward)
VEHICLE_TO_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def lidar_hmr_record(i: int) -> dict:
    \"\"\"One published LiDAR-HMR record, in our camera convention.\"\"\"
    rec = hmr[i]
    verts = np.asarray(rec["smpl_verts"], dtype=np.float64)
    points = np.asarray(rec["human_points"][0], dtype=np.float64)
    kp = np.asarray(rec["keypoints"], dtype=np.float64)
    ok = np.asarray(rec["flag"], dtype=bool)[HSEL]
    joints = smpl.coco_joints(verts)[HCSEL][ok]
    where = rec["location_dict"]
    return {
        "segment": str(where["frame"]),
        "time": str(where["time"]),
        "verts": verts @ VEHICLE_TO_CAMERA.T,
        "points": points @ VEHICLE_TO_CAMERA.T,
        "keypoints": kp[HSEL][ok] @ VEHICLE_TO_CAMERA.T,
        "p2m": float(np.median(cKDTree(verts).query(points)[0])),
        "kp_err": float(np.linalg.norm(joints - kp[HSEL][ok], axis=1).mean()) if ok.any() else float("nan"),
        "range_m": float(np.linalg.norm(verts.mean(0))),
        "n_points": len(points),
    }


def lidar_hmr_page(page: int):
    \"\"\"Three of their records, their meshes on their own returns.\"\"\"
    recs = [lidar_hmr_record(int(g)) for g in HMR_ORDER[page * PER_PAGE : (page + 1) * PER_PAGE]]
    titles = [f"{r['segment'][:22]}  t{r['time'][-7:]}"
              f"<br><sub>{r['range_m']:.1f} m away, {r['n_points']} returns | point-to-mesh {r['p2m'] * 100:.1f} cm, "
              f"keypoints {r['kp_err'] * 100:.1f} cm</sub>" for r in recs]
    fig = make_subplots(rows=1, cols=len(recs), specs=[[{"type": "scene"}] * len(recs)],
                        subplot_titles=titles, horizontal_spacing=0.02)
    for col, rec in enumerate(recs, start=1):
        centre = rec["verts"].mean(0)
        traces = [
            mesh_trace(rec["verts"] - centre, smpl.faces, "LiDAR-HMR SMPL", color="plum", opacity=0.55),
            points_trace(rec["points"] - centre, "their LiDAR returns", color="black", size=2.5),
            points_trace(rec["keypoints"] - centre, "their 12 limb keypoints", color="limegreen", size=5),
        ]
        for t in traces:
            fig.add_trace(t.update(legendgroup=t.name, showlegend=col == 1), row=1, col=col)
    scene = {
        "aspectmode": "data",
        "xaxis_title": "x right [m]", "yaxis_title": "z forward [m]", "zaxis_title": "up [m]",
        "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 0.8}},
    }
    fig.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"orientation": "h", "x": 0, "y": 0, "itemsizing": "constant"},
        **{f"scene{'' if c == 1 else c}": scene for c in range(1, len(recs) + 1)},
    )
    fig.update_annotations(font_size=11)
    return fig


if HMR_N:
    look = [lidar_hmr_record(int(g)) for g in HMR_ORDER[: min(200, HMR_N)]]
    print(f"{hmr_path.name}: {HMR_N} records -> {HMR_PAGES} pages")
    print(f"medians over {len(look)} of them: point-to-mesh {np.median([r['p2m'] for r in look]) * 100:.1f} cm, "
          f"keypoints {np.median([r['kp_err'] for r in look]) * 100:.1f} cm, "
          f"{np.median([r['n_points'] for r in look]):.0f} returns at {np.median([r['range_m'] for r in look]):.1f} m")
    show(lidar_hmr_page, HMR_PAGES,
         lambda p: f"&nbsp;&nbsp;<b>LiDAR-HMR records {p * PER_PAGE + 1}&ndash;{min((p + 1) * PER_PAGE, HMR_N)}"
                   f" of {HMR_N}</b> &nbsp;(page {p + 1} of {HMR_PAGES})")
else:
    print(f"no LiDAR-HMR pickles under {HMR_DIR}")
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
