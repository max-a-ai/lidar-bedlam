"""Cell bodies of ``001-waymo-pseudo-gt.ipynb``: one function per cell.

Waymo labels pedestrians with keypoints and boxes, not meshes. The fitter
in ``lidar_bedlam/generate/pseudo_fit_v2.py`` produces SMPL labels for
them; this notebook shows those labels in 3D, compares every label set we
have for the same record, and looks at the labels LiDAR-HMR trained on.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.schema import LIDARHMR14_TO_COCO17, WAYMO15_TO_COCO17
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.utils.viz import mesh_trace, points_trace

ROOT = Path.cwd() if (Path.cwd() / "resources").exists() else Path.cwd().parent
DATA = ROOT / "resources" / "data" / "generated"
WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]  # the 13 Waymo joints SMPL has
CSEL = WAYMO15_TO_COCO17[WSEL]
SCENE = {
    "aspectmode": "data",
    "xaxis_title": "x right [m]",
    "yaxis_title": "z forward [m]",
    "zaxis_title": "up [m]",
    "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 0.8}},
}
LEGEND = {"orientation": "h", "x": 0, "y": 0, "itemsizing": "constant"}
Array = NDArray[Any]


@dataclass
class Labels:
    """One Waymo label set: its shards and the accepted (shard, row) pairs."""

    name: str
    shards: list[Shard]
    accepted: list[tuple[int, int]]
    smpl: SmplModel
    order: Array


def _shards(folder: Path) -> list[Shard]:
    return [
        Shard(p)
        for p in sorted(folder.glob("waymo_train_*.npz"))
        if ".fit." not in p.name
    ]


def _params(sh: Shard, i: int) -> SmplParams:
    return SmplParams(
        *(
            sh.row(k, i).astype(np.float64)
            for k in ("global_orient", "body_pose", "betas", "transl")
        )
    )


def _scenes(fig: go.Figure, n: int, height: int, title: str = "") -> None:
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 90 if title else 70, "b": 0},
        legend=LEGEND,
        title={"text": title, "x": 0.02, "font": {"size": 13}},
        **{f"scene{'' if c == 1 else c}": SCENE for c in range(1, n + 1)},
    )
    fig.update_annotations(font_size=11)


def load_labels(label_set: str = "real/v1_pseudo3") -> Labels:
    """Open the shards of one label set and count the accepted fits."""
    smpl = SmplModel(DATA / "body_models")
    shards = _shards(DATA / label_set)
    accepted = [
        (si, int(i))
        for si, sh in enumerate(shards)
        for i in np.nonzero(sh.array("has_smpl"))[0]
    ]
    n = sum(len(s) for s in shards)
    print(f"{label_set}: {n} records, {len(accepted)} with a pseudo-GT mesh")
    order = np.random.default_rng(0).permutation(len(accepted))
    return Labels(label_set, shards, accepted, smpl, order)


def _record(lab: Labels, si: int, i: int) -> dict[str, Any]:
    sh = lab.shards[si]
    verts, _ = lab.smpl.forward(_params(sh, i))
    valid = sh.array("joints3d_valid")[i][:15]
    err = (
        float(sh.array("pseudo_kp_error_m")[i]) * 1000
        if sh.has("pseudo_kp_error_m")
        else float("nan")
    )
    return {
        "key": str(sh.array("key")[i]),
        "verts": verts,
        "points": sh.scan(i, "real").points.astype(np.float64),
        "keypoints": sh.array("joints3d")[i][:15][valid].astype(np.float64),
        "range_m": float(np.linalg.norm(verts.mean(0))),
        "error_mm": err,
    }


def label_pages(lab: Labels, pages: int = 2, per_page: int = 3) -> None:
    """Accepted records in a fixed order, three rotatable 3D panels per

    page.
    """
    for page in range(pages):
        picks = lab.order[page * per_page : (page + 1) * per_page]
        recs = [_record(lab, *lab.accepted[int(g)]) for g in picks]
        titles = [
            f"{r['key'].split('/')[-2]}<br><sub>{r['range_m']:.1f} m away, "
            f"{len(r['points'])} returns, kp error {r['error_mm']:.0f} mm"
            "</sub>"
            for r in recs
        ]
        fig = make_subplots(
            rows=1,
            cols=len(recs),
            specs=[[{"type": "scene"}] * len(recs)],
            subplot_titles=titles,
            horizontal_spacing=0.02,
        )
        for col, rec in enumerate(recs, start=1):
            c = rec["verts"].mean(0)
            traces = [
                mesh_trace(
                    rec["verts"] - c,
                    lab.smpl.faces,
                    "pseudo-GT SMPL mesh",
                    color="lightblue",
                    opacity=0.55,
                ),
                points_trace(
                    rec["points"] - c, "LiDAR returns", color="black", size=2.5
                ),
                points_trace(
                    rec["keypoints"] - c,
                    "Waymo 3D keypoints",
                    color="limegreen",
                    size=5,
                ),
            ]
            for t in traces:
                fig.add_trace(
                    t.update(legendgroup=t.name, showlegend=col == 1),
                    row=1,
                    col=col,
                )
        _scenes(fig, len(recs), 560)
        display(HTML(f"<b>page {page + 1}</b> of the fixed order"))
        fig.show()


# every Waymo label set: (title, folder under generated/, mesh colour)
LABEL_SETS = [
    ("pseudo-GT v3", "real/v1_pseudo3", "lightblue"),
    ("pseudo-GT v2", "real/v1_pseudo2", "lightskyblue"),
    ("LiDAR-HMR", "real/v1_lidarhmr", "plum"),
    ("pedestrian generation", "real/v1_pedgen", "lightsalmon"),
    ("ours, v1", "real/v1_pseudo", "lightgreen"),
]


@dataclass
class Compare:
    """The same Waymo records in every label set we have."""

    sets: dict[str, dict[str, Shard]]
    colours: dict[str, str]
    keys: list[tuple[str, int, str]]
    order: Array
    smpl: SmplModel


def load_all_label_sets() -> Compare:
    """Open every label set under ``real/`` and print how many meshes each

    has.
    """
    sets = {
        title: {p.path.name: p for p in _shards(DATA / folder)}
        for title, folder, _ in LABEL_SETS
        if (DATA / folder).exists()
    }
    base = next(iter(sets))
    keys = [
        (name, int(i), str(k))
        for name, sh in sets[base].items()
        for i, k in enumerate(sh.array("key"))
    ]
    for title, d in sets.items():
        n = sum(int(sh.array("has_smpl").sum()) for sh in d.values())
        print(f"{title:24s} {n:5d} meshes")
    return Compare(
        sets,
        {t: c for t, _, c in LABEL_SETS},
        keys,
        np.random.default_rng(1).permutation(len(keys)),
        SmplModel(DATA / "body_models"),
    )


def _mesh(cmp: Compare, title: str, name: str, i: int) -> Array | None:
    sh = cmp.sets[title].get(name)
    if sh is None or not bool(sh.array("has_smpl")[i]):
        return None
    verts, _ = cmp.smpl.forward(_params(sh, i))
    return verts


def _project(k: Array, pts: Array) -> Array:
    z = np.clip(pts[:, 2], 1e-3, None)
    return np.stack(
        [k[0, 0] * pts[:, 0] / z + k[0, 2], k[1, 1] * pts[:, 1] / z + k[1, 2]],
        -1,
    )


def _wire(uv: Array, faces: Array, step: int = 3) -> go.Scatter:
    f = faces[::step]
    tri = uv[f]
    gap = np.full((len(f), 1), np.nan)
    xs = np.concatenate([tri[:, [0, 1, 2, 0], 0], gap], 1).ravel()
    ys = np.concatenate([tri[:, [0, 1, 2, 0], 1], gap], 1).ravel()
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": "red", "width": 0.6},
        opacity=0.6,
        hoverinfo="skip",
        showlegend=False,
    )


def _kp_error_cm(cmp: Compare, verts: Array, rec: dict[str, Any]) -> float:
    ok = rec["valid15"][WSEL]
    if not ok.any():
        return float("nan")
    joints = cmp.smpl.coco_joints(verts)[CSEL][ok]
    d = np.linalg.norm(joints - rec["kp15"][WSEL][ok], axis=1)
    return float(d.mean() * 100)


def _title(
    cmp: Compare,
    title: str,
    name: str,
    i: int,
    verts: Array | None,
    rec: dict[str, Any],
) -> str:  # noqa: E501
    if verts is None:
        return f"{title}<br><sub>no mesh label</sub>"
    extra = ""
    sh = cmp.sets[title][name]
    if sh.has("label_conf"):
        extra = f", conf {float(sh.array('label_conf')[i]):.2f}"
    return (
        f"{title}<br><sub>keypoints {_kp_error_cm(cmp, verts, rec):.1f} cm"
        f"{extra}</sub>"
    )


def compare_record(cmp: Compare, position: int) -> None:
    """One record in every label set: crop with the projected mesh above, 3D

    below.
    """
    name, i, key = cmp.keys[int(cmp.order[position])]
    base = cmp.sets[next(iter(cmp.sets))][name]
    rec = {
        "image": base.array("image")[i],
        "K": base.row("intrinsics", i).astype(np.float64),
        "points": base.scan(i, "real").points.astype(np.float64),
        "kp15": base.array("joints3d")[i][:15].astype(np.float64),
        "valid15": base.array("joints3d_valid")[i][:15].astype(bool),
        "kp2d": base.array("kp2d")[i][:15].astype(np.float64),
    }
    titles = list(cmp.sets)
    meshes = [_mesh(cmp, t, name, i) for t in titles]
    heads = [
        _title(cmp, t, name, i, v, rec)
        for t, v in zip(titles, meshes, strict=True)
    ]
    n = len(titles)
    fig = make_subplots(
        rows=2,
        cols=n,
        row_heights=[0.42, 0.58],
        vertical_spacing=0.04,
        horizontal_spacing=0.02,
        subplot_titles=heads + [""] * n,
        specs=[[{"type": "xy"}] * n, [{"type": "scene"}] * n],
    )
    valid = rec["valid15"]
    c = rec["kp15"][valid].mean(0) if valid.any() else rec["points"].mean(0)
    h, w = rec["image"].shape[:2]
    for col, (title, verts) in enumerate(zip(titles, meshes, strict=True), 1):
        fig.add_trace(
            go.Image(z=rec["image"], hoverinfo="skip"), row=1, col=col
        )
        if verts is not None:
            uv = _project(rec["K"], verts)
            fig.add_trace(_wire(uv, cmp.smpl.faces), row=1, col=col)
        ok2 = rec["kp2d"][:, 2] > 0
        fig.add_trace(
            go.Scatter(
                x=rec["kp2d"][ok2, 0],
                y=rec["kp2d"][ok2, 1],
                mode="markers",
                marker={"color": "limegreen", "size": 6},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(visible=False, range=[0, w], row=1, col=col)
        anchor = f"x{'' if col == 1 else col}"
        fig.update_yaxes(
            visible=False, range=[h, 0], scaleanchor=anchor, row=1, col=col
        )
        traces = []
        if verts is not None:
            traces.append(
                mesh_trace(
                    verts - c,
                    cmp.smpl.faces,
                    title,
                    color=cmp.colours[title],
                    opacity=0.55,
                )
            )
        traces += [
            points_trace(
                rec["points"] - c, "LiDAR returns", color="black", size=2.5
            ),
            points_trace(
                rec["kp15"][valid] - c,
                "Waymo 3D keypoints",
                color="limegreen",
                size=5,
            ),
        ]
        for t in traces:
            show = col == 1 or t.name == title
            fig.add_trace(
                t.update(legendgroup=t.name, showlegend=show), row=2, col=col
            )
    _scenes(
        fig,
        n,
        900,
        f"{key}<br><sub>{np.linalg.norm(c):.1f} m away, "
        f"{len(rec['points'])} returns</sub>",
    )
    fig.show()


def label_set_errors(cmp: Compare, n: int = 300) -> None:
    """Median keypoint error and point-to-mesh distance of every label set over

    ``n`` records.
    """
    base_title = next(iter(cmp.sets))
    picks = cmp.order[:n]
    rows: dict[str, list[tuple[float, float]]] = {t: [] for t in cmp.sets}
    for g in picks:
        name, i, _ = cmp.keys[int(g)]
        base = cmp.sets[base_title][name]
        rec = {
            "kp15": base.array("joints3d")[i][:15].astype(np.float64),
            "valid15": base.array("joints3d_valid")[i][:15].astype(bool),
        }
        points = base.scan(i, "real").points.astype(np.float64)
        for title in cmp.sets:
            verts = _mesh(cmp, title, name, i)
            if verts is None:
                continue
            p2m = float(np.median(cKDTree(verts).query(points)[0])) * 100
            rows[title].append((_kp_error_cm(cmp, verts, rec), p2m))
    print(
        f"{'label set':24s} {'meshes':>6s} {'kp err cm':>10s} {'p2m cm':>8s}"
    )
    for title, vals in rows.items():
        if not vals:
            continue
        a = np.array(vals)
        kp, p2m = np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1])
        print(f"{title:24s} {len(vals):6d} {kp:10.1f} {p2m:8.1f}")


# Waymo vehicle frame (x fwd, y left, z up) -> camera (x right, y down, z fwd)
VEHICLE_TO_CAMERA = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
)
HSEL = np.nonzero(LIDARHMR14_TO_COCO17 >= 0)[0]  # their 12 limb joints
HCSEL = LIDARHMR14_TO_COCO17[HSEL]


def _hmr_record(smpl: SmplModel, rec: dict[str, Any]) -> dict[str, Any]:
    verts = np.asarray(rec["smpl_verts"], dtype=np.float64)
    points = np.asarray(rec["human_points"][0], dtype=np.float64)
    kp = np.asarray(rec["keypoints"], dtype=np.float64)
    ok = np.asarray(rec["flag"], dtype=bool)[HSEL]
    joints = smpl.coco_joints(verts)[HCSEL][ok]
    err = (
        float(np.linalg.norm(joints - kp[HSEL][ok], axis=1).mean())
        if ok.any()
        else float("nan")
    )
    where = rec["location_dict"]
    return {
        "segment": str(where["frame"]),
        "time": str(where["time"]),
        "verts": verts @ VEHICLE_TO_CAMERA.T,
        "points": points @ VEHICLE_TO_CAMERA.T,
        "keypoints": kp[HSEL][ok] @ VEHICLE_TO_CAMERA.T,
        "p2m": float(np.median(cKDTree(verts).query(points)[0])),
        "kp_err": err,
        "range_m": float(np.linalg.norm(verts.mean(0))),
    }


def lidar_hmr_labels(split: str = "test", pages: int = 2) -> None:
    """LiDAR-HMR's published Waymo fits on their own returns and keypoints,

    with medians.
    """
    path = DATA.parent / "external" / "lidar_hmr_waymov2" / f"{split}.pkl"
    if not path.exists():
        print("no LiDAR-HMR pickle at", path)
        return
    smpl = SmplModel(DATA / "body_models")
    with path.open("rb") as fh:
        hmr = pickle.load(fh)
    order = np.random.default_rng(0).permutation(len(hmr))
    look = [_hmr_record(smpl, hmr[int(g)]) for g in order[:200]]
    p2m = np.median([r["p2m"] for r in look]) * 100
    kp = np.median([r["kp_err"] for r in look]) * 100
    print(
        f"{path.name}: {len(hmr)} records; medians over {len(look)}: "
        f"point-to-mesh {p2m:.1f} cm, keypoints {kp:.1f} cm"
    )
    for page in range(pages):
        recs = look[page * 3 : (page + 1) * 3]
        titles = [
            f"{r['segment'][:22]} t{r['time'][-7:]}<br><sub>"
            f"{r['range_m']:.1f} m, {len(r['points'])} returns | "
            f"p2m {r['p2m'] * 100:.1f} cm, kp {r['kp_err'] * 100:.1f} cm</sub>"
            for r in recs
        ]
        fig = make_subplots(
            rows=1,
            cols=len(recs),
            specs=[[{"type": "scene"}] * len(recs)],
            subplot_titles=titles,
            horizontal_spacing=0.02,
        )
        for col, rec in enumerate(recs, start=1):
            c = rec["verts"].mean(0)
            traces = [
                mesh_trace(
                    rec["verts"] - c,
                    smpl.faces,
                    "LiDAR-HMR SMPL",
                    color="plum",
                    opacity=0.55,
                ),
                points_trace(
                    rec["points"] - c,
                    "their LiDAR returns",
                    color="black",
                    size=2.5,
                ),
                points_trace(
                    rec["keypoints"] - c,
                    "their 12 limb keypoints",
                    color="limegreen",
                    size=5,
                ),
            ]
            for t in traces:
                fig.add_trace(
                    t.update(legendgroup=t.name, showlegend=col == 1),
                    row=1,
                    col=col,
                )
        _scenes(fig, len(recs), 560)
        fig.show()
