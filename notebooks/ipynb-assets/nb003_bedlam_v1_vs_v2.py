"""Cell bodies of ``003-bedlam-v1-vs-v2.ipynb``: one function per cell.

BEDLAM 1.0 (the training source) against the BEDLAM 2.0 download on the
NAS: inventory, cameras against the target rigs, motion configs, frames,
bodies and assets, and the two LiDAR simulation routes.
"""

from __future__ import annotations

import csv
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

ROOT = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
V1_NAS = Path("/home/max/nas_drive/publicdatasets/bedlam")
V1_LOCAL = ROOT / "resources/data/generated/bedlam_raw"
V1_LABELS = ROOT / "resources/data/generated/bedlam_labels/smpl/bedlam-labels"
V2_NAS = Path("/home/max/nas_drive/publicdatasets/bedlam2")
CACHE = ROOT / "outputs/bedlam2_cache"

# the rigs this project targets: (camera height m, pitch deg, colour)
TARGETS = {
    "Waymo / nuScenes (car roof)": (1.9, -1.0, "tab:blue"),
    "SLOPER4D (head-mounted)": (1.6, -12.0, "tab:green"),
    "TUMTraf S110 (pole)": (7.0, -40.0, "tab:orange"),
    "drone band": (20.0, -70.0, "tab:red"),
}


def _table(rows: list[list[Any]], header: list[str]) -> None:
    h = "".join(
        f"<th style='text-align:left;padding:2px 10px'>{c}</th>"
        for c in header
    )
    b = "".join(
        "<tr>"
        + "".join(f"<td style='padding:2px 10px'>{c}</td>" for c in r)
        + "</tr>"
        for r in rows
    )
    display(
        HTML(f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>")
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _span(a: np.ndarray[Any, Any], digits: int) -> str:
    """``median (p10..p90)`` of an array."""
    lo, hi = np.percentile(a, 10), np.percentile(a, 90)
    return f"{np.median(a):.{digits}f} ({lo:.{digits}f}..{hi:.{digits}f})"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def v2_scenes() -> list[str]:
    """The BEDLAM 2.0 scene groups on the NAS."""
    return sorted(
        p.name
        for p in (V2_NAS / "images").iterdir()
        if p.is_dir() and p.name[0].isdigit() and (p / "ground_truth").is_dir()
    )


def v1_groups() -> list[str]:
    """The BEDLAM 1.0 groups extracted locally."""
    return sorted(p.name for p in V1_LOCAL.iterdir() if p.is_dir())


def prepare_cache() -> None:
    """Unpack the small BEDLAM 2.0 archives (camera CSVs, placements, overview

    sheets) into the cache.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in v2_scenes():
        if not (CACHE / name / "be_seq.csv").exists():
            with tarfile.open(
                next(
                    (V2_NAS / "images" / name / "ground_truth").glob(
                        "*_exr_meta_csv.tar.gz"
                    )
                )
            ) as tf:
                tf.extractall(CACHE, filter="data")
        if not (CACHE / name / "overview").exists():
            with tarfile.open(
                next(
                    (V2_NAS / "images" / name / "overview").glob(
                        "*_overview.tar"
                    )
                )
            ) as tf:
                tf.extractall(CACHE, filter="data")
    nas_groups = {
        re.sub(r"_(depth|gt|masks|mp4|png)\..*$", "", p.name)
        for p in V1_NAS.iterdir()
        if p.suffix in (".tar", ".gz") and "_" in p.name
    }
    print(
        f"BEDLAM 1.0: {len(nas_groups)} groups on the NAS, "
        f"{len(v1_groups())} extracted locally"
    )
    print(f"BEDLAM 2.0: {len(v2_scenes())} scene groups on the NAS")


def inventory() -> None:
    """What each release has on disk: frames, PNG, depth, masks, body ground

    truth.
    """
    rows = []
    for name in v2_scenes():
        base = V2_NAS / "images" / name
        png = sum(p.stat().st_size for p in (base / "png").glob("*.tar"))
        depth = sum(
            p.stat().st_size
            for p in (V2_NAS / "depth" / name).rglob("*")
            if p.is_file()
        )
        csvs = list((CACHE / name / "ground_truth/meta_exr_csv").glob("*.csv"))
        frames = sum(len(_rows(f)) for f in csvs)
        rows.append(
            [
                name,
                len(csvs),
                f"{frames:,}",
                _human(png),
                "none" if depth == 0 else _human(depth),
                "none",
                "camera only",
            ]
        )
    _table(
        rows,
        [
            "BEDLAM 2.0 scene",
            "sequences",
            "frames",
            "PNG",
            "depth",
            "masks",
            "body GT",
        ],
    )
    rows = []
    for name in v1_groups():
        base = V1_LOCAL / name
        frames = sum(1 for _ in (base / "depth").rglob("*_depth.exr"))
        masks = sum(1 for _ in (base / "masks").rglob("*_body.png"))
        lab = sorted(V1_LABELS.glob(f"{name}_*fps.npz"))
        rows.append(
            [
                name,
                len(list((base / "png").glob("seq_*"))),
                f"{frames:,}",
                "clothed, EXR",
                f"{masks:,} person masks",
                "SMPL table" if lab else "missing",
            ]
        )
    _table(
        rows,
        [
            "BEDLAM 1.0 group (local)",
            "sequences",
            "frames",
            "depth",
            "masks",
            "body GT",
        ],
    )


def _camera_rows(files: list[Path]) -> dict[str, np.ndarray[Any, Any]]:
    z: list[Any] = []
    pitch: list[Any] = []
    hfov: list[Any] = []
    speed: list[Any] = []
    for f in files:
        rows = _rows(f)
        if not rows:
            continue
        xyz = (
            np.array(
                [[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]
            )
            / 100.0
        )
        z.append(xyz[:, 2])
        pitch.append([float(r["pitch"]) for r in rows])
        hfov.append([float(r["hfov"]) for r in rows])
        if len(xyz) > 1:
            speed.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 30.0)
    return {
        k: np.concatenate(v) if v else np.zeros(0)
        for k, v in (
            ("z", z),
            ("pitch", pitch),
            ("hfov", hfov),
            ("speed", speed),
        )
    }


def camera_stats() -> dict[tuple[str, str], dict[str, np.ndarray[Any, Any]]]:
    """Per group: camera height, pitch, FOV and speed from every camera CSV, as

    a table and a dict.
    """
    cams: dict[tuple[str, str], dict[str, np.ndarray[Any, Any]]] = {}
    for name in v1_groups():
        cams[("1.0", name)] = _camera_rows(
            sorted((V1_LOCAL / name / "ground_truth/camera").glob("*.csv"))
        )
    for name in v2_scenes():
        cams[("2.0", name)] = _camera_rows(
            sorted((CACHE / name / "ground_truth/meta_exr_csv").glob("*.csv"))
        )
    rows = []
    for (rel, name), c in cams.items():
        rows.append(
            [
                rel,
                name,
                f"{len(c['z']):,}",
                _span(c["z"], 1),
                _span(c["pitch"], 0),
                f"{np.median(c['hfov']):.0f}",
                f"{np.median(c['speed']):.2f}" if len(c["speed"]) else "-",
            ]
        )
    _table(
        rows,
        [
            "release",
            "group",
            "frames",
            "height m (p10..p90)",
            "pitch deg (p10..p90)",
            "HFOV deg",
            "camera speed m/s",
        ],
    )
    return cams


def plot_cameras(
    cams: dict[tuple[str, str], dict[str, np.ndarray[Any, Any]]],
) -> None:
    """Height, pitch and FOV histograms per release, and a height-vs-pitch

    scatter with the target rigs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for rel, color in (("1.0", "0.35"), ("2.0", "crimson")):
        z = np.concatenate([c["z"] for (r, _), c in cams.items() if r == rel])
        p = np.concatenate(
            [c["pitch"] for (r, _), c in cams.items() if r == rel]
        )
        h = np.concatenate(
            [c["hfov"] for (r, _), c in cams.items() if r == rel]
        )
        axes[0].hist(
            z,
            bins=np.linspace(0, 8, 41),
            histtype="step",
            lw=2,
            color=color,
            label=f"BEDLAM {rel}",
            density=True,
        )
        axes[1].hist(
            p,
            bins=np.linspace(-60, 20, 41),
            histtype="step",
            lw=2,
            color=color,
            density=True,
        )
        axes[2].hist(
            h,
            bins=np.linspace(20, 130, 45),
            histtype="step",
            lw=2,
            color=color,
            label=f"BEDLAM {rel}",
            density=True,
        )
    for name, (hz, pt, col) in TARGETS.items():
        axes[0].axvline(hz, color=col, ls="--", lw=1, label=name)
        axes[1].axvline(pt, color=col, ls="--", lw=1)
    axes[0].set_xlabel("camera height above ground (m)")
    axes[1].set_xlabel("pitch (deg, negative = looking down)")
    axes[2].set_xlabel("horizontal FOV (deg)")
    for ax in axes:
        ax.set_yticks([])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(0)
    for (rel, _name), c in cams.items():
        n = min(600, len(c["z"]))
        idx = rng.choice(len(c["z"]), n, replace=False)
        ax.scatter(
            c["pitch"][idx],
            c["z"][idx],
            s=4,
            alpha=0.35,
            color="0.4" if rel == "1.0" else "crimson",
        )
    for name, (hz, pt, col) in TARGETS.items():
        ax.scatter(
            [pt],
            [hz],
            marker="*",
            s=260,
            color=col,
            edgecolor="k",
            zorder=5,
            label=name,
        )
    ax.axhspan(10, 30, xmin=0.0, xmax=0.45, color="tab:red", alpha=0.06)
    ax.set_xlabel("pitch (deg)")
    ax.set_ylabel("camera height (m)")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 40)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "grey: BEDLAM 1.0 frames, red: BEDLAM 2.0 frames, stars: target rigs",
        fontsize=10,
    )
    plt.show()


def motion_configs() -> None:
    """The camera animation config of every group: type, follow, shake, height

    and FOV ranges.
    """
    rows = []
    for name in v2_scenes():
        with (CACHE / name / "be_camera_animations.json").open() as fh:
            info = json.load(fh).get("info", {})
        cfg = info.get("config", {})
        rows.append(
            [
                "2.0",
                name,
                info.get("config_type", "?"),
                "yes" if cfg.get("follow") else "no",
                cfg.get("camera_shake") or "none",
                f"{cfg.get('camera_height_min', '?')}.."
                f"{cfg.get('camera_height_max', '?')} cm",
                ", ".join(
                    f"{a:.0f}-{b:.0f}" for a, b in cfg.get("hfov_ranges", [])
                ),
            ]
        )
    for name in v1_groups():
        comment = _rows(V1_LOCAL / name / "be_seq.csv")[0]["Comment"]
        kv = dict(p.split("=", 1) for p in comment.split(";") if "=" in p)
        rows.append(
            [
                "1.0",
                name,
                kv.get("cam_config", "?"),
                "no",
                "none",
                f"pitch {kv.get('cam_pitch_min')}.."
                f"{kv.get('cam_pitch_max')} deg",
                "fixed per group",
            ]
        )
    _table(
        rows,
        [
            "release",
            "group",
            "camera config",
            "follows a body",
            "shake",
            "height / pitch range",
            "HFOV range",
        ],
    )


def _cam_of(
    csv_path: Path, frame_png: str
) -> tuple[float, float, float] | None:
    for r in _rows(csv_path):
        if r["name"] == frame_png:
            return float(r["z"]) / 100, float(r["pitch"]), float(r["hfov"])
    return None


def frames_gallery(
    v1_picks: list[tuple[str, str, str]], n_v2: int = 4
) -> None:
    """1.0 frames from the local extraction and 2.0 frames from the cache,

    camera pose in the titles.
    """
    from PIL import Image

    fig, axes = plt.subplots(
        1, len(v1_picks), figsize=(4.4 * len(v1_picks), 2.9)
    )
    for ax, (g, s, f) in zip(np.atleast_1d(axes), v1_picks, strict=True):
        p = V1_LOCAL / g / "png" / s / f"{s}_{f}.png"
        if not p.exists():
            p = next((V1_LOCAL / g / "png" / s).glob("*.png"))
        ax.imshow(Image.open(p))
        ax.axis("off")
        c = _cam_of(
            V1_LOCAL / g / "ground_truth/camera" / f"{s}_camera.csv", p.name
        )
        ax.set_title(
            f"1.0 {g.split('_', 3)[-1]}\n"
            + (
                f"h {c[0]:.1f} m, pitch {c[1]:.0f}, hfov {c[2]:.0f}"
                if c
                else ""
            ),
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()
    by_scene: dict[str, list[Path]] = {}
    for p in sorted(CACHE.rglob("png/seq_*/*.png")):
        by_scene.setdefault(p.parts[-4], []).append(p)
    picks = [ps[len(ps) // 2] for ps in by_scene.values()][:n_v2]
    if picks:
        fig, axes = plt.subplots(
            1, len(picks), figsize=(4.4 * len(picks), 2.9)
        )
        for ax, p in zip(np.atleast_1d(axes), picks, strict=True):
            scene, seq = p.parts[-4], p.parts[-2]
            ax.imshow(Image.open(p))
            ax.axis("off")
            c = _cam_of(
                CACHE
                / scene
                / "ground_truth/meta_exr_csv"
                / f"{seq}_camera.csv",
                p.name,
            )
            ax.set_title(
                f"2.0 {scene.split('_', 3)[-1]}\n"
                + (
                    f"h {c[0]:.1f} m, pitch {c[1]:.0f}, hfov {c[2]:.0f}"
                    if c
                    else ""
                ),
                fontsize=8,
            )
        plt.tight_layout()
        plt.show()
    else:
        print(
            "no 2.0 frames in the cache: extract a few with tar --wildcards "
            "from <scene>/png/*_png.0.tar into outputs/bedlam2_cache"
        )


def overview_sheets() -> None:
    """The official overview sheet of every 2.0 scene (one sequence, first

    half).
    """
    from PIL import Image

    for name in v2_scenes():
        sheets = sorted((CACHE / name / "overview/images").glob("*.png"))
        if not sheets:
            continue
        im = Image.open(sheets[len(sheets) // 2])
        fig, ax = plt.subplots(
            figsize=(14, 14 * im.size[1] / im.size[0] * 0.5)
        )
        ax.imshow(im.crop((0, 0, im.size[0], im.size[1] // 2)))
        ax.axis("off")
        ax.set_title(f"2.0 {name}", fontsize=9)
        plt.show()


def _placement(f: Path) -> tuple[int, int, float, int, int, int, int]:
    per: Counter[str] = Counter()
    g = ""
    subj, cloth, hair, shoe = set(), set(), set(), set()
    for r in _rows(f):
        if r["Type"] == "Group":
            g = r["Comment"].split(";")[0]
        elif r["Type"] == "Body":
            per[g] += 1
            subj.add(r["Body"])
            kv = dict(
                p.split("=", 1) for p in r["Comment"].split(";") if "=" in p
            )
            cloth.add(kv.get("texture_clothing", ""))
            hair.add(kv.get("hair", ""))
            shoe.add(kv.get("shoe", ""))
    shoe.discard("")
    return (
        len(per),
        sum(per.values()),
        float(np.median(list(per.values()))) if per else 0.0,
        len(subj),
        len(cloth),
        len(hair),
        len(shoe),
    )


def bodies_and_assets() -> None:
    """Bodies per sequence, subjects, clothing, hair and shoes per group; 1.0

    label statistics.
    """
    rows = []
    for rel, names, base in (
        ("2.0", v2_scenes(), CACHE),
        ("1.0", v1_groups(), V1_LOCAL),
    ):
        for name in names:
            n, b, med, s, c, h, sh = _placement(base / name / "be_seq.csv")
            rows.append(
                [rel, name, n, b, f"{med:.0f}", s, c, h, sh if sh else "none"]
            )
    _table(
        rows,
        [
            "release",
            "group",
            "sequences",
            "bodies",
            "bodies / seq",
            "subjects",
            "clothing textures",
            "hair grooms",
            "shoes",
        ],
    )
    rows = []
    for name in v1_groups():
        lab = sorted(V1_LABELS.glob(f"{name}_*fps.npz"))
        if not lab:
            continue
        d = np.load(lab[0], allow_pickle=True)
        depth = d["trans_cam"][:, 2] + d["cam_ext"][:, 2, 3]
        rows.append(
            [
                name,
                f"{len(d['imgname']):,}",
                _span(depth, 1),
                f"{np.median(200 * d['scale']):.0f}",
                f"{np.std(d['shape'][:, 0]):.2f}",
            ]
        )
    _table(
        rows,
        [
            "BEDLAM 1.0 group",
            "label rows (6 fps)",
            "person depth m (p10..p90)",
            "bbox side px",
            "beta_0 std",
        ],
    )
    print("BEDLAM 2.0: no body parameters on disk.")


def lidar_routes(group: str, frame_stride: int = 50) -> None:
    """One 1.0 frame scanned two ways: against the clothed depth map, and

    against the SMPL body mesh.
    """
    from scipy.spatial import cKDTree

    from lidar_bedlam.body.smpl import SmplModel
    from lidar_bedlam.data.bedlam import BedlamFramesSource
    from lidar_bedlam.lidar.mesh_depth import offset_mesh, render_depth
    from lidar_bedlam.lidar.placement import BallPlacement
    from lidar_bedlam.lidar.simulate import ouster, simulate
    from lidar_bedlam.utils.io import read_exr_depth

    smpl = SmplModel(ROOT / "resources/data/generated/body_models")
    src = BedlamFramesSource(
        V1_LOCAL,
        groups=[group],
        frame_stride=frame_stride,
        labels_dir=V1_LABELS,
        smpl_model=smpl,
    )
    sample = next(
        s for s in (src.load(i) for i in range(len(src))) if s.smpl is not None
    )
    assert sample.smpl is not None
    cam = sample.camera
    rng = np.random.default_rng(0)
    spec = ouster("OS1", 128, 1024)
    pose = BallPlacement().sample(rng)
    seq = sample.meta.sequence.split("/")[1]
    depth_cm = read_exr_depth(
        V1_LOCAL
        / group
        / "depth"
        / seq
        / f"{seq}_{sample.meta.frame}_depth.exr"
    )
    depth_m = np.where(depth_cm < 1e7, depth_cm * 0.01, 1e4).astype(np.float64)
    scan_a = simulate(depth_m, cam, spec, pose, rng)
    uv = scan_a.pixel
    assert sample.mask is not None
    keep = sample.mask[
        np.clip(uv[:, 1], 0, cam.height - 1),
        np.clip(uv[:, 0], 0, cam.width - 1),
    ]
    pts_a = scan_a.points[keep]
    verts, _ = smpl.forward(sample.smpl)
    depth_b, _ = render_depth(
        [(offset_mesh(verts, smpl.faces, 0.02), smpl.faces)], cam
    )
    pts_b = simulate(
        depth_b.astype(np.float64), cam, spec, pose, np.random.default_rng(0)
    ).points
    tree = cKDTree(verts)
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(sample.image)
    ax.axis("off")
    x0, y0, x1, y1 = sample.bbox_xyxy
    ax.set_xlim(x0 - 40, x1 + 40)
    ax.set_ylim(y1 + 40, y0 - 40)
    ax.set_title(f"1.0 frame, person {sample.meta.person}", fontsize=9)
    for k, (pts, title) in enumerate(
        (
            (pts_a, "clothed depth (the 1.0 route)"),
            (pts_b, "SMPL mesh + 2 cm (the 2.0 route)"),
        )
    ):
        dd = tree.query(pts)[0] * 100
        ax3 = fig.add_subplot(1, 3, 2 + k, projection="3d")
        ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=3, c="0.3")
        ax3.scatter(
            verts[::6, 0],
            verts[::6, 2],
            -verts[::6, 1],
            s=0.4,
            c="red",
            alpha=0.4,
        )
        c = verts.mean(0)
        ax3.set_xlim(c[0] - 1, c[0] + 1)
        ax3.set_ylim(c[2] - 1, c[2] + 1)
        ax3.set_zlim(-c[1] - 1, -c[1] + 1)
        ax3.view_init(elev=12, azim=-60)
        ax3.set_title(
            f"{title}\n{len(pts)} returns, point-to-body median "
            f"{np.median(dd):.1f} cm, p90 {np.percentile(dd, 90):.1f} cm",
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()
