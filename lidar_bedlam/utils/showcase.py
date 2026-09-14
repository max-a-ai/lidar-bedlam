"""Showcase helpers for the capabilities notebook.

Runs our model on the validation shards, loads the baseline meshes written
by ``lidar_bedlam/scripts/baselines``, computes per-record metrics for every
method with the paper protocol and draws the comparison figures (crop,
bird's-eye and side view of every method's mesh in the LiDAR returns), the
knowledge-transfer bars and curves, the error-by-distance curves and the
learned gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.geometry.rotations import matrix_to_axis_angle
from lidar_bedlam.metrics.protocol import SampleMetrics, sample_metrics
from lidar_bedlam.scripts.score_baselines import PredictionTable, _select
from lidar_bedlam.train.config import TrainConfig, load_config
from lidar_bedlam.train.loop import (
    build_dataset,
    build_model,
    source_shards,
    tensors_only,
    to_device,
)

FloatArray = NDArray[np.float64]

# published pipelines: label -> prediction folder under outputs/baselines
BASELINES = {
    "HMR2.0": "hmr2-full",
    "TokenHMR": "tokenhmr-tight",
    "CameraHMR": "camerahmr-full",
    "LiDAR-HMR": "lidar-hmr",
}
COLORS = {
    "GT": "limegreen",
    "ours": "crimson",
    "LiDAR-HMR": "darkorange",
    "CameraHMR": "royalblue",
    "TokenHMR": "mediumorchid",
    "HMR2.0": "teal",
}


@torch.no_grad()
def run_model(
    config: Path,
    checkpoint: Path,
    out_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
    overrides: list[str] | None = None,
) -> dict[str, Path]:
    """Predict every validation record; one npz per source (baseline format).

    Existing files are reused, so the notebook re-runs in seconds.
    """
    cfg: TrainConfig = load_config(config, overrides or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s.name: out_dir / f"{s.name}.npz" for s in cfg.data.val}
    todo = [s for s in cfg.data.val if not paths[s.name].exists()]
    if not todo:
        return paths
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    for src in todo:
        loader = DataLoader(
            build_dataset(src, cfg, train=False),
            batch_size=batch_size,
            num_workers=4,
        )
        keys: list[str] = []
        verts, transl, orient, betas, gates = [], [], [], [], []
        for batch in loader:
            b = to_device(batch, torch.device(device))
            pred = model(tensors_only(b))
            keys.extend(str(k) for k in batch["key"])
            verts.append(pred["vertices"].cpu().numpy().astype(np.float16))
            transl.append(pred["transl"].double().cpu().numpy())
            rot = pred["global_orient"].double().cpu().numpy()
            orient.append(matrix_to_axis_angle(rot.reshape(-1, 3, 3)))
            betas.append(pred["betas"].double().cpu().numpy())
            g = pred["gates"].float().cpu().numpy()  # (L, B, G)
            gates.append(np.transpose(g, (1, 0, 2)).astype(np.float16))
        arrays: dict[str, Any] = {
            "key": np.array(keys),
            "vertices": np.concatenate(verts),
            "transl": np.concatenate(transl),
            "global_orient": np.concatenate(orient),
            "betas": np.concatenate(betas),
            "gates": np.concatenate(gates),
            "meta/method": np.array("ours"),
            "meta/step": np.array(int(state.get("step", -1))),
        }
        np.savez(paths[src.name], **arrays)
    return paths


class ValIndex:
    """Validation records of one source by key (shard-backed)."""

    def __init__(self, config: Path, source: str) -> None:
        cfg = load_config(config)
        src = next(s for s in cfg.data.val if s.name == source)
        self.shards = [Shard(p) for p in source_shards(src)]
        self.where: dict[str, tuple[Shard, int]] = {}
        for sh in self.shards:
            for i, k in enumerate(sh.array("key")):
                self.where[str(k)] = (sh, i)

    def __len__(self) -> int:
        return len(self.where)

    def row(self, key: str) -> dict[str, Any]:
        """Image, crop intrinsics, LiDAR returns and labels of one record."""
        sh, i = self.where[key]
        out: dict[str, Any] = {
            "key": key,
            "dataset": str(sh.array("dataset")[i]),
            "image": np.asarray(sh.row("image", i)),
            "intrinsics": sh.row("intrinsics", i).astype(np.float64),
            "points": sh.scan(i, "real").points.astype(np.float64),
            "joints3d": sh.row("joints3d", i).astype(np.float64),
            "joints3d_valid": sh.row("joints3d_valid", i).astype(bool),
            "joint_convention": str(sh.array("joint_convention")[i]),
            "has_smpl": bool(sh.array("has_smpl")[i]),
            "kp2d": sh.row("kp2d", i).astype(np.float64),
        }
        if out["has_smpl"]:
            out["smpl"] = SmplParams(
                sh.row("global_orient", i).astype(np.float64),
                sh.row("body_pose", i).astype(np.float64),
                sh.row("betas", i).astype(np.float64),
                sh.row("transl", i).astype(np.float64),
            )
        return out


class MeshTable(PredictionTable):
    """PredictionTable whose vertices stay on disk (memory-mapped).

    Five methods x two splits of float16 meshes would take ~4 GB in RAM;
    the npz is unpacked once into ``<stem>.mm/`` and read lazily.
    """

    def __init__(self, path: Path) -> None:  # noqa: D107
        cache = path.with_suffix(".mm")
        if not (cache / "vertices.npy").exists():
            cache.mkdir(exist_ok=True)
            with np.load(path) as z:
                for k in ("vertices", "transl", "global_orient"):
                    np.save(cache / f"{k}.npy", z[k])
                meta = {
                    k[5:]: z[k].item()
                    for k in z.files
                    if k.startswith("meta/")
                }
                (cache / "keys.json").write_text(
                    json.dumps(
                        {"keys": [str(k) for k in z["key"]], "meta": meta}
                    )
                )
        info = json.loads((cache / "keys.json").read_text())
        self.index = {k: i for i, k in enumerate(info["keys"])}
        self.vertices = np.load(cache / "vertices.npy", mmap_mode="r")
        self.transl = np.load(cache / "transl.npy").astype(np.float64)
        self.global_orient = np.load(cache / "global_orient.npy").astype(
            np.float64
        )
        self.meta = info["meta"]
        self.name = path.parent.name
        self.split = path.stem


def per_record_metrics(
    tables: dict[str, PredictionTable],
    config: Path,
    source: str,
    smpl: SmplModel,
) -> dict[str, dict[str, SampleMetrics]]:
    """Paper-protocol metrics per record for every method (one data pass)."""
    cfg = load_config(config)
    src = next(s for s in cfg.data.val if s.name == source)
    loader = DataLoader(
        build_dataset(src, cfg, train=False), batch_size=64, num_workers=4
    )
    out: dict[str, dict[str, SampleMetrics]] = {n: {} for n in tables}
    for batch in loader:
        keys = [str(k) for k in batch["key"]]
        for name, table in tables.items():
            pred, rows = table.batch(keys, smpl)
            if rows:
                for m in sample_metrics(pred, _select(batch, rows), smpl):
                    out[name][m.key] = m
    return out


def load_tables(
    ours: Path, baselines_dir: Path, source: str
) -> dict[str, PredictionTable]:
    """Mesh tables of ours and every baseline that has the source."""
    tables: dict[str, PredictionTable] = {"ours": MeshTable(ours)}
    for label, folder in BASELINES.items():
        p = baselines_dir / folder / f"{source}.npz"
        if p.exists():
            tables[label] = MeshTable(p)
    return tables


@dataclass
class Pick:
    """A record chosen for the showcase and why."""

    key: str
    reason: str


def pick_records(
    metrics: dict[str, dict[str, SampleMetrics]],
    source: str,
) -> list[Pick]:
    """Records where our model's advantage is largest, by story."""
    ours = metrics["ours"]
    picks: list[Pick] = []
    image = [m for m in ("CameraHMR", "TokenHMR", "HMR2.0") if m in metrics]
    if image:
        # image methods misplace, we do not (placement margin, pose sane)
        best_key, best_margin = "", -1.0
        for k, m in ours.items():
            others = [
                metrics[n][k].transl_err_m for n in image if k in metrics[n]
            ]
            if not others or not np.isfinite(m.transl_err_m):
                continue
            if m.mpjpe < 90 and m.transl_err_m < 0.25:
                margin = float(min(others)) - m.transl_err_m
                if margin > best_margin:
                    best_key, best_margin = k, margin
        if best_key:
            picks.append(
                Pick(
                    best_key,
                    f"image methods misplace {best_margin:.2f} m more",
                )
            )
    if "LiDAR-HMR" in metrics:
        best_key, best_margin = "", -1.0
        for k, m in ours.items():
            if k not in metrics["LiDAR-HMR"] or m.mpjpe > 60:
                continue
            margin = metrics["LiDAR-HMR"][k].mpjpe - m.mpjpe
            if margin > best_margin:
                best_key, best_margin = k, margin
        if best_key:
            picks.append(
                Pick(
                    best_key,
                    f"pose: LiDAR-HMR {best_margin:.0f} mm worse than ours",
                )
            )
    if source.startswith("waymo"):
        far = [
            (m.gt_depth_m, k)
            for k, m in ours.items()
            if np.isfinite(m.gt_depth_m)
            and m.transl_err_m < 0.3
            and m.mpjpe < 90
        ]
        if far:
            depth, key = max(far)
            picks.append(
                Pick(key, f"far range: {depth:.0f} m from the camera")
            )
    return picks


def _project(k: FloatArray, pts: FloatArray) -> FloatArray:
    z = np.clip(pts[:, 2], 1e-3, None)
    return np.stack(
        [k[0, 0] * pts[:, 0] / z + k[0, 2], k[1, 1] * pts[:, 1] / z + k[1, 2]],
        -1,
    )


def comparison_figure(
    rec: dict[str, Any],
    tables: dict[str, PredictionTable],
    metrics: dict[str, dict[str, SampleMetrics]],
    smpl: SmplModel,
    title: str = "",
) -> Any:
    """Crop with GT joints and our projected mesh; BEV and side views of
    every method's mesh (vertices subsampled) in the LiDAR returns."""
    import matplotlib.pyplot as plt

    key = rec["key"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax = axes[0]
    ax.imshow(rec["image"])
    kp = rec["kp2d"]
    ok = kp[:, 2] > 0
    ax.scatter(kp[ok, 0], kp[ok, 1], s=18, c=COLORS["GT"], label="GT joints")
    meshes: dict[str, FloatArray] = {}
    for name, t in tables.items():
        if key in t.index:
            meshes[name] = t.vertices[t.index[key]].astype(np.float64)
    if "ours" in meshes:
        uv = _project(rec["intrinsics"], meshes["ours"][::8])
        ax.scatter(
            uv[:, 0],
            uv[:, 1],
            s=1.5,
            c=COLORS["ours"],
            alpha=0.5,
            label="ours",
        )
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{rec['dataset']}  {key.split('/')[-1][:28]}", fontsize=9)
    pts = rec["points"]
    gt = rec["joints3d"][rec["joints3d_valid"]]
    if rec["has_smpl"]:
        gt_verts, _ = smpl.forward(rec["smpl"])
    for ax, (a, b, la, lb, flip) in zip(
        axes[1:],
        [
            (0, 2, "x [m]  (right)", "z [m]  (depth)", False),
            (2, 1, "z [m]  (depth)", "y [m]  (down)", True),
        ],
        strict=True,
    ):
        ax.scatter(pts[:, a], pts[:, b], s=2, c="0.6", label="LiDAR returns")
        if rec["has_smpl"]:
            ax.scatter(
                gt_verts[::6, a],
                gt_verts[::6, b],
                s=1,
                c=COLORS["GT"],
                alpha=0.5,
            )
        ax.scatter(
            gt[:, a], gt[:, b], s=30, c=COLORS["GT"], marker="x", label="GT"
        )
        for name, v in meshes.items():
            m = metrics.get(name, {}).get(key)
            tag = (
                f"{name}: {m.mpjpe:.0f} mm, {m.transl_err_m:.2f} m"
                if m
                else name
            )
            ax.scatter(
                v[::6, a],
                v[::6, b],
                s=1,
                c=COLORS.get(name, "k"),
                alpha=0.6,
                label=tag,
            )
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        # square window around the labelled joints, wide enough for the
        # farthest misplaced mesh along the depth axis
        allv = np.concatenate([gt] + [v for v in meshes.values()] + [pts])
        lo, hi = allv[:, 2].min() - 0.5, allv[:, 2].max() + 0.5
        half = max((hi - lo) / 2.0, 1.5)
        c = gt.mean(0) if len(gt) else pts.mean(0)
        ax.set_xlim(*((c[a] - half, c[a] + half) if a != 2 else (lo, hi)))
        ax.set_ylim(*((c[b] - half, c[b] + half) if b != 2 else (lo, hi)))
        ax.set_aspect("equal")
        if flip:
            ax.invert_yaxis()
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, markerscale=6, loc="upper left")
    axes[1].set_title("bird's-eye view: placement along the ray", fontsize=9)
    axes[2].set_title("side view", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _val_lines(metrics_file: Path) -> list[dict[str, Any]]:
    return [
        json.loads(ln)
        for ln in metrics_file.read_text().splitlines()
        if '"val/' in ln
    ]


def transfer_figure(root: Path, runs: dict[str, str]) -> Any:
    """Final metrics per run (bars) and Waymo / SLOPER4D MPJPE over training
    (curves) for runs trained with and without synthetic data."""
    import matplotlib.pyplot as plt

    from lidar_bedlam.scripts.build_scoreboard import load_run

    keys = [
        ("W_mpjpe", "Waymo MPJPE [mm]"),
        ("W_transl_err_m", "Waymo placement [m]"),
        ("W_map", "Waymo box mAP"),
        ("S_mpjpe", "SLOPER4D MPJPE [mm]"),
        ("S_transl_err_m", "SLOPER4D placement [m]"),
        ("S_map", "SLOPER4D box mAP"),
    ]
    values = {label: load_run(root, run) for label, run in runs.items()}
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    for ax, (k, name) in zip(axes[0], keys[:3], strict=True):
        _bars(ax, values, k, name)
    for ax, (k, name) in zip(axes[1, :2], keys[3:5], strict=True):
        _bars(ax, values, k, name)
    ax = axes[1, 2]
    for label, run in runs.items():
        f = root / "runs" / run / "metrics.jsonl"
        if not f.exists():
            continue
        lines = _val_lines(f)
        steps = [ln["step"] / 1000 for ln in lines]
        ax.plot(
            steps,
            [ln["val/waymo_val/mpjpe"] for ln in lines],
            label=f"{label} · Waymo",
        )
        ax.plot(
            steps,
            [ln["val/sloper4d_test/mpjpe"] for ln in lines],
            "--",
            label=f"{label} · SLOPER4D",
        )
    ax.set_xlabel("step [k]")
    ax.set_ylabel("MPJPE [mm]")
    ax.set_title("validation MPJPE over training", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "knowledge transfer: real only vs synthetic added (same schedule)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _bars(ax: Any, values: dict[str, Any], key: str, name: str) -> None:
    labels, ys = [], []
    for label, v in values.items():
        if v is None or v[0].get(key) is None:
            continue
        labels.append(label)
        ys.append(float(v[0][key]))
    if not ys:
        ax.text(
            0.5,
            0.5,
            "awaiting re-evaluation with\nthe corrected placement metric",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="0.4",
        )
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    bars = ax.bar(
        labels,
        ys,
        color=[
            "0.5" if "real" in lb or "synth" in lb else COLORS["ours"]
            for lb in labels
        ],
    )
    for b, y in zip(bars, ys, strict=True):
        ax.annotate(
            f"{y:.3g}",
            (b.get_x() + b.get_width() / 2, y),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(name, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def distance_figure(
    static_results: Path,
    ours_results: Path | None,
    source: str = "waymo_val",
    ours_name: str = "full-main-mixed",
) -> Any:
    """Placement error per ground-truth distance bin for every method.

    Both files come from ``score_baselines.py`` (method -> split -> summary).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    res = json.loads(static_results.read_text())
    series: dict[str, dict[str, float]] = {}
    for label, folder in BASELINES.items():
        if folder in res and source in res[folder]:
            series[label] = res[folder][source]["by_distance"]
    if ours_results is not None and ours_results.exists():
        ours = json.loads(ours_results.read_text())
        if ours_name in ours and source in ours[ours_name]:
            series["ours"] = ours[ours_name][source]["by_distance"]
    for label, bins in series.items():
        names = [n for n in bins if np.isfinite(bins[n])]
        ax.plot(
            names,
            [bins[n] for n in names],
            "o-",
            label=label,
            color=COLORS.get(label, "k"),
        )
    ax.set_ylabel("placement error [m]")
    ax.set_xlabel("ground-truth distance")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"placement error by distance ({source})", fontsize=10)
    fig.tight_layout()
    return fig


def gates_figure(
    gates: NDArray[np.floating], group_names: list[str], title: str
) -> Any:
    """Mean image gate per decoder layer and joint group over records."""
    import matplotlib.pyplot as plt

    g = np.asarray(gates, dtype=np.float64).mean(0)  # (L, G)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(g, vmin=0, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticks(range(g.shape[0]))
    ax.set_ylabel("decoder layer")
    fig.colorbar(im, ax=ax, label="image gate (1 = image, 0 = LiDAR)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
