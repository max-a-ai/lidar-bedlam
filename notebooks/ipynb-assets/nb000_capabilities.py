"""Cell bodies of ``000-capabilities.ipynb``: one function per cell.

Each function is one capability of the repository, in the order it was
built, and draws on real files under ``resources/``. The notebook cell is
the parameters plus the call; hover the call to read what it does.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.base import add_smpl_derived, crop_sample
from lidar_bedlam.data.bedlam import CM_TO_M, BedlamFramesSource
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17, Sample
from lidar_bedlam.lidar.simulate import (
    PRESETS,
    azimuth_window,
    camera_hfov_deg,
    select_mask,
    sensor_pose,
    simulate,
)
from lidar_bedlam.utils.io import read_exr_depth, read_mask
from lidar_bedlam.utils.viz import (
    box_trace,
    draw_mask_outline,
    draw_points,
    figure_3d,
    mesh_trace,
    point_to_surface_distance,
    points_trace,
)

ROOT = Path.cwd() if (Path.cwd() / "resources").exists() else Path.cwd().parent
DATA = ROOT / "resources" / "data"
GEN = DATA / "generated"
BEDLAM_RAW = GEN / "bedlam_raw"
WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
CSEL = WAYMO15_TO_COCO17[WSEL]


@dataclass
class Bedlam:
    """Frames picked from the local BEDLAM extraction, with depth maps."""

    src: BedlamFramesSource
    samples: list[Sample]
    depths: list[NDArray[Any]]
    persons: list[list[Sample]]
    smpl: SmplModel
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0)
    )
    scans: list[Any] = field(default_factory=list)


def load_bedlam(frame_stride: int = 5, n_sequences: int = 5) -> Bedlam:
    """Index the local BEDLAM frames and pick one frame per spread sequence."""
    src = BedlamFramesSource(BEDLAM_RAW, frame_stride=frame_stride)
    metas = [src.meta(i) for i in range(len(src))]
    sequences = sorted({m.sequence for m in metas})
    picked = [
        sequences[i]
        for i in np.linspace(0, len(sequences) - 1, n_sequences).astype(int)
    ]
    ids = []
    for seq in picked:
        idx = [
            i
            for i, m in enumerate(metas)
            if m.sequence == seq and m.person == "00"
        ]
        ids.append(idx[len(idx) // 2])
    ctx = Bedlam(
        src=src,
        samples=[src.load(i) for i in ids],
        depths=[
            read_exr_depth(src.depth_path(i)).astype(np.float64) * CM_TO_M
            for i in ids
        ],
        persons=[[src.load(j) for j in src.frame_persons(i)] for i in ids],
        smpl=SmplModel(GEN / "body_models"),
    )
    print(
        f"{len(src)} person-frames in {len(sequences)} sequences; showing",
        [s.meta.key for s in ctx.samples],
    )
    return ctx


def show_frames(ctx: Bedlam) -> None:
    """RGB with person boxes and the planar depth map of every picked frame."""
    fig, axes = plt.subplots(
        len(ctx.samples), 2, figsize=(13, 3.4 * len(ctx.samples))
    )
    for row, (s, depth, ppl) in enumerate(
        zip(ctx.samples, ctx.depths, ctx.persons, strict=True)
    ):
        ax = axes[row, 0]
        ax.imshow(s.image)
        ax.set_title(
            f"{_seq(s)} frame {s.meta.frame}: {len(ppl)} persons",
            fontsize=9,
        )
        for p in ppl:
            x0, y0, x1, y1 = p.bbox_xyxy
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    color="yellow",
                    lw=1,
                )
            )
        ax = axes[row, 1]
        d = np.where(depth < 1e4, depth, np.nan)
        im = ax.imshow(d, cmap="turbo", vmin=0, vmax=np.nanpercentile(d, 99))
        ax.set_title("depth (planar z)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, label="m")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def show_hair_mask(ctx: Bedlam, key: tuple[str, str, str, str]) -> None:
    """The person mask with and without the hair part, and the returns it

    adds.
    """
    group, seq, frame, person = key
    src = ctx.src
    idx = next(
        i
        for i in range(len(src))
        if src.meta(i).sequence == f"{group}/{seq}"
        and src.meta(i).frame == frame
        and src.meta(i).person == person
    )
    hs = src.load(idx)
    hdepth = read_exr_depth(src.depth_path(idx)).astype(np.float64) * CM_TO_M
    parts: dict[str, NDArray[Any]] = {}
    for part in ("body", "clothing", "hair"):
        path = (
            BEDLAM_RAW
            / group
            / "masks"
            / seq
            / f"{seq}_{frame}_{person}_{part}.png"
        )
        if path.exists():
            parts[part] = read_mask(path)
    without = parts["body"] | parts["clothing"]
    with_hair = without | parts.get("hair", np.zeros_like(without))
    scan = simulate(
        hdepth, hs.camera, PRESETS["OS1-128"], rng=np.random.default_rng(0)
    )
    n_before = len(select_mask(scan, without).points)
    n_after = len(select_mask(scan, with_hair).points)
    x0, y0, x1, y1 = hs.bbox_xyxy.astype(int)
    pad = 40
    crop = (
        slice(max(0, y0 - pad), y1 + pad),
        slice(max(0, x0 - pad), x1 + pad),
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    axes[0].imshow(hs.image[crop])
    axes[0].set_title("RGB crop")
    for ax, (title, m) in zip(
        axes[1:3],
        (("body | clothing", without), ("body | clothing | hair", with_hair)),
        strict=True,
    ):
        ax.imshow(hs.image[crop])
        ax.imshow(
            np.ma.masked_where(~m[crop], m[crop]), alpha=0.5, cmap="autumn"
        )
        ax.set_title(title)
    axes[3].imshow(hs.image[crop])
    pc = select_mask(scan, with_hair)
    ph = select_mask(scan, parts.get("hair", np.zeros_like(with_hair)))
    axes[3].scatter(
        pc.pixel[:, 0] - crop[1].start,
        pc.pixel[:, 1] - crop[0].start,
        s=6,
        c="cyan",
        label=f"person returns: {n_after} (was {n_before})",
    )
    axes[3].scatter(
        ph.pixel[:, 0] - crop[1].start,
        ph.pixel[:, 1] - crop[0].start,
        s=10,
        c="red",
        label=f"on hair: {len(ph.points)}",
    )
    axes[3].legend(loc="lower right")
    axes[3].set_title("OS1-128 returns on the person")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def _seq(s: Sample) -> str:
    return s.meta.sequence.split("/")[-1]


def _mask(s: Sample) -> NDArray[np.bool_]:
    assert s.mask is not None
    return s.mask


def _occlude(pts: NDArray[Any], kind: str, frac: float) -> NDArray[Any]:
    from lidar_bedlam.lidar.augment import (
        axis_cut_fraction,
        height_cut_fraction,
    )

    if kind == "legs":
        return height_cut_fraction(pts, frac, True)
    if kind == "head":
        return height_cut_fraction(pts, frac, False)
    if kind == "left side":
        return axis_cut_fraction(pts, 0, frac, True)
    if kind == "right side":
        return axis_cut_fraction(pts, 0, frac, False)
    return axis_cut_fraction(pts, 2, frac, True)


def _surface(
    p: Sample, rng: np.random.Generator, n: int = 4000
) -> NDArray[Any]:
    pts = p.points.astype(np.float64)
    return pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]


def show_surfaces_3d(ctx: Bedlam) -> None:
    """Every person of every picked frame as a dense clothed surface,

    rotatable.
    """
    for s, ppl in zip(ctx.samples, ctx.persons, strict=True):
        traces = [
            points_trace(
                _surface(p, ctx.rng),
                f"person {p.meta.person}",
                color=i,
                size=1.5,
            )
            for i, p in enumerate(ppl)
        ]
        figure_3d(
            traces,
            f"{_seq(s)} frame {s.meta.frame}: dense surfaces",
            height=500,
        ).show()


def scan_overlay(ctx: Bedlam, preset: str = "OS1-64") -> None:
    """Simulate one sensor per frame and draw its returns on the image by

    channel.
    """
    print(f"{'sample':60s} {'HFOV':>7s} {'512':>5s} {'1024':>5s} {'2048':>5s}")
    for s in ctx.samples:
        cols = [
            azimuth_window(s.camera, PRESETS[preset]).columns * m // 1024
            for m in (512, 1024, 2048)
        ]
        print(
            f"{s.meta.key:60s} {camera_hfov_deg(s.camera):6.1f} deg "
            f"{cols[0]:5d} {cols[1]:5d} {cols[2]:5d}"
        )
    fig, axes = plt.subplots(
        len(ctx.samples), 1, figsize=(14, 8 * len(ctx.samples))
    )
    ctx.scans = []
    for ax, s, depth, ppl in zip(
        axes, ctx.samples, ctx.depths, ctx.persons, strict=True
    ):
        scan = simulate(depth, s.camera, PRESETS[preset], rng=ctx.rng)
        ctx.scans.append(scan)
        ax.imshow(s.image)
        hit = np.zeros(_mask(s).shape, bool)
        for p in ppl:
            draw_mask_outline(ax, _mask(p), "yellow")
            hit |= _mask(p)
        on_person = select_mask(scan, hit)
        sc = draw_points(ax, scan.pixel, scan.channel, size=2, cmap="turbo")
        draw_points(
            ax,
            on_person.pixel,
            size=10,
            label=f"on persons: {len(on_person.points)}",
        )
        w = azimuth_window(s.camera, PRESETS[preset])
        ax.legend(loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{s.meta.key}: {preset}, {w.columns} columns in "
            f"{w.hfov_deg:.1f} deg, {len(scan.points)} scene returns",
            fontsize=9,
        )
        fig.colorbar(sc, ax=ax, fraction=0.02, label="channel (0 = top)")
    plt.tight_layout()
    plt.show()


def returns_vs_surface(ctx: Bedlam) -> None:
    """Per frame: the first person's returns (red) against its dense surface

    (grey).
    """
    for scan, ppl in zip(ctx.scans, ctx.persons, strict=True):
        p = ppl[0]
        sel = select_mask(scan, _mask(p))
        traces = [
            points_trace(
                _surface(p, ctx.rng),
                "dense clothed surface",
                color="lightgrey",
                size=1.2,
            ),
            points_trace(
                sel.points, f"returns ({len(sel.points)})", color="red", size=3
            ),
        ]
        figure_3d(
            traces, f"{p.meta.key}: returns vs surface", height=500
        ).show()


def resolution_grid(ctx: Bedlam, sample: int = 0) -> None:
    """OS1 32 to 256 channels from four sensor placements, on one frame and

    in 3D.
    """
    s, depth, ppl = (
        ctx.samples[sample],
        ctx.depths[sample],
        ctx.persons[sample],
    )
    person_mask = np.zeros(_mask(s).shape, bool)
    for p in ppl:
        person_mask |= _mask(p)
    viewpoints = {
        "10 cm above camera": sensor_pose(np.array([0.0, -0.10, 0.0])),
        "1 m right": sensor_pose(np.array([1.0, 0.0, 0.0])),
        "5 m right": sensor_pose(np.array([5.0, 0.0, 0.0])),
        "10 m right": sensor_pose(np.array([10.0, 0.0, 0.0])),
    }
    beams = ["OS1-32", "OS1-64", "OS1-128", "OS1-256"]
    x0, y0, x1, y1 = s.bbox_xyxy
    pad = 60
    fig, axes = plt.subplots(len(beams), len(viewpoints), figsize=(22, 19))
    grid: dict[tuple[str, str], Any] = {}
    for r, name in enumerate(beams):
        for c, (vname, pose) in enumerate(viewpoints.items()):
            sel = select_mask(
                simulate(depth, s.camera, PRESETS[name], pose, ctx.rng),
                person_mask,
            )
            grid[(name, vname)] = sel
            ax = axes[r, c]
            ax.imshow(s.image)
            draw_points(ax, sel.pixel, sel.channel, size=12, cmap="turbo")
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y1 + pad, y0 - pad)
            ax.set_title(
                f"{name}, {vname}: {len(sel.points)} returns", fontsize=9
            )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    for vname in viewpoints:
        traces = [
            points_trace(
                _surface(ppl[0], ctx.rng),
                "dense surface",
                color="lightgrey",
                size=1.0,
            )
        ]
        for name in beams:
            sel = grid[(name, vname)]
            pm = _mask(ppl[0])[sel.pixel[:, 1], sel.pixel[:, 0]]
            traces.append(
                points_trace(
                    sel.points[pm], f"{name} ({int(pm.sum())})", size=3
                )
            )
        figure_3d(
            traces,
            f"{ppl[0].meta.key}: channel counts from {vname}",
            height=520,
        ).show()


def simulator_widget(ctx: Bedlam) -> None:
    """Interactive LiDAR simulator: resolutions, placement, occlusion, noise,

    ego speed.
    """
    import ipywidgets as w
    from IPython.display import clear_output, display

    from lidar_bedlam.data.image_augment import cover_side
    from lidar_bedlam.lidar.augment import (
        SensorCover,
        add_outliers,
        cover_mask,
        drop_channels,
        jitter,
        miscalibrated_pose,
    )
    from lidar_bedlam.lidar.motion import (
        EgoMotion,
        SpeedSetting,
        apply_rolling_shutter,
        sweep_duration_s,
    )
    from lidar_bedlam.lidar.simulate import ouster

    channels, steps = [32, 64, 128, 256], [512, 1024, 2048]
    sample_dd = w.Dropdown(
        options=[
            (f"{s.meta.sequence.split('/')[-1]} f{s.meta.frame}", i)
            for i, s in enumerate(ctx.samples)
        ],
        description="sample",
    )
    person_dd = w.Dropdown(description="person")

    def refresh_persons(*_args: object) -> None:
        person_dd.options = [
            (f"person {p.meta.person}", j)
            for j, p in enumerate(ctx.persons[sample_dd.value])
        ]
        person_dd.value = 0

    sample_dd.observe(refresh_persons, "value")
    refresh_persons()
    boxes = {
        (c, st): w.Checkbox(
            value=(c == 64 and st == 1024),
            indent=False,
            layout=w.Layout(width="70px"),
        )
        for c in channels
        for st in steps
    }
    cells = [w.Label("channels \\ steps")] + [
        w.Label(str(st), layout=w.Layout(width="70px")) for st in steps
    ]
    for c in channels:
        cells += [w.Label(f"{c} channels", layout=w.Layout(width="110px"))] + [
            boxes[(c, st)] for st in steps
        ]
    grid = w.GridBox(
        cells, layout=w.Layout(grid_template_columns="110px 70px 70px 70px")
    )
    family = w.Dropdown(
        options=["OS0", "OS1", "OS2"], value="OS1", description="family"
    )
    viewpoint = w.Dropdown(
        options=[("camera", 0), ("10 cm above", 1), ("1 m right", 2)],
        description="viewpoint",
    )
    cam_side = w.Dropdown(
        options=["none", "left", "right", "top", "bottom"],
        description="camera cover",
    )
    cam_frac = w.FloatSlider(
        0.3,
        min=0.0,
        max=0.8,
        step=0.05,
        description="cover frac",
        continuous_update=False,
    )
    sides = {
        k: w.FloatSlider(
            0.0,
            min=0.0,
            max=0.9,
            step=0.05,
            description=f"LiDAR {k}",
            continuous_update=False,
        )
        for k in ("left", "right", "top", "bottom")
    }
    obj_occ = w.Dropdown(
        options=["none", "legs", "head", "left side", "right side", "front"],
        description="object occl.",
    )
    obj_frac = w.FloatSlider(
        0.4,
        min=0.05,
        max=0.9,
        step=0.05,
        description="occl. frac",
        continuous_update=False,
    )
    jitter_s = w.FloatSlider(
        0.0,
        min=0.0,
        max=0.1,
        step=0.005,
        description="jitter [m]",
        continuous_update=False,
        readout_format=".3f",
    )
    outlier_s = w.FloatSlider(
        0.0,
        min=0.0,
        max=0.3,
        step=0.01,
        description="outliers",
        continuous_update=False,
    )
    chdrop_s = w.FloatSlider(
        0.0,
        min=0.0,
        max=0.8,
        step=0.05,
        description="chan. dropout",
        continuous_update=False,
    )
    miscal_s = w.FloatSlider(
        0.0,
        min=0.0,
        max=3.0,
        step=0.1,
        description="miscalib [deg]",
        continuous_update=False,
    )
    speed_mean = w.IntSlider(
        0,
        min=0,
        max=130,
        step=10,
        description="speed E [km/h]",
        continuous_update=False,
    )
    speed_std = w.IntSlider(
        0,
        min=0,
        max=60,
        step=10,
        description="speed SD [km/h]",
        continuous_update=False,
    )
    spin_hz = w.Dropdown(options=[10, 20], value=10, description="spin [Hz]")
    out = w.Output()

    def render(*_args: object) -> None:
        with out:
            clear_output(wait=True)
            if sample_dd.value is None or person_dd.value is None:
                return
            s, depth = (
                ctx.samples[sample_dd.value],
                ctx.depths[sample_dd.value],
            )
            p = ctx.persons[sample_dd.value][person_dd.value]
            pose = [
                np.eye(4),
                sensor_pose(np.array([0.0, -0.1, 0.0])),
                sensor_pose(np.array([1.0, 0.0, 0.0])),
            ][viewpoint.value]
            rng_i = np.random.default_rng(0)
            if miscal_s.value > 0:
                pose = miscalibrated_pose(
                    pose, miscal_s.value, 0.01 * miscal_s.value, rng_i
                )
            image = (
                s.image
                if cam_side.value == "none"
                else cover_side(s.image, cam_side.value, cam_frac.value)
            )
            cover = SensorCover(
                sides["left"].value,
                sides["right"].value,
                sides["top"].value,
                sides["bottom"].value,
            )
            results: dict[tuple[int, int], tuple[Any, Any, Any, int]] = {}
            spec = ouster(family.value, 64, 1024)
            win = azimuth_window(s.camera, spec, pose)
            speed = 0.0
            for (c, st), cb in boxes.items():
                if not cb.value:
                    continue
                spec = ouster(family.value, c, st)
                scan = select_mask(
                    simulate(depth, s.camera, spec, pose, rng_i), _mask(p)
                )
                win = azimuth_window(s.camera, spec, pose)
                speed = SpeedSetting(
                    float(speed_mean.value), float(speed_std.value)
                ).sample_mps(np.random.default_rng(1))
                scan = apply_rolling_shutter(
                    scan,
                    EgoMotion(speed_mps=speed, spin_hz=float(spin_hz.value)),
                    win,
                )
                keep = cover_mask(
                    scan.channel,
                    scan.azimuth_deg,
                    c,
                    win.az_min_deg,
                    win.az_max_deg,
                    cover,
                )
                if chdrop_s.value > 0:
                    keep &= drop_channels(
                        scan.channel, c, chdrop_s.value, rng_i
                    )
                pts: NDArray[Any] = scan.points[keep].astype(np.float64)
                pix, ch = scan.pixel[keep], scan.channel[keep]
                if obj_occ.value != "none" and len(pts):
                    k = _occlude(pts, obj_occ.value, obj_frac.value)
                    pts, pix, ch = pts[k], pix[k], ch[k]
                pts = jitter(pts, jitter_s.value, rng_i)
                if outlier_s.value > 0:
                    pts = add_outliers(pts, outlier_s.value, 0.3, rng_i)
                results[(c, st)] = (pts, pix, ch, win.columns)
            if speed_mean.value or speed_std.value:
                dur = sweep_duration_s(win, spec, float(spin_hz.value))
                print(
                    f"rolling shutter: ego {speed * 3.6:.0f} km/h, sweep "
                    f"over the window {dur * 1000:.1f} ms, up to "
                    f"{speed * dur * 100:.1f} cm shift"
                )
            n = max(len(results), 1)
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 7 * nrows), squeeze=False
            )
            x0, y0, x1, y1 = p.bbox_xyxy
            pad = 60
            for ax, ((c, st), (pts, pix, ch, ncol)) in zip(
                axes.flat, results.items(), strict=False
            ):
                ax.imshow(image)
                draw_mask_outline(ax, _mask(p), "yellow")
                if len(pix):
                    draw_points(ax, pix, ch, size=14, cmap="turbo")
                ax.set_xlim(x0 - pad, x1 + pad)
                ax.set_ylim(y1 + pad, y0 - pad)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"{family.value}-{c} @ {st} steps ({ncol} cols): "
                    f"{len(pts)} pts"
                )
            for ax in list(axes.flat)[len(results) :]:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            traces = [
                points_trace(
                    _surface(p, ctx.rng),
                    "dense surface",
                    color="lightgrey",
                    size=1.0,
                )
            ]
            for (c, st), (pts, _, _, _) in results.items():
                if len(pts):
                    traces.append(
                        points_trace(
                            pts,
                            f"{family.value}-{c} @ {st} ({len(pts)})",
                            size=3,
                        )
                    )
            figure_3d(
                traces, f"{p.meta.key}: selected resolutions", height=600
            ).show()

    for widget in [
        sample_dd,
        person_dd,
        family,
        viewpoint,
        cam_side,
        cam_frac,
        *sides.values(),
        obj_occ,
        obj_frac,
        jitter_s,
        outlier_s,
        chdrop_s,
        miscal_s,
        speed_mean,
        speed_std,
        spin_hz,
        *boxes.values(),
    ]:
        widget.observe(render, "value")
    ui = w.VBox([
        w.HBox([sample_dd, person_dd, family, viewpoint]), grid,
        w.HBox([cam_side, cam_frac]),
        w.HBox([sides["left"], sides["right"]]),
        w.HBox([sides["top"], sides["bottom"]]),
        w.HBox([obj_occ, obj_frac]),
        w.HBox([jitter_s, outlier_s]),
        w.HBox([chdrop_s, miscal_s]),
        w.HBox([speed_mean, speed_std, spin_hz]),
    ])  # fmt: skip
    display(ui, out)
    render()


@dataclass
class RealSets:
    """One loaded sample per real dataset plus the sources themselves."""

    sources: dict[str, Any]
    samples: dict[str, Sample]
    smpl: SmplModel


def load_real(ctx: Bedlam, sloper_seq: str = "seq008_running_001") -> RealSets:
    """Load one sample each from SLOPER4D, LiDARHuman26M and Waymo and show the

    crops.
    """
    from lidar_bedlam.data.lidarhuman26m import LidarHuman26mSource
    from lidar_bedlam.data.sloper4d import Sloper4dSource
    from lidar_bedlam.data.waymo import WaymoSource

    smpl = ctx.smpl
    real = {
        "sloper4d": (
            Sloper4dSource(DATA / "sloper4d", smpl, [sloper_seq]),
            100,
        ),
        "lidarhuman26m": (
            LidarHuman26mSource(DATA / "lidarhuman26m", smpl, "test"),
            500,
        ),
        "waymo": (
            WaymoSource(
                DATA / "waymo_perception" / "waymo_pose_complete_4", "val"
            ),
            3,
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    samples: dict[str, Sample] = {}
    for ax, (name, (source, idx)) in zip(axes, real.items(), strict=True):
        rs = source.load(idx)
        add_smpl_derived(rs, smpl)
        samples[name] = rs
        c = crop_sample(rs, 512)
        ax.imshow(c.image)
        if "vertices" in rs.extra:
            draw_points(
                ax,
                c.camera.project(rs.extra["vertices"]),
                size=1,
                label="SMPL vertices",
            )
        if rs.joints3d is not None and rs.joints3d_valid is not None:
            uv = c.camera.project(rs.joints3d[rs.joints3d_valid])
            ax.scatter(
                uv[:, 0],
                uv[:, 1],
                s=40,
                c="lime",
                marker="x",
                label=f"joints ({rs.joint_convention})",
            )
        pv = c.camera.project(rs.points.astype(np.float64))
        ax.scatter(
            pv[:, 0],
            pv[:, 1],
            s=6,
            c="cyan",
            label=f"LiDAR ({len(rs.points)})",
        )
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.legend(loc="lower right")
        ax.set_title(f"{name}: {rs.meta.key}", fontsize=9)
    plt.tight_layout()
    plt.show()
    return RealSets({k: v[0] for k, v in real.items()}, samples, smpl)


def real_points_vs_mesh(real: RealSets) -> None:
    """Each real sample in 3D: returns coloured by distance to the labelled

    mesh.
    """
    for name, rs in real.samples.items():
        traces = []
        if "vertices" in rs.extra:
            v = rs.extra["vertices"]
            traces.append(
                mesh_trace(v, real.smpl.faces, "GT SMPL mesh", opacity=0.5)
            )
            d = point_to_surface_distance(rs.points, v)
            traces.append(
                points_trace(
                    rs.points,
                    "LiDAR (colour = distance)",
                    color=d,
                    size=3,
                    colorscale="Turbo",
                    colorbar_title="m",
                )
            )
            print(
                f"{name}: LiDAR-to-mesh median {np.median(d):.3f} m, "
                f"mean {d.mean():.3f} m, p90 {np.percentile(d, 90):.3f} m"
            )
        else:
            traces.append(
                points_trace(rs.points, "LiDAR", color="red", size=3)
            )
            if rs.joints3d is not None and rs.joints3d_valid is not None:
                traces.append(
                    points_trace(
                        rs.joints3d[rs.joints3d_valid],
                        "GT keypoints",
                        color="lime",
                        size=5,
                    )
                )
        if rs.box3d is not None:
            traces.append(box_trace(rs.box3d, "3D box"))
        figure_3d(traces, f"{name}: {rs.meta.key}", height=520).show()


def training_batch(real: RealSets) -> dict[str, Any]:
    """Collate four records from the real sources and evaluate the loss on

    their labels.
    """
    from torch.utils.data import DataLoader, Subset

    from lidar_bedlam.data.torch_dataset import HumanPoseDataset
    from lidar_bedlam.losses.smpl import FusionLoss

    sources = [
        real.sources["sloper4d"],
        real.sources["lidarhuman26m"],
        real.sources["waymo"],
    ]
    ds = HumanPoseDataset(sources, real.smpl, out_size=256, n_points=1024)
    ids = [
        100,
        len(sources[0]) + 500,
        len(sources[0]) + len(sources[1]) + 3,
        101,
    ]
    batch: dict[str, Any] = next(
        iter(DataLoader(Subset(ds, ids), batch_size=len(ids)))
    )
    for k, v in batch.items():
        print(f"{k:18s}", tuple(v.shape) if hasattr(v, "shape") else v)
    pred_gt = {
        k: batch[k]
        for k in [
            "global_orient",
            "body_pose",
            "betas",
            "transl",
            "joints3d",
            "box3d",
        ]
    }
    pred_gt["kp2d"] = batch["kp2d"][..., :2]
    total, parts = FusionLoss()(pred_gt, batch)
    print(
        "loss on ground truth:",
        float(total),
        {k: round(float(v), 5) for k, v in parts.items()},
    )
    return batch


def model_forward(batch: dict[str, Any]) -> dict[str, Any]:
    """Build the fusion model with the TokenHMR ViT, run the batch, print the

    loss and the gate priors.
    """
    from lidar_bedlam.losses.smpl import FusionLoss
    from lidar_bedlam.models.fusion import ModelConfig, SelectiveFusionModel
    from lidar_bedlam.models.selective_attention import JOINT_GROUPS
    from lidar_bedlam.models.vit import VIT_H, load_backbone_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SelectiveFusionModel(
        ModelConfig(vit=VIT_H, smpl_model_dir=GEN / "body_models")
    )
    ckpt = ROOT / "resources" / "pretrained-checkpoints" / "tokenhmr_vith"
    if ckpt.exists() and model.backbone is not None:
        load_backbone_weights(model.backbone, ckpt)
    model = model.to(device).eval()
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"parameters: {n_all / 1e6:.1f} M total, "
        f"{n_train / 1e6:.1f} M trainable (backbone frozen)"
    )
    with torch.no_grad():
        out = model(
            {k: batch[k].to(device) for k in ["image", "points", "intrinsics"]}
        )
    for k, v in out.items():
        print(f"{k:14s}", tuple(v.shape))
    total, parts = FusionLoss()({k: v.cpu() for k, v in out.items()}, batch)
    print(
        "untrained loss:",
        float(total),
        {k: round(float(v), 4) for k, v in parts.items()},
    )
    gates = out["gates"].mean(1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(11, 3))
    im = ax.imshow(gates, vmin=0, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(JOINT_GROUPS)))
    ax.set_xticklabels([g.name for g in JOINT_GROUPS], rotation=45, ha="right")
    ax.set_yticks(range(gates.shape[0]))
    ax.set_ylabel("decoder layer")
    fig.colorbar(im, ax=ax, label="image gate (1 = image, 0 = LiDAR)")
    ax.set_title("selective-attention gates at initialisation")
    plt.tight_layout()
    plt.show()
    return {k: v.cpu() for k, v in out.items()}


def sensor_rigs(max_rigs: int = 3) -> None:
    """Every dataset's calibration tree, and a checkbox table drawing up to

    three rigs.
    """
    import ipywidgets as w
    from IPython.display import clear_output, display

    from lidar_bedlam.rigs.plot import rigs_figure
    from lidar_bedlam.rigs.registry import load_all_rigs

    loaded, skipped = load_all_rigs(DATA)
    rigs = {rig.title: rig for rig in loaded.values()}
    for slug, why in skipped.items():
        print("skipped:", slug, why)
    for rig in rigs.values():
        print(rig.tree_text())
        print()
    boxes = {
        t: w.Checkbox(
            value=(i < max_rigs), indent=False, layout=w.Layout(width="40px")
        )
        for i, t in enumerate(rigs)
    }
    rows = [
        w.HBox(
            [
                w.Label("show", layout=w.Layout(width="40px")),
                w.Label("dataset (mount)", layout=w.Layout(width="220px")),
                w.Label("lidars", layout=w.Layout(width="60px")),
                w.Label("cameras", layout=w.Layout(width="70px")),
                w.Label("base frame"),
            ]
        )
    ]
    for title, rig in rigs.items():
        rows.append(
            w.HBox(
                [
                    boxes[title],
                    w.Label(title, layout=w.Layout(width="220px")),
                    w.Label(
                        str(len(rig.by_kind("lidar"))),
                        layout=w.Layout(width="60px"),
                    ),
                    w.Label(
                        str(len(rig.by_kind("camera"))),
                        layout=w.Layout(width="70px"),
                    ),
                    w.Label(rig.base_frame),
                ]
            )
        )
    out = w.Output()

    def render(*_args: object) -> None:
        with out:
            clear_output(wait=True)
            chosen = [rigs[t] for t, cb in boxes.items() if cb.value]
            if not chosen:
                print("tick at least one rig")
                return
            rigs_figure(chosen[:max_rigs]).show()

    for cb in boxes.values():
        cb.observe(render, "value")
    display(w.VBox(rows), out)
    render()


def _load_shards(pattern: str) -> list[Any]:
    from lidar_bedlam.generate.records import Shard

    paths = sorted(
        p
        for p in GEN.glob(pattern)
        if ".fit." not in p.name and not p.name.endswith("stats.npz")
    )
    return [Shard(p) for p in paths]


def _project(k: NDArray[Any], pts: NDArray[Any]) -> NDArray[Any]:
    z = np.maximum(pts[:, 2], 1e-3)
    return np.stack(
        [k[0, 0] * pts[:, 0] / z + k[0, 2], k[1, 1] * pts[:, 1] / z + k[1, 2]],
        1,
    )


def _locate(shards: list[Any], g: int) -> tuple[Any, int]:
    si, i = 0, g
    while i >= len(shards[si]):
        i -= len(shards[si])
        si += 1
    return shards[si], i


def mesh_lidar_3dpw(
    smpl: SmplModel, pattern: str = "meshlidar/v1/threedpw_*.npz", n: int = 4
) -> None:
    """3DPW records with LiDAR simulated on the clothing-offset mesh; a re-roll

    button picks new ones.
    """
    import ipywidgets as w
    from IPython.display import clear_output, display

    shards = _load_shards(pattern)
    total = sum(len(s) for s in shards)
    state = {"roll": 0}
    btn = w.Button(description="re-roll", icon="refresh")
    out = w.Output()

    def render(*_args: object) -> None:
        with out:
            clear_output(wait=True)
            if not shards:
                print(
                    "no 3DPW shards here: run generate_3dpw.py "
                    "or pull them from the cluster"
                )
                return
            rng = np.random.default_rng(1000 + state["roll"])
            picks = rng.integers(0, total, n)
            print(
                f"set {state['roll']}: {total} records, showing", list(picks)
            )
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
            traces: list[Any] = []
            for j, (ax, g) in enumerate(zip(axes, picks, strict=True)):
                sh, i = _locate(shards, int(g))
                k = sh.array("intrinsics")[i]
                ax.imshow(sh.array("image")[i])
                draw_mask_outline(ax, sh.array("mask")[i], "yellow")
                sc = sh.scan(i, "main_0")
                uv = _project(k, sc.points)
                sp = ax.scatter(
                    uv[:, 0], uv[:, 1], c=sc.points[:, 2], s=8, cmap="turbo"
                )
                key = str(sh.array("key")[i]).split("/")
                ax.set_title(
                    f"{key[2]} f{key[3]} p{key[4]}\n{sc.channels}ch x "
                    f"{sc.steps}: {len(sc.points)} returns",
                    fontsize=9,
                )
                ax.set_xlim(0, 256)
                ax.set_ylim(256, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    v, _ = smpl.forward(
                        SmplParams(
                            sh.row("global_orient", i),
                            sh.row("body_pose", i),
                            sh.row("betas", i),
                            sh.row("transl", i),
                        )
                    )
                    traces += [
                        mesh_trace(v, smpl.faces, "SMPL label", opacity=0.45),
                        points_trace(
                            sc.points,
                            "returns (clothing offset)",
                            color=sc.points[:, 2],
                            size=3,
                            colorscale="Turbo",
                        ),
                    ]
            fig.colorbar(sp, ax=axes[-1], fraction=0.03, label="depth [m]")
            plt.tight_layout()
            plt.show()
            figure_3d(
                traces, "first sample: label mesh and simulated returns"
            ).show()
        state["roll"] += 1

    btn.on_click(render)
    display(btn, out)
    render()


def pseudo_gt_waymo(
    smpl: SmplModel, labels: str = "real/v1_pseudo3", n: int = 3
) -> None:
    """Waymo records with their pseudo-GT SMPL fit on the crop and in the

    returns; re-roll picks new ones.
    """
    import ipywidgets as w
    from IPython.display import clear_output, display

    shards = _load_shards(f"{labels}/waymo_train_*.npz")
    total = sum(len(s) for s in shards)
    j_coco = np.load(GEN / "body_models" / "J_regressor_coco.npy")
    state = {"roll": 0}
    btn = w.Button(description="re-roll", icon="refresh")
    out = w.Output()

    def render(*_args: object) -> None:
        with out:
            clear_output(wait=True)
            if not shards:
                print(f"no shards under {labels}")
                return
            rng = np.random.default_rng(2000 + state["roll"])
            picks = rng.integers(0, total, n)
            print(
                f"set {state['roll']}: {total} records, showing", list(picks)
            )
            fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.5))
            traces: list[Any] = []
            for j, (ax, g) in enumerate(zip(axes, picks, strict=True)):
                sh, i = _locate(shards, int(g))
                k = sh.array("intrinsics")[i]
                v, _ = smpl.forward(
                    SmplParams(
                        sh.row("global_orient", i),
                        sh.row("body_pose", i),
                        sh.row("betas", i),
                        sh.row("transl", i),
                    )
                )
                coco = j_coco @ v
                kp = sh.array("kp2d")[i]
                ok = kp[:15, 2] > 0
                err = (
                    float(sh.array("pseudo_kp_error_m")[i]) * 1000
                    if sh.has("pseudo_kp_error_m")
                    else float("nan")
                )
                ax.imshow(sh.array("image")[i])
                uvv = _project(k, v[::10])
                ax.scatter(uvv[:, 0], uvv[:, 1], s=1, c="red", alpha=0.4)
                ax.scatter(
                    kp[:15][ok, 0],
                    kp[:15][ok, 1],
                    s=25,
                    c="lime",
                    label="Waymo kp",
                )
                uvc = _project(k, coco[CSEL])
                ax.scatter(
                    uvc[:, 0],
                    uvc[:, 1],
                    s=25,
                    c="red",
                    marker="x",
                    label="fitted COCO",
                )
                ax.set_title(
                    f"{str(sh.array('key')[i]).split('/')[-1][:24]}\n"
                    f"accepted={bool(sh.array('has_smpl')[i])}, "
                    f"kp err {err:.0f} mm",
                    fontsize=9,
                )
                ax.set_xlim(0, 256)
                ax.set_ylim(256, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.legend(loc="lower left", fontsize=8)
                    sc = sh.scan(i, "real")
                    traces += [
                        mesh_trace(
                            v, smpl.faces, "pseudo-GT SMPL", opacity=0.5
                        ),
                        points_trace(
                            sc.points, "LiDAR returns", color="black", size=3
                        ),
                        points_trace(
                            sh.array("joints3d")[i][:15][
                                sh.array("joints3d_valid")[i][:15]
                            ],
                            "Waymo keypoints",
                            color="lime",
                            size=6,
                        ),
                    ]
            plt.tight_layout()
            plt.show()
            figure_3d(
                traces, "first sample: fitted mesh in the point cloud"
            ).show()
        state["roll"] += 1

    btn.on_click(render)
    display(btn, out)
    render()


def architecture_figure(width: int = 1400) -> None:
    """The block diagram of the model from .docs/figures."""
    from IPython.display import Image, display

    display(
        Image(
            filename=str(ROOT / ".docs" / "figures" / "architecture.png"),
            width=width,
        )
    )


def scoreboard(pull: bool = True, groups: tuple[int, ...] = (0,)) -> None:
    """Pull the Helma results and render the chosen scoreboard groups as HTML

    tables.
    """
    from IPython.display import HTML, display

    from lidar_bedlam.scripts import build_scoreboard as sb

    if pull:
        subprocess.run(
            [
                "bash",
                str(ROOT / "lidar_bedlam/scripts/pull_helma_results.sh"),
                str(ROOT / "outputs/helma"),
            ],
            capture_output=True,
            check=False,
        )
    helma, base = ROOT / "outputs" / "helma", ROOT / "outputs" / "baselines"
    display(HTML(f"<style>{sb.STYLE}</style>"))
    for gi in groups:
        g = sb.GROUPS[gi]
        rows = sb.group_rows(helma, g.rows, g.pending)
        if gi == 0:
            rows = (
                sb.static_rows(base / "results_static.json", sb.STATIC_ROWS)
                + rows
            )
        display(
            HTML(f"<h3>{g.title}</h3><p>{g.desc}</p>" + sb.table_html(rows))
        )


@dataclass
class Showcase:
    """Our predictions on the validation sets next to the baselines'."""

    config: Path
    paths: dict[str, Path]
    sets: dict[str, tuple[Any, Any, Any, Any]]
    smpl: SmplModel


def run_showcase(checkpoint: str, config: str, out_dir: str) -> Showcase:
    """Predict every validation record with one checkpoint and pick the records

    to show.
    """
    from lidar_bedlam.utils import showcase as sc

    cfg, ckpt, out = ROOT / config, ROOT / checkpoint, ROOT / out_dir
    smpl = SmplModel(GEN / "body_models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = (
        sc.run_model(cfg, ckpt, out, device=device) if ckpt.exists() else {}
    )
    if not paths:
        print("checkpoint missing:", ckpt)
    sets: dict[str, tuple[Any, Any, Any, Any]] = {}
    for split, path in paths.items():
        tables = sc.load_tables(path, ROOT / "outputs" / "baselines", split)
        per_record = sc.per_record_metrics(tables, cfg, split, smpl)
        index = sc.ValIndex(cfg, split)
        picks = sc.pick_records(per_record, split)
        sets[split] = (tables, per_record, index, picks)
        print(split, {n: len(m) for n, m in per_record.items()})
        for p in picks:
            print("  pick:", p.key, "-", p.reason)
    return Showcase(cfg, paths, sets, smpl)


def showcase_figures(show: Showcase) -> None:
    """Bird's-eye and side views of the picked records, every method's mesh,

    plus one 3D view per set.
    """
    from lidar_bedlam.utils import showcase as sc

    for split, (tables, per_record, index, picks) in show.sets.items():
        for p in picks:
            sc.comparison_figure_interactive(
                index.row(p.key),
                tables,
                per_record,
                show.smpl,
                title=f"{split}: {p.reason}",
            ).show()
        if not picks:
            continue
        rec = index.row(picks[0].key)
        traces = [
            points_trace(rec["points"], "LiDAR returns", color="black", size=2)
        ]
        if rec["has_smpl"]:
            v_gt, _ = show.smpl.forward(rec["smpl"])
            traces.append(
                mesh_trace(
                    v_gt,
                    show.smpl.faces,
                    "GT SMPL",
                    color="limegreen",
                    opacity=0.35,
                )
            )
        for name, t in tables.items():
            if picks[0].key in t.index:
                traces.append(
                    mesh_trace(
                        t.vertices[t.index[picks[0].key]].astype(np.float64),
                        show.smpl.faces,
                        name,
                        color=sc.COLORS.get(name, "grey"),
                        opacity=0.45,
                    )
                )
        figure_3d(traces, f"{split}: {picks[0].reason}").show()


def transfer_figure(runs: dict[str, str]) -> None:
    """Validation curves of real-only, synth-only and mixed trainings from the

    pulled metrics.
    """
    from lidar_bedlam.utils import showcase as sc

    sc.transfer_figure(ROOT / "outputs" / "helma", runs)
    plt.show()


def distance_and_gates(show: Showcase, out_dir: str) -> None:
    """Waymo placement error per distance bin, and the learned gates of the

    showcased checkpoint.
    """
    from lidar_bedlam.models.selective_attention import JOINT_GROUPS
    from lidar_bedlam.utils import showcase as sc

    res = ROOT / out_dir / "results.json"
    if not res.exists() and show.paths:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(ROOT / "lidar_bedlam/scripts/score_baselines.py"),
                "--config",
                str(show.config),
                "--pred",
                *[str(p) for p in show.paths.values()],
                "--out",
                str(res),
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    sc.distance_figure(
        ROOT / "outputs" / "baselines" / "results_static.json",
        res if res.exists() else None,
    )
    plt.show()
    for split, path in show.paths.items():
        g = np.load(path)["gates"].astype(np.float64)
        sc.gates_figure(
            g,
            [gr.name for gr in JOINT_GROUPS],
            f"learned gates on {split} (mean over {len(g)} records)",
        )
        plt.show()


def sam3d_conversion(
    npz: str = "outputs/baselines/sam3d-body/smoke/mhr_vs_smpl.npz",
) -> None:
    """SAM 3D Body's MHR mesh against the SMPL fitted to it, front and side,

    with the vertex distances.
    """
    from scipy.spatial import cKDTree

    z = np.load(ROOT / npz)
    keys = [str(k) for k in z["keys"]]
    mhr, smpl_v = z["mhr_vertices"].astype(np.float64), z["smpl_vertices"]
    d_ms = [
        cKDTree(s).query(m)[0].mean() for m, s in zip(mhr, smpl_v, strict=True)
    ]
    d_sm = [
        cKDTree(m).query(s)[0].mean() for m, s in zip(mhr, smpl_v, strict=True)
    ]
    print(
        f"{len(keys)} crops: MHR to SMPL {np.mean(d_ms) * 100:.2f} cm, "
        f"SMPL to MHR {np.mean(d_sm) * 100:.2f} cm (mean nearest vertex)"
    )
    pick = [0, 1, len(keys) - 1]
    fig = plt.figure(figsize=(12, 8))
    for c, i in enumerate(pick):
        m, s = mhr[i], smpl_v[i]
        for r, (a, b, view) in enumerate(((0, 1, "front"), (2, 1, "side"))):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1)
            ax.scatter(m[:, a], -m[:, b], s=0.4, c="0.55", label="MHR")
            ax.scatter(s[:, a], -s[:, b], s=0.4, c="#0E6B64", label="SMPL")
            ax.set_aspect("equal")
            ax.set_title(f"{keys[i].split('/')[0]} #{i}, {view}", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(markerscale=12, fontsize=8, loc="lower right")
    fig.tight_layout()
    plt.show()


def waymo_header(
    folder: str = "outputs/figures/waymo_header", width: int = 1400
) -> None:
    """A whole Waymo sweep with our meshes: image and 3D renders from the

    header figure folder.
    """
    from IPython.display import Image, display

    for name in ("header_image.png", "header_sweep.png"):
        display(Image(filename=str(ROOT / folder / name), width=width))
