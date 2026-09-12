"""Export rigs for documentation: PNG, interactive HTML and GLB 3D files.

The GLB is a real mesh (axes as coloured cylinders, links as thin grey
rods, camera frustums as orange rods), viewable in any glTF viewer and in
Obsidian through a 3D-model plugin. The HTML is the plotly figure and
references a shared ``plotly.min.js`` next to it, so it stays small.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.rigs.plot import rig_traces, rigs_figure
from lidar_bedlam.rigs.schema import Sensor, SensorRig

FloatArray = NDArray[np.float64]
AXIS_RGBA = ((220, 40, 40, 255), (40, 170, 40, 255), (40, 70, 220, 255))
LINK_RGBA = (140, 140, 140, 255)
FRUSTUM_RGBA = (255, 160, 20, 255)


def _rod(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    radius: float,
    rgba: tuple[int, int, int, int],
) -> Any:
    """Cylinder from a to b with a flat colour."""
    import trimesh

    a3, b3 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b3 - a3))
    if length < 1e-9:
        return None
    rod = trimesh.creation.cylinder(radius=radius, segment=np.stack([a3, b3]))
    rod.visual.vertex_colors = np.tile(
        np.array(rgba, dtype=np.uint8), (len(rod.vertices), 1)
    )
    return rod


def _scale(rig: SensorRig) -> float:
    """Axis length adapted to the rig size (cars 0.5 m, helmets 0.1 m)."""
    span = max(
        (float(np.linalg.norm(s.position)) for s in rig.sensors), default=1.0
    )
    return float(np.clip(span * 0.2, 0.05, 0.6))


def _frustum_corners(sensor: Sensor, depth: float) -> FloatArray:
    k = sensor.intrinsics
    assert k is not None
    xs = np.array([0, k.width, k.width, 0]) - k.cx
    ys = np.array([0, 0, k.height, k.height]) - k.cy
    corners_opt = np.stack(
        [xs / k.fx * depth, ys / k.fy * depth, np.full(4, depth)], 1
    )
    sensor_from_optical = sensor.optical_from_sensor.T
    return np.asarray(
        corners_opt @ sensor_from_optical.T @ sensor.rotation.T
        + sensor.position,
        dtype=np.float64,
    )


def rig_mesh(rig: SensorRig) -> Any:
    """``trimesh.Scene`` of the rig in its base frame (metres)."""
    import trimesh

    axis = _scale(rig)
    radius = axis * 0.03
    scene = trimesh.Scene()
    eye = np.eye(3)
    for i in range(3):
        scene.add_geometry(
            _rod(
                np.zeros(3), eye[:, i] * axis * 1.5, radius * 1.5, AXIS_RGBA[i]
            ),
            node_name=f"base_{'xyz'[i]}",
        )
    for s in rig.sensors:
        link = _rod(np.zeros(3), s.position, radius * 0.5, LINK_RGBA)
        if link is not None:
            scene.add_geometry(link, node_name=f"link_{s.name}")
        for i in range(3):
            scene.add_geometry(
                _rod(
                    s.position,
                    s.position + s.rotation[:, i] * axis,
                    radius,
                    AXIS_RGBA[i],
                ),
                node_name=f"{s.name}_{'xyz'[i]}",
            )
        if s.kind == "camera" and s.intrinsics is not None:
            corners = _frustum_corners(s, axis * 2.0)
            for j in range(4):
                scene.add_geometry(
                    _rod(s.position, corners[j], radius * 0.4, FRUSTUM_RGBA),
                    node_name=f"{s.name}_ray{j}",
                )
                scene.add_geometry(
                    _rod(
                        corners[j],
                        corners[(j + 1) % 4],
                        radius * 0.4,
                        FRUSTUM_RGBA,
                    ),
                    node_name=f"{s.name}_edge{j}",
                )
    return scene


def export_png(rig: SensorRig, path: Path, dpi: int = 130) -> None:
    """Static matplotlib 3D view (isometric-ish) with labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = _scale(rig)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for tr in rig_traces(rig, axis_len=axis, frustum_depth=axis * 2.0):
        if tr.mode == "lines":
            ax.plot(
                tr.x, tr.y, tr.z, color=tr.line.color, lw=tr.line.width / 2
            )
        elif tr.mode == "text":
            ax.text(tr.x[0], tr.y[0], tr.z[0], tr.text[0], fontsize=7)
    pts = np.array([s.position for s in rig.sensors] + [np.zeros(3)])
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()), axis * 2.0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(rig.title)
    ax.view_init(elev=25, azim=-135)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_html(
    rig: SensorRig, path: Path, plotly_js: str = "plotly.min.js"
) -> None:
    """Interactive figure referencing a shared plotly.min.js next to it."""
    fig = rigs_figure([rig], height=700)
    fig.write_html(path, include_plotlyjs=plotly_js, full_html=True)


def write_plotly_js(directory: Path) -> Path:
    """Copy plotly's bundled JS once into ``directory``."""
    from plotly.offline import get_plotlyjs

    out = directory / "plotly.min.js"
    if not out.exists():
        out.write_text(get_plotlyjs(), encoding="utf-8")
    return out


def export_rig(rig: SensorRig, directory: Path, slug: str) -> dict[str, Path]:
    """Write ``<slug>.{png,html,glb,txt}`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    write_plotly_js(directory)
    out = {
        "png": directory / f"{slug}.png",
        "html": directory / f"{slug}.html",
        "glb": directory / f"{slug}.glb",
        "txt": directory / f"{slug}.txt",
    }
    export_png(rig, out["png"])
    export_html(rig, out["html"])
    rig_mesh(rig).export(out["glb"])
    out["txt"].write_text(rig.tree_text() + "\n", encoding="utf-8")
    return out
