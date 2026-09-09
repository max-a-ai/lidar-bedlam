"""Plotly drawing of sensor rigs: base link, sensor triads, frustums."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.rigs.schema import Sensor, SensorRig

FloatArray = NDArray[np.float64]
AXIS_COLORS = ("red", "green", "blue")  # x, y, z


def _line(
    points: Sequence[NDArray[np.floating]],
    color: str,
    width: float,
    name: str,
    show: bool = False,
    dash: str | None = None,
) -> Any:
    import plotly.graph_objects as go

    p = np.asarray(points, dtype=np.float64)
    line: dict[str, Any] = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", line=line,
        name=name, showlegend=show, hoverinfo="name",
    )  # fmt: skip


def triad_traces(
    origin: FloatArray, rotation: FloatArray, length: float, label: str
) -> list[Any]:
    """Three axis lines (x red, y green, z blue) plus a text label."""
    import plotly.graph_objects as go

    traces = [
        _line(
            [origin, origin + rotation[:, i] * length],
            AXIS_COLORS[i],
            5,
            f"{label} {'xyz'[i]}",
        )
        for i in range(3)
    ]
    traces.append(
        go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode="text",
            text=[label],
            textposition="top center",
            showlegend=False,
            hoverinfo="text",
        )  # fmt: skip
    )
    return traces


def frustum_trace(sensor: Sensor, depth: float) -> Any | None:
    """Pyramid of a camera's field of view, in the base frame."""
    k = sensor.intrinsics
    if k is None:
        return None
    # corners in the OpenCV optical frame at the given depth
    xs = np.array([0, k.width, k.width, 0]) - k.cx
    ys = np.array([0, 0, k.height, k.height]) - k.cy
    corners_opt = np.stack(
        [xs / k.fx * depth, ys / k.fy * depth, np.full(4, depth)], 1
    )
    sensor_from_optical = sensor.optical_from_sensor.T
    corners = (
        corners_opt @ sensor_from_optical.T @ sensor.rotation.T
        + sensor.position
    )
    o = sensor.position
    segs: list[FloatArray] = []
    nan = np.full(3, np.nan)
    for i in range(4):
        segs += [o, corners[i], nan, corners[i], corners[(i + 1) % 4], nan]
    return _line(segs, "orange", 2, f"{sensor.name} FOV")


def rig_traces(
    rig: SensorRig, axis_len: float = 0.5, frustum_depth: float = 1.0
) -> list[Any]:
    """All traces of one rig."""
    traces = triad_traces(np.zeros(3), np.eye(3), axis_len * 1.5, "base_link")
    for s in rig.sensors:
        traces.append(
            _line(
                [np.zeros(3), s.position],
                "grey",
                2,
                f"link {s.name}",
                dash="dot",
            )
        )
        traces += triad_traces(s.position, s.rotation, axis_len, s.name)
        if s.kind == "camera":
            f = frustum_trace(s, frustum_depth)
            if f is not None:
                traces.append(f)
    return traces


def rigs_figure(rigs: Sequence[SensorRig], height: int = 650) -> Any:
    """Up to three rigs side by side, each in its own 3D scene."""
    from plotly.subplots import make_subplots

    rigs = list(rigs)[:3]
    fig = make_subplots(
        rows=1, cols=len(rigs), specs=[[{"type": "scene"}] * len(rigs)],
        subplot_titles=[r.title for r in rigs], horizontal_spacing=0.02,
    )  # fmt: skip
    for col, rig in enumerate(rigs, start=1):
        for tr in rig_traces(rig):
            fig.add_trace(tr, row=1, col=col)
        scene = f"scene{'' if col == 1 else col}"
        fig.update_layout(
            {
                scene: {
                    "aspectmode": "data",
                    "xaxis_title": "x [m]",
                    "yaxis_title": "y [m]",
                    "zaxis_title": "z [m]",
                }
            }
        )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,
    )
    return fig
