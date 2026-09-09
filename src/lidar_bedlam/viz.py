"""Plotting helpers for notebooks (matplotlib for 2D, plotly for 3D).

3D figures are shown in a display frame with up pointing up: camera
(x right, y down, z forward) is mapped to (x, z, -y) = (right, forward, up).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from lidar_bedlam.geometry.boxes import box_corners_bev

FloatArray = NDArray[np.float64]


def to_display(points: NDArray[np.floating]) -> FloatArray:
    """Camera frame (x, y, z) -> display frame (x, z, -y)."""
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def mesh_trace(
    verts: NDArray[np.floating],
    faces: NDArray[np.integer],
    name: str,
    color: str = "lightblue",
    opacity: float = 0.6,
) -> Any:
    """Plotly Mesh3d of a camera-frame mesh."""
    import plotly.graph_objects as go

    v = to_display(verts)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
        flatshading=True,
    )  # fmt: skip


def points_trace(
    points: NDArray[np.floating],
    name: str,
    color: Any = "red",
    size: float = 2.0,
    colorscale: str | None = None,
    colorbar_title: str | None = None,
) -> Any:
    """Plotly Scatter3d of camera-frame points (colour may be per point)."""
    import plotly.graph_objects as go

    p = to_display(points)
    marker: dict[str, Any] = {"size": size, "color": color}
    if colorscale is not None:
        marker["colorscale"] = colorscale
        marker["showscale"] = True
        marker["colorbar"] = {"title": colorbar_title or ""}
    return go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", marker=marker,
        name=name,
    )  # fmt: skip


def box_trace(
    box: NDArray[np.floating], name: str, color: str = "black"
) -> Any:
    """Plotly wireframe of a 7-vector box (camera frame, up = -y)."""
    import plotly.graph_objects as go

    bev = box_corners_bev(box, up_axis=1)  # (4, 2) in (x, z)
    y0, y1 = box[1] - box[4] / 2.0, box[1] + box[4] / 2.0
    bottom = np.array([[x, y1, z] for x, z in bev])
    top = np.array([[x, y0, z] for x, z in bev])
    segs = []
    for i in range(4):
        segs += [bottom[i], bottom[(i + 1) % 4], [np.nan] * 3]
        segs += [top[i], top[(i + 1) % 4], [np.nan] * 3]
        segs += [bottom[i], top[i], [np.nan] * 3]
    s = to_display(np.array(segs, dtype=np.float64))
    return go.Scatter3d(
        x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="lines",
        line={"color": color, "width": 3}, name=name,
    )  # fmt: skip


def figure_3d(traces: list[Any], title: str, height: int = 600) -> Any:
    """Plotly figure with equal axes and the display-frame labels."""
    import plotly.graph_objects as go

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, height=height, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "x right [m]", "yaxis_title": "z forward [m]",
            "zaxis_title": "up [m]", "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
    )  # fmt: skip
    return fig


def point_to_surface_distance(
    points: NDArray[np.floating], verts: NDArray[np.floating]
) -> FloatArray:
    """Distance of each point to the nearest mesh vertex (metres)."""
    d, _ = cKDTree(np.asarray(verts, dtype=np.float64)).query(
        np.asarray(points, dtype=np.float64)
    )
    return np.asarray(d, dtype=np.float64)


def draw_points(
    ax: Any,
    uv: NDArray[np.floating],
    values: NDArray[np.floating] | None = None,
    size: float = 4.0,
    cmap: str = "turbo",
    label: str | None = None,
) -> Any:
    """Scatter pixel coordinates on a matplotlib axis; returns the artist."""
    p = np.asarray(uv)
    ok = np.isfinite(p).all(axis=1)
    if values is None:
        return ax.scatter(p[ok, 0], p[ok, 1], s=size, c="red", label=label)
    return ax.scatter(
        p[ok, 0], p[ok, 1], s=size, c=np.asarray(values)[ok], cmap=cmap,
        label=label,
    )  # fmt: skip


def draw_mask_outline(ax: Any, mask: NDArray[np.bool_], color: str) -> None:
    """Contour of a binary mask."""
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1)
