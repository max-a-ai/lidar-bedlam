"""LiDAR-to-surface losses.

Every LiDAR return that hit the person should lie on the predicted mesh,
a clothing offset outside it, and only on the part of the surface that
faces the sensor (a beam cannot hit the far side). The rule mirrors the
simulator's facing test: a vertex counts when one of its faces has a
normal pointing against the direction from the sensor.

Two losses share the same correspondences (return -> nearest facing
vertex):

- ``chamfer``: mean |distance - clothing offset| over the returns (the
  offset is a learned model parameter, a few centimetres). It deforms the
  body towards the returns vertex by vertex.
- ``icp``: a few closest-point iterations solve the rigid transform that
  best aligns the facing vertices with the returns (Kabsch, differentiable
  through the SVD); the loss is the size of that residual transform, so it
  moves the body as a whole (placement and heading), not its shape.
"""

from __future__ import annotations

import torch
from torch import Tensor


def facing_vertex_mask(verts: Tensor, faces: Tensor, origin: Tensor) -> Tensor:
    """(B, V) True where a vertex belongs to a face that faces ``origin``."""
    v0, v1, v2 = (
        verts[:, faces[:, 0]],
        verts[:, faces[:, 1]],
        verts[:, faces[:, 2]],
    )
    normal = torch.cross(v1 - v0, v2 - v0, dim=-1)
    centroid = (v0 + v1 + v2) / 3.0
    facing = (normal * (centroid - origin[:, None])).sum(-1) < 0  # (B, F)
    mask = torch.zeros(verts.shape[:2], device=verts.device)
    for k in range(3):
        mask.index_add_(1, faces[:, k], facing.to(mask.dtype))
    return mask > 0


def facing_subset(
    verts: Tensor,
    mask: Tensor,
    n: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Up to ``n`` facing vertices per sample (random) and a validity mask."""
    b, v = mask.shape
    score = torch.rand(b, v, device=verts.device, generator=generator)
    score = score + (~mask).to(score.dtype) * 2.0  # facing first
    idx = score.topk(min(n, v), dim=1, largest=False).indices  # (B, n)
    sub = torch.gather(verts, 1, idx[..., None].expand(-1, -1, 3))
    valid = torch.gather(mask, 1, idx)
    return sub, valid


def nearest_vertex(
    points: Tensor, sub: Tensor, sub_valid: Tensor, chunk: int = 256
) -> tuple[Tensor, Tensor]:
    """Distance and index of the nearest valid vertex for every point."""
    dists, idxs = [], []
    for s in range(0, points.shape[0], chunk):
        d = torch.cdist(points[s : s + chunk], sub[s : s + chunk])  # (b, N, n)
        d = d.masked_fill(~sub_valid[s : s + chunk, None, :], float("inf"))
        m = d.min(dim=-1)
        dists.append(m.values)
        idxs.append(m.indices)
    return torch.cat(dists), torch.cat(idxs)


def _correspondences(
    verts: Tensor,
    faces: Tensor,
    origin: Tensor,
    points: Tensor,
    n_sub: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Facing subset, its validity, and per-point nearest distance / index."""
    mask = facing_vertex_mask(verts, faces, origin)
    sub, valid = facing_subset(verts, mask, n_sub)
    d, idx = nearest_vertex(points, sub, valid)
    return sub, valid, d, idx


def chamfer_loss(
    verts: Tensor,
    faces: Tensor,
    origin: Tensor,
    points: Tensor,
    points_valid: Tensor,
    clothing_offset: Tensor,
    n_sub: int = 512,
) -> Tensor:
    """Mean |return-to-facing-surface distance - clothing offset| (metres)."""
    _, _, d, _ = _correspondences(verts, faces, origin, points, n_sub)
    ok = points_valid & torch.isfinite(d)
    err = (d - clothing_offset).abs()
    per_sample = (err * ok).sum(1) / ok.sum(1).clamp_min(1)
    has = ok.any(1)
    if not has.any():
        return verts.sum() * 0.0
    return per_sample[has].mean()


def _kabsch(src: Tensor, dst: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Weighted rigid transform (R, t) with dst ~ R src + t, per batch."""
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    cs = (w[..., None] * src).sum(1, keepdim=True)
    cd = (w[..., None] * dst).sum(1, keepdim=True)
    h = torch.einsum("bni,bnj->bij", w[..., None] * (src - cs), dst - cd)
    u, _, vt = torch.linalg.svd(h + 1e-8 * torch.eye(3, device=h.device))
    det = torch.det(vt.transpose(1, 2) @ u.transpose(1, 2))
    fix = torch.diag_embed(torch.stack([torch.ones_like(det)] * 2 + [det], -1))
    rot = vt.transpose(1, 2) @ fix @ u.transpose(1, 2)
    t = cd[:, 0] - torch.einsum("bij,bj->bi", rot, cs[:, 0])
    return rot, t


def icp_loss(
    verts: Tensor,
    faces: Tensor,
    origin: Tensor,
    points: Tensor,
    points_valid: Tensor,
    iterations: int = 3,
    n_sub: int = 512,
) -> Tensor:
    """How far ICP has to move the mesh to meet the returns (metres).

    A few closest-point iterations align the facing vertices rigidly with
    the returns. The transform is estimated without gradient and turned
    into a fixed target position per vertex; the loss is the mean distance
    of the vertices to those targets, so its value is the rigid
    displacement and its gradient moves the body as a whole towards the
    returns (a gradient through the SVD itself is unstable and diverged).
    """
    sub, valid, _, _ = _correspondences(verts, faces, origin, points, n_sub)
    with torch.no_grad():
        cur = sub.detach()
        for _ in range(iterations):
            d, idx = nearest_vertex(points, cur, valid)
            ok = (points_valid & torch.isfinite(d)).to(verts.dtype)
            matched = torch.gather(cur, 1, idx[..., None].expand(-1, -1, 3))
            rot, t = _kabsch(matched, points, ok)
            cur = torch.einsum("bij,bnj->bni", rot, cur) + t[:, None]
        target = torch.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        bad = ~torch.isfinite(cur).all(-1)
    has = points_valid.any(1)
    if not has.any():
        return verts.sum() * 0.0
    moved = (sub - target).norm(dim=-1)
    vw = (valid & ~bad).to(verts.dtype)
    per_sample = (moved * vw).sum(1) / vw.sum(1).clamp_min(1)
    out: Tensor = per_sample[has].mean()
    return out
