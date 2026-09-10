"""Virtual distance by moving the camera back."""

from __future__ import annotations

import numpy as np

from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.lidar.distance import SKY_M, move_camera_back, shrink_factor


def test_move_back_shrinks_and_deepens() -> None:
    cam = PinholeCamera(400.0, 400.0, 199.5, 149.5, 400, 300)
    depth = np.full((300, 400), 20.0)  # far wall
    mask = np.zeros((300, 400), bool)
    depth[100:200, 160:240] = 5.0  # a person-sized blob at 5 m
    mask[100:200, 160:240] = True
    new_depth, new_masks = move_camera_back(depth, {"p": mask}, cam, 5.0)
    m = new_masks["p"]
    ys, xs = np.nonzero(m)
    # blob at 10 m now: half the pixel size, centred on the principal point
    assert 0.45 < (xs.max() - xs.min()) / 80 < 0.55
    assert 0.45 < (ys.max() - ys.min()) / 100 < 0.55
    assert np.allclose(new_depth[m], 10.0)
    # the wall behind is at 25 m and stays where it was rendered
    assert np.isclose(np.median(new_depth[~m & (new_depth < SKY_M)]), 25.0)
    # unseen border regions are sky
    assert new_depth[0, 0] == SKY_M
    assert np.isclose(shrink_factor(5.0, 5.0), 0.5)
