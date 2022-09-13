"""Unit tests for bird's eye view texture map rendering utilities."""

import numpy as np

import salve.utils.bev_rendering_utils as bev_rendering_utils


def test_prune_to_2d_bbox() -> None:
    """Tests that a 2d point cloud can be pruned to rectangular box boundaries."""
    # fmt: off
    pts = np.array(
    	[[-2, 2], # will be discarded
    	[2, 0], # will be discarded
    	[1, 2],
    	[0, 1]
    ])
    # fmt: on
    xmin = -1
    ymin = -1
    xmax = 1
    ymax = 2

    pts = bev_rendering_utils.prune_to_2d_bbox(pts, xmin, ymin, xmax, ymax)
    gt_pts = np.array([[1, 2], [0, 1]])
    assert np.allclose(pts, gt_pts)
