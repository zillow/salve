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

    rgb = np.array(
        [[255,128,0],
        [1,2,3],
        [4,5,6],
        [7,8,9]
        ]
    )

    # fmt: on
    xmin = -1
    ymin = -1
    xmax = 1
    ymax = 2

    valid_pts, valid_rgb = bev_rendering_utils.prune_to_2d_bbox(
        pts=pts, rgb=rgb, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
    )
    expected_valid_pts = np.array([[1, 2], [0, 1]])

    expected_valid_rgb = np.array([[4, 5, 6], [7, 8, 9]])

    assert np.allclose(valid_pts, expected_valid_pts)
    assert np.allclose(valid_rgb, expected_valid_rgb)
