"""Unit tests for utilities that set BEV parameters."""

import numpy as np

import salve.common.bevparams as bev_params_utils
from salve.common.bevparams import BEVParams


def test_bevimg_Sim2_world() -> None:
    """Ensure that class constructor works, and Sim(2) generated is correct."""
    # 10 meters x 10 meters in total.
    params = BEVParams(img_h=20, img_w=20, meters_per_px=0.5)

    # fmt: off
    world_pts = np.array(
        [
            [2,2],
            [-5,-5],
            [5,5]  # out of bounds
        ]
    )
    # fmt: on
    img_pts = params.bevimg_Sim2_world.transform_from(world_pts)

    # fmt: off
    expected_img_pts = np.array(
        [
            [14,14],
            [0,0],
            [20,20]
        ]
    )
    # fmt: on
    assert np.allclose(img_pts, expected_img_pts)



def test_get_line_width_by_resolution() -> None:
    """Ensure polyline thickness is computed properly."""

    line_width = bev_params_utils.get_line_width_by_resolution(resolution=0.005)
    assert line_width == 30
    assert isinstance(line_width, int)

    line_width = bev_params_utils.get_line_width_by_resolution(resolution=0.01)
    assert line_width == 15
    assert isinstance(line_width, int)

    line_width = bev_params_utils.get_line_width_by_resolution(resolution=0.02)
    assert line_width == 8
    assert isinstance(line_width, int)
