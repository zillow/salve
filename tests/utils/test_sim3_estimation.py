"""Unit tests to make sure that Sim(3) estimation between two point clouds works properly."""

import numpy as np

import salve.utils.sim3_estimation as sim3_estimation


def test_align_points_sim3_horseshoe() -> None:
    """Ensure align_points_sim3() works properly."""
    # fmt: off
    # small horseshoe
    pts_a = np.array(
        [
            [3, 1, 0],
            [1, 1, 0],
            [1, 3, 0],
            [3, 3, 0]
        ]
    )

    # large horseshoe
    pts_b = np.array(
        [
            [3, 1, 10],
            [-1, 1, 10],
            [-1, 5, 10],
            [3, 5, 10]
        ]
    )

    # fmt: on
    # a is the reference, and b will be transformed to a_
    aSb, pts_a_ = sim3_estimation.align_points_sim3(pts_a, pts_b)

    # fmt: off

    # shift:
    # [ 3,1]         = [6,2]       = [3,1]
    # [-1,1] + [3,1] = [2,2] * 0.5 = [1,1]
    # [-1,5]         = [2,6]       = [1,3]
    # [ 3,5]         = [6,6]       = [3,3]

    expected_pts_a_ = np.array([
        [3, 1, 0],
        [1, 1, 0],
        [1, 3, 0],
        [3, 3, 0]
    ])
    # fmt: on

    assert np.allclose(expected_pts_a_, pts_a)
    assert aSb.scale == 0.5
    assert np.allclose(aSb.rotation, np.eye(2))
    assert np.allclose(aSb.translation, np.array([3, 1]))


# def test_reorthonormalize():
#     """ """
#     # from afp.common.pano_data import generate_Sim2_from_floorplan_transform
#     pano3_data = {
#         "translation": [0.01553549307166846, -0.002272521859178478],
#         "rotation": -352.5305535406924,
#         "scale": 0.4042260417272217,
#     }
#     pano4_data = {"translation": [0.0, 0.0], "rotation": 0.0, "scale": 0.4042260417272217}

#     global_SIM2_i3 = generate_Sim2_from_floorplan_transform(pano3_data)
#     global_SIM2_i4 = generate_Sim2_from_floorplan_transform(pano4_data)

#     import pdb

#     pdb.set_trace()

#     i2Ti1_gt = global_SIM2_i4.inverse().compose(global_SIM2_i3)

#     reorthonormalize_sim2(i2Ti1_gt)

