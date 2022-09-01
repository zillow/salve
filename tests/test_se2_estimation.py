"""Unit tests to make sure that SE(2) estimation between two point clouds works properly."""

import numpy as np

import salve.utils.se2_estimation as se2_estimation


def test_align_points_SE2() -> None:
    """
    Two horseshoes of the same size, just rotated and translated.

    TODO: fix bug in GTSAM where the inverse() object bTa is inappropriately returned for ab pairs.
    """
    # fmt: off
    pts_a = np.array(
        [
            [3, 1],
            [1, 1],
            [1, 3],
            [3, 3]
        ])
    pts_b = np.array(
        [
            [1, -3],
            [1, -5],
            [-1, -5],
            [-1, -3]
        ])

    # fmt: on
    aTb, pts_a_ = se2_estimation.align_points_SE2(pts_a, pts_b)
    assert aTb is not None

    for pt_a, pt_b in zip(pts_a, pts_b):

        pt_a_ = aTb.transform_from(pt_b.reshape(1,2) ).squeeze()
        assert np.allclose(pt_a, pt_a_)
        print("match")

    assert np.allclose(pts_a, pts_a_)


def test_align_points_SE2_doorway() -> None:
    """
    Check least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.

    X o -- o X

    """
    # fmt: off
    pts_a = np.array(
        [
            [-4, 2],
            [-2, 2]
        ])
    pts_b = np.array(
        [
            [-5, 2],
            [-1, 2]
        ])

    # fmt: on
    aTb, _ = se2_estimation.align_points_SE2(pts_a, pts_b)

    assert aTb.theta_deg == 0.0
    assert np.allclose(aTb.translation, np.zeros(2))


def test_align_points_SE2_doorway_rotated() -> None:
    """
    Check least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.
    
    X
    |
    | o -- o
    X

    """
    # fmt: off
    pts_a = np.array(
        [
            [7, 3],
            [9, 3]
        ])
    pts_b = np.array(
        [
            [5, 2],
            [5, 6]
        ])

    # fmt: on
    aTb, _ = se2_estimation.align_points_SE2(pts_a, pts_b)
    bTa = aTb.inverse()

    expected_ptb1 = np.array([5., 3.])
    assert np.allclose(expected_ptb1, bTa.transform_from(np.array([7,3]).reshape(1,2) ) )

    expected_ptb2 = np.array([5., 5.])
    assert np.allclose(expected_ptb2, bTa.transform_from(np.array([9,3]).reshape(1,2) ) )



# NOTE: TEST BELOW IS BASED OFF OF AN ANNOTATION ERROR I THINK -- CAN DELETE.
# def test_align_points_SE2_door() -> None:
#     """ """
#     # fmt: off
#     pano2_wd_pts = np.array(
#         [
#             [ 0.31029039,  5.0409614 ],
#             [ 0.31029039,  5.0409614 ],
#             [-0.19424723,  4.51055183],
#             [-0.19424723,  4.51055183],
#             [ 0.31029039,  5.0409614 ]
#         ]
#     )
#     pano1_wd_pts = np.array(
#         [
#             [ 1.44758631,  0.30243336],
#             [ 1.44758631,  0.30243336],
#             [ 1.38693234, -0.40765391],
#             [ 1.38693234, -0.40765391],
#             [ 1.44758631,  0.30243336]
#         ]
#     )
#     # fmt: on
#     # passed in as target, source
#     i2Ti1, pano1_pts_aligned = align_points_SE2(pts_a=pano2_wd_pts, pts_b=pano1_wd_pts)

    

#     # fmt: off
#     i2Ti1_gt = Sim2(
#         R = np.array(
#             [
#                 [ 0.65135753, -0.758771  ],
#                 [ 0.758771  ,  0.65135753]
#             ], dtype=np.float32
#         ),
#         t = np.array([-3.9725296 ,  0.15346308], dtype=np.float32),
#         s = 0.9999999999999999
#     )

#     import pdb; pdb.set_trace()



if __name__ == "__main__":

    test_align_points_SE2()
    test_align_points_SE2_doorway()
    test_align_points_SE2_doorway_rotated()
