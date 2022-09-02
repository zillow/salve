"""Unit tests to make sure that SE(2) estimation between two point clouds works properly."""

import numpy as np

import salve.utils.se2_estimation as se2_estimation


def test_align_points_SE2() -> None:
    """Check SE(2) least squares fit for two unaligned horseshoe-shaped rooms of the same size.

    The two rooms are unaligned because rotated and translated away from each other.
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
    """Check SE(2) least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.

    W/D/O "A" represented by X--X, and W/D/O "B" represented by o--o.

    X o -- o X

    """
    # fmt: off
    # Door with width of 2 units.
    pts_a = np.array(
        [
            [-4, 2],
            [-2, 2]
        ])
    # Door with width of 4 units.
    pts_b = np.array(
        [
            [-5, 2],
            [-1, 2]
        ])

    # fmt: on
    aTb, _ = se2_estimation.align_points_SE2(pts_a, pts_b)

    # Since the small door is sandwiched in the middle of the large door, the SE(2) alignment won't move the small one.
    assert aTb.theta_deg == 0.0
    assert np.allclose(aTb.translation, np.zeros(2))


def test_align_points_SE2_doorway_rotated() -> None:
    """Check SE(2) least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.
    
    W/D/O "A" represented by o--o, and W/D/O "B" represented by X--X.

    X
    |
    | o -- o
    |
    |
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

    # Vertices from W/D/O "A" at (7,3) and (9,3) will be moved as close as possible to W/D/O "B".
    # Note that since scale is not fit here, we get vertices close -- to (5,3) and to (5,5), but
    # we cannot fit a perfect match to W/D/O "B"s vertices at (5,2) and (5,6).
    expected_ptb1 = np.array([5., 3.])
    assert np.allclose(expected_ptb1, bTa.transform_from(np.array([7,3]).reshape(1,2) ) )

    expected_ptb2 = np.array([5., 5.])
    assert np.allclose(expected_ptb2, bTa.transform_from(np.array([9,3]).reshape(1,2) ) )
