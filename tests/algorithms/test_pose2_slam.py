"""Unit tests on landmark-based SLAM utilities."""

import math

import numpy as np
from gtsam import Rot2, Point2, Pose2

import salve.algorithms.pose2_slam as pose2_slam
from salve.algorithms.pose2_slam import BearingRangeMeasurement, OdometryMeasurement


def test_planar_slam_pgo_only() -> None:
    """Test pose-graph optimization only (no landmarks) with 5 estimated cameras and 1 unknown camera.

    Requires an initial estimate.
    Based off of https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/Pose2SLAMExample.py

    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vector3(0.3, 0.3, 0.1))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vector3(0.2, 0.2, 0.1))

              5       4
              O-- . --O
              |       |
      .
      .
      |       |       |
    . O-- . . O-- . --O .
      .1      2       3
      .
    """
    wTi_list_init = [
        None,
        Pose2(0.5, 0.0, 0.2),
        Pose2(2.3, 0.1, -0.2),
        Pose2(4.1, 0.1, math.pi / 2),
        Pose2(4.0, 2.0, math.pi),
        Pose2(2.1, 2.1, -math.pi / 2),
    ]
    # Create odometry (Between) factors between consecutive poses as (x,y,theta).
    i2Ti1_measurements = [
        OdometryMeasurement(i1=1, i2=2, i2Ti1=Pose2(2, 0, 0).inverse()),
        OdometryMeasurement(i1=2, i2=3, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=3, i2=4, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=4, i2=5, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=2, i2=5, i2Ti1=Pose2(2, 0, math.pi / 2)),  #  Adds the loop closure constraint
    ]

    wTi_list, landmark_positions = pose2_slam.planar_slam(
        wTi_list_init,
        i2Ti1_measurements,
        landmark_positions_init={},
        landmark_measurements=[],
        optimize_poses_only=True,
        use_robust=False,
    )

    expected_wTi_list = [
        None, # Pose 0 unknown.
        Pose2(0, 0, 0), # Pose 1
        Pose2(2, 0, 0), # Pose 2
        Pose2(4, 0, math.pi/2), # Pose 3
        Pose2(4, 2, math.pi), # Pose 4
        Pose2(2, 2, -math.pi/2), # Pose 5
    ]
    for i, wTi_expected in enumerate(expected_wTi_list):
        wTi = wTi_list[i]
        if wTi_expected is None:
            assert wTi is None
            continue
        assert np.isclose(wTi_expected.theta(), wTi.theta())
        assert np.allclose(wTi_expected.translation(), wTi.translation())


def test_planar_slam() -> None:
    """Test planar SLAM with both odometry measurements and bearing-range measurements to landmarks.

    Scenario: (with 3 poses and 2 landmarks as `X`) in cartesian coordinate system below:
       .    X    X
       . /  |    |
       ./   |    |
    ...1....2....3...
       .
       .
    """
    # fmt: off
    # Create (deliberately inaccurate) initial estimate.
    wTi_list_init = [
        None,
        Pose2(-0.25, 0.20, 0.15),
        Pose2(2.30, 0.10, -0.20),
        Pose2(4.10, 0.10, 0.10)
    ]
    # Relative pose measurements as (x,y,theta).
    i2Ti1_measurements = [
        OdometryMeasurement(i1=1, i2=2, i2Ti1=Pose2(-2.0, 0.0, 0.0)),
        OdometryMeasurement(i1=2, i2=3, i2Ti1=Pose2(-2.0, 0.0, 0.0))
    ]
    landmark_positions_init = {
        1: Point2(1.80, 2.10),
        2: Point2(4.10, 1.80)
    }
    # fmt: on
    # Add Bearing-Range measurements to two different landmarks L1 and L2
    # angle to reach landmark, from given pose.
    landmark_measurements = [
        BearingRangeMeasurement(pano_id=1, l_idx=1, bearing_deg=45, range=np.sqrt(4.0 + 4.0)),
        BearingRangeMeasurement(pano_id=2, l_idx=1, bearing_deg=90, range=2),
        BearingRangeMeasurement(pano_id=3, l_idx=2, bearing_deg=90, range=2),
    ]
    wTi_list, landmark_positions = pose2_slam.planar_slam(
        wTi_list_init=wTi_list_init,
        i2Ti1_measurements=i2Ti1_measurements,
        landmark_positions_init=landmark_positions_init,
        landmark_measurements=landmark_measurements,
        optimize_poses_only=False
    )

    # Global poses as (x,y,theta).
    # fmt: off
    expected_wTi_list = [
        None,
        Pose2(0.0, 0.0, 0.0),
        Pose2(2.0, 0.0, 0.0),
        Pose2(4.0, 0.0, 0.0)
    ]
    # fmt: on

    # Two landmarks should now be correctly localized, despite inaccurate initialization.
    expected_landmark_positions = {1: Point2(2, 2), 2: Point2(4, 2)}
    assert expected_landmark_positions.keys() == landmark_positions.keys()

    # Here `k` is the landmark index.
    for k in expected_landmark_positions.keys():
        assert np.allclose(expected_landmark_positions[k], landmark_positions[k])

    for i, wTi in enumerate(expected_wTi_list):
        if wTi is None:
            assert wTi_list[i] is None
        else:
            assert np.isclose(wTi_list[i].theta(), expected_wTi_list[i].theta())
            assert np.isclose(wTi_list[i].x(), expected_wTi_list[i].x())
            assert np.isclose(wTi_list[i].y(), expected_wTi_list[i].y())
