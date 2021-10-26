
import math

import numpy as np
from gtsam import Point2, Pose2

import afp.algorithms.pose2_slam as pose2_slam
from afp.algorithms.pose2_slam import BearingRangeMeasurement, OdometryMeasurement


def test_planar_slam_pgo_only() -> None:
    """Test pose-graph optimization only (no landmarks).

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
        Pose2(2.1, 2.1, -math.pi / 2)
    ]
    # Create odometry (Between) factors between consecutive poses
    # as (x,y,theta)
    i2Ti1_measurements = [
        OdometryMeasurement(i1=1, i2=2, i2Ti1=Pose2(2, 0, 0).inverse()),
        OdometryMeasurement(i1=2, i2=3, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=3, i2=4, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=4, i2=5, i2Ti1=Pose2(2, 0, math.pi / 2).inverse()),
        OdometryMeasurement(i1=2, i2=5, i2Ti1=Pose2(2, 0, math.pi / 2)) #  Adds the loop closure constraint
    ]

    wTi_list, landmark_positions = pose2_slam.planar_slam(
        wTi_list_init, i2Ti1_measurements, landmark_positions_init={}, landmark_measurements=[], optimize_poses_only=True, use_robust=False
    )
    print(wTi_list)

    expected_wTi_list = [
        None,
        Pose2(0, 0, 0),
        Pose2(2, 0, 0),
        Pose2(4, 0, 1.5708),
        Pose2(4, 2, 3.14159),
        Pose2(2, 2, -1.5708)
    ]
    for i, wTi_expected in enumerate(expected_wTi_list):
        wTi = wTi_list[i]
        if wTi_expected is None:
            assert wTi is None
            continue
        assert np.isclose(wTi_expected.theta(), wTi.theta())
        assert np.allclose(wTi_expected.translation(), wTi.translation())

    # # 4. Optimize the initial values using a Gauss-Newton nonlinear optimizer
    # # The optimizer accepts an optional set of configuration parameters,
    # # controlling things like convergence criteria, the type of linear
    # # system solver to use, and the amount of information displayed during
    # # optimization. We will set a few parameters as a demonstration.
    # parameters = gtsam.GaussNewtonParams()

    # # Stop iterating once the change in error between steps is less than this value
    # parameters.setRelativeErrorTol(1e-5)
    # # Do not perform more than N iteration steps
    # parameters.setMaxIterations(100)
    # # Create the optimizer ...
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    # # ... and optimize
    # result = optimizer.optimize()
    # print("Final Result:\n{}".format(result))

    # # 5. Calculate and print marginal covariances for all variables
    # marginals = gtsam.Marginals(graph, result)
    # for i in range(1, 6):
    #     print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

    # fig = plt.figure(0)
    # for i in range(1, 6):
    #     gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    # plt.axis("equal")
    # plt.show()




def test_planar_slam() -> None:
    """

    Scenario: (with O as pose and X as landmark)
       .    X    X
       . /  |    |
       ./   |    |
    ...O....O....O...
       .
       .
    """
    # fmt: off
    # Create (deliberately inaccurate) initial estimate
    wTi_list_init = [
        None,
        Pose2(-0.25, 0.20, 0.15),
        Pose2(2.30, 0.10, -0.20),
        Pose2(4.10, 0.10, 0.10)
    ]
    # # as (x,y,theta)
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
        wTi_list_init, i2Ti1_measurements, landmark_positions_init, landmark_measurements, optimize_poses_only=False
    )

    # as (x,y,theta)
    expected_wTi_list = [None, Pose2(0.0, 0.0, 0.0), Pose2(2.0, 0.0, 0.0), Pose2(4.0, 0.0, 0.0)]
    import pdb

    pdb.set_trace()

    expected_landmark_positions = {1: Point2(2, 2), 2: Point2(4, 2)}

    for lkey in expected_landmark_positions.keys():
        assert np.allclose(expected_landmark_positions[lkey], landmark_positions[lkey])

    for i, wTi in enumerate(expected_wTi_list):
        if wTi is None:
            assert wTi_list[i] is None
        else:
            assert np.isclose(wTi_list[i].theta(), expected_wTi_list[i].theta())
            assert np.isclose(wTi_list[i].x(), expected_wTi_list[i].x())
            assert np.isclose(wTi_list[i].y(), expected_wTi_list[i].y())



if __name__ == "__main__":
    #test_planar_slam()
    test_planar_slam_pgo_only()

