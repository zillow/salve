"""
A 2D Pose SLAM example that reads input from g2o, and solve the Pose2 problem
using LAGO (Linear Approximation for Graph Optimization).
Output is written to a file, in g2o format

Reference:
L. Carlone, R. Aragues, J. Castellanos, and B. Bona, A fast and accurate
approximation for planar pose graph optimization, IJRR, 2014.

L. Carlone, R. Aragues, J.A. Castellanos, and B. Bona, A linear approximation
for graph-based simultaneous localization and mapping, RSS, 2011.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import gtsam
from argoverse.utils.sim2 import Sim2
from gtsam import Rot2, Point2, Point3, Pose2, PriorFactorPose2, Values
from gtsam.symbol_shorthand import X, L

from afp.common.floor_reconstruction_report import FloorReconstructionReport


# Create noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  # np.array([1e-6, 1e-6, 1e-8])
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))


def estimate_poses_lago(i2Ti1_dict: Dict[Tuple[int, int], Pose2]) -> List[Pose2]:
    """Does not require an initial estimate.

    Can also estimate over landmarks if we can solve data association (s & e for shared door).

    We add factors as load2D (https://github.com/borglab/gtsam/blob/develop/gtsam/slam/dataset.cpp#L500) does.
    """
    initial_estimate = Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))

    graph = gtsam.NonlinearFactorGraph()

    # Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

    for (i1, i2), i2Ti1 in i2Ti1_dict.items():
        # Add odometry factors between X1,X2 and X2,X3, respectively
        graph.add(gtsam.BetweenFactorPose2(X(i1), X(i2), i2Ti1, ODOMETRY_NOISE))

    print(graph)

    import pdb

    pdb.set_trace()

    print("Computing LAGO estimate")
    estimate_lago: Values = gtsam.lago.initialize(graph)

    max_frame_id = max([max(i1, i2) for (i1, i2) in i2Ti1_dict.keys()]) + 1
    for i in range(max_frame_id):
        wTi_list.append(estimate_lago.atPose2(i))


def test_estimate_poses_lago() -> None:
    """Ensure pose graph is correctly estimated for simple 4-pose scenario."""
    wTi_list = [
        Pose2(Rot2(), np.array([2, 0])),
        Pose2(Rot2(), np.array([2, 2])),
        Pose2(Rot2(), np.array([0, 2])),
        Pose2(Rot2(), np.array([0, 0])),
    ]
    i2Ti1_dict = {
        (0, 1): wTi_list[1].between(wTi_list[0]),
        (1, 2): wTi_list[2].between(wTi_list[1]),
        (2, 3): wTi_list[3].between(wTi_list[2]),
        (0, 3): wTi_list[3].between(wTi_list[0]),
    }

    wTi_list_computed = estimate_poses_lago(i2Ti1_dict)


def pose2slam():
    """
    Pose-graph optimization only (no landmarks)

    Requires an initial estimate.
    Based off of https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/Pose2SLAMExample.py
    """

    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vector3(0.3, 0.3, 0.1))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vector3(0.2, 0.2, 0.1))

    # 1. Create a factor graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()

    # 2a. Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise ODOMETRY_NOISE (covariance matrix)
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), PRIOR_NOISE))

    # 2b. Add odometry factors
    # Create odometry (Between) factors between consecutive poses
    graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2, 0, 0), ODOMETRY_NOISE))
    graph.add(gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE))
    graph.add(gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE))
    graph.add(gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE))

    # 2c. Add the loop closure constraint
    # This factor encodes the fact that we have returned to the same pose. In real
    # systems, these constraints may be identified in many ways, such as appearance-based
    # techniques with camera images. We will use another Between Factor to enforce this constraint:
    graph.add(gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE))
    print("\nFactor Graph:\n{}".format(graph))  # print

    # 3. Create the data structure to hold the initial_estimate estimate to the
    # solution. For illustrative purposes, these have been deliberately set to incorrect values
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(4.1, 0.1, math.pi / 2))
    initial_estimate.insert(4, gtsam.Pose2(4.0, 2.0, math.pi))
    initial_estimate.insert(5, gtsam.Pose2(2.1, 2.1, -math.pi / 2))
    print("\nInitial Estimate:\n{}".format(initial_estimate))  # print

    # 4. Optimize the initial values using a Gauss-Newton nonlinear optimizer
    # The optimizer accepts an optional set of configuration parameters,
    # controlling things like convergence criteria, the type of linear
    # system solver to use, and the amount of information displayed during
    # optimization. We will set a few parameters as a demonstration.
    parameters = gtsam.GaussNewtonParams()

    # Stop iterating once the change in error between steps is less than this value
    parameters.setRelativeErrorTol(1e-5)
    # Do not perform more than N iteration steps
    parameters.setMaxIterations(100)
    # Create the optimizer ...
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    # ... and optimize
    result = optimizer.optimize()
    print("Final Result:\n{}".format(result))

    # 5. Calculate and print marginal covariances for all variables
    marginals = gtsam.Marginals(graph, result)
    for i in range(1, 6):
        print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

    fig = plt.figure(0)
    for i in range(1, 6):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis("equal")
    plt.show()


from gtsam.symbol_shorthand import L, X
from dataclasses import dataclass


@dataclass
class BearingRangeMeasurement:
    """Bearing is provided in degrees.

    Args:
        l_idx: landmark index.
    """

    pano_id: int
    l_idx: int
    bearing_deg: float
    range: float


@dataclass
class OdometryMeasurement:
    i1: int
    i2: int
    i2Ti1: Pose2


def planar_slam(
    wTi_list_init: List[Optional[Pose2]],
    i2Ti1_measurements: List[Pose2],
    landmark_positions_init: Dict[int, Point2],
    landmark_measurements: List[BearingRangeMeasurement],
    optimize_poses_only: bool = False
) -> Tuple[List[Optional[Pose2]], Dict[int, Point2]]:
    """

    See https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py

    using odometry measurements and bearing-range (laser) measurements

    Args:
        wTi_list_init:
        i2Ti1_measurement: odometry measurements
        landmark_positions_init
        landmark_measurements

    Returns:
        wTi_list:
    """
    # TODO: use robust noise model.
    # measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, MEASUREMENT_NOISE_SIGMA)
    # if self._robust_measurement_noise:
    #     measurement_noise = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), measurement_noise)

    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    # MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0]))

    PRIOR_NOISE = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), PRIOR_NOISE)
    ODOMETRY_NOISE = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), ODOMETRY_NOISE)
    MEASUREMENT_NOISE = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), MEASUREMENT_NOISE)

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # find the first pano for which the initial pose estimate is not None
    origin_pano_id = np.argmax([wTi is not None for wTi in wTi_list_init])

    # keys corresponding to unknown variables in the factor graph.
    # Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    graph.add(gtsam.PriorFactorPose2(X(origin_pano_id), gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

    # Add odometry factors between poses
    for om in i2Ti1_measurements:
        # for unestimated pose, cannot use this odometry measurement
        if wTi_list_init[om.i1] is None:
            continue

        if wTi_list_init[om.i2] is None:
            continue

        graph.add(gtsam.BetweenFactorPose2(X(om.i2), X(om.i1), om.i2Ti1, ODOMETRY_NOISE))

    if not optimize_poses_only:
        # Add Bearing, Range measurements to two different landmarks L1 and L2
        for lm in landmark_measurements:
            # unestimated pose, cannot use this bearing-range landmark measurement
            if wTi_list_init[lm.pano_id] is None:
                continue

            graph.add(
                gtsam.BearingRangeFactor2D(
                    X(lm.pano_id), L(lm.l_idx), Rot2.fromDegrees(lm.bearing_deg), lm.range, MEASUREMENT_NOISE
                )
            )

    # Print graph
    # print("Factor Graph:\n{}".format(graph))

    # Create initial estimate
    initial_estimate = gtsam.Values()

    for i, wTi in enumerate(wTi_list_init):
        if wTi is None:
            continue
        initial_estimate.insert(X(i), wTi)

    if not optimize_poses_only:
        for l, wTl in landmark_positions_init.items():
            initial_estimate.insert(L(l), landmark_positions_init[l])

    # print("Initial Estimate:\n{}".format(initial_estimate))

    # Optimize using Levenberg-Marquardt optimization. The optimizer
    # accepts an optional set of configuration parameters, controlling
    # things like convergence criteria, the type of linear system solver
    # to use, and the amount of information displayed during optimization.
    # Here we will use the default set of parameters.  See the
    # documentation for the full set of parameters.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    # print("\nFinal Result:\n{}".format(result))

    # # Calculate and print marginal covariances for all variables
    # marginals = gtsam.Marginals(graph, result)
    # for (key, s) in [(X(1), "X1"), (X(2), "X2"), (X(3), "X3"), (L(1), "L1"), (L(2), "L2")]:
    #     print("{} covariance:\n{}\n".format(s, marginals.marginalCovariance(key)))

    num_poses = len(wTi_list_init)
    wTi_list = [None] * num_poses
    for i in range(num_poses):
        if wTi_list_init[i] is None:
            continue
        wTi_list[i] = result.atPose2(X(i))

    landmark_positions = {}
    if not optimize_poses_only:
        for lkey in landmark_positions_init.keys():
            landmark_positions[lkey] = result.atPoint2(L(lkey))

    return wTi_list, landmark_positions


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
    wTi_list, landmark_positions = planar_slam(wTi_list_init, i2Ti1_measurements, landmark_positions_init, landmark_measurements)

    # as (x,y,theta)
    expected_wTi_list = [
        None,
        Pose2(0.0, 0.0, 0.0),
        Pose2(2.0, 0.0, 0.0),
        Pose2(4.0, 0.0, 0.0)
    ]
    import pdb; pdb.set_trace()

    expected_landmark_positions = {
        1: Point2(2,2),
        2: Point2(4,2)
    }

    for lkey in expected_landmark_positions.keys():
        assert np.allclose(expected_landmark_positions[lkey], landmark_positions[lkey])

    for i, wTi in enumerate(expected_wTi_list):
        if wTi is None:
            assert wTi_list[i] is None
        else:
            assert np.isclose(wTi_list[i].theta(), expected_wTi_list[i].theta())
            assert np.isclose(wTi_list[i].x(), expected_wTi_list[i].x())
            assert np.isclose(wTi_list[i].y(), expected_wTi_list[i].y())



def execute_planar_slam(
    measurements: List[EdgeClassification],
    gt_floor_pg: "PoseGraph2d",
    hypotheses_save_root: str,
    building_id: str,
    floor_id: str,
    wSi_list: List[Sim2],
    verbose: bool = True
) -> None:
    """Gather odometry and landmark measurements for planar Pose(2) SLAM.

    Args:

    Returns:

    """
    # load up the 3d point locations for each WDO.
    from read_prod_predictions import load_inferred_floor_pose_graphs

    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    floor_pose_graphs = load_inferred_floor_pose_graphs(
        query_building_id=building_id, raw_dataset_dir=raw_dataset_dir
    )
    pano_dict_inferred = floor_pose_graphs[floor_id].nodes

    wTi_list_init = [ Pose2(Rot2.fromDegrees(wSi.theta_deg), wSi.translation) if wSi is not None else None for wSi in wSi_list]

    # # as (x,y,theta). We don't use a dict, as we may have multiple measurements for each pair of poses.
    i2Ti1_measurements = []
    for m in measurements:
        i2Si1 = get_alignment_hypothesis_for_measurement(m, hypotheses_save_root, building_id, floor_id)
        theta_rad = np.deg2rad(i2Si1.theta_deg)
        x, y = i2Si1.translation
        om = OdometryMeasurement(m.i1, m.i2, Pose2(x, y, theta_rad))
        i2Ti1_measurements.append(om)
    
    # Add Bearing-Range measurements to different landmarks
    # angle to reach landmark, from given pose.
    # for each (s,e)

    tracks_2d = data_association.perform_data_association(measurements, pano_dict_inferred)

    landmark_measurements = []
    landmark_positions_init = {}
    for j, track_2d in enumerate(tracks_2d):
        color = np.random.rand(3)
        for k, m in enumerate(track_2d.measurements):

            if wTi_list_init[m.i] is None:
                continue

            if j not in landmark_positions_init:
                # TODO: initialize the landmark positions by using the spanning tree poses
                landmark_positions_init[j] = wTi_list_init[m.i].transformFrom(m.uv)

            # an abuse of "uv",this really just means "xy"
            bearing_deg, range = bearing_range_from_vertex(m.uv)
            landmark_measurements += [BearingRangeMeasurement(pano_id=m.i, l_idx=j, bearing_deg=bearing_deg, range=range)]

            pt_w = wTi_list_init[m.i].transformFrom(m.uv)
            plt.scatter(pt_w[0], pt_w[1], 10, color=color, marker='+')
            plt.text(pt_w[0], pt_w[1], f"j={j},i={m.i}", color=color)

            draw_coordinate_frame(wTi_list_init[m.i], text=str(m.i))

    plt.axis("equal")
    plt.show()

    wTi_list, landmark_positions = pose2_slam.planar_slam(
        wTi_list_init, i2Ti1_measurements, landmark_positions_init, landmark_measurements
    )

    wSi_list = [None] * len(wTi_list)
    for i, wTi in enumerate(wTi_list):
        if wTi is None:
            continue
        wSi_list[i] = Sim2(R=wTi.rotation().matrix(), t=wTi.translation(), s=1.0)

    report = FloorReconstructionReport.from_wSi_list(wSi_list, gt_floor_pg, plot_save_dir="BLAH")
    return report


def draw_coordinate_frame(wTi: Pose2, text: str) -> None:
    """Draw a 2d coordinate frame using matplotlib."""
    # camera center
    cc = wTi.translation()
    plt.text(cc[0], cc[1], text)

    # loop over the x-axis and then y-axis
    for a, color in zip(range(2), ["r", "g"]):
        axis = np.zeros(2)
        axis[a] = 1
        w_axis = wTi.transformFrom(axis)
        plt.plot([cc[0],w_axis[0]], [cc[1], w_axis[1]], c=color)


def bearing_range_from_vertex(v: Tuple[float,float]) -> float:
    """Return bearing in degrees and range."""
    x, y = v
    bearing_rad = np.arctan2(y, x)
    range = np.linalg.norm(v)
    return np.rad2deg(bearing_rad), range



if __name__ == "__main__":
    test_planar_slam()

