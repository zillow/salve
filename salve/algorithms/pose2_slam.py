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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Rot2, Point2, Pose2, PriorFactorPose2, Values
from gtsam.symbol_shorthand import L, X

import salve.algorithms.data_association as data_association
import salve.common.edge_classification as edge_classification
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.axis_alignment_utils as axis_alignment_utils
import salve.utils.graph_rendering_utils as graph_rendering_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.common.edge_classification import EdgeClassification
from salve.common.edgewdopair import EdgeWDOPair
from salve.common.floor_reconstruction_report import FloorReconstructionReport
from salve.common.sim2 import Sim2


#RAW_DATASET_DIR = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
RAW_DATASET_DIR = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"


# Create noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  # np.array([1e-6, 1e-6, 1e-8])
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))


@dataclass
class BearingRangeMeasurement:
    """Bearing is provided in degrees.

    Args:
        pano_id: panorama ID
        l_idx: landmark index.
        bearing_deg: angle to landmark, in degrees.
        range: distance to landmark.
    """

    pano_id: int
    l_idx: int
    bearing_deg: float
    range: float


@dataclass
class OdometryMeasurement:
    """
    Args:
        i1: unique ID of panorama 1
        i2: unique ID of panorama 2
        i2Ti1: relative pose
    """
    i1: int
    i2: int
    i2Ti1: Pose2


def estimate_poses_lago(i2Ti1_dict: Dict[Tuple[int, int], Pose2]) -> List[Pose2]:
    """Does not require an initial estimate.

    Can also estimate over landmarks if we can solve data association (s & e for shared door).

    We add factors as load2D (https://github.com/borglab/gtsam/blob/develop/gtsam/slam/dataset.cpp#L500) does.
    """
    initial_estimate = Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))

    graph = gtsam.NonlinearFactorGraph()

    # Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    graph.add(PriorFactorPose2(X(0), gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

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


def planar_slam(
    wTi_list_init: List[Optional[Pose2]],
    i2Ti1_measurements: List[Pose2],
    landmark_positions_init: Dict[int, Point2],
    landmark_measurements: List[BearingRangeMeasurement],
    optimize_poses_only: bool,
    use_robust: bool = True,
) -> Tuple[List[Optional[Pose2]], Dict[int, Point2]]:
    """Execute SLAM in the 2d plane.

    Uses odometry measurements and possibly also bearing-range measurements.
    Based on:
    https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py

    Args:
        wTi_list_init: initialization for global poses.
        i2Ti1_measurement: odometry measurements
        landmark_positions_init:
        landmark_measurements:
        optimize_poses_only: whether to ignore landmarks and perform pose-graph optimization only.
        use_robust: whether to use robust optimization (Huber loss).

    Returns:
        wTi_list: list of length (N,) representing optimized global poses.
        landmark_positions: mapping from landmark ID to 2d position.
            Note: returns an empty dict if landmarks were not used (i.e. for PGO-only case).
    """
    # measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, MEASUREMENT_NOISE_SIGMA)

    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
    # MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0]))

    if use_robust:
        huber_loss = gtsam.noiseModel.mEstimator.Huber(1.345)
        PRIOR_NOISE = gtsam.noiseModel.Robust(huber_loss, PRIOR_NOISE)
        ODOMETRY_NOISE = gtsam.noiseModel.Robust(huber_loss, ODOMETRY_NOISE)
        MEASUREMENT_NOISE = gtsam.noiseModel.Robust(huber_loss, MEASUREMENT_NOISE)

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # find the first pano for which the initial pose estimate is not None
    origin_pano_id = np.argmax([wTi is not None for wTi in wTi_list_init])

    # keys corresponding to unknown variables in the factor graph.
    # Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    graph.add(PriorFactorPose2(X(origin_pano_id), gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

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

    # Create initial estimate
    initial_estimate = gtsam.Values()

    for i, wTi in enumerate(wTi_list_init):
        if wTi is None:
            continue
        initial_estimate.insert(X(i), wTi)

    if not optimize_poses_only:
        for l, wTl in landmark_positions_init.items():
            initial_estimate.insert(L(l), landmark_positions_init[l])

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


def execute_planar_slam(
    measurements: List[EdgeClassification],
    gt_floor_pg: "PoseGraph2d",
    hypotheses_save_root: str,
    building_id: str,
    floor_id: str,
    wSi_list: List[Sim2],
    plot_save_dir: str,
    use_axis_alignment: bool,
    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair],
    optimize_poses_only: bool = False,
    verbose: bool = True,
) -> None:
    """Gathers odometry and landmark measurements, and then executes planar Pose(2) SLAM.

    Args:
        measurements: per-edge measurements.
        gt_floor_pg: ground truth 2d pose graph for this ZinD building floor.
        hypotheses_save_root:
        building_id: unique ID of a ZInD building.
        floor_id: unique ID of floor of a ZInD building.
        wSi_list:
        plot_save_dir:
        use_axis_alignment: whether to refine relative rotations by vanishing angle.
        per_edge_wdo_dict
        optimize_poses_only: whether to only optimize poses, or instead to optimize poses + landmarks jointly.
        verbose:

    Returns:
        report: reconstruction report.
    """
    if (not optimize_poses_only) or use_axis_alignment:
        floor_pose_graphs = hnet_prediction_loader.load_inferred_floor_pose_graphs(
            query_building_id=building_id, raw_dataset_dir=RAW_DATASET_DIR
        )
        pano_dict_inferred = floor_pose_graphs[floor_id].nodes

    wTi_list_init = [
        Pose2(Rot2.fromDegrees(wSi.theta_deg), wSi.translation) if wSi is not None else None for wSi in wSi_list
    ]

    # # as (x,y,theta). We don't use a dict, as we may have multiple measurements for each pair of poses.
    i2Ti1_measurements = []
    for m in measurements:
        i2Si1 = edge_classification.get_alignment_hypothesis_for_measurement(
            m, hypotheses_save_root, building_id, floor_id
        )

        if use_axis_alignment:
            edge_wdo_pair = per_edge_wdo_dict[(m.i1, m.i2)]
            i2rSi1 = axis_alignment_utils.align_pair_measurement_by_vanishing_angle(
                i1=m.i1,
                i2=m.i2,
                i2Si1=i2Si1,
                edge_wdo_pair=edge_wdo_pair,
                pano_dict_inferred=pano_dict_inferred,
                visualize=False,
            )
            if i2rSi1 is not None:
                # use the corrected version
                i2Si1 = i2rSi1

        theta_rad = np.deg2rad(i2Si1.theta_deg)
        x, y = i2Si1.translation
        om = OdometryMeasurement(m.i1, m.i2, Pose2(x, y, theta_rad))
        i2Ti1_measurements.append(om)

    # Add Bearing-Range measurements to different landmarks
    # angle to reach landmark, from given pose.
    # for each start and end vertex of W/D/O, denoted by (s,e)
    landmark_measurements = []
    landmark_positions_init = {}

    if not optimize_poses_only:
        # will use loaded 3d point locations for each WDO inside pano_dict_inferred.
        tracks_2d = data_association.perform_data_association(measurements, pano_dict_inferred)
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
                landmark_measurements += [
                    BearingRangeMeasurement(pano_id=m.i, l_idx=j, bearing_deg=bearing_deg, range=range)
                ]

                pt_w = wTi_list_init[m.i].transformFrom(m.uv)
                # plt.scatter(pt_w[0], pt_w[1], 10, color=color, marker="+")
                # plt.text(pt_w[0], pt_w[1], f"j={j},i={m.i}", color=color)

        #         draw_coordinate_frame(wTi_list_init[m.i], text=str(m.i))

        # plt.axis("equal")
        # plt.show()

    wTi_list, landmark_positions = planar_slam(
        wTi_list_init=wTi_list_init,
        i2Ti1_measurements=i2Ti1_measurements,
        landmark_positions_init=landmark_positions_init,
        landmark_measurements=landmark_measurements,
        optimize_poses_only=optimize_poses_only
    )
    wSi_list = [None] * len(wTi_list)
    for i, wTi in enumerate(wTi_list):
        if wTi is None:
            continue
        wSi_list[i] = Sim2(R=wTi.rotation().matrix(), t=wTi.translation(), s=1.0)


    est_floor_pose_graph = PoseGraph2d.from_wSi_list(wSi_list, gt_floor_pg)
    report = FloorReconstructionReport.from_est_floor_pose_graph(
        est_floor_pose_graph,
        gt_floor_pose_graph=gt_floor_pg,
        plot_save_dir=plot_save_dir
    )
    #graph_rendering_utils.draw_multigraph(measurements, est_floor_pose_graph, raw_dataset_dir=raw_dataset_dir, confidence_threshold=0.93)

    return report


def draw_coordinate_frame(wTi: Pose2, text: str) -> None:
    """Draw a 2d coordinate frame using matplotlib."""
    """
    Args:
        wTi: camera pose in global frame.
        text:
    """
    # camera center
    cc = wTi.translation()
    plt.text(cc[0], cc[1], text)

    # loop over the x-axis and then y-axis
    for a, color in zip(range(2), ["r", "g"]):
        axis = np.zeros(2)
        axis[a] = 1
        w_axis = wTi.transformFrom(axis)
        plt.plot([cc[0], w_axis[0]], [cc[1], w_axis[1]], c=color)


def bearing_range_from_vertex(v: Tuple[float, float]) -> float:
    """Return bearing in degrees and range.

    Args:
        v: coordinate in Cartesian frame.

    Returns:
        bearing: angle (in degrees) to vertex/landmark
        range: distance (in 2d) to vertex/landmark.
    """
    x, y = v
    bearing_rad = np.arctan2(y, x)
    range = np.linalg.norm(v)
    return np.rad2deg(bearing_rad), range
