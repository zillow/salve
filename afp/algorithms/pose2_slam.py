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

from typing import Dict, List, Tuple

import numpy as np

import gtsam
from gtsam import Rot2, Point3, Pose2, PriorFactorPose2, Values
from gtsam.symbol_shorthand import X, L


# Create noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1])) # np.array([1e-6, 1e-6, 1e-8])
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

    import pdb; pdb.set_trace()

    print("Computing LAGO estimate")
    estimate_lago: Values = gtsam.lago.initialize(graph)

    max_frame_id = max([max(i1,i2) for (i1,i2) in i2Ti1_dict.keys()]) + 1
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
        (0, 3): wTi_list[3].between(wTi_list[0])
    }

    wTi_list_computed = estimate_poses_lago(i2Ti1_dict)


def pose2slam():
    """
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
