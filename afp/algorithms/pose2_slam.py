

import gtsam

def pose2slam_lago():
	"""
	Linear Approximation for Graph Optimization (LAGO)
	Based off of https://github.com/borglab/gtsam/blob/develop/examples/Pose2SLAMExample_lago.cpp
	
	L. Carlone, R. Aragues, J. Castellanos, and B. Bona, A fast and accurate
	approximation for planar pose graph optimization, IJRR, 2014.
	
	L. Carlone, R. Aragues, J.A. Castellanos, and B. Bona, A linear approximation
	for graph-based simultaneous localization and mapping, RSS, 2011.
	"""
	g2oFile = findExampleDataFile("noisyToyGraph.txt")

	NonlinearFactorGraph::shared_ptr graph
	Values::shared_ptr initial
	boost::tie(graph, initial) = readG2o(g2oFile)

	# Add prior on the pose having index (key) = 0
	auto priorModel = noiseModel::Diagonal::Variances(Vector3(1e-6, 1e-6, 1e-8))
	graph->addPrior(0, Pose2(), priorModel)
	graph->print()

	print("Computing LAGO estimate")
	Values estimateLago = lago::initialize(*graph)
	std::cout << "done!" << std::endl;

	if (argc < 3):
		estimateLago.print("estimateLago");
	else:
		const string outputFile = argv[2]
		print("Writing results to file: ", outputFile)
		NonlinearFactorGraph::shared_ptr graphNoKernel;
		Values::shared_ptr initial2;
		boost::tie(graphNoKernel, initial2) = readG2o(g2oFile)
		writeG2o(*graphNoKernel, estimateLago, outputFile)
		print("done! ")

	return 


def pose2slam():
	"""
	Based off of https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/Pose2SLAMExample.py
	"""

	def vector3(x, y, z):
	    """Create 3d double numpy array."""
	    return np.array([x, y, z], dtype=float)

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

	plt.axis('equal')
	plt.show()

