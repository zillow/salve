"""Utilities for greedily assembling a spanning tree.

Greedy search identifies shortest paths from each node to an origin node (within a single connected component).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import gtsfm.utils.graph as graph_utils
import networkx as nx
import numpy as np
import scipy

import salve.utils.rotation_utils as rotation_utils
import salve.utils.graph_rendering_utils as graph_rendering_utils
import salve.common.edge_classification as edge_classification
from salve.common.edge_classification import EdgeClassification
from salve.common.posegraph2d import PoseGraph2d, ENDCOLOR, REDTEXT
from salve.common.sim2 import Sim2

logger = logging.getLogger(__name__)


def greedily_construct_st(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[np.ndarray]:
    """Greedily assemble a spanning tree (not a minimum spanning tree) from Rot(2) measurements.

    Args:
        i2Ri1_dict: mapping from pano ID pair (i1,i2) to relative rotation i2Ri1.

    Returns:
        wRi_list: global 2d rotations
    """
    edges = i2Ri1_dict.keys()

    num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

    # Find the largest connected component
    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
    cc_nodes = sorted(cc_nodes)

    wRi_list = [None] * num_nodes
    # Choose origin node
    origin_node = cc_nodes[0]
    wRi_list[origin_node] = np.eye(2)

    G = nx.Graph()
    G.add_edges_from(edges)

    # Ignore 0th node, as we already set its global pose as the origin
    for dst_node in cc_nodes[1:]:

        # Determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
        path = nx.shortest_path(G, source=origin_node, target=dst_node)

        wRi = np.eye(2)
        for (i1, i2) in zip(path[:-1], path[1:]):

            # i1, i2 may not be in sorted order here. May need to reverse ordering
            if i1 < i2:
                i1Ri2 = i2Ri1_dict[(i1, i2)].T  # use inverse
            else:
                i1Ri2 = i2Ri1_dict[(i2, i1)]

            # wRi = wR0 * 0R1
            wRi = wRi @ i1Ri2

        wRi_list[dst_node] = wRi

    return wRi_list


def greedily_construct_st_Sim2(
    i2Si1_dict: Dict[Tuple[int, int], Sim2], verbose: bool = True
) -> Optional[List[np.ndarray]]:
    """Greedily assembles a spanning tree (not a minimum spanning tree) from Sim(2) measurements.

    Find the largest connected component, then sort the nodes in the largest CC by their pano ID.
    Choose the smallest pano ID in that CC as the origin. Then, walk through the rest of the nodes
    in the CC one at a time, and for each next node, find the shortest path to it through the graph.
    Walk along each edge in that path, and chain the relative pose to the previous absolute (global) pose,
    to get a new absolute (global) pose. Cache away this new global pose in a dictionary of Sim(2) objects.

    Intuition: shortest path is a way to try to minimize drift.

    Note: The scale of the Sim(2) objects is set to 1, though, this is just an artifact of the old way I
    had implemented this.

    TODOs:
        - could use a MST.
        - likely better to start at degree with highest degree to minimize drift the most.

    Args:
        i2Si1_dict: relative poses as Sim(2).

    Returns:
        wSi_list: global 2d Sim(2) transformations / poses, or None if no edges
    """
    edges = i2Si1_dict.keys()

    if len(edges) == 0:
        return None

    num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

    # Find the largest connected component.
    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
    cc_nodes = sorted(cc_nodes)

    wSi_list = [None] * num_nodes
    # Choose origin node.
    origin_node = cc_nodes[0]
    wSi_list[origin_node] = Sim2(R=np.eye(2), t=np.zeros(2), s=1.0)

    G = nx.Graph()
    G.add_edges_from(edges)

    # Ignore 0th node, as we already set its global pose as the origin.
    for dst_node in cc_nodes[1:]:

        # Determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
        path = nx.shortest_path(G, source=origin_node, target=dst_node)

        if verbose:
            print(f"\tPath from {origin_node}->{dst_node}: {str(path)}")

        wSi = Sim2(R=np.eye(2), t=np.zeros(2), s=1.0)
        for (i1, i2) in zip(path[:-1], path[1:]):

            # i1, i2 may not be in sorted order here. May need to reverse ordering
            if i1 < i2:
                i1Si2 = i2Si1_dict[(i1, i2)].inverse()  # use inverse
            else:
                i1Si2 = i2Si1_dict[(i2, i1)]

            # wRi = wR0 * 0R1
            wSi = wSi.compose(i1Si2)

        wSi_list[dst_node] = wSi

    return wSi_list


def compute_objective_function_improvement(
    avg_rot_error: float,
    avg_rot_error_best: float,
    avg_trans_error: float,
    avg_trans_error_best: float,
    num_poses_estimated: int,
    num_poses_estimated_best: int,
) -> float:
    """Computes cost function should trade-off maximizing the number of poses in the ST, while minimizing errors
        per pano-pano edge.

    Search for optimal point on Pareto curve.
    Absolute vs. relative wins for accuracy (divide by some absolute number that means something, be relative to that )

    Returns:
        Scalar (positive is win, negative is loss)
    """
    # Naive: ignore completeness
    # avg_rot_error <= avg_rot_error_best and avg_trans_error <= avg_trans_error_best:

    EPS = 1e-10

    # Compute relative improvements
    rot_improvement = (avg_rot_error_best - avg_rot_error) / 5  #  avg_rot_error_best
    trans_improvement = (avg_trans_error_best - avg_trans_error) / 1  # avg_trans_error_best
    loc_completeness_improvement = -(num_poses_estimated_best - num_poses_estimated) / (num_poses_estimated_best + EPS)

    print(f"\tRot win: {rot_improvement:.2f}")
    print(f"\tTrans win: {trans_improvement:.2f}")
    print(f"\tLoc % win: {loc_completeness_improvement:.2f}")

    # Normalize all to unit-based scale, for fair scaling.
    return rot_improvement + trans_improvement + 1.33 * loc_completeness_improvement


def ransac_spanning_trees(
    measurements: List[EdgeClassification],
    num_hypotheses: int = 10,
    # min_num_edges_for_hypothesis: int = 10,
    gt_floor_pose_graph: Optional[PoseGraph2d] = None,
    visualize: bool = True,
    sampling_fraction: float = 0.5  # 0.95
) -> List[Optional[Sim2]]:
    """Generate global poses by sampling random spanning trees from relative pose measurements.

    Sample with a bias towards images captured close by each other (by capture order).

    See V. M. Govindu. Robustness in motion averaging. In ACCV, 2006.
    Count the number of relative motions (i.e. edges) that fall within this distance from the global motion.
    C. Olsson and O. Enqvist. Stable structure from motion for unordered image collections. In SCIA, 2011. LNCS 6688.

    Note: this may be accelerated by techniques such as https://arxiv.org/pdf/1611.07451.pdf.

    Args:
        measurements: relative pose measurements, likely already pruned by confidence.
        num_hypotheses: number of hypotheses to sample in RANSAC.
        min_num_edges_for_hypothesis: minimum number of edges in each RANSAC hypothesis.
        gt_floor_pose_graph: ground truth floor pose graph.
        visualize:
        sampling_fraction: percent of poses to sample for hypothesis. We Leave out X% of measurements to allow
            exclusion of possible outliers.

    Returns:
        Most probable global poses.
        Best hypothesis (subset of input measurements).
    """
    if len(measurements) == 0:
        raise ValueError("At least one edge prediction/measurement must be provided.")

    K = len(measurements)

    avg_rot_error_best = float("inf")  # lower is better
    avg_trans_error_best = float("inf")  # lower is better
    num_poses_estimated_best = 0  # -float("inf") # lower is better
    best_wSi_list = None

    min_num_edges_for_hypothesis = None
    if min_num_edges_for_hypothesis is None:
        # Alternatively, could use 2 * number of unique nodes, or 80% of total, etc.
        min_num_edges_for_hypothesis = int(math.ceil(sampling_fraction * K))
    print(f"Sampling {min_num_edges_for_hypothesis} edges for each hypothesis.")

    try:
        # Large values will make return infinity.
        max_unique_hypotheses = int(scipy.special.binom(K, min_num_edges_for_hypothesis))  # {n choose k}
    except:
        max_unique_hypotheses = 1000
    num_hypotheses = min(max_unique_hypotheses, num_hypotheses)

    # Sample per confidence.
    # probabilities = np.array([m.prob for m in measurements])
    # Sample randomly
    # probabilities = np.ones(K)
    capture_distance = np.array([np.absolute(m.i2 - m.i1) for m in measurements])
    probabilities = 1 / capture_distance

    probabilities = probabilities / np.sum(probabilities)
    # Generate random hypotheses.
    hypotheses = [
        np.random.choice(a=K, size=min_num_edges_for_hypothesis, replace=False, p=probabilities)
        for _ in range(num_hypotheses)
    ]

    best_hypothesis = None

    # For each hypothesis, count inliers.
    for h_counter, h_idxs in enumerate(hypotheses):

        hypothesis_measurements = [m for k, m in enumerate(measurements) if k in h_idxs]

        i2Si1_dict = {}
        edge_classification_dict = {}
        # Randomly overwrite by order
        for m in hypothesis_measurements:
            i2Si1_dict[(m.i1, m.i2)] = m.i2Si1
            edge_classification_dict[(m.i1, m.i2)] = m

        if visualize:
            two_view_reports_dict = edge_classification.create_two_view_reports_dict_from_edge_classification_dict(
                edge_classification_dict, gt_floor_pose_graph
            )

        # Create the i2Si1_dict. Generate a spanning tree greedily.
        wSi_list = greedily_construct_st_Sim2(i2Si1_dict, verbose=False)

        if wSi_list is None:
            import pdb; pdb.set_trace()

        avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(
            measurements=measurements, wSi_list=wSi_list
        )
        num_poses_estimated = sum([1 if wSi is not None else 0 for wSi in wSi_list])

        print(
            f"Hypothesis {h_counter} had Rot errors {avg_rot_error:.1f}(mean) {med_rot_error:.1f} (med),"
            f" Trans errors {avg_trans_error:.2f}(mean) {med_trans_error:.2f} (med), "
            f" Num poses estimated: {num_poses_estimated}"
        )

        # Cost function should trade-off maximizing the number of poses in the ST, while minimizing errors 
        # per pano-pano edge. If hypothesis has most inliers of any hypothesis so far, set as best hypothesis.
        # We use average error, although `median` could be used instead.
        if (
            compute_objective_function_improvement(
                avg_rot_error,
                avg_rot_error_best,
                avg_trans_error,
                avg_trans_error_best,
                num_poses_estimated,
                num_poses_estimated_best,
            )
            > 0
        ):
            avg_rot_error_best = avg_rot_error
            avg_trans_error_best = avg_trans_error
            num_poses_estimated_best = num_poses_estimated
            best_wSi_list = wSi_list
            best_hypothesis = hypothesis_measurements

            print(f"Selecting hypothesis {h_counter} as next best.")

        if visualize and gt_floor_pose_graph is not None and two_view_reports_dict is not None:
            # Render the error magnitude on each edge (red is highest error, green is lowest error).
            floor_id = gt_floor_pose_graph.floor_id
            building_id = gt_floor_pose_graph.building_id
            vis_save_fpath = "random_spanning_trees/topology_by_rotation_angular_error"
            vis_save_fpath += f"/{building_id}_{floor_id}_hypothesis_{h_counter}.png"
            plot_title = f"Avg. rot error {avg_rot_error:.2f}, Avg trans error: {avg_trans_error:.2f},"
            plot_title += f" #Poses: {num_poses_estimated}"

            graph_rendering_utils.draw_graph_topology(
                edges=list(i2Si1_dict.keys()),
                gt_floor_pose_graph=gt_floor_pose_graph,
                two_view_reports_dict=two_view_reports_dict,
                title=plot_title,
                show_plot=False,
                save_fpath=vis_save_fpath,
                color_scheme="by_error_magnitude",
            )

    # TODO: throw away all edges with the highest error.
    K_consensus = len(best_hypothesis)

    avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(
        measurements, best_wSi_list
    )
    print(
        REDTEXT + f"Chose hypothesis with {avg_rot_error:.1f}, {med_rot_error:.1f},"
        f" {avg_trans_error:.2f}, {med_trans_error:.2f}. Left with {K_consensus} / {K} original measurements."
        + ENDCOLOR
    )
    return best_wSi_list, best_hypothesis


def compute_hypothesis_errors(
    measurements: List[EdgeClassification], wSi_list: List[Optional[Sim2]]
) -> Tuple[float, float, float, float]:
    """Measure the quality of global poses (from a single random spanning tree) vs. input measurements.

    Args:
        measurements: relative pose measurements, likely already pruned by confidence.
        wSi_list: list of Sim(2) poses in global frame.

    Returns:
        avg_rot_error: Average rotation error (in degrees) per edge/measurement.
        med_rot_error: Median rotation error (in degrees) per edge/measurement.
        avg_trans_error: Average translation error per edge/measurement.
        med_trans_error: Median translation error per edge/measurement.
    """
    rot_errors = []
    trans_errors = []
    # Count the number of inliers / error against ALL measurements now.
    # Can use just rotation error, or just translation error, or both
    for m in measurements:

        if m.i1 >= len(wSi_list) or m.i2 >= len(wSi_list):
            continue

        wSi1 = wSi_list[m.i1]
        wSi2 = wSi_list[m.i2]

        if wSi1 is None or wSi2 is None:
            continue

        i2Si1_simulated = wSi2.inverse().compose(wSi1)

        rot_error_deg = rotation_utils.wrap_angle_deg(angle1=i2Si1_simulated.theta_deg, angle2=m.i2Si1.theta_deg)
        trans_error = np.linalg.norm(i2Si1_simulated.translation - m.i2Si1.translation)

        rot_errors.append(rot_error_deg)
        trans_errors.append(trans_error)

    avg_rot_error = np.mean(rot_errors)
    med_rot_error = np.median(rot_errors)

    avg_trans_error = np.mean(trans_errors)
    med_trans_error = np.median(trans_errors)

    return avg_rot_error, med_rot_error, avg_trans_error, med_trans_error
