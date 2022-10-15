"""Utilities for greedily assembling a spanning tree.

Greedy search identifies shortest paths from each node to an origin node (within a single connected component).
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gtsfm.utils.graph as graph_utils
import networkx as nx
import numpy as np
import scipy

import salve.utils.rotation_utils as rotation_utils
from salve.common.posegraph2d import ENDCOLOR, REDTEXT
from salve.common.sim2 import Sim2

logger = logging.getLogger(__name__)


@dataclass
class RelativePoseMeasurement:
    """Represents a Sim(2) relative pose, with named panos for To() and From() indices."""
    i1: int
    i2: int
    i2Si1: Sim2


def greedily_construct_st(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[np.ndarray]:
    """Greedily assemble a spanning tree (not a minimum spanning tree).

    Args:
        i2Ri1_dict: mapping from pano ID pair (i1,i2) to relative rotation i2Ri1.

    Returns:
        wRi_list: global 2d rotations
    """
    # find the largest connected component
    edges = i2Ri1_dict.keys()

    num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
    cc_nodes = sorted(cc_nodes)

    wRi_list = [None] * num_nodes
    # choose origin node
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
    """Greedily assemble a spanning tree (not a minimum spanning tree).

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
        i2Ri1_dict: relative rotations

    Returns:
        wSi_list: global 2d Sim(2) transformations / poses, or None if no edges
    """
    # find the largest connected component
    edges = i2Si1_dict.keys()

    if len(edges) == 0:
        return None

    num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
    cc_nodes = sorted(cc_nodes)

    wSi_list = [None] * num_nodes
    # choose origin node
    origin_node = cc_nodes[0]
    wSi_list[origin_node] = Sim2(R=np.eye(2), t=np.zeros(2), s=1.0)

    G = nx.Graph()
    G.add_edges_from(edges)

    # ignore 0th node, as we already set its global pose as the origin
    for dst_node in cc_nodes[1:]:

        # determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
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


def ransac_spanning_trees(
    high_conf_measurements: List[RelativePoseMeasurement],
    num_hypotheses: int = 10,
    min_num_edges_for_hypothesis: int = 20,
) -> List[Optional[Sim2]]:
    """Generate global poses by sampling random spanning trees from relative pose measurements.

    See V. M. Govindu. Robustness in motion averaging. In ACCV, 2006.
    Count the number of relative motions (i.e. edges) that fall within this distance from the global motion.
    C. Olsson and O. Enqvist. Stable structure from motion for unordered image collections. In SCIA, 2011. LNCS 6688.

    Note: this may be accelerated by techniques such as https://arxiv.org/pdf/1611.07451.pdf.

    Args:
        high_conf_measurements: relative pose measurements, already pruned by confidence.
        num_hypotheses: number of hypotheses to sample in RANSAC.
        min_num_edges_for_hypothesis: minimum number of edges in each RANSAC hypothesis.
    
    Returns:
        Most probable global poses.
    """
    K = len(high_conf_measurements)

    avg_rot_error_best = float("inf")
    avg_trans_error_best = float("inf")
    best_wSi_list = None

    # Leave out 5% to allow exclusion of possible outliers.
    SAMPLING_FRAC = 0.95  # 0.8

    if min_num_edges_for_hypothesis is None:
        # Alternatively, could use 2 * number of unique nodes, or 80% of total, etc.
        min_num_edges_for_hypothesis = int(math.ceil(SAMPLING_FRAC * K))

    max_unique_hypotheses = int(scipy.special.binom(K, min_num_edges_for_hypothesis))  # {n choose k}
    num_hypotheses = min(max_unique_hypotheses, num_hypotheses)

    # Generate random hypotheses.
    hypotheses = [
        np.random.choice(a=K, size=min_num_edges_for_hypothesis, replace=False) for _ in range(num_hypotheses)
    ]

    # For each hypothesis, count inliers.
    for h_counter, h_idxs in enumerate(hypotheses):

        hypothesis_measurements = [m for k, m in enumerate(high_conf_measurements) if k in h_idxs]

        i2Si1_dict = {}
        # Randomly overwrite by order
        for m in hypothesis_measurements:
            i2Si1_dict[(m.i1, m.i2)] = m.i2Si1

        # Create the i2Si1_dict. Generate a spanning tree greedily.
        wSi_list = greedily_construct_st_Sim2(i2Si1_dict, verbose=False)

        if wSi_list is None:
            import pdb; pdb.set_trace()

        avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(
            high_conf_measurements, wSi_list
        )

        print(
            f"Hypothesis {h_counter} had Rot errors {avg_rot_error:.1f}(mean) {med_rot_error:.1f} (med),"
            f" Trans errors {avg_trans_error:.2f}(mean) {med_trans_error:.2f} (med)"
        )

        # If hypothesis has most inliers of any hypothesis so far, set as best hypothesis.
        # We use average error, although `median` could be used instead.
        if avg_rot_error <= avg_rot_error_best and avg_trans_error <= avg_trans_error_best:
            avg_rot_error_best = avg_rot_error
            avg_trans_error_best = avg_trans_error
            best_wSi_list = wSi_list

    avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(
        high_conf_measurements, best_wSi_list
    )
    print(
        REDTEXT
        + f"Chose hypothesis with {avg_rot_error:.1f}, {med_rot_error:.1f},"
        f" {avg_trans_error:.2f}, {med_trans_error:.2f} "
        + ENDCOLOR
    )
    return best_wSi_list


def compute_hypothesis_errors(
    high_conf_measurements: List[RelativePoseMeasurement], wSi_list: List[Optional[Sim2]]
) -> Tuple[float, float, float, float]:
    """Measure the quality of global poses (from a single random spanning tree) vs. input measurements.

    Args:
        high_conf_measurements: relative pose measurements, already pruned by confidence.
        wSi_list: list of Sim(2) poses in global frame.

    Returns:
        avg_rot_error: average rotation error (in degrees) per edge/measurement.
        med_rot_error: median rotation error (in degrees) per edge/measurement.
        avg_trans_error: average translation error per edge/measurement.
        med_trans_error: median translation error per edge/measurement.
    """
    rot_errors = []
    trans_errors = []
    # count the number of inliers / error against ALL measurements now.
    # Can use just rotation error, or just translation error, or both
    for m in high_conf_measurements:

        if m.i1 >= len(wSi_list) or m.i2 >= len(wSi_list):
            # or set some standard penalty on a less complete graph?
            continue

        wSi1 = wSi_list[m.i1]
        wSi2 = wSi_list[m.i2]

        if wSi1 is None or wSi2 is None:
            # or set some standard penalty on a less complete graph?
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
