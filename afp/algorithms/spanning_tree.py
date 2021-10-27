
"""
Utilities for greedily assembling a spanning tree, by shortest paths to an origin node within
a single connected component.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
import networkx as nx
import numpy as np
import scipy
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Point3, Pose2, Rot2, Rot3, Unit3

import afp.utils.rotation_utils as rotation_utils
from afp.algorithms.rotation_averaging import globalaveraging2d
from afp.utils.rotation_utils import rotmat2d, rotmat2theta_deg
from afp.common.edge_classification import EdgeClassification
from afp.common.posegraph2d import REDTEXT, ENDCOLOR

logger = logger_utils.get_logger()


def greedily_construct_st(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[np.ndarray]:
    """Greedily assemble a spanning tree (not a minimum spanning tree).

    Args:
        i2Ri1_dict: relative rotations

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

    # ignore 0th node, as we already set its global pose as the origin
    for dst_node in cc_nodes[1:]:

        # determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
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


def greedily_construct_st_Sim2(i2Si1_dict: Dict[Tuple[int, int], Sim2], verbose: bool = True) -> Optional[List[np.ndarray]]:
    """Greedily assemble a spanning tree (not a minimum spanning tree).

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
    wSi_list[origin_node] = Sim2(R=np.eye(2),t=np.zeros(2), s=1.0)

    G = nx.Graph()
    G.add_edges_from(edges)

    # ignore 0th node, as we already set its global pose as the origin
    for dst_node in cc_nodes[1:]:

        # determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
        path = nx.shortest_path(G, source=origin_node, target=dst_node)

        if verbose:
            print(f"\tPath from {origin_node}->{dst_node}: {str(path)}")

        wSi = Sim2(R=np.eye(2),t=np.zeros(2), s=1.0)
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


def test_greedily_construct_st():
    """
    Below, we specify i1 < i2 for all edges (i1,i2)

    Graph topology:

              | 2     | 3
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    0         1       4

    """
    # ground truth 2d rotations
    wRi_list_gt = [
        rotmat2d(0),
        rotmat2d(90),
        rotmat2d(0),
        rotmat2d(0),
        rotmat2d(90)
        # Rot3(), # 0
        # Rot3.Rz(np.deg2rad(90)), # 1
        # Rot3(), # 2
        # Rot3(), # 3
        # Rot3.Rz(np.deg2rad(90))# 4
    ]

    i2Ri1_dict = {}
    for i1 in range(4):
        i2 = i1 + 1
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.T @ wRi1

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        print(f"EDGE_SE2 {i1} {i2} 0 0 {rotmat2theta_deg(i2Ri1)}")

    wRi_list_greedy = greedily_construct_st(i2Ri1_dict)

    # expected angles
    wRi_list_euler_deg_exp = [
        0,
        90,
        0,
        0,
        90,
    ]
    # wRi_list_euler_deg_est = [ np.rad2deg(wRi.xyz()).tolist() for wRi in wRi_list_greedy]
    wRi_list_euler_deg_est = [rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est

    wRi_list_shonan = globalaveraging2d(i2Ri1_dict)

    wRi_list_shonan_est = [rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # Note that:
    # 360 - 125.812 =  234.188
    # 234.188 - 144.188 = 90.0
    wRi_list_shonan_exp = [-125.81, 144.18, -125.81, -125.81, 144.18]
    assert np.allclose(wRi_list_shonan_exp, wRi_list_shonan_est, atol=0.01)

    # # cast to a 2d problem
    # wRi_list_Rot3_shonan = global_averaging(i2Ri1_dict)
    # wRi_list_shonan = posegraph3d_to_posegraph2d(wRi_list_Rot3_shonan)

    # wRi_list_shonan_est = [ rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # # corresponds to 110.5 and 200.4 degrees (as if 0 and 90 degrees)
    # wRi_list_shonan_exp = [110.52, -159.61, 110.52, 110.52, -159.61]
    # assert np.allclose(wRi_list_shonan_exp, wRi_list_shonan_est, atol=0.01)


def test_greedily_construct_st2():
    """
    Below, we do NOT specify i1 < i2 for all edges (i1,i2).

    Graph topology:

              | 3     | 0
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    4         1       2

    """
    # ground truth 2d rotations
    wRi_list_gt = [
        rotmat2d(0),  # 0
        rotmat2d(90),  # 1
        rotmat2d(90),  # 2
        rotmat2d(0),
        rotmat2d(0)
        # Rot3(), # 0
        # Rot3.Rz(np.deg2rad(90)), # 1
        # Rot3(), # 2
        # Rot3(), # 3
        # Rot3.Rz(np.deg2rad(90))# 4
    ]

    edges = [(1, 4), (1, 3), (0, 3), (0, 2)]

    i2Ri1_dict = {}
    for (i1, i2) in edges:
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.T @ wRi1

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        print(f"EDGE_SE2 {i1} {i2} 0 0 {rotmat2theta_deg(i2Ri1)}")

    import pdb

    pdb.set_trace()
    wRi_list_greedy = greedily_construct_st(i2Ri1_dict)

    # expected angles
    wRi_list_euler_deg_exp = [
        0,
        90,
        90,
        0,
        0,
    ]
    # wRi_list_euler_deg_est = [ np.rad2deg(wRi.xyz()).tolist() for wRi in wRi_list_greedy]
    wRi_list_euler_deg_est = [rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est


@dataclass
class RelativePoseMeasurement:
    i1: int
    i2: int
    i2Si1: Sim2



def ransac_spanning_trees(
    high_conf_measurements: List[RelativePoseMeasurement], num_hypotheses: int = 10, min_num_edges_for_hypothesis: int = 20
) -> List[Optional[Sim2]]:
    """
    Generate random spanning trees
    https://arxiv.org/pdf/1611.07451.pdf

    Args:
        high_conf_measurements
    """
    K = len(high_conf_measurements)

    avg_rot_error_best = float("inf")
    avg_trans_error_best = float("inf")
    best_wSi_list = None

    SAMPLING_FRAC = 0.95  # 0.8

    if min_num_edges_for_hypothesis is None:
        min_num_edges_for_hypothesis = int(math.ceil(SAMPLING_FRAC * K)) # or 2 * number of unique nodes, or 80% of total, etc.

    max_unique_hypotheses = int(scipy.special.binom(K, min_num_edges_for_hypothesis)) # {n choose k}
    num_hypotheses = min(max_unique_hypotheses, num_hypotheses)

    # generate random hypotheses
    hypotheses = [np.random.choice(a=K, size=min_num_edges_for_hypothesis, replace=False) for _ in range(num_hypotheses)]

    #for each hypothesis
    for h_counter, h_idxs in enumerate(hypotheses):

        hypothesis_measurements = [m for k, m in enumerate(high_conf_measurements) if k in h_idxs]

        i2Si1_dict = {}
        # randomly overwrite by order
        for m in hypothesis_measurements:
            i2Si1_dict[(m.i1,m.i2)] = m.i2Si1

        # create the i2Si1_dict
        # generate a spanning tree greedily
        wSi_list = greedily_construct_st_Sim2(i2Si1_dict, verbose=False)

        if wSi_list is None:
            import pdb; pdb.set_trace()

        avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(high_conf_measurements, wSi_list)

        print(
            f"Hypothesis {h_counter} had Rot errors {avg_rot_error:.1f}(mean) {med_rot_error:.1f} (med), Trans errors {avg_trans_error:.2f}(mean) {med_trans_error:.2f} (med)"
        )
        
        # TODO: or could use MEDIAN instead
        #if most inliers so far, set as best hypothesis
        if avg_rot_error <= avg_rot_error_best and avg_trans_error <= avg_trans_error_best:
            avg_rot_error_best = avg_rot_error
            avg_trans_error_best = avg_trans_error
            best_wSi_list = wSi_list

    avg_rot_error, med_rot_error, avg_trans_error, med_trans_error = compute_hypothesis_errors(high_conf_measurements, best_wSi_list)
    print(REDTEXT + f"Chose hypothesis with {avg_rot_error:.1f}, {med_rot_error:.1f}, {avg_trans_error:.2f}, {med_trans_error:.2f} " + ENDCOLOR)
    return best_wSi_list


def compute_hypothesis_errors(
    high_conf_measurements: List[RelativePoseMeasurement], wSi_list: List[Optional[Sim2]]
) -> Tuple[float, float, float, float]:
    """ """
    rot_errors = []
    trans_errors = []
    #count the number of inliers / error against ALL measurements now.
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



def convert_Pose2_to_Sim2(i2Ti1: Pose2) -> Sim2:
    """ """
    return Sim2(R=i2Ti1.rotation().matrix(), t=i2Ti1.translation(), s=1.0)


def test_ransac_spanning_trees() -> None:
    """Toy scenario with 3 nodes (3 accurate edges, and 1 noisy edge)."""

    np.random.seed(0)

    wT0 = Pose2(Rot2(), np.array([0,0]))
    wT1 = Pose2(Rot2(), np.array([2,0]))
    wT2 = Pose2(Rot2(), np.array([2,2]))

    wT2_noisy = Pose2(Rot2(), np.array([3,3]))

    i1Ti0 = wT1.between(wT0)
    i2Ti1 = wT2.between(wT1)
    i2Ti0 = wT2.between(wT0)

    i2Ti0_noisy = wT2_noisy.between(wT0)

    i1Si0 = convert_Pose2_to_Sim2(i1Ti0)
    i2Si1 = convert_Pose2_to_Sim2(i2Ti1)
    i2Si0 = convert_Pose2_to_Sim2(i2Ti0)
    i2Si0_noisy = convert_Pose2_to_Sim2(i2Ti0_noisy)

    # fmt: off
    high_conf_measurements = [
        RelativePoseMeasurement(i1=0, i2=1, i2Si1=i1Si0),
        RelativePoseMeasurement(i1=1, i2=2, i2Si1=i2Si1),
        RelativePoseMeasurement(i1=0, i2=2, i2Si1=i2Si0),
        RelativePoseMeasurement(i1=0, i2=2, i2Si1=i2Si0_noisy)
    ]
    # fmt: on
    wSi_list = ransac_spanning_trees(high_conf_measurements, num_hypotheses=10, min_num_edges_for_hypothesis=3)

    assert len(wSi_list) == 3
    assert wSi_list[0] == convert_Pose2_to_Sim2(wT0)
    assert wSi_list[1] == convert_Pose2_to_Sim2(wT1)
    assert wSi_list[2] == convert_Pose2_to_Sim2(wT2)


if __name__ == "__main__":
    test_ransac_spanning_trees()
