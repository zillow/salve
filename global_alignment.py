"""
Global alignment.
"""

import copy
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Rot2, Rot3

import cycle_consistency as cycle_utils

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

from vis_depth import rotmat2d
from posegraph2d import PoseGraph2d, get_gt_pose_graph


def main(hypotheses_dir: str, raw_dataset_dir: str) -> None:
    """ """
    np.random.seed(0)

    inside_triplet_percent_list = []
    accs = []

    label_dict = {"incorrect_alignment": 0, "gt_alignment_approx": 1}

    all_floor_rot_errs = []

    building_ids = [int(Path(dirpath).stem) for dirpath in glob.glob(f"{hypotheses_dir}/*")]

    for building_id in building_ids:

        if building_id != 1442:
            continue

        # if len(inside_triplet_percent_list) > 10:
        # 	break

        floor_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_dir}/{building_id}/*")]
        for floor_id in floor_ids:

            # if not (building_id == 338 and floor_id == "floor_01"):
            # 	continue

            logger.info(f"Building {building_id}, floor {floor_id}")

            floor_label_idxs = []
            floor_sim2_json_fpaths = []

            for label_type, label_idx in label_dict.items():
                label_json_fpaths = glob.glob(f"{hypotheses_dir}/{building_id}/{floor_id}/{label_type}/*")
                label_idxs = [label_idx] * len(label_json_fpaths)

                floor_sim2_json_fpaths.extend(label_json_fpaths)
                floor_label_idxs.extend(label_idxs)

            floor_label_idxs = np.array(floor_label_idxs)

            # TODO: cache all of the model results beforehand (suppose we randomly pollute 8.5% of the results)
            POLLUTION_FRAC = 0.085

            num_floor_labels = len(floor_label_idxs)
            idxs_to_pollute = np.random.choice(a=num_floor_labels, size=int(POLLUTION_FRAC * num_floor_labels))

            corrupted_floor_label_idxs = copy.deepcopy(floor_label_idxs)
            corrupted_floor_label_idxs[idxs_to_pollute] = 1 - corrupted_floor_label_idxs[idxs_to_pollute]

            # for a single floor, find all of the triplets
            two_view_reports_dict = {}

            i2Ri1_dict = {}
            i2ti1_dict = {}

            floor_pano_ids = []
            for sim2_json_fpath, gt_class_idx in zip(floor_sim2_json_fpaths, floor_label_idxs):

                if gt_class_idx != 1:
                    continue

                i1, i2 = Path(sim2_json_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                floor_pano_ids.append(i1)
                floor_pano_ids.append(i2)

            for sim2_json_fpath, corrupted_class_idx, gt_class_idx in zip(
                floor_sim2_json_fpaths, corrupted_floor_label_idxs, floor_label_idxs
            ):

                # add if potentially corrupted label said it was a match
                if corrupted_class_idx != 1:
                    continue

                # if gt_class_idx != 1:
                # 	continue

                i1, i2 = Path(sim2_json_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                i2Ti1 = Sim2.from_json(json_fpath=sim2_json_fpath)
                i2Ri1_dict[(i1, i2)] = i2Ti1.rotation
                i2ti1_dict[(i1, i2)] = i2Ti1.translation

                two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(gt_class=gt_class_idx)

            # print(i2Ri1_dict.keys())

            method = "shonan" # "greedy"  # 
            if method == "shonan":
                import pdb; pdb.set_trace()
                wRi_list = globalaveraging2d(i2Ri1_dict)
                # wRi_list_Rot3 = global_averaging(i2Ri1_dict)
                # #print(wRi_list_Rot3)

                # # # have to align the pose graphs up to a Sim(3) transformation
                # # import gtsfm.utils.geometry_comparisons as comp_utils

                # # wRi_list_Rot3 = comp_utils.align_rotations(wRi_list_Rot3_gt, wRi_list_Rot3)
                # wRi_list = posegraph3d_to_posegraph2d(wRi_list_Rot3)

            elif method == "greedy":
                wRi_list = greedily_construct_st(i2Ri1_dict)


            est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)
            gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

            mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)

            # plt.hist(all_floor_rot_errs, bins=10)
            # plt.ylabel("Counts")
            # plt.xlabel("Absolute rotation error (degrees)")
            # plt.show()

            import pdb

            pdb.set_trace()

            # try greedy composition
            # draw the edges in a top-down floor plan, visible edges
            # use confidence value. would be good for it to be a reliable number.
            # visualize the O(n^2k^2) graph, put it in a confusion matrix

            # triplets = cycle_utils.extract_triplets_adjacency_list_intersection(i2Ri1_dict)

            # present_in_any_triplet = np.zeros((len(floor_pano_ids)), dtype=bool)
            # # for each node
            # for i, node in enumerate(floor_pano_ids):
            # 	# check if present in any triplet
            # 	present_in_any_triplet[i] = node_present_in_any_triplet(node, triplets)

            # #import pdb; pdb.set_trace()

            # inside_triplet_percent = np.mean(present_in_any_triplet)
            # logger.info(f"Building {building_id}, floor {floor_id}, % nodes in any triplet? {inside_triplet_percent:.3f}")

            # inside_triplet_percent_list.append(inside_triplet_percent)

            # check which triplets are self consistent. if so, admit the 3 edges to the graph
            # i2Ri1_consistent, i2Ui1_consistent = cycle_utils.filter_to_cycle_consistent_edges(
            #  			i2Ri1_dict,
            #  			i2ti1_dict,
            #  			two_view_reports_dict,
            #  			visualize=False
            #  		)

            # # import pdb; pdb.set_trace()

            # predictions = []
            # for (i1,i2) in i2Ri1_consistent.keys():
            # 	predictions.append(two_view_reports_dict[(i1,i2)].gt_class)

            # predictions = np.array(predictions)
            # acc = np.mean(predictions)
            # accs.append(acc)
            # logger.info(f"Building {building_id}, floor {floor_id}, Mean accuracy: {acc:.2f}\n")

    plt.scatter(inside_triplet_percent_list, accs)
    plt.show()

    plt.hist(inside_triplet_percent_list, bins=50)
    plt.xlabel("Fraction of nodes found in any triplet")
    plt.ylabel("Counts")
    plt.show()


# def wrap_angle_deg(angles: np.ndarray, period: float = 360) -> np.ndarray:
#     """Map angles (in degrees) from domain [-∞, ∞] to [0, 180). This function is
#         the inverse of `np.unwrap`.
#     Returns:
#         Angles (in radians) mapped to the interval [0, 180).
#     """
#     # Map angles to [0, ∞].
#     angles = np.abs(angles)

#     # Calculate floor division and remainder simultaneously.
#     divs, mods = np.divmod(angles, period)

#     # Select angles which exceed specified period.
#     angle_complement_mask = np.nonzero(divs)

#     # Take set complement of `mods` w.r.t. the set [0, π].
#     # `mods` must be nonzero, thus the image is the interval [0, π).
#     angles[angle_complement_mask] = period - mods[angle_complement_mask]
#     return angles




def posegraph2d_to_posegraph3d(wRi_list: List[Optional[np.ndarray]]) -> List[Optional[Rot3]]:
    """ """
    num_images = len(wRi_list)
    wRi_list_Rot3 = [None] * num_images
    for i, wRi in enumerate(wRi_list_Rot3):

        if wRi is None:
            continue

        wRi_Rot3 = np.eye(3)
        wRi_Rot3[:2, :2] = wRi

        wRi_list_Rot3[i] = Rot3(wRi_Rot3)
    return wRi_list_Rot3


def posegraph3d_to_posegraph2d(wRi_list_Rot3: List[Optional[Rot3]]) -> List[Optional[np.ndarray]]:
    """ """
    num_images = len(wRi_list_Rot3)
    wRi_list = [None] * num_images
    for i, wRi_Rot3 in enumerate(wRi_list_Rot3):

        if wRi_Rot3 is None:
            continue

        wRi = wRi_Rot3.matrix()[:2, :2]
        wRi_list[i] = wRi
    return wRi_list


def global_averaging(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[Optional[Rot3]]:
    """Lift 2d rotation matrix to 3d, then run Shonan rotation averaging.


    Args:
        i2Ri1_dict: mapping from image pair indices to 2x2 array representing 2d rotation matrix
    """
    i2Ri1_dict_Rot3 = {}

    max_pano_id = 0

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():

        R = np.eye(3)
        R[:2, :2] = i2Ri1

        i2Ri1_dict_Rot3[(i1, i2)] = Rot3(R)

        max_pano_id = max(i1, max_pano_id)
        max_pano_id = max(i2, max_pano_id)

    # import pdb; pdb.set_trace()
    from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging

    shonan_obj = ShonanRotationAveraging()
    num_images = max_pano_id + 1

    print(i2Ri1_dict_Rot3)

    wRi_list_Rot3 = shonan_obj.run(num_images, i2Ri1_dict_Rot3)

    return wRi_list_Rot3


def node_present_in_any_triplet(node: int, triplets: List[Tuple[int, int, int]]):
    """ """
    return any([node in triplet for triplet in triplets])


def test_node_present_in_any_triplet():
    """ """
    triplets = [(1, 2, 3), (4, 5, 6), (3, 5, 6)]

    assert not node_present_in_any_triplet(8, triplets)
    assert node_present_in_any_triplet(3, triplets)


@dataclass(frozen=False)
class TwoViewEstimationReport:

    gt_class: int
    R_error_deg: Optional[float] = None
    U_error_deg: Optional[float] = None


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

    # form adjacency list
    adj_list = defaultdict(set)

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():

        adj_list[i1].add(i2)
        adj_list[i2].add(i1)

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
        print(f"EDGE_SE2 {i1} {i2} 0 0 {cycle_utils.rotmat2theta_deg(i2Ri1)}")

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
    wRi_list_euler_deg_est = [cycle_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est

    wRi_list_shonan = globalaveraging2d(i2Ri1_dict)

    wRi_list_shonan_est = [cycle_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # Note that:
    # 360 - 125.812 =  234.188
    # 234.188 - 144.188 = 90.0
    wRi_list_shonan_exp = [-125.81, 144.18, -125.81, -125.81, 144.18]
    assert np.allclose(wRi_list_shonan_exp, wRi_list_shonan_est, atol=0.01)

    # # cast to a 2d problem
    # wRi_list_Rot3_shonan = global_averaging(i2Ri1_dict)
    # wRi_list_shonan = posegraph3d_to_posegraph2d(wRi_list_Rot3_shonan)

    # wRi_list_shonan_est = [ cycle_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

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
        print(f"EDGE_SE2 {i1} {i2} 0 0 {cycle_utils.rotmat2theta_deg(i2Ri1)}")

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
    wRi_list_euler_deg_est = [cycle_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est


def globalaveraging2d_consecutive_ordering(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[np.ndarray]:
    """ """
    input_file = "shonan_input.g2o"
    with open(input_file, "w") as f:
        x = 0
        y = 0
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            theta_deg = cycle_utils.rotmat2theta_deg(i2Ri1)
            theta_rad = np.deg2rad(theta_deg)
            f.write(f"EDGE_SE2 {i2} {i1} {x} {y} {theta_rad} 1.000000 0.000000 0.000000 1.000000 0.000000 1.000000\n")

    shonan = gtsam.ShonanAveraging2(input_file)
    if shonan.nrUnknowns() == 0:
        raise ValueError("No 2D pose constraints found, try -d 3.")
    initial = shonan.initializeRandomly()

    pmin = 2
    pmax = 100
    rotations, _ = shonan.run(initial, pmin, pmax)

    wRi_list = [rotations.atRot2(j).matrix() for j in range(rotations.size())]
    return wRi_list


def globalaveraging2d(i2Ri1_dict: Dict[Tuple[int, int], Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.
    Note: run() functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
    All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
    camera in a single, global coordinate frame.
    Args:
    num_images: number of images. Since we have one pose per image, it is also the number of poses.
    i2Ri1_dict: relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
    Returns:
    Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
    `num_images`. The list may contain `None` where the global rotation could not be computed (either
    underconstrained system or ill-constrained system), or where the camera pose had no valid observation
    in the input to run().
    """
    edges = i2Ri1_dict.keys()
    num_images = max([max(i1, i2) for i1, i2 in edges]) + 1

    connected_nodes = set()
    for (i1, i2) in i2Ri1_dict.keys():
        connected_nodes.add(i1)
        connected_nodes.add(i2)

    connected_nodes = sorted(list(connected_nodes))

    # given original index, this map gives back a new temporary index, starting at 0
    reordered_idx_map = {}
    for (new_idx, i) in enumerate(connected_nodes):
        reordered_idx_map[i] = new_idx

    # now, map the original indices to reordered indices
    i2Ri1_dict_reordered = {}
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        i1_ = reordered_idx_map[i1]
        i2_ = reordered_idx_map[i2]
        i2Ri1_dict_reordered[(i1_, i2_)] = i2Ri1

    wRi_list_subset = globalaveraging2d_consecutive_ordering(i2Ri1_dict=i2Ri1_dict_reordered)

    wRi_list = [None] * num_images
    for remapped_i, original_i in enumerate(connected_nodes):
        wRi_list[original_i] = wRi_list_subset[remapped_i]

    return wRi_list


# def test_shonanaveraging2():
# 	""" """
# input_file = "shonanaveraging2_example.g2o"

# 	# # Load 2D toy example
# lmParams = gtsam.LevenbergMarquardtParams.CeresDefaults()
# # lmParams.setVerbosityLM("SUMMARY")
# g2oFile = gtsam.findExampleDataFile("noisyToyGraph.txt")
# parameters = gtsam.ShonanAveragingParameters2(lmParams)
# shonan = gtsam.ShonanAveraging2(g2oFile, parameters)
# self.assertAlmostEqual(4, shonan.nrUnknowns())


if __name__ == "__main__":
    """ """

    # test_greedily_construct_st2()

    # test_greedily_construct_st()

    # test_shonanaveraging2()

    # hypotheses_dir = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_06_25"
    hypotheses_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_alignment_hypotheses_2021_06_25"

    raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    main(hypotheses_dir, raw_dataset_dir)
    # test_node_present_in_any_triplet()

    # test_wrap_angle_deg()
