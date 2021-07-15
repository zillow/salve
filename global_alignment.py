"""
Global alignment.
"""

import copy
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Point3, Rot2, Rot3, Unit3

import cycle_consistency as cycle_utils
from cycle_consistency import TwoViewEstimationReport

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

from vis_depth import rotmat2d
from posegraph2d import PoseGraph2d, get_gt_pose_graph, rot2x2_to_Rot3
from rotation_averaging import globalaveraging2d
from spanning_tree import greedily_construct_st


def main(hypotheses_dir: str, raw_dataset_dir: str) -> None:
    """
    Evaluate the quality of a global alignment, when noise is synthetically injected into the binary measurements.
    """
    # TODO: cache all of the model results beforehand (suppose we randomly pollute 8.5% of the results)
    #POLLUTION_FRAC = 0.085
    POLLUTION_FRAC = 0.050
    #POLLUTION_FRAC = 0.025
    #POLLUTION_FRAC = 0.0

    enforce_cycle_consistency =  True # False #
    method =  "shonan" #  "greedy"  #

    np.random.seed(0)

    inside_triplet_percent_list = []
    accs = []

    label_dict = {"incorrect_alignment": 0, "gt_alignment_exact":1}# "gt_alignment_approx": 1}

    all_floor_rot_errs = []

    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_dir}/*")]
    building_ids.sort()

    mean_rel_rot_errs = []

    cycle_precs = []
    cycle_recs = []
    cycle_mAccs = []

    for building_id in building_ids:

        # if len(inside_triplet_percent_list) > 10:
        # 	break

        floor_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_dir}/{building_id}/*")]
        for floor_id in floor_ids:

            # #  or 
            # # 
            if  (building_id == "004" and floor_id == "floor_02") or (building_id == "007" and floor_id == "floor_01") or (building_id == "008" and floor_id == "floor_01") or (building_id == "009" and floor_id == "floor_01"):# ):
                continue

            # if (building_id == "012" and floor_id == "floor_00"):
            #     import pdb; pdb.set_trace()


            logger.info(f"Building {building_id}, {floor_id}")

            floor_label_idxs = []
            floor_sim2_json_fpaths = []

            for label_type, label_idx in label_dict.items():
                label_json_fpaths = glob.glob(f"{hypotheses_dir}/{building_id}/{floor_id}/{label_type}/*")
                label_idxs = [label_idx] * len(label_json_fpaths)

                floor_sim2_json_fpaths.extend(label_json_fpaths)
                floor_label_idxs.extend(label_idxs)

            floor_label_idxs = np.array(floor_label_idxs)

            num_floor_labels = len(floor_label_idxs)
            idxs_to_pollute = np.random.choice(a=num_floor_labels, size=int(POLLUTION_FRAC * num_floor_labels))

            print(f"{len(idxs_to_pollute)} of {num_floor_labels} were polluted")

            predicted_idxs = copy.deepcopy(floor_label_idxs)
            # perform logical NOT operation via arithmetic
            predicted_idxs[idxs_to_pollute] = 1 - predicted_idxs[idxs_to_pollute]

            # for a single floor, find all of the triplets
            two_view_reports_dict = {}

            i2Ri1_dict = {}
            i2ti1_dict = {}

            floor_pano_ids = []
            gt_edges = []
            for sim2_json_fpath, gt_class_idx in zip(floor_sim2_json_fpaths, floor_label_idxs):

                if gt_class_idx != 1:
                    continue

                i1, i2 = Path(sim2_json_fpath).stem.split("_")[:2]
                i1, i2 = int(i1), int(i2)

                floor_pano_ids.append(i1)
                floor_pano_ids.append(i2)

                gt_edges.append((i1,i2))

            for sim2_json_fpath, predicted_idx, gt_class_idx in zip(
                floor_sim2_json_fpaths, predicted_idxs, floor_label_idxs
            ):

                # add if prediction (i.e. a potentially corrupted label) said it was a match
                if predicted_idx != 1:
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

            if enforce_cycle_consistency:
                #check which triplets are self consistent. if so, admit the 3 edges to the graph
                i2Ri1_dict_consistent, i2ti1_dict_consistent = cycle_utils.filter_to_rotation_cycle_consistent_edges(
                    i2Ri1_dict,
                    i2ti1_dict,
                    two_view_reports_dict,
                    visualize=False
                )

                cycle_prec, cycle_rec, cycle_mAcc = cycle_utils.estimate_rot_cycle_filtering_classification_acc(i2Ri1_dict, i2Ri1_dict_consistent, two_view_reports_dict)
                print(f"Rotation cycle accuracy: {cycle_mAcc:.2f}, prec: {cycle_prec:.2f}, rec: {cycle_rec:.2f}")

                cycle_precs.append(cycle_prec)
                cycle_recs.append(cycle_rec)
                cycle_mAccs.append(cycle_mAcc)

                i2Ri1_dict = i2Ri1_dict_consistent
                i2ti1_dict = i2ti1_dict_consistent

                # filter to cycle consistent translation directions
                i2Ri1_dict_consistent, i2ti1_dict_consistent = cycle_utils.filter_to_translation_cycle_consistent_edges(i2Ri1_dict, i2ti1_dict)

            print(f"Estimate global rotations using {method} from {len(i2Ri1_dict)} edges.")

            if method == "shonan":
                wRi_list = globalaveraging2d(i2Ri1_dict)
                # wRi_list_Rot3 = global_averaging(i2Ri1_dict)
                # #print(wRi_list_Rot3)

                # # # have to align the pose graphs up to a Sim(3) transformation
                # # import gtsfm.utils.geometry_comparisons as comp_utils

                # # wRi_list_Rot3 = comp_utils.align_rotations(wRi_list_Rot3_gt, wRi_list_Rot3)
                # wRi_list = posegraph3d_to_posegraph2d(wRi_list_Rot3)

            elif method == "greedy":
                wRi_list = greedily_construct_st(i2Ri1_dict)

            # run 1dsfm
            wti_list = run_translation_averaging(i2ti1_dict, wRi_list)
            for i in range(len(wti_list)):
                if wti_list[i] is not None:
                    wti_list[i] /= 25

            # est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)
            gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)


            est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pose_graph, building_id, floor_id)

            mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)
            print(f"Avg translation error: {mean_abs_trans_err}")
            est_floor_pose_graph.render_estimated_layout()
            #continue

            #mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)

            verbose = False
            mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=verbose)
            mean_rel_rot_errs.append(mean_rel_rot_err)

            # plt.hist(all_floor_rot_errs, bins=10)
            # plt.ylabel("Counts")
            # plt.xlabel("Absolute rotation error (degrees)")
            # plt.show()

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

            # # import pdb; pdb.set_trace()

            # predictions = []
            # for (i1,i2) in i2Ri1_consistent.keys():
            # 	predictions.append(two_view_reports_dict[(i1,i2)].gt_class)

            # predictions = np.array(predictions)
            # acc = np.mean(predictions)
            # accs.append(acc)
            # logger.info(f"Building {building_id}, floor {floor_id}, Mean accuracy: {acc:.2f}\n")

    print(f"Over {len(mean_rel_rot_errs)} floors of all buildings, mean relative rotation error was {np.mean(mean_rel_rot_errs):.1f} +- {np.std(mean_rel_rot_errs):.2f}")
    
    print(f"Over {len(cycle_mAccs)} floors of all buildings, mean rot-based cycle filtering mAccs. was {np.mean(cycle_mAccs):2f}")
    print(f"Over {len(cycle_precs)} floors of all buildings, mean rot-based cycle filtering Precision was {np.mean(cycle_precs):2f}")
    print(f"Over {len(cycle_recs)} floors of all buildings, mean rot-based cycle filtering Recall was {np.mean(cycle_recs):2f}")

    # plt.scatter(inside_triplet_percent_list, accs)
    # plt.show()

    # plt.hist(inside_triplet_percent_list, bins=50)
    # plt.xlabel("Fraction of nodes found in any triplet")
    # plt.ylabel("Counts")
    # plt.show()

def form_adjacency_matrix(edges: List[Tuple[int,int]]) -> np.ndarray:
    """ """
    num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    for (i1,i2) in edges:
        adj_matrix[i1,i2] = 1
        adj_matrix[i2,i1] = 1

    return adj_matrix


def run_translation_averaging(i2ti1_dict: Dict[Tuple[int,int],np.ndarray], wRi_list: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    """
    What if we fix the scale?

    Args:
        Given 2x2 rotations
        Given (2,) translation directions

    Returns:
        wti_list: global translations/positions
    """
    from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM


    wRi_Rot3_list = [ rot2x2_to_Rot3(wRi)  if wRi is not None else None for wRi in wRi_list ]
    i2Ui1_dict = {(i1,i2): Unit3(np.array([i2ti1[0], i2ti1[1], 0])) for (i1,i2), i2ti1 in i2ti1_dict.items()}
    trans_avg = TranslationAveraging1DSFM()
    wti_list = trans_avg.run(num_images=len(wRi_Rot3_list), i2Ui1_dict=i2Ui1_dict, wRi_list=wRi_Rot3_list)

    wti_list = wti_list_3d_to_2d(wti_list)
    return wti_list


def test_run_translation_averaging() -> None:
    """
    Ensure translation averaging can recover translations for a simple 2d case.

    GT pose graph:

       | pano 1 = (0,4)
     --o
       | .
       .   .
       .     .
       |       |
       o-- ... o--
    pano 0          pano 2 = (4,0)
      (0,0)

    """
    from gtsam import Pose2, Rot2
    wTi_list = [
        Pose2(Rot2.fromDegrees(0), np.array([0,0])),
        Pose2(Rot2.fromDegrees(90), np.array([0,4])),
        Pose2(Rot2.fromDegrees(0), np.array([4,0]))
    ]

    i2ti1_dict = {
        (0,1): wTi_list[1].between(wTi_list[0]).translation(),
        (1,2): wTi_list[2].between(wTi_list[1]).translation(),
        (0,2): wTi_list[2].between(wTi_list[0]).translation()
    }
    wRi_list = [ wTi.rotation().matrix() for wTi in wTi_list ]

    wti_list = run_translation_averaging(i2ti1_dict, wRi_list)
    
    # fmt: off
    wti_list_expected = np.array(
        [
        [-0., -1.],
        [-0., -0.],
        [ 1., -1.]
    ])

    # fmt: on
    assert np.allclose(wti_list_expected, np.array(wti_list))


def wti_list_3d_to_2d(wti_list_3d: List[Point3]):
    """ """
    num_images = len(wti_list_3d)
    wti_list = [None] * num_images
    for i, wti in enumerate(wti_list_3d):
        if wti is None:
            continue
        wti_list[i] = wti[:2]

    return wti_list



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


def test_globalaveraging2d_shonan() -> None:
    """ """
    # Building 007, floor_01
    i2Ri1_dict = {
        (3, 4): np.array(
            [
                [ 0.72099364,  0.6929417 ],
                [-0.6929417 ,  0.72099364]
            ], dtype=np.float32),
        (3, 7): np.array(
            [
                [-0.7078902,  0.7063225],
                [-0.7063225, -0.7078902]
            ], dtype=np.float32),
        (4, 6): np.array(
            [
                [ 0.29828665, -0.95447636],
                [ 0.95447636,  0.29828665]
            ], dtype=np.float32),
        (5, 7): np.array(
            [
                [-0.46576393, -0.88490903],
                [ 0.88490903, -0.46576393]
            ], dtype=np.float32),
        (8, 9): np.array(
            [
                [ 0.85045004,  0.5260558 ],
                [-0.52605575,  0.85045004]
            ], dtype=np.float32),
        (9, 14): np.array(
            [
                [-0.99782497, -0.04734466],
                [ 0.04428102, -0.99685955]
            ], dtype=np.float32),
        (8, 12): np.array(
            [
                [-0.68945694,  0.7243267 ],
                [-0.7243267 , -0.68945694]
            ], dtype=np.float32),
        (9, 11): np.array(
            [
                [-0.01023931,  0.99994755],
                [-0.9999476 , -0.01023931]
            ], dtype=np.float32),
        (2, 5): np.array(
            [
                [-0.6022301, -0.7983226],
                [ 0.7983226, -0.6022301]
            ], dtype=np.float32),
        (11, 14): np.array(
            [
                [-0.03764843,  0.99786335],
                [-0.99651164, -0.03350743]
            ], dtype=np.float32),
        (6, 11): np.array(
            [
                [ 0.6206304 , -0.7841033 ],
                [ 0.7841033 ,  0.62063044]
            ], dtype=np.float32),
        (6, 8): np.array(
            [
                [-0.9944558 , -0.10515608],
                [ 0.10515605, -0.99445575]
            ], dtype=np.float32),
        (6, 14): np.array(
            [
                [-0.7614757, -0.6481934],
                [ 0.6481934, -0.7614757]
            ], dtype=np.float32),
        (4, 5): np.array(
            [
                [-0.87495995, -0.48419532],
                [ 0.48419532, -0.87496   ]
            ], dtype=np.float32),
        (5, 6): np.array(
            [
                [ 0.20116411,  0.9795576 ],
                [-0.9795576 ,  0.20116411]
            ], dtype=np.float32),
        (3, 6): np.array(
            [
                [ 0.87645924, -0.48147613],
                [ 0.48147613,  0.87645924]
            ], dtype=np.float32),
        (8, 11): np.array(
            [
                [-0.5347363 ,  0.84501904],
                [-0.84501904, -0.5347363 ]
            ], dtype=np.float32),
        (2, 4): np.array(
            [
                [ 0.9134712 ,  0.40690336],
                [-0.40690336,  0.9134713 ]
            ], dtype=np.float32),
        (7, 9): np.array(
            [
                [ 0.5887703 ,  0.80829966],
                [-0.80829835,  0.58877134]
            ], dtype=np.float32),
        (6, 7): np.array(
            [
                [-0.9605143 ,  0.27823064],
                [-0.27823064, -0.9605143 ]
            ], dtype=np.float32),
        (4, 7): np.array(
            [
                [-0.02094402,  0.99978065],
                [-0.99978065, -0.02094402]
            ], dtype=np.float32),
        (3, 5): np.array(
            [
                [-0.2953214 , -0.95539796],
                [ 0.955398  , -0.29532143]
            ], dtype=np.float32),
        (9, 12): np.array(
            [
                [-0.21155617,  0.9738934 ],
                [-0.9657355 , -0.19534206]
            ], dtype=np.float32),
        (2, 3): np.array(
            [
                [ 0.94056726, -0.33960763],
                [ 0.3396076 ,  0.94056726]
            ], dtype=np.float32),
        (7, 11): np.array(
            [
                [-0.81428593,  0.580464  ],
                [-0.580464  , -0.81428593]
            ], dtype=np.float32),
        (7, 8): np.array(
            [
                [ 0.92593133,  0.37769195],
                [-0.37769195,  0.92593133]
            ], dtype=np.float32),
        (7, 14): np.array(
            [
                [ 0.55106103,  0.8344649 ],
                [-0.8344649 ,  0.55106103]
            ], dtype=np.float32)
        }
    edges = i2Ri1_dict.keys()
    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
    wRi_list = globalaveraging2d(i2Ri1_dict)



def posegraph2d_to_posegraph3d(wRi_list: List[Optional[np.ndarray]]) -> List[Optional[Rot3]]:
    """ """
    num_images = len(wRi_list)
    wRi_list_Rot3 = [None] * num_images
    for i, wRi in enumerate(wRi_list_Rot3):

        if wRi is None:
            continue

        wRi_Rot3 = rot2x2_to_Rot3(wRi)
        wRi_list_Rot3[i] = wRi_Rot3
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


def count_connected_components(edges: List[Tuple[int,int]]) -> int:
    """ """
    input_graph = nx.Graph()
    input_graph.add_edges_from(edges)

    # get the largest connected component
    cc = nx.connected_components(input_graph)
    import pdb; pdb.set_trace()


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
    #hypotheses_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_alignment_hypotheses_2021_06_25"
    hypotheses_dir = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_07"

    raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    main(hypotheses_dir, raw_dataset_dir)
    # test_node_present_in_any_triplet()

    # test_wrap_angle_deg()
    #test_globalaveraging2d_shonan()

    #test_run_translation_averaging()
    # test_estimate_rot_cycle_filtering_classification_acc()


