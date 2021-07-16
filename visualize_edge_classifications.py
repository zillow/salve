"""
"""

import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

import cycle_consistency as cycle_utils
import rotation_averaging
from cycle_consistency import TwoViewEstimationReport
from posegraph2d import PoseGraph2d, get_gt_pose_graph
from pr_utils import assign_tp_fp_fn_tn
from rotation_utils import rotmat2d, wrap_angle_deg, rotmat2theta_deg
from spanning_tree import greedily_construct_st_Sim2


@dataclass(frozen=False)
class EdgeClassification:
    """
    i1 and i2 are panorama id's
    """

    i1: int
    i2: int
    prob: float
    y_hat: int
    y_true: int
    pair_idx: int
    wdo_pair_uuid: str
    configuration: str


def get_edge_classifications_from_serialized_preds(
    serialized_preds_json_dir: str,
) -> Dict[Tuple[str, str], List[EdgeClassification]]:
    """

    Args:
        serialized_preds_json_dir:

    Returns:
        floor_edgeclassifications_dict
    """
    floor_edgeclassifications_dict = defaultdict(list)

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    for json_fpath in json_fpaths:

        json_data = read_json_file(json_fpath)
        y_hat_list = json_data["y_hat"]
        y_true_list = json_data["y_true"]
        y_hat_prob_list = json_data["y_hat_probs"]
        fp0_list = json_data["fp0"]
        fp1_list = json_data["fp1"]
        fp2_list = json_data["fp2"]
        fp3_list = json_data["fp3"]

        for y_hat, y_true, y_hat_prob, fp0, fp1, fp2, fp3 in zip(
            y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list, fp2_list, fp3_list
        ):
            i1 = int(Path(fp0).stem.split("_")[-1])
            i2 = int(Path(fp1).stem.split("_")[-1])
            building_id = Path(fp0).parent.stem

            s = Path(fp0).stem.find("floor_")
            e = Path(fp0).stem.find("_partial")
            floor_id = Path(fp0).stem[s:e]

            pair_idx = Path(fp0).stem.split("_")[1]

            is_identity = "identity" in Path(fp0).stem
            configuration = "identity" if is_identity else "rotated"

            k = Path(fp0).stem.split("___")[1].find(f"_{configuration}")
            assert k != -1

            wdo_pair_uuid = Path(fp0).stem.split("___")[1][:k]
            assert any([wdo_type in wdo_pair_uuid for wdo_type in ["door", "window", "opening"]])

            floor_edgeclassifications_dict[(building_id, floor_id)] += [
                EdgeClassification(
                    i1=i1,
                    i2=i2,
                    prob=y_hat_prob,
                    y_hat=y_hat,
                    y_true=y_true,
                    pair_idx=pair_idx,
                    wdo_pair_uuid=wdo_pair_uuid,
                    configuration=configuration,
                )
            ]
    return floor_edgeclassifications_dict


def vis_edge_classifications(serialized_preds_json_dir: str, raw_dataset_dir: str) -> None:
    """ """
    floor_edgeclassifications_dict = get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

    color_dict = {"TP": "green", "FP": "red", "FN": "orange", "TN": "blue"}

    # loop over each building and floor
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        # if building_id != '1490': # '1394':# '1635':
        # 	continue

        print(f"On building {building_id}, {floor_id}")
        gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

        # gather all of the edge classifications
        y_hat = np.array([m.y_hat for m in measurements])
        y_true = np.array([m.y_true for m in measurements])

        # classify into TPs, FPs, FNs, TNs
        is_TP, is_FP, is_FN, is_TN = assign_tp_fp_fn_tn(y_true, y_pred=y_hat)
        for m, is_tp, is_fp, is_fn, is_tn in zip(measurements, is_TP, is_FP, is_FN, is_TN):

            # then render the edges
            if is_tp:
                color = color_dict["TP"]
                # gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)
                print(f"\tFP: ({m.i1},{m.i2}) for pair {m.pair_idx}")

            elif is_fp:
                color = color_dict["FP"]

            elif is_fn:
                color = color_dict["FN"]
                # gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)

            elif is_tn:
                color = color_dict["TN"]
                gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)

            # if m.i1 or m.i2 not in gt_floor_pose_graph.nodes:
            # 	import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # render the pose graph first
        gt_floor_pose_graph.render_estimated_layout()
        # continue


def run_incremental_reconstruction(
    hypotheses_save_root: str, serialized_preds_json_dir: str, raw_dataset_dir: str
) -> None:
    """ """
    method = "growing_consensus"
    confidence_threshold = 0.95

    floor_edgeclassifications_dict = get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

    # loop over each building and floor
    # for each building/floor tuple
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        i2Si1_dict = {}
        i2Ri1_dict = {}
        i2Ui1_dict = {}
        two_view_reports_dict = {}

        # if not (building_id == "1635" and floor_id == "floor_01"):
        #     continue

        gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        print(f"On building {building_id}, {floor_id}")

        visualize_confidence_histograms = False
        if visualize_confidence_histograms:
            probs = np.array([m.prob for m in measurements])
            y_true_array = np.array([m.y_true for m in measurements])
            y_hat_array = np.array([m.y_hat for m in measurements])
            is_TP, is_FP, is_FN, is_TN = assign_tp_fp_fn_tn(y_true_array, y_hat_array)

            plt.subplot(2,2,1)
            plt.hist(probs[is_TP], bins=15)
            plt.title("TP")

            plt.subplot(2,2,2)
            plt.hist(probs[is_FP], bins=15)
            plt.title("FP")

            plt.subplot(2,2,3)
            plt.hist(probs[is_FN], bins=15)
            plt.title("FN")

            plt.subplot(2,2,4)
            plt.hist(probs[is_TN], bins=15)
            plt.title("TN")

            plt.show()


        num_gt_positives = 0
        num_gt_negatives = 0

        gt_edges = []
        for m in measurements:

            if m.y_true == 1:
                num_gt_positives += 1
            else:
                num_gt_negatives += 1

            # TODO: remove this debugging condition
            # if m.y_true != 1:
            #     continue

            # find all of the predictions where pred class is 1
            if m.y_hat != 1:
                continue

            if m.prob < confidence_threshold:
                continue

            # TODO: this should never happen bc sorted, figure out why it occurs
            if m.i1 >= m.i2:
                i2 = m.i1
                i1 = m.i2

                m.i1 = i1
                m.i2 = i2

            # look up the associated Sim(2) file for this prediction, by looping through the pair idxs again
            label_dirname = "gt_alignment_approx" if m.y_true else "incorrect_alignment"
            fpaths = glob.glob(
                f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_dirname}/{m.i1}_{m.i2}__{m.wdo_pair_uuid}_{m.configuration}.json"
            )

            # label_dirname = "gt_alignment_exact"
            # fpaths = glob.glob(f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_dirname}/{m.i1}_{m.i2}.json")

            if not len(fpaths) == 1:
                import pdb

                pdb.set_trace()
            i2Si1 = Sim2.from_json(fpaths[0])

            # TODO: choose the most confident score. How often is the most confident one, the right one, among all of the choices?
            # use model confidence
            # CHECK WEIRD IMBALANCE RATE

            i2Si1_dict[(m.i1, m.i2)] = i2Si1

            i2Ri1_dict[(m.i1, m.i2)] = i2Si1.rotation
            i2Ui1_dict[(m.i1, m.i2)] = i2Si1.translation

            R_error_deg = 0
            U_error_deg = 0
            two_view_reports_dict[(m.i1, m.i2)] = cycle_utils.TwoViewEstimationReport(
                gt_class=m.y_true, R_error_deg=R_error_deg, U_error_deg=U_error_deg
            )

            if m.y_true == 1:
                gt_edges.append((m.i1, m.i2))

        class_imbalance_ratio = num_gt_negatives / num_gt_positives
        print(f"\tClass imbalance ratio {class_imbalance_ratio:.2f}")

        unfiltered_edge_acc = get_edge_accuracy(edges=i2Si1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
        print(f"\tUnfiltered Edge Acc = {unfiltered_edge_acc:.2f}")

        if method == "spanning_tree":
            build_filtered_spanning_tree(
                building_id,
                floor_id,
                i2Si1_dict,
                i2Ri1_dict,
                i2Ui1_dict,
                gt_edges,
                two_view_reports_dict,
                gt_floor_pose_graph,
            )

        elif method == "growing_consensus":
            growing_consensus(
                building_id,
                floor_id,
                i2Si1_dict,
                i2Ri1_dict,
                i2Ui1_dict,
                gt_edges,
                two_view_reports_dict,
                gt_floor_pose_graph,
            )
        else:
            raise RuntimeError("Unknown method.")


def find_max_degree_vertex(i2Ti1_dict: Dict[Tuple[int,int], Any]) -> int:
    """Find the node inside of a graph G=(V,E) with highest degree.

    Args:
        i2Ti1_dict: edges E of a graph G=(V,E)

    Returns:
        seed_node: integer id of 
    """
    # find the seed (vertex with highest degree)
    adj_list = cycle_utils.create_adjacency_list(i2Ti1_dict)
    seed_node = -1
    max_neighbors = 0
    for v, neighbors in adj_list.items():
        if len(neighbors) > max_neighbors:
            seed_node = v
            max_neighbors = len(neighbors)

    return seed_node


def growing_consensus(
    building_id: str,
    floor_id: str,
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    i2Ui1_dict: Dict[Tuple[int, int], np.ndarray],
    gt_edges,
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    gt_floor_pose_graph: PoseGraph2d,
) -> None:
    """Implements a variant of the Growing Consensus" algorithm described below:

    Reference:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf
    
    Args:
    """
    seed_node = find_max_degree_vertex(i2Si1_dict)
    unused_triplets = cycle_utils.extract_triplets(i2Si1_dict)
    unused_triplets = set(unused_triplets)

    i2Si1_dict_consensus = {}

    # compute cycle errors for each triplet connected to seed_node
    # find one with low cycle_error

    while True:
        for triplet in unused_triplets:
            import pdb; pdb.set_trace()

        # compute all cycle errors with these new edges added
        # if all errors are low, then:
            # unused_triplets.remove(triplet)

    


def build_filtered_spanning_tree(
    building_id: str,
    floor_id: str,
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    i2Ui1_dict: Dict[Tuple[int, int], np.ndarray],
    gt_edges,
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    gt_floor_pose_graph: PoseGraph2d,
) -> None:
    """ """

    i2Ri1_dict, i2Ui1_dict_consistent = cycle_utils.filter_to_rotation_cycle_consistent_edges(
        i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize=False
    )

    filtered_edge_acc = get_edge_accuracy(edges=i2Ri1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
    print(f"\tFiltered by rot cycles Edge Acc = {filtered_edge_acc:.2f}")

    # # could count how many nodes or edges never appeared in any triplet
    # triplets = cycle_utils.extract_triplets(i2Ri1_dict)
    # dropped_edges = set(i2Ri1_dict.keys()) - set(i2Ri1_dict_consistent.keys())
    # print("Dropped ")

    wRi_list = rotation_averaging.globalaveraging2d(i2Ri1_dict)
    if wRi_list is None:
        print(f"Rotation averaging failed, because {len(i2Ri1_dict)} measurements provided.")
        print()
        print()
        return
    # TODO: measure the error in rotations

    # filter to rotations that are consistent with global
    i2Ri1_dict = filter_measurements_to_absolute_rotations(
        wRi_list, i2Ri1_dict, max_allowed_deviation=5, two_view_reports_dict=two_view_reports_dict
    )

    consistent_edge_acc = get_edge_accuracy(edges=i2Ri1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
    print(f"\tFiltered relative by abs. rotation deviation Edge Acc = {consistent_edge_acc:.2f}")

    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=False
    )
    print(f"\tMean relative rotation error {mean_rel_rot_err:.2f} deg.")

    # remove the edges that we deem to be outliers
    outlier_edges = set(i2Si1_dict.keys()) - set(i2Ri1_dict.keys())
    for outlier_edge in outlier_edges:
        del i2Si1_dict[outlier_edge]

    i2Si1_dict_consistent = cycle_utils.filter_to_translation_cycle_consistent_edges(
        wRi_list, i2Si1_dict, translation_cycle_thresh=0.25, two_view_reports_dict=two_view_reports_dict
    )
    filtered_edge_acc = get_edge_accuracy(
        edges=i2Si1_dict_consistent.keys(), two_view_reports_dict=two_view_reports_dict
    )
    print(f"\tFiltered by trans. cycle Edge Acc = {filtered_edge_acc:.2f}")

    labels = np.array([two_view_reports_dict[(i1, i2)].gt_class for (i1, i2) in i2Si1_dict_consistent.keys()])
    num_fps_for_st = (labels == 0).sum()
    print(f"{num_fps_for_st} FPs were fed to spanning tree.")

    wSi_list = greedily_construct_st_Sim2(i2Si1_dict_consistent, verbose=False)

    if wSi_list is None:
        print(f"Could not build spanning tree, since {len(i2Si1_dict_consistent)} edges in i2Si1 dictionary.")
        print()
        print()
        return

    num_localized_panos = np.array([wSi is not None for wSi in wSi_list]).sum()
    num_floor_panos = len(gt_floor_pose_graph.nodes)
    print(
        f"Localized {num_localized_panos/num_floor_panos*100:.2f}% of panos: {num_localized_panos} / {num_floor_panos}"
    )

    # TODO: try spanning tree version, vs. Shonan version
    wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
    wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(
        wRi_list, wti_list, gt_floor_pose_graph, building_id, floor_id
    )

    mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)
    print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
    #est_floor_pose_graph.render_estimated_layout()

    print()
    print()


def filter_measurements_to_absolute_rotations(
    wRi_list: List[Optional[np.ndarray]],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    max_allowed_deviation: float = 5,
    verbose: bool = False,
    two_view_reports_dict=None,
    visualize: bool = False
) -> Dict[Tuple[int, int], np.ndarray]:
    """ """

    deviations = []
    is_gt_edge = []

    edges = i2Ri1_dict.keys()

    i2Ri1_dict_consistent = {}

    for (i1, i2) in edges:

        if wRi_list[i1] is None or wRi_list[i2] is None:
            continue

        # find estimate, given absolute values
        wTi1 = wRi_list[i1]
        wTi2 = wRi_list[i2]
        i2Ri1_inferred = wTi2.T @ wTi1

        i2Ri1_measured = i2Ri1_dict[(i1, i2)]

        theta_deg_inferred = rotmat2theta_deg(i2Ri1_inferred)
        theta_deg_measured = rotmat2theta_deg(i2Ri1_measured)

        if verbose:
            print(f"\tPano pair ({i1},{i2}): Measured {theta_deg_measured:.1f} vs. Inferred {theta_deg_inferred:.1f}")

        # need to wrap around at 360
        err = wrap_angle_deg(theta_deg_inferred, theta_deg_measured)
        if err < max_allowed_deviation:
            i2Ri1_dict_consistent[(i1, i2)] = i2Ri1_measured

        if two_view_reports_dict is not None:
            is_gt_edge.append(two_view_reports_dict[(i1, i2)].gt_class)
            deviations.append(err)

    if two_view_reports_dict is not None and visualize:
        deviations = np.array(deviations)
        is_gt_edge = np.array(is_gt_edge)
        misclassified_errs = deviations[is_gt_edge == 0]
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.hist(misclassified_errs, bins=30)
        plt.ylabel("Counts")
        plt.xlabel("Deviation from measurement (degrees)")
        plt.title("False Positive Edge")
        plt.ylim(0, deviations.size)
        plt.xlim(0, 180)

        correct_classified_errs = deviations[is_gt_edge == 1]
        plt.subplot(1, 2, 2)
        plt.hist(correct_classified_errs, bins=30)
        plt.title("GT edge")
        plt.ylim(0, deviations.size)
        plt.xlim(0, 180)

        plt.suptitle("Filtering rot avg result to be consistent with measurements")
        plt.show()

    print(
        f"\tFound that {len(i2Ri1_dict_consistent)} of {len(i2Ri1_dict)} rotations were consistent w/ global rotations"
    )
    return i2Ri1_dict_consistent


def get_edge_accuracy(
    edges: List[Tuple[int, int]], two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport]
) -> float:
    """
    Check GT for each predicted (i.e. allowed) edge.
    """
    preds = []
    # what is the purity of what comes out?
    for (i1, i2) in edges:
        preds.append(two_view_reports_dict[(i1, i2)].gt_class)

    acc = np.mean(preds)
    return acc


if __name__ == "__main__":
    """ """
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_binary_model_edge_classifications"
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_edge_classifications_fixed_argmax_bug/2021_07_13_edge_classifications_fixed_argmax_bug"
    serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26/2021_07_15_serialized_edge_classifications/2021_07_15_serialized_edge_classifications"

    raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # vis_edge_classifications(serialized_preds_json_dir, raw_dataset_dir)

    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"

    run_incremental_reconstruction(hypotheses_save_root, serialized_preds_json_dir, raw_dataset_dir)
