"""
Another translation-based filtering step is 1dsfm projection directions:
https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_relative_translation.cc
Must happen after rotation averaging.

2d SLAM baseline: https://github.com/borglab/gtsam/blob/develop/examples/Pose2SLAMExample_lago.cpp
"""

import copy
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gtsfm.utils.graph as graph_utils
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2

import afp.algorithms.cycle_consistency as cycle_utils
import afp.algorithms.rotation_averaging as rotation_averaging
import afp.utils.pr_utils as pr_utils
from afp.algorithms.cycle_consistency import TwoViewEstimationReport
from afp.algorithms.spanning_tree import greedily_construct_st_Sim2
from afp.algorithms.cluster_merging import EdgeWDOPair
from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
from afp.utils.graph_rendering_utils import draw_multigraph, draw_graph_topology
from afp.utils.rotation_utils import rotmat2d, wrap_angle_deg, rotmat2theta_deg

from visualize_edge_classifications import get_edge_classifications_from_serialized_preds, EdgeClassification


@dataclass(frozen=True)
class FloorReconstructionReport:
    avg_abs_rot_err: float
    avg_abs_trans_err: float
    percent_panos_localized: float


def get_conf_thresholded_edges(
    measurements: List[EdgeClassification],
    hypotheses_save_root: str,
    confidence_threshold: float,
    building_id: str,
    floor_id: str,
) -> Tuple[
    Dict[Tuple[int, int], Sim2],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], TwoViewEstimationReport],
    List[Tuple[int, int]],
    Dict[Tuple[int, int], EdgeWDOPair],
]:
    """
    Args:
        measurements
        hypotheses_save_root
        confidence_threshold
        building_id
        floor_id

    Returns:
        i2Si1_dict
        i2Ri1_dict
        i2Ui1_dict
        two_view_reports_dict
        gt_edges
        per_edge_wdo_dict
    """
    # for each edge, choose the most confident prediction over all WDO pair alignments
    most_confident_edge_dict = defaultdict(list)

    i2Si1_dict = {}
    i2Ri1_dict = {}
    i2Ui1_dict = {}
    two_view_reports_dict = {}

    # compute_precision_recall(measurements)

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

        most_confident_edge_dict[(m.i1, m.i2)] += [m]

    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair] = {}

    most_confident_was_correct = []
    for (i1, i2), measurements in most_confident_edge_dict.items():

        most_confident_idx = np.argmax([m.prob for m in measurements])
        m = measurements[most_confident_idx]
        if len(measurements) > 1:
            most_confident_was_correct.append(m.y_true == 1)

        # look up the associated Sim(2) file for this prediction, by looping through the pair idxs again
        label_dirname = "gt_alignment_approx" if m.y_true else "incorrect_alignment"
        fpaths = glob.glob(
            f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_dirname}/{m.i1}_{m.i2}__{m.wdo_pair_uuid}_{m.configuration}.json"
        )

        per_edge_wdo_dict[(i1, i2)] = EdgeWDOPair(i1=i1, i2=i2, wdo_pair_uuid=m.wdo_pair_uuid)

        # label_dirname = "gt_alignment_exact"
        # fpaths = glob.glob(f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_dirname}/{m.i1}_{m.i2}.json")

        if not len(fpaths) == 1:
            import pdb

            pdb.set_trace()
        i2Si1 = Sim2.from_json(fpaths[0])

        # TODO: choose the most confident score. How often is the most confident one, the right one, among all of the choices?
        # use model confidence

        i2Si1_dict[(m.i1, m.i2)] = i2Si1

        i2Ri1_dict[(m.i1, m.i2)] = i2Si1.rotation
        i2Ui1_dict[(m.i1, m.i2)] = i2Si1.translation

        R_error_deg = 0
        U_error_deg = 0
        two_view_reports_dict[(m.i1, m.i2)] = cycle_utils.TwoViewEstimationReport(
            gt_class=m.y_true, R_error_deg=R_error_deg, U_error_deg=U_error_deg, confidence=m.prob
        )

        if m.y_true == 1:
            gt_edges.append((m.i1, m.i2))

        # print(m)

    print(f"most confident was correct {np.array(most_confident_was_correct).mean():.2f}")

    class_imbalance_ratio = num_gt_negatives / num_gt_positives
    print(f"\tClass imbalance ratio {class_imbalance_ratio:.2f}")

    return i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, gt_edges, per_edge_wdo_dict


def run_incremental_reconstruction(
    hypotheses_save_root: str, serialized_preds_json_dir: str, raw_dataset_dir: str
) -> None:
    """
    Can get multi-graph out of classification model.

    Args:
        hypotheses_save_root
        serialized_preds_json_dir
        raw_dataset_dir
    """
    # TODO: determine why some FPs have zero cycle error? why so close to GT?

    method = "spanning_tree"  # "SE2_cycles" # # "growing_consensus"
    confidence_threshold = 0.98  # 0.95 # 0.95 # 0.90 # 0.95 # 1.01 #= 0.95

    plot_save_dir = (
        f"2021_07_29_{method}_floorplans_with_gt_conf_{confidence_threshold}_mostconfident_edge_trainingv1_old"
    )
    os.makedirs(plot_save_dir, exist_ok=True)

    floor_edgeclassifications_dict = get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

    # import pdb; pdb.set_trace()

    reconstruction_reports = []

    # loop over each building and floor
    # for each building/floor tuple
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        # if not (building_id == "1635" and floor_id == "floor_02"):
        #     continue

        gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        print(f"On building {building_id}, {floor_id}")

        # continue

        visualize_confidence_histograms = False
        if visualize_confidence_histograms:
            probs = np.array([m.prob for m in measurements])
            y_true_array = np.array([m.y_true for m in measurements])
            y_hat_array = np.array([m.y_hat for m in measurements])
            is_TP, is_FP, is_FN, is_TN = pr_utils.assign_tp_fp_fn_tn(y_true_array, y_hat_array)

            plt.subplot(2, 2, 1)
            plt.hist(probs[is_TP], bins=15)
            plt.title("TP")

            plt.subplot(2, 2, 2)
            plt.hist(probs[is_FP], bins=15)
            plt.title("FP")

            plt.subplot(2, 2, 3)
            plt.hist(probs[is_FN], bins=15)
            plt.title("FN")

            plt.subplot(2, 2, 4)
            plt.hist(probs[is_TN], bins=15)
            plt.title("TN")

            plt.show()

        render_multigraph = False
        if render_multigraph:
            draw_multigraph(measurements, gt_floor_pose_graph)

        i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, gt_edges, _ = get_conf_thresholded_edges(
            measurements, hypotheses_save_root, confidence_threshold, building_id, floor_id
        )

        unfiltered_edge_acc = get_edge_accuracy(edges=i2Si1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
        print(f"\tUnfiltered Edge Acc = {unfiltered_edge_acc:.2f}")

        cc_nodes = graph_utils.get_nodes_in_largest_connected_component(i2Si1_dict.keys())
        print(
            f"Before any filtering, the largest CC contains {len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
        )

        if method == "spanning_tree":
            est_floor_pose_graph, i2Si1_dict_consistent, report = build_filtered_spanning_tree(
                building_id,
                floor_id,
                i2Si1_dict,
                i2Ri1_dict,
                i2Ui1_dict,
                gt_edges,
                two_view_reports_dict,
                gt_floor_pose_graph,
                plot_save_dir,
            )
            reconstruction_reports.append(report)

            # i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, _, per_edge_wdo_dict = get_conf_thresholded_edges(
            #     measurements,
            #     hypotheses_save_root,
            #     confidence_threshold=0.5,
            #     building_id=building_id,
            #     floor_id=floor_id
            # )
            # from afp.algorithms.cluster_merging import merge_clusters
            # est_pose_graph = merge_clusters(i2Si1_dict, i2Si1_dict_consistent, per_edge_wdo_dict, gt_floor_pose_graph, two_view_reports_dict)

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
        elif method == "SE2_cycles":
            cycles_SE2_spanning_tree(
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

    print()
    print()
    print(f"Test set contained {len(reconstruction_reports)} total floors.")
    error_metrics = reconstruction_reports[0].__dict__.keys()
    for error_metric in error_metrics:
        avg_val = np.nanmean([getattr(r, error_metric) for r in reconstruction_reports])
        print(f"Averaged over all tours, {error_metric} = {avg_val:.2f}")

        median_val = np.nanmedian([getattr(r, error_metric) for r in reconstruction_reports])
        print(f"Median over all tours, {error_metric} = {median_val:.2f}")


def cycles_SE2_spanning_tree(
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
    i2Si1_dict_consistent = cycle_utils.filter_to_SE2_cycle_consistent_edges(i2Si1_dict, two_view_reports_dict)

    filtered_edge_acc = get_edge_accuracy(
        edges=i2Si1_dict_consistent.keys(), two_view_reports_dict=two_view_reports_dict
    )
    print(f"\tFiltered by SE(2) cycles Edge Acc = {filtered_edge_acc:.2f}")

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

    mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_unaligned_abs_pose_error(
        gt_floor_pg=gt_floor_pose_graph
    )
    print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
    est_floor_pose_graph.render_estimated_layout()

    print()
    print()


def find_max_degree_vertex(i2Ti1_dict: Dict[Tuple[int, int], Any]) -> int:
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
    i2Ri1_dict = {(i1, i2): i2Si1.rotation for (i1, i2), i2Si1 in i2Si1_dict.items()}

    seed_node = find_max_degree_vertex(i2Si1_dict)
    unused_triplets = cycle_utils.extract_triplets(i2Si1_dict)
    unused_triplets = set(unused_triplets)

    seed_cycle_errors = []
    # compute cycle errors for each triplet connected to seed_node
    connected_triplets = [t for t in unused_triplets if seed_node in t]
    for triplet in connected_triplets:
        cycle_error, _, _ = cycle_utils.compute_rot_cycle_error(
            i2Ri1_dict,
            cycle_nodes=triplet,
            two_view_reports_dict=two_view_reports_dict,
            verbose=True,
        )
        seed_cycle_errors.append(cycle_error)

    import pdb

    pdb.set_trace()

    min_error_triplet_idx = np.argmin(np.array(seed_cycle_errors))
    seed_triplet = connected_triplets[min_error_triplet_idx]
    # find one with low cycle_error

    unused_triplets.remove(seed_triplet)

    edges = [(i0, i1), (i1, i2), (i0, i2)]

    i2Si1_dict_consensus = {edge: i2Si1_dict[edge] for edge in edges}

    while True:
        candidate_consensus = copy.deepcopy(i2Si1_dict_consensus)

        for triplet in unused_triplets:
            import pdb

            pdb.set_trace()

        # can we do this? i3Ui3 = i3Ri2 * i2Ui3 = i2Ri1 * i1Ui3 (cycle is 3->1->2)

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
    plot_save_dir: str,
) -> None:
    """ """
    draw_graph_topology(
        edges=list(i2Ri1_dict.keys()),
        gt_floor_pose_graph=gt_floor_pose_graph,
        two_view_reports_dict=two_view_reports_dict,
        title=f"Building {building_id} {floor_id}: topology after CNN prediction of positives",
        show_plot=False,
        save_fpath=f"{plot_save_dir}/{building_id}_{floor_id}_topology_CNN_predicted_positives.jpg",
    )

    i2Ri1_dict, i2Ui1_dict_consistent = cycle_utils.filter_to_rotation_cycle_consistent_edges(
        i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize=False
    )
    draw_graph_topology(
        edges=list(i2Ri1_dict.keys()),
        gt_floor_pose_graph=gt_floor_pose_graph,
        two_view_reports_dict=two_view_reports_dict,
        title=f"Building {building_id} {floor_id}: topology after rot. cycle consistency filtering",
        show_plot=False,
        save_fpath=f"{plot_save_dir}/{building_id}_{floor_id}_topology_after_rot_cycle_consistency.jpg",
    )

    filtered_edge_acc = get_edge_accuracy(edges=i2Ri1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
    print(f"\tFiltered by rot cycles Edge Acc = {filtered_edge_acc:.2f}")

    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(i2Ri1_dict.keys())
    print(
        f"After triplet rot cycle filtering, the largest CC contains {len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
    )

    # # could count how many nodes or edges never appeared in any triplet
    # triplets = cycle_utils.extract_triplets(i2Ri1_dict)
    # dropped_edges = set(i2Ri1_dict.keys()) - set(i2Ri1_dict_consistent.keys())
    # print("Dropped ")

    wRi_list = rotation_averaging.globalaveraging2d(i2Ri1_dict)
    if wRi_list is None:
        print(f"Rotation averaging failed, because {len(i2Ri1_dict)} measurements provided.")
        print()
        print()

        report = FloorReconstructionReport(
            avg_abs_rot_err=np.nan, avg_abs_trans_err=np.nan, percent_panos_localized=0.0
        )
        return None, None, report
    # TODO: measure the error in rotations

    # filter to rotations that are consistent with global
    i2Ri1_dict = filter_measurements_to_absolute_rotations(
        wRi_list, i2Ri1_dict, max_allowed_deviation=5, two_view_reports_dict=two_view_reports_dict
    )
    draw_graph_topology(
        edges=list(i2Ri1_dict.keys()),
        gt_floor_pose_graph=gt_floor_pose_graph,
        two_view_reports_dict=two_view_reports_dict,
        title=f"Building {building_id} {floor_id}: topology after filtering relative by global",
        show_plot=False,
        save_fpath=f"{plot_save_dir}/{building_id}_{floor_id}_topology_after_filtering_relative_by_global.jpg",
    )

    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(i2Ri1_dict.keys())
    print(
        f"After filtering by rel. vs. composed abs., the largest CC contains {len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
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

    draw_graph_topology(
        edges=list(i2Si1_dict_consistent.keys()),
        gt_floor_pose_graph=gt_floor_pose_graph,
        two_view_reports_dict=two_view_reports_dict,
        title=f"Building {building_id} {floor_id}: topology after filtering by translations",
        show_plot=False,
        save_fpath=f"{plot_save_dir}/{building_id}_{floor_id}_topology_after_filtering_translations.jpg",
    )

    cc_nodes = graph_utils.get_nodes_in_largest_connected_component(i2Si1_dict.keys())
    print(
        f"After triplet trans. cycle filtering, the largest CC contains {len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
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

        report = FloorReconstructionReport(
            avg_abs_rot_err=np.nan, avg_abs_trans_err=np.nan, percent_panos_localized=0.0
        )
        return None, None, report

    num_localized_panos = np.array([wSi is not None for wSi in wSi_list]).sum()
    num_floor_panos = len(gt_floor_pose_graph.nodes)
    percent_panos_localized = num_localized_panos / num_floor_panos * 100
    print(f"Localized {percent_panos_localized:.2f}% of panos: {num_localized_panos} / {num_floor_panos}")

    # TODO: try spanning tree version, vs. Shonan version
    wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
    wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(
        wRi_list, wti_list, gt_floor_pose_graph, building_id, floor_id
    )

    mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_unaligned_abs_pose_error(
        gt_floor_pg=gt_floor_pose_graph
    )
    print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
    est_floor_pose_graph.render_estimated_layout(
        show_plot=False, save_plot=True, plot_save_dir=plot_save_dir, gt_floor_pg=gt_floor_pose_graph
    )

    print()
    print()

    report = FloorReconstructionReport(
        avg_abs_rot_err=mean_abs_rot_err,
        avg_abs_trans_err=mean_abs_trans_err,
        percent_panos_localized=percent_panos_localized,
    )
    return est_floor_pose_graph, i2Si1_dict_consistent, report


def filter_measurements_to_absolute_rotations(
    wRi_list: List[Optional[np.ndarray]],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    max_allowed_deviation: float = 5.0,
    verbose: bool = False,
    two_view_reports_dict=None,
    visualize: bool = False,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Simulate the relative pose measurement, given the global rotations:
    wTi0 = (wRi0, wti0). wTi1 = (wRi1, wti1) -> wRi0, wRi1 -> i1Rw * wRi0 -> i1Ri0_ vs. i1Ri0

    Reference: See FilterViewPairsFromOrientation()
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_orientation.h
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_orientation.cc

    Theia also uses a 5 degree threshold:
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/reconstruction_estimator_options.h#L122

    Args:
        wRi_list:
        i2Ri1_dict:
        max_allowed_deviation:
        verbose:
        two_view_reports_dict:
        visualize:

    Returns:
        i2Ri1_dict_consistent:
    """
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


def visualize_deviations_from_ground_truth(hypotheses_save_root: str) -> None:
    """ """
    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_save_root}/*")]

    for building_id in building_ids:

        floor_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{hypotheses_save_root}/{building_id}/*")]
        for floor_id in floor_ids:

            # for each pano-pair
            for i1 in range(200):

                for i2 in range(200):

                    correct_Sim2_fpaths = glob.glob(
                        f"{hypotheses_save_root}/{building_id}/{floor_id}/gt_alignment_approx/{i1}_{i2}_*.json"
                    )
                    incorrect_Sim2_fpaths = glob.glob(
                        f"{hypotheses_save_root}/{building_id}/{floor_id}/incorrect_alignment/{i1}_{i2}_*.json"
                    )

                    if len(incorrect_Sim2_fpaths) == 0 or len(correct_Sim2_fpaths) == 0:
                        continue

                    correct_Sim2 = Sim2.from_json(correct_Sim2_fpaths[0])

                    # how far is each from GT?
                    for incorrect_Sim2_fpath in incorrect_Sim2_fpaths:
                        incorrect_Sim2 = Sim2.from_json(incorrect_Sim2_fpath)

                        if np.isclose(correct_Sim2.theta_deg, incorrect_Sim2.theta_deg, atol=0.1):
                            print(
                                f"{building_id} {floor_id}: Correct {correct_Sim2.theta_deg:.1f} vs {incorrect_Sim2.theta_deg:.1f}, Correct {np.round(correct_Sim2.translation,2)} vs {np.round(incorrect_Sim2.translation,2)}"
                            )

                    print()
                    print()


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

    # # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_binary_model_edge_classifications"
    # # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_edge_classifications_fixed_argmax_bug/2021_07_13_edge_classifications_fixed_argmax_bug"
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26/2021_07_15_serialized_edge_classifications/2021_07_15_serialized_edge_classifications"

    # # training, v2
    # # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_28_serialized_edge_classifications"

    # raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    # # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # # vis_edge_classifications(serialized_preds_json_dir, raw_dataset_dir)

    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"


    # 186 tours, low-res, RGB only floor and ceiling.
    serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20/2021_10_22_serialized_edge_classifications"
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"

    run_incremental_reconstruction(hypotheses_save_root, serialized_preds_json_dir, raw_dataset_dir)
