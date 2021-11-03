"""
Another translation-based filtering step is 1dsfm projection directions:
https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_relative_translation.cc
Must happen after rotation averaging.
"""

import copy
import glob
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import argoverse.utils.json_utils as json_utils
import argoverse.utils.geometry as geometry_utils
import gtsfm.utils.graph as graph_utils
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Rot2, Pose2

import afp.algorithms.cycle_consistency as cycle_utils
import afp.algorithms.mfas as mfas
import afp.algorithms.pose2_slam as pose2_slam
import afp.algorithms.rotation_averaging as rotation_averaging
import afp.algorithms.spanning_tree as spanning_tree
import afp.common.edge_classification as edge_classification
import afp.common.posegraph2d as posegraph2d
import afp.utils.axis_alignment_utils as axis_alignment_utils
import afp.utils.graph_rendering_utils as graph_rendering_utils
import afp.utils.iou_utils as iou_utils
import afp.utils.rotation_utils as rotation_utils
import afp.utils.pr_utils as pr_utils
from afp.algorithms.cycle_consistency import TwoViewEstimationReport
from afp.algorithms.cluster_merging import EdgeWDOPair
from afp.common.edge_classification import EdgeClassification
from afp.common.floor_reconstruction_report import FloorReconstructionReport
from afp.common.pano_data import PanoData
from afp.common.posegraph2d import PoseGraph2d, REDTEXT, ENDCOLOR


def get_conf_thresholded_edges(
    measurements: List[EdgeClassification],
    hypotheses_save_root: str,
    confidence_threshold: float,
    building_id: str,
    floor_id: str,
    gt_floor_pose_graph: Optional[PoseGraph2d] = None
) -> Tuple[
    Dict[Tuple[int, int], Sim2],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], TwoViewEstimationReport],
    List[Tuple[int, int]],
    Dict[Tuple[int, int], EdgeWDOPair],
    List[EdgeClassification]
]:
    """Among all model predictions for a particular floor of a home, select only the positive predictions
    with sufficiently high confidence.

    Note: We select the most confident prediction for each edge.

    Args:
        measurements: list containing the model's prediction for each edge.
        hypotheses_save_root: path to directory where alignment hypotheses are saved as JSON files.
        confidence_threshold: minimum confidence to treat a model's prediction as a positive.
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.

    Returns:
        i2Si1_dict: Similarity(2) for each edge.
        i2Ri1_dict: 2d relative rotation for each edge
        i2Ui1_dict:
        two_view_reports_dict:
        gt_edges:
        per_edge_wdo_dict:
        high_conf_measurements:
    """
    # for each edge, choose the most confident prediction over all WDO pair alignments
    most_confident_edge_dict = defaultdict(list)
    high_conf_measurements = []

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
        # (happens because we sort by partial room, not by i1, i2 in dataloader?)
        if m.i1 >= m.i2:
            i2 = m.i1
            i1 = m.i2

            m.i1 = i1
            m.i2 = i2

        # Useful for understanding tracks.
        #print(m)

        most_confident_edge_dict[(m.i1, m.i2)] += [m]
        high_conf_measurements.append(m)

    
    wdo_type_counter = defaultdict(float)
    for m in high_conf_measurements:
        alignment_object, _, _ = m.wdo_pair_uuid.split("_")
        wdo_type_counter[alignment_object] += 1/len(high_conf_measurements)
    print("WDO Type Distribution: ", wdo_type_counter)

    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair] = {}

    if len(high_conf_measurements) > 0:
        # Compute errors over all edges (repeated per pano)
        measure_avg_relative_pose_errors(high_conf_measurements, gt_floor_pose_graph,
            hypotheses_save_root,
            building_id,
            floor_id,
        )

    # keep track of how often the most confident prediction per edge was the correct one.
    most_confident_was_correct = []
    for (i1, i2), measurements in most_confident_edge_dict.items():

        most_confident_idx = np.argmax([m.prob for m in measurements])
        m = measurements[most_confident_idx]
        if len(measurements) > 1:
            most_confident_was_correct.append(m.y_true == 1)

        per_edge_wdo_dict[(i1, i2)] = EdgeWDOPair(i1=i1, i2=i2, wdo_pair_uuid=m.wdo_pair_uuid)

        # TODO: choose the most confident score. How often is the most confident one, the right one, among all of the choices?
        # use model confidence
        i2Si1 = edge_classification.get_alignment_hypothesis_for_measurement(m, hypotheses_save_root, building_id, floor_id)
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
        # TODO: fix this
        #mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=verbose)

    # #import pdb; pdb.set_trace()
    # wRi_list = rotation_averaging.globalaveraging2d(i2Ri1_dict)
    # est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)
    # mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
    #     gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=False
    # )
    # print(f"\tMean relative rotation error {mean_rel_rot_err:.2f} deg.")

    # mfas.run_mfas(wRi_list, i2Ui1_dict)

    print(f"most confident was correct {np.array(most_confident_was_correct).mean():.2f}")

    EPS = 1e-10
    class_imbalance_ratio = num_gt_negatives / (num_gt_positives + EPS)
    print(f"\tClass imbalance ratio {class_imbalance_ratio:.2f}")

    return i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, gt_edges, per_edge_wdo_dict, high_conf_measurements, wdo_type_counter


def measure_avg_relative_pose_errors(measurements: List[EdgeClassification], gt_floor_pg: "PoseGraph2d",
        hypotheses_save_root: str,
        building_id: str,
        floor_id: str,
        verbose: bool = False
    ) -> float:
    """Measure the error on each edge for Similarity(2) or SE(2) measurements, for rotation and for translation.

    Created for evaluation of measurements without an estimated pose graph (only edges). Can be multiple edges (multigraph)
    between two nodes.

    Args:
        
    """
    rot_errs = []
    trans_errs = []
    for m in measurements:

        i1 = m.i1
        i2 = m.i2

        wTi1_gt = gt_floor_pg.nodes[i1].global_Sim2_local
        wTi2_gt = gt_floor_pg.nodes[i2].global_Sim2_local
        i2Ti1_gt = wTi2_gt.inverse().compose(wTi1_gt)

        # technically it is i2Si1, but scale will always be 1 with inferred WDO.
        i2Ti1 = edge_classification.get_alignment_hypothesis_for_measurement(m, hypotheses_save_root, building_id, floor_id)

        theta_deg_est = i2Ti1.theta_deg
        theta_deg_gt = i2Ti1_gt.theta_deg

        # need to wrap around at 360
        rot_err = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
        rot_errs.append(rot_err)

        trans_err = np.linalg.norm(i2Ti1_gt.translation - i2Ti1.translation)
        trans_errs.append(trans_err)

        if verbose:
            print(f"\tPano pair ({i1},{i2}): (Rot) GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}, Trans Error {trans_err:.1f} from {np.round(i2Ti1_gt.translation,1)} vs. {np.round(i2Ti1.translation,1)}")

    print(REDTEXT + f"Max rot error: {max(rot_errs):.1f}, Max trans error: {max(trans_errs):.1f}" + ENDCOLOR)

    mean_rot_err = np.mean(rot_errs)
    mean_trans_err = np.mean(trans_errs)
    print_str = f"Mean relative rot. error: {mean_rot_err:.1f}. trans. error: {mean_trans_err:.1f}. Over  {len(gt_floor_pg.nodes)} of {len(gt_floor_pg.nodes)} GT panos"
    print_str += f", estimated {len(rot_errs)} edges"
    print(REDTEXT + print_str + ENDCOLOR)

    return mean_rot_err, mean_trans_err


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

    wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict_consistent, verbose=False)

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

    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pose_graph)

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

    import pdb; pdb.set_trace()

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
    """Uses chained cycle consistency.

    """
    graph_rendering_utils.draw_graph_topology(
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
    graph_rendering_utils.draw_graph_topology(
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
        return None,  report
    # TODO: measure the error in rotations

    # filter to rotations that are consistent with global
    i2Ri1_dict = filter_measurements_to_absolute_rotations(
        wRi_list, i2Ri1_dict, max_allowed_deviation=5, two_view_reports_dict=two_view_reports_dict
    )
    graph_rendering_utils.draw_graph_topology(
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

    graph_rendering_utils.draw_graph_topology(
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

    wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict_consistent, verbose=False)

    if wSi_list is None:
        print(f"Could not build spanning tree, since {len(i2Si1_dict_consistent)} edges in i2Si1 dictionary.")
        print()
        print()

        report = FloorReconstructionReport(
            avg_abs_rot_err=np.nan, avg_abs_trans_err=np.nan, percent_panos_localized=0.0
        )
        return None, report


    report = FloorReconstructionReport.from_wSi_list(wSi_list, gt_floor_pose_graph, plot_save_dir=plot_save_dir)
    return i2Si1_dict_consistent, report


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

        theta_deg_inferred = rotation_utils.rotmat2theta_deg(i2Ri1_inferred)
        theta_deg_measured = rotation_utils.rotmat2theta_deg(i2Ri1_measured)

        if verbose:
            print(f"\tPano pair ({i1},{i2}): Measured {theta_deg_measured:.1f} vs. Inferred {theta_deg_inferred:.1f}")

        # need to wrap around at 360
        err = rotation_utils.wrap_angle_deg(theta_deg_inferred, theta_deg_measured)
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
    """Compute the average accuracy per-edge (as classified by GT supervision).

    We check the GT for each predicted (i.e. allowed) edge.
    """
    preds = []
    # what is the purity of what comes out?
    for (i1, i2) in edges:
        preds.append(two_view_reports_dict[(i1, i2)].gt_class)

    acc = np.mean(preds)
    return acc


def run_incremental_reconstruction(
    hypotheses_save_root: str, serialized_preds_json_dir: str, raw_dataset_dir: str
) -> None:
    """
    Can get multi-graph out of classification model.

    Args:
        hypotheses_save_root: path to directory where alignment hypotheses are saved as JSON files.
        serialized_preds_json_dir: path to directory where model predictions (per edge) have been serialized as JSON.
        raw_dataset_dir: path to directory where the full ZinD dataset is stored (in raw form as downloaded from Bridge API).
    """
    # TODO: determine why some FPs have zero cycle error? why so close to GT?

    method = "spanning_tree" 
    # method = "SE2_cycles"
    # method = "growing_consensus"
    # method = "filtered_spanning_tree"
    # method = "random_spanning_trees"
    # method = "pose2_slam"
    # method = "pgo"

    # TODO: add axis alignment.

    confidence_threshold =  0.97 # 0.97 #8 # 0.98  # 0.95 # 0.95 # 0.90 # 0.95 # 1.01 #= 0.95

    plot_save_dir = (
        f"2021_10_26_testsplit_morerendered_{method}_floorplans_with_gt_conf_{confidence_threshold}"
    )
    os.makedirs(plot_save_dir, exist_ok=True)

    floor_edgeclassifications_dict = edge_classification.get_edge_classifications_from_serialized_preds(serialized_preds_json_dir)

    reconstruction_reports = []

    averaged_wdo_type_counter = defaultdict(list)

    # loop over each building and floor
    # for each building/floor tuple
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        # if not (building_id == "1635" and floor_id == "floor_02"):
        #     continue

        gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        print(f"On building {building_id}, {floor_id}")

        # continue

        visualize_confidence_histograms = False
        if visualize_confidence_histograms:
            probs = np.array([m.prob for m in measurements])
            y_true_array = np.array([m.y_true for m in measurements])
            y_hat_array = np.array([m.y_hat for m in measurements])
            is_TP, is_FP, is_FN, is_TN = pr_utils.assign_tp_fp_fn_tn(y_true_array, y_hat_array)

            plt.subplot(2, 2, 1)
            plt.hist(probs[is_TP], bins=30) # 15 bins are too few, to see 98%-99% confidence bin
            plt.title("TP")

            plt.subplot(2, 2, 2)
            plt.hist(probs[is_FP], bins=30)
            plt.title("FP")

            plt.subplot(2, 2, 3)
            plt.hist(probs[is_FN], bins=30)
            plt.title("FN")

            plt.subplot(2, 2, 4)
            plt.hist(probs[is_TN], bins=30)
            plt.title("TN")

            plt.show()

        render_multigraph = False
        if render_multigraph:
            graph_rendering_utils.draw_multigraph(measurements, gt_floor_pose_graph)

        i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, gt_edges, _, high_conf_measurements, wdo_type_counter = get_conf_thresholded_edges(
            measurements, hypotheses_save_root, confidence_threshold, building_id, floor_id, gt_floor_pose_graph
        )
        # TODO: edge accuracy doesn't mean anything (too many FPs). Use average error on each edge, instead.

        for wdo_type, percent in wdo_type_counter.items():
            averaged_wdo_type_counter[wdo_type].append(percent)

        print("On average, over all tours, WDO types used were:")
        for wdo_type, percents in averaged_wdo_type_counter.items():
            print(REDTEXT + f"For {wdo_type}, {np.mean(percents)*100:.1f}%" + ENDCOLOR)

        if len(high_conf_measurements) == 0:
            print(f"Skip Building {building_id}, {floor_id} -> no high conf measurements from {len(measurements)} measurements.")
            continue

        unfiltered_edge_acc = get_edge_accuracy(edges=i2Si1_dict.keys(), two_view_reports_dict=two_view_reports_dict)
        print(f"\tUnfiltered Edge Acc = {unfiltered_edge_acc:.2f}")

        cc_nodes = graph_utils.get_nodes_in_largest_connected_component(i2Si1_dict.keys())
        print(
            f"Before any filtering, the largest CC contains {len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
        )

        # TODO: apply axis alignment pre-processing (or post-processing) before evaluation

        if method == "spanning_tree":

            # i2Si1_dict = align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)

            wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict, verbose=False)
            report = FloorReconstructionReport.from_wSi_list(
                wSi_list, gt_floor_pose_graph, plot_save_dir=f"raw_spanning_tree_only_{confidence_threshold}"
            )
            reconstruction_reports.append(report)

        elif method in ["pose2_slam", "pgo"]:
            #graph_rendering_utils.draw_multigraph(high_conf_measurements, gt_floor_pose_graph)
            wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict, verbose=False)
            report = pose2_slam.execute_planar_slam(
                measurements=high_conf_measurements,
                gt_floor_pg=gt_floor_pose_graph,
                hypotheses_save_root=hypotheses_save_root,
                building_id=building_id,
                floor_id=floor_id,
                wSi_list=wSi_list,
                plot_save_dir=plot_save_dir,
                optimize_poses_only="pgo"==method
            )
            reconstruction_reports.append(report)

            # TODO: build better unit tests.

        elif method == "random_spanning_trees":
            # V. M. Govindu. Robustness in motion averaging. In ACCV, 2006.
            # count the number of relative motions (i.e. edges) that fall within this distance from the global motion.
            # C. Olsson and O. Enqvist. Stable structure from motion for unordered image collections. In SCIA, 2011. LNCS 6688.

            # generate 100 spanning trees.
            from afp.algorithms.spanning_tree import RelativePoseMeasurement
            high_conf_measurements_rel = []
            for m in high_conf_measurements:
                i2Si1 = edge_classification.get_alignment_hypothesis_for_measurement(m, hypotheses_save_root, building_id, floor_id)
                high_conf_measurements_rel.append(RelativePoseMeasurement(i1=m.i1, i2=m.i2, i2Si1=i2Si1))

            wSi_list = spanning_tree.ransac_spanning_trees(high_conf_measurements_rel, num_hypotheses=100, min_num_edges_for_hypothesis=None)
            report = FloorReconstructionReport.from_wSi_list(
                wSi_list, gt_floor_pose_graph, plot_save_dir=plot_save_dir
            )
            reconstruction_reports.append(report)

        elif method == "filtered_spanning_tree":
            # filtered by cycle consistency.
            i2Si1_dict_consistent, report = build_filtered_spanning_tree(
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

            # i2Si1_dict, i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, _, per_edge_wdo_dict, high_conf_measurements, wdo_type_counter = get_conf_thresholded_edges(
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


def align_pairs_by_vanishing_angle(
    i2Si1_dict: Dict[Tuple[int,int], Sim2], gt_floor_pose_graph: PoseGraph2d, visualize: bool = True
) -> Dict[Tuple[int,int], Sim2]:
    """ """

    for (i1,i2), i2Si1 in i2Si1_dict.items():

        verts_i1 = gt_floor_pose_graph.nodes[i1].room_vertices_local_2d
        verts_i2 = gt_floor_pose_graph.nodes[i2].room_vertices_local_2d
        
        if visualize:
            plt.subplot(1,2,1)
            draw_polygon(verts_i1, color='r', linewidth=5)
            draw_polygon(verts_i2, color='g', linewidth=1)
            plt.axis("equal")

        dominant_angle_deg1, angle_frac1 = axis_alignment_utils.determine_rotation_angle(verts_i1)
        dominant_angle_deg2, angle_frac2 = axis_alignment_utils.determine_rotation_angle(verts_i2)

        # Below: using the oracle.
        # wSi1 = gt_floor_pose_graph.nodes[i1].global_Sim2_local
        # wSi2 = gt_floor_pose_graph.nodes[i2].global_Sim2_local
        # wSi1 = i2Si1_dict[i1]
        # wSi2 = i2Si1_dict[i2]
        # i2Si1 = wSi2.inverse().compose(wSi1)

        # import pdb; pdb.set_trace()
        i2Ri1_dominant = rotation_utils.rotmat2d(theta_deg=dominant_angle_deg2 - dominant_angle_deg1)
        i2Si1_dominant = Sim2(R=i2Ri1_dominant, t=np.zeros(2), s=1.0)
        # verts_i1_ = i2Si1_dominant.transform_from(verts_i1)


        # method = "rotate_about_origin_first"
        # method = "rotate_about_origin_last"
        #method = "rotate_about_centroid_first"
        method = "none"
        
        # i1a and i2a represent the aligned frames.
        if method == "rotate_about_origin_first":
            # apply rotation first (delta pose)
            i1Si1a = i2Si1_dominant
            i2Si1a = i2Si1.compose(i1Si1a)
            verts_i1_ = i2Si1a.transform_from(verts_i1)

        elif method == "rotate_about_origin_last":
            # apply rotation second (much worse)
            i2aSi2 = i2Si1_dominant
            i2aSi1 = i2aSi2.compose(i2Si1)
            verts_i1_ = i2aSi1.transform_from(verts_i1)

        elif method == "rotate_about_centroid_first":

            import pdb; pdb.set_trace()
            verts_i1_ = geometry_utils.rotate_polygon_about_pt(
                verts_i1, rotmat=i2Ri1_dominant, center_pt=np.mean(verts_i1, axis=0)
            )
            verts_i1_ = i2Si1.transform_from(verts_i1_)
            # TODO: compute new translation that will accomplish this via Pose2.align()

        elif method == "none":

            # TODO: wrong, since layouts need to be expressed in the body frame.
            verts_i1_ = i2Si1.transform_from(verts_i1)


        if visualize:
            plt.subplot(1,2,2)
            draw_polygon(verts_i1_, color='r', linewidth=5)
            draw_polygon(verts_i2, color='g', linewidth=1)
            plt.axis("equal")
            plt.show()

        import pdb; pdb.set_trace()

    return i2Si1_dict


def draw_polygon(poly: np.ndarray, color: str, linewidth: float = 1) -> None:
    """ """
    verts = np.vstack([poly, poly[0]]) # allow connection between the last and first vertex

    plt.plot(verts[:,0], verts[:,1], color=color, linewidth=linewidth)
    plt.scatter(verts[:,0], verts[:,1], 10, color=color, marker='.')



def test_align_pairs_by_vanishing_angle() -> None:
    """Ensure we can use vanishing angles to make small corrections to rotations, with perfect layouts (no noise)."""
    
    # cameras are at the center of each room. doorway is at (0,1.5)
    wTi1 = Pose2(Rot2(), np.array([1.5,1.5]))
    wTi2 = Pose2(Rot2(), np.array([-1.5,1.5]))

    i2Ri1 = wTi2.between(wTi1).rotation().matrix() # 2x2 identity matrix.
    i2ti1 = wTi2.between(wTi1).translation()

    i2Ri1_noisy = rotation_utils.rotmat2d(theta_deg=30)

    # simulate noisy rotation, but correct translation
    i2Si1_dict = {(1,2): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

    # We cannot specify these layouts in the world coordinate frame. They must be in the local body frame.
    # fmt: off
    # Note: this is provided in i1's frame.
    layout1_w = np.array(
        [
            [0,0],
            [3,0],
            [3,3],
            [0,3]
        ]
    ).astype(np.float32)
    # TODO: figure out how to undo the effect on the translation, as well!

    #(align at WDO only! and then recompute!)
    # Note: this is provided in i2's frame.
    #layout2 = (layout1 - np.array([3,0])) @ i2Ri1_noisy.T
    # layout2 = (layout1 @ i2Ri1_noisy.T) - np.array([3,0])
    layout2_w = geometry_utils.rotate_polygon_about_pt(layout1_w - np.array([3.,0]), rotmat=i2Ri1_noisy, center_pt=np.array([-1.5,1.5]))
    # fmt: on

    draw_polygon(layout1_w, color='r', linewidth=5)
    draw_polygon(layout2_w, color='g', linewidth=1)
    plt.axis("equal")
    plt.show()

    def transform_point_cloud(pts_w: np.ndarray, iTw: Pose2) -> np.ndarray:
        """Transfer from world frame to camera i's frame."""
        return np.vstack([iTw.transformFrom(pt_w) for pt_w in pts_w])

    layout1_i1 = transform_point_cloud(layout1_w, iTw=wTi1.inverse())
    layout2_i2 = transform_point_cloud(layout2_w, iTw=wTi2.inverse())

    pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1_i1})
    pano_data_2 = SimpleNamespace(**{"room_vertices_local_2d": layout2_i2})

    # use square and rotated square
    gt_floor_pose_graph = SimpleNamespace(**{"nodes": {1: pano_data_1, 2: pano_data_2}})

    import pdb; pdb.set_trace()
    # ensure delta pose is accounted for properly.
    i2Si1_dict_aligned = align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)


def test_align_pairs_by_vanishing_angle_noisy() -> None:
    """Ensure delta pose is accounted for properly, with dominant rotation directions computed from noisy layouts.

    Using noisy contours, using Douglas-Peucker to capture the rough manhattanization.

    """
    assert False
    i2Ri1 = None
    i2ti1 = None

    # simulate noisy rotation, but correct translation
    i2Si1_dict = {(0,1): Sim2(R=i2Ri1, t=i2ti1, s=1.0)}

    layout0 = np.array([[]])
    layout1 = np.array([[]])

    pano_data_0 = SimpleNamespace(**{"room_vertices_local_2d": layout0})
    pano_data_1 = SimpleNamespace(**{"room_vertices_local_2d": layout1})

    # use square and rotated square
    gt_floor_pose_graph = SimpleNamespace(**{"nodes": {0: pano_data_0, 1: pano_data_1}})

    i2Si1_dict_aligned = align_pairs_by_vanishing_angle(i2Si1_dict, gt_floor_pose_graph)



def measure_acc_vs_visual_overlap(serialized_preds_json_dir: str) -> None:
    """
    ONLY FOR POSITIVE EXAMPLES! DOES IT MAKE SENSE TO EVALUTE THIS FOR NEGATIVE EXAMPLES?
    """
    import imageio

    pairs = []

    # maybe interesting to also check histograms at different confidence thresholds
    confidence_threshold = 0.0

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    # import random
    # random.shuffle(json_fpaths)
    for json_idx, json_fpath in enumerate(json_fpaths):
        print(f"On {json_idx}/{len(json_fpaths)}")

        # if json_idx > 300:
        #     continue

        json_data = json_utils.read_json_file(json_fpath)
        y_hat_list = json_data["y_hat"]
        y_true_list = json_data["y_true"]
        y_hat_prob_list = json_data["y_hat_probs"]
        fp0_list = json_data["fp0"]
        fp1_list = json_data["fp1"]

         # for each GT positive
        for y_hat, y_true, y_hat_prob, fp0, fp1 in zip(y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list):

            if y_true != 1:
                continue
       
            if y_hat_prob < confidence_threshold:
                continue            

            f1 = imageio.imread(fp0)
            f2 = imageio.imread(fp1)
            floor_iou = iou_utils.texture_map_iou(f1, f2)

            pairs += [(floor_iou, y_hat)]

    bin_edges = np.linspace(0,1,11)
    counts = np.zeros(10)
    iou_bins = np.zeros(10)

    # running computation of the mean.
    for (iou, y_pred) in pairs:

        bin_idx = np.digitize(iou, bins=bin_edges)
        # digitize puts it into `bins[i-1] <= x < bins[i]` so we have to subtract 1
        iou_bins[bin_idx - 1] += y_pred
        counts[bin_idx - 1] += 1

    normalized_iou_bins = np.divide(iou_bins.astype(np.float32), counts.astype(np.float32))

    # bar chart.
    plt.bar(np.arange(10), normalized_iou_bins)
    plt.xlabel("Floor-Floor Texture Map IoU")
    plt.ylabel("Mean Accuracy")

    xtick_labels = []
    for i in range(len(bin_edges)-1):
        xtick_labels += [ f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})" ]
    plt.xticks(ticks=np.arange(10), labels=xtick_labels, rotation=20)
    plt.tight_layout()

    plt.savefig(f"{Path(serialized_preds_json_dir).stem}___bar_chart_iou_positives_only__confthresh{confidence_threshold}.jpg", dpi=500)

    


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


    # 186 tours, low-res, RGB only floor and ceiling. custom hacky val split
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20/2021_10_22_serialized_edge_classifications"
    # raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    # hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"

    # 373 training tours, low-res, RGB only floor and ceiling, true ZinD train/val/test split
    #serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26_serialized_edge_classifications"
    #serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26_serialized_edge_classifications_v2_more_rendered"
    
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_22___ResNet50_186tours_serialized_edge_classifications_test2021_11_02"
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26__ResNet50_373tours_serialized_edge_classifications_test2021_11_02"
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"

    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"

    #run_incremental_reconstruction(hypotheses_save_root, serialized_preds_json_dir, raw_dataset_dir)

    #test_align_pairs_by_vanishing_angle()

    serialized_preds_json_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"
    measure_acc_vs_visual_overlap(serialized_preds_json_dir)


    # cluster ID, pano ID, (x, y, theta). Share JSON for layout.


