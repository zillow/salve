

import copy
import glob
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import argoverse.utils.geometry as geometry_utils
import gtsfm.utils.graph as gtsfm_graph_utils
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Rot2, Pose2

from gtsam import Point3, Point3Pairs, Similarity3

import salve.algorithms.cycle_consistency as cycle_utils
import salve.algorithms.mfas as mfas
import salve.algorithms.pose2_slam as pose2_slam
import salve.algorithms.rotation_averaging as rotation_averaging
import salve.algorithms.spanning_tree as spanning_tree
import salve.common.edge_classification as edge_classification
import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
import salve.utils.axis_alignment_utils as axis_alignment_utils
import salve.utils.graph_utils as graph_utils
import salve.utils.graph_rendering_utils as graph_rendering_utils
import salve.utils.rotation_utils as rotation_utils
import salve.utils.pr_utils as pr_utils
from salve.algorithms.cycle_consistency import TwoViewEstimationReport
from salve.common.edge_classification import EdgeClassification
from salve.common.edgewdopair import EdgeWDOPair
from salve.common.floor_reconstruction_report import FloorReconstructionReport
from salve.common.posegraph2d import PoseGraph2d, REDTEXT, ENDCOLOR


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
    """Uses chained cycle consistency."""
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

    cc_nodes = gtsfm_graph_utils.get_nodes_in_largest_connected_component(i2Ri1_dict.keys())
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
        return None, report
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

    cc_nodes = gtsfm_graph_utils.get_nodes_in_largest_connected_component(i2Ri1_dict.keys())
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

    cc_nodes = gtsfm_graph_utils.get_nodes_in_largest_connected_component(i2Si1_dict.keys())
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
