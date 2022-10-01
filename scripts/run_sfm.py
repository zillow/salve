"""Threshold front-end measurements and run back-end optimization to compute global poses.

Visualizations are saved for each floor.
"""

import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import click
import gtsfm.utils.graph as gtsfm_graph_utils
import matplotlib.pyplot as plt
import numpy as np

import salve.algorithms.cycle_consistency as cycle_utils
import salve.algorithms.pose2_slam as pose2_slam
import salve.algorithms.rotation_averaging as rotation_averaging
import salve.algorithms.spanning_tree as spanning_tree
import salve.common.edge_classification as edge_classification
import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
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
from salve.common.sim2 import Sim2


def get_conf_thresholded_edges(
    measurements: List[EdgeClassification],
    hypotheses_save_root: str,
    confidence_threshold: float,
    building_id: str,
    floor_id: str,
    gt_floor_pose_graph: Optional[PoseGraph2d] = None,
) -> Tuple[
    Dict[Tuple[int, int], Sim2],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], TwoViewEstimationReport],
    List[Tuple[int, int]],
    Dict[Tuple[int, int], EdgeWDOPair],
    List[EdgeClassification],
    DefaultDict[str, float],
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
        gt_floor_pose_graph: ground truth pose graph for this particular building floor, to allow
            computation of per-edge errors w.r.t. GT.

    Returns:
        i2Si1_dict: Similarity(2) for each edge.
        i2Ri1_dict: 2d relative rotation for each edge
        i2ti1_dict: 2d relative translation for each edge.
        two_view_reports_dict:
        gt_edges: list of tuples (i1,i2) representing all edges in true adjacency graph.
        per_edge_wdo_dict: mapping from edge (i1,i2) to EdgeWDOPair information.
        high_conf_measurements: all measurements of sufficient confidence, even if forming a multigraph.
        wdo_type_counter: dictionary containing counts of W/D/O types.
    """
    # for each edge, choose the most confident prediction over all WDO pair alignments
    most_confident_edge_dict = defaultdict(list)
    high_conf_measurements = []

    i2Si1_dict = {}
    i2Ri1_dict = {}
    i2ti1_dict = {}
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
        # print(m)

        most_confident_edge_dict[(m.i1, m.i2)] += [m]
        high_conf_measurements.append(m)

    wdo_type_counter = defaultdict(float)
    for m in high_conf_measurements:
        alignment_object, _, _ = m.wdo_pair_uuid.split("_")
        wdo_type_counter[alignment_object] += 1 / len(high_conf_measurements)
    print("WDO Type Distribution: ", wdo_type_counter)

    per_edge_wdo_dict: Dict[Tuple[int, int], EdgeWDOPair] = {}

    if len(high_conf_measurements) > 0:
        # Compute errors over all edges (repeated per pano)
        measure_avg_relative_pose_errors(
            high_conf_measurements,
            gt_floor_pose_graph,
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

        per_edge_wdo_dict[(i1, i2)] = EdgeWDOPair.from_wdo_pair_uuid(i1=i1, i2=i2, wdo_pair_uuid=m.wdo_pair_uuid)

        # TODO: choose the most confident score. How often is the most confident one,
        # the right one, among all of the choices?
        # use model confidence
        i2Si1 = edge_classification.get_alignment_hypothesis_for_measurement(
            m, hypotheses_save_root, building_id, floor_id
        )
        i2Si1_dict[(m.i1, m.i2)] = i2Si1

        i2Ri1_dict[(m.i1, m.i2)] = i2Si1.rotation
        i2ti1_dict[(m.i1, m.i2)] = i2Si1.translation

        R_error_deg = 0
        U_error_deg = 0
        two_view_reports_dict[(m.i1, m.i2)] = cycle_utils.TwoViewEstimationReport(
            gt_class=m.y_true, R_error_deg=R_error_deg, U_error_deg=U_error_deg, confidence=m.prob
        )

        if m.y_true == 1:
            gt_edges.append((m.i1, m.i2))

        # print(m)
        # TODO: fix this
        # mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=verbose)

    # wRi_list = rotation_averaging.globalaveraging2d(i2Ri1_dict)
    # est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)
    # mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
    #     gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges, verbose=False
    # )
    # print(f"\tMean relative rotation error {mean_rel_rot_err:.2f} deg.")

    print(f"most confident was correct {np.array(most_confident_was_correct).mean():.2f}")

    EPS = 1e-10
    class_imbalance_ratio = num_gt_negatives / (num_gt_positives + EPS)
    print(f"\tClass imbalance ratio {class_imbalance_ratio:.2f}")

    return (
        i2Si1_dict,
        i2Ri1_dict,
        i2ti1_dict,
        two_view_reports_dict,
        gt_edges,
        per_edge_wdo_dict,
        high_conf_measurements,
        wdo_type_counter,
    )


def measure_avg_relative_pose_errors(
    measurements: List[EdgeClassification],
    gt_floor_pg: "PoseGraph2d",
    hypotheses_save_root: str,
    building_id: str,
    floor_id: str,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Measure the error on each edge for Similarity(2) or SE(2) measurements, for rotation and for translation.

    Created for evaluation of measurements without an estimated pose graph (only edges). Can be multiple edges (multigraph)
    between two nodes.

    Note: edge accuracy doesn't mean anything (too many FPs from noisy GT). A much more reliable metric is the average
    relative pose error on each edge (w.r.t. GT poses).

    Args:
        hypotheses_save_root: Path to where alignment hypotheses are saved on disk.
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.
        verbose:

    Returns:
        mean_rot_err: average rotation error on each relative pose prediction.
        mean_trans_err: average translation error on each relative pose prediction.
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
        i2Ti1 = edge_classification.get_alignment_hypothesis_for_measurement(
            m, hypotheses_save_root, building_id, floor_id
        )

        theta_deg_est = i2Ti1.theta_deg
        theta_deg_gt = i2Ti1_gt.theta_deg

        # need to wrap around at 360
        rot_err = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
        rot_errs.append(rot_err)

        trans_err = np.linalg.norm(i2Ti1_gt.translation - i2Ti1.translation)
        trans_errs.append(trans_err)

        if verbose:
            print(
                f"\tPano pair ({i1},{i2}): (Rot) GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}, Trans Error {trans_err:.1f} from {np.round(i2Ti1_gt.translation,1)} vs. {np.round(i2Ti1.translation,1)}"
            )

    print(
        REDTEXT
        + f"Max relative rot error: {max(rot_errs):.1f}, Max relative trans error: {max(trans_errs):.1f} (normalized)"
        + ENDCOLOR
    )

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

    filtered_edge_error = float("inf")  # compute error w.r.t. GT from i2Si1_dict_consistent
    print(f"\tFiltered by SE(2) cycles Edge Acc = {filtered_edge_error:.2f}")

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

    mean_abs_rot_err, mean_abs_trans_err, _, _ = est_floor_pose_graph.measure_unaligned_abs_pose_error(
        gt_floor_pg=gt_floor_pose_graph
    )
    print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
    est_floor_pose_graph.render_estimated_layout()

    print()
    print()


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
                            print_str = f"{building_id} {floor_id}: Correct {correct_Sim2.theta_deg:.1f}"
                            print_str += (
                                f" vs {incorrect_Sim2.theta_deg:.1f}, Correct {np.round(correct_Sim2.translation,2)}"
                            )
                            print_str += f" vs {np.round(incorrect_Sim2.translation,2)}"
                            print(print_str)

                    print()
                    print()


def plot_confidence_histograms(measurements: List[EdgeClassification]) -> None:
    """Create a histogram of network confidences for each of the following groups separately:
    (True Positives), (False Positives), (False Negatives), (True Negatives).

    Useful for tuning confidence thresholds. TPs should ideally have all high confidences,
    and TNs should have all low confidences.

    Args:
        measurements: possible relative pose hypotheses, before confidence thresholding is applied.
    """
    probs = np.array([m.prob for m in measurements])
    y_true_array = np.array([m.y_true for m in measurements])
    y_hat_array = np.array([m.y_hat for m in measurements])
    is_TP, is_FP, is_FN, is_TN = pr_utils.assign_tp_fp_fn_tn(y_true_array, y_hat_array)

    plt.subplot(2, 2, 1)
    plt.hist(probs[is_TP], bins=30)  # 15 bins are too few, to see 98%-99% confidence bin
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


def run_incremental_reconstruction(
    hypotheses_save_root: str,
    serialized_preds_json_dir: str,
    raw_dataset_dir: str,
    method: str,
    confidence_threshold: float,
    use_axis_alignment: bool,
    allowed_wdo_types: List[str],
    predictions_data_root: str,
) -> None:
    """Run the global optimization stage on confidence-thresholded relative pose hypotheses, in the largest CC.

    Note: Can get multi-graph out of classification model.

    We optionally apply axis alignment pre-processing before evaluation. Axis-alignment cannot be added via
    post-processing, as in that case translations will not be updated.

    Note: edge accuracy doesn't mean anything (too many FPs from noisy GT). A much more reliable metric is the average
    relative pose error on each edge (w.r.t. GT poses).

    Args:
        hypotheses_save_root: path to directory where alignment hypotheses are saved as JSON files.
        serialized_preds_json_dir: path to directory where model predictions (per edge) have been serialized as JSON.
        raw_dataset_dir: path to directory where the full ZinD dataset is stored (in raw form as downloaded from Bridge API).
        method: Global pose aggregation method (e.g. `pgo`, `spanning_tree`, etc.)
        confidence_threshold: Minimum required SALVe network confidence to accept a prediction.
        use_axis_alignment: whether to refine relative rotations by vanishing angle.
        allowed_wdo_types: types of W/D/O objects to use for localization (only these edge types will be inserted into the graph).
        predictions_data_root: Path to directory containing ModifiedHorizonNet (MHNet) predictions.
    """
    # TODO: determine why some FPs have zero cycle error? why so close to GT?

    allowed_wdo_types_summary = "_".join(allowed_wdo_types)
    plot_save_dir = f"{Path(serialized_preds_json_dir).name}___2022_07_05_{method}_floorplans_with_conf_{confidence_threshold}_{allowed_wdo_types_summary}_axisaligned{use_axis_alignment}"
    os.makedirs(plot_save_dir, exist_ok=True)

    floor_edgeclassifications_dict = edge_classification.get_edge_classifications_from_serialized_preds(
        serialized_preds_json_dir, allowed_wdo_types=allowed_wdo_types
    )

    reconstruction_reports = []

    averaged_wdo_type_counter = defaultdict(list)

    # Probability distribution function & cumulative dist. fn of #panos in first N CCs, for each building floor.
    pdfs = []
    cdfs = []

    # Loop over each building/floor tuple.
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        inferred_floor_pose_graph = hnet_prediction_loader.load_inferred_floor_pose_graph(
            building_id=building_id,
            floor_id=floor_id,
            raw_dataset_dir=raw_dataset_dir,
            predictions_data_root=predictions_data_root
        )

        gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        print(f"On building {building_id}, {floor_id}")

        # (building_id == "0564" and floor_id == "floor_01") or \
        is_demo = (
            (building_id == "0519" and floor_id == "floor_01")
            or (building_id == "1214" and floor_id == "floor_01")
            # or (building_id == "0308" and floor_id == "floor_02")
            or (building_id == "0438" and floor_id == "floor_01")
            or (building_id == "0715" and floor_id == "floor_01")
        )
        if not is_demo:
            continue

        visualize_confidence_histograms = False
        if visualize_confidence_histograms:
            plot_confidence_histograms(measurements)

        render_multigraph = True
        if render_multigraph:
            graph_rendering_utils.draw_multigraph(
                measurements=measurements,
                gt_floor_pose_graph=gt_floor_pose_graph,
                inferred_floor_pose_graph=None,
                use_gt_positions=True,
                confidence_threshold=confidence_threshold,
                save_dir=plot_save_dir
            )

        (
            i2Si1_dict,
            i2Ri1_dict,
            i2Ui1_dict,
            two_view_reports_dict,
            gt_edges,
            per_edge_wdo_dict,
            high_conf_measurements,
            wdo_type_counter,
        ) = get_conf_thresholded_edges(
            measurements, hypotheses_save_root, confidence_threshold, building_id, floor_id, gt_floor_pose_graph
        )

        pdf, cdf = graph_utils.analyze_cc_distribution(
            nodes=list(gt_floor_pose_graph.nodes.keys()), edges=list(i2Si1_dict.keys())
        )
        pdfs.append(pdf)
        cdfs.append(cdf)

        # update statistics about average edge type from W/D/O-based edges, and log a summary.
        for wdo_type, percent in wdo_type_counter.items():
            averaged_wdo_type_counter[wdo_type].append(percent)
        print("On average, over all tours, WDO types used were:")
        for wdo_type, percents in averaged_wdo_type_counter.items():
            print(REDTEXT + f"For {wdo_type}, {np.mean(percents)*100:.1f}%" + ENDCOLOR)

        if len(high_conf_measurements) == 0:
            print_str = f"Skip global optimization for Building {building_id}, {floor_id}"
            print_str += f" -> no high conf. measurements from {len(measurements)} measurements."
            print(print_str)
            report = FloorReconstructionReport(
                avg_abs_rot_err=np.nan, avg_abs_trans_err=np.nan, percent_panos_localized=0.0, floorplan_iou=0.0
            )
            reconstruction_reports.append(report)
            continue

        cc_nodes = gtsfm_graph_utils.get_nodes_in_largest_connected_component(i2Si1_dict.keys())
        print(
            "Before any filtering, the largest CC contains "
            f"{len(cc_nodes)} / {len(gt_floor_pose_graph.nodes.keys())} panos ."
        )
        if method == "spanning_tree":

            if use_axis_alignment:
                i2Si1_dict = axis_alignment_utils.align_pairs_by_vanishing_angle(
                    i2Si1_dict=i2Si1_dict,
                    inferred_floor_pose_graph=inferred_floor_pose_graph,
                    per_edge_wdo_dict=per_edge_wdo_dict,
                )

            wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict, verbose=False)
            est_floor_pose_graph = PoseGraph2d.from_wSi_list(wSi_list, gt_floor_pose_graph)
            report = FloorReconstructionReport.from_est_floor_pose_graph(
                est_floor_pose_graph, gt_floor_pose_graph, plot_save_dir=plot_save_dir
            )
            reconstruction_reports.append(report)

        elif method in ["pose2_slam", "pgo"]:
            # graph_rendering_utils.draw_multigraph(high_conf_measurements, gt_floor_pose_graph)

            if use_axis_alignment:
                # Align here to compute initialization with aligned pairs.
                i2Si1_dict = axis_alignment_utils.align_pairs_by_vanishing_angle(
                    i2Si1_dict=i2Si1_dict,
                    inferred_floor_pose_graph=inferred_floor_pose_graph,
                    per_edge_wdo_dict=per_edge_wdo_dict,
                )

            wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict, verbose=False)
            report = pose2_slam.execute_planar_slam(
                measurements=high_conf_measurements,
                gt_floor_pg=gt_floor_pose_graph,
                hypotheses_save_root=hypotheses_save_root,
                building_id=building_id,
                floor_id=floor_id,
                wSi_list=wSi_list,
                plot_save_dir=plot_save_dir,
                optimize_poses_only="pgo" == method,
                use_axis_alignment=use_axis_alignment,
                per_edge_wdo_dict=per_edge_wdo_dict,
                inferred_floor_pose_graph=inferred_floor_pose_graph,
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
                i2Si1 = edge_classification.get_alignment_hypothesis_for_measurement(
                    m, hypotheses_save_root, building_id, floor_id
                )
                high_conf_measurements_rel.append(RelativePoseMeasurement(i1=m.i1, i2=m.i2, i2Si1=i2Si1))

            wSi_list = spanning_tree.ransac_spanning_trees(
                high_conf_measurements_rel, num_hypotheses=100, min_num_edges_for_hypothesis=None
            )
            est_floor_pose_graph = PoseGraph2d.from_wSi_list(wSi_list, gt_floor_pose_graph)
            report = FloorReconstructionReport.from_est_floor_pose_graph(
                est_floor_pose_graph, gt_floor_pose_graph, plot_save_dir=plot_save_dir
            )
            reconstruction_reports.append(report)

        elif method == "filtered_spanning_tree":
            # filtered by cycle consistency.
            import sandbox.filtered_spanning_tree as filtered_spanning_tree

            i2Si1_dict_consistent, report = filtered_spanning_tree.build_filtered_spanning_tree(
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

    show_pdf_cdf = False
    if show_pdf_cdf:
        aggregate_cc_distributions(pdfs, cdfs)

    floor_reconstruction_report.summarize_reports(reconstruction_reports)

    print("Completed Eval with:")
    print(f"\tconfidence>={confidence_threshold}")
    print(f"\tmethod={method}")
    print(f"\tfrom serializations {Path(serialized_preds_json_dir).name}")
    print("\tUsing object types", allowed_wdo_types)
    print(f"\tUsed axis alignment: {use_axis_alignment}")


def aggregate_cc_distributions(pdfs: List[np.ndarray], cdfs: List[np.ndarray]) -> None:
    """Summarize PDFs and CDFs of per-floor connected component distributions into a single PDF and CDF.

    Args:
        pdfs: probability distribution function for each building floor.
        cdfs: cumulative distribution function for each building floor.
    """
    max_num_ccs = max([len(pdf) for pdf in pdfs])

    avg_pdf = np.zeros((max_num_ccs))
    avg_cdf = np.zeros((max_num_ccs))

    for pdf, cdf in zip(pdfs, cdfs):
        C = pdf.shape[0]

        # pad the rest (long tail) of the PDF with 0s
        padded_pdf = np.zeros(max_num_ccs)
        padded_pdf[:C] = pdf

        # pad the rest of the CDF with 1s
        padded_cdf = np.ones(max_num_ccs)
        padded_cdf[:C] = cdf

        avg_pdf += padded_pdf
        avg_cdf += padded_cdf

    avg_pdf /= len(pdfs)
    avg_cdf /= len(cdfs)

    graph_utils.plot_pdf_cdf(avg_pdf, avg_cdf)


@click.command(help="Script to run SfM using SALVe verifier predictions.")
@click.option(
    "--serialized_preds_json_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where serialized predictions were saved to (from executing `test.py`).",
)
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--hypotheses_save_root",
    type=click.Path(exists=True),
    required=True,
    help="Directory where JSON files with alignment hypotheses have been saved to (from executing"
    " `export_alignment_hypotheses.py`)",
)
@click.option(
    "--method",
    type=click.Choice([
            "spanning_tree",
            "SE2_cycles",
            "filtered_spanning_tree",
            "random_spanning_trees",
            "pose2_slam",
            "pgo",

        ]),
    required=True,
    help="Global aggregation method.",
)
@click.option(
    "--mhnet_predictions_data_root",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing ModifiedHorizonNet (MHNet) predictions.",
)
@click.option(
    "--confidence_threshold",
    type=float,
    default=0.93,
    help="Minimum required SALVe network confidence to accept a prediction.",
)
@click.option(
    "--use_axis_alignment",
    type=bool,
    default=True,
    help="Whether to use estimated vanishing points to postprocess relatiive pose hypotheses.",
)
def launch_run_incremental_reconstruction(
    serialized_preds_json_dir: str,
    raw_dataset_dir: str,
    hypotheses_save_root: str,
    method: str,
    mhnet_predictions_data_root: str,
    confidence_threshold: float,
    use_axis_alignment: bool, 
) -> None:
    """Click entry point for SfM using SALVe predictions."""

    allowed_wdo_types = ["door", "window", "opening"]  #    ["window"] #  ["opening"] # ["door"] #

    print("Run SfM with settings:")
    print(f"\thypotheses_save_root={hypotheses_save_root}")
    print(f"\tserialized_preds_json_dir={serialized_preds_json_dir}")
    print(f"\traw_dataset_dir={raw_dataset_dir}")
    print(f"\tmethod={method}")
    print(f"\tconfidence_threshold={confidence_threshold}")
    print(f"\tuse_axis_alignment={use_axis_alignment}")
    print(f"\tallowed_wdo_types={allowed_wdo_types}")
    print(f"\tpredictions_data_root={mhnet_predictions_data_root}")

    run_incremental_reconstruction(
        hypotheses_save_root=hypotheses_save_root,
        serialized_preds_json_dir=serialized_preds_json_dir,
        raw_dataset_dir=raw_dataset_dir,
        method=method,
        confidence_threshold=confidence_threshold,
        use_axis_alignment=use_axis_alignment,
        allowed_wdo_types=allowed_wdo_types,
        predictions_data_root=mhnet_predictions_data_root
    )

    # cluster ID, pano ID, (x, y, theta). Share JSON for layout.


if __name__ == "__main__":
    """Example CLI usage:

    python scripts/run_sfm.py \
        --raw_dataset_dir ../zind_bridgeapi_2021_10_05/
        --method pgo
        --serialized_preds_json_dir ../2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23
        --hypotheses_save_root ../ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65

    python scripts/run_sfm.py \
        --raw_dataset_dir ../zind_bridgeapi_2021_10_05/
        --method pgo
        --serialized_preds_json_dir ../2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16
        --hypotheses_save_root ../ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65

    Predictions will be saved in a new directory named:
       {SALVE_REPO_ROOT}/{serialized_preds_json_dir.name}_{FLAGS}_serialized

    Visualizations will be saved in a new directory named:
       {SALVE_REPO_ROOT/{serialized_preds_json_dir.name}_{FLAGS}
    """
    launch_run_incremental_reconstruction()

