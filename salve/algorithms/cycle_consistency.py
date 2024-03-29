"""Utilities for cycle triplet extraction and cycle error computation.

Reference: see GTSFM
https://github.com/borglab/gtsfm/blob/master/gtsfm/view_graph_estimator/cycle_consistent_rotation_estimator.py
"""

import logging
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

import salve.utils.pr_utils as pr_utils
from salve.common.sim2 import Sim2
from salve.common.two_view_estimation_report import TwoViewEstimationReport
from salve.utils.rotation_utils import rotmat2theta_deg

logger = logging.getLogger(__name__)


ROT_CYCLE_ERROR_THRESHOLD = 0.5  # 1.0 # 5.0 # 2.5 # 1.0 # 0.3


def extract_triplets(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> List[Tuple[int, int, int]]:
    """Discover triplets from a graph, without O(n^3) complexity, by using intersection within adjacency lists.

    Based off of Theia's implementation:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/math/graph/triplet_extractor.h

    If we have an edge a<->b, if we can find any node c such that a<->c and b<->c, then we have
    discovered a triplet. In other words, we need only look at the intersection between the nodes
    connected to `a` and the nodes connected to `b`.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    adj_list = create_adjacency_list(i2Ri1_dict)

    # only want to keep the unique ones
    triplets = set()

    # find intersections
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        if i1 >= i2:
            raise RuntimeError("Graph edges (i1,i2) must be ordered with i1 < i2 in the image loader.")

        nodes_from_i1 = adj_list[i1]
        nodes_from_i2 = adj_list[i2]
        node_intersection = (nodes_from_i1).intersection(nodes_from_i2)
        for node in node_intersection:
            cycle_nodes = tuple(sorted([i1, i2, node]))
            if cycle_nodes not in triplets:
                triplets.add(cycle_nodes)

    return list(triplets)


def create_adjacency_list(i2Ri1_dict: Dict[Tuple[int, int], np.ndarray]) -> DefaultDict[int, Set[int]]:
    """Create an adjacency-list representation of a **rotation** graph G=(V,E) when provided its edges E.

    Note: this is specific to the rotation averaging use case, where some edges may be unestimated
    (i.e. their relative rotation is None), in which case they are not incorporated into the graph.

    In an adjacency list, the neighbors of each vertex may be listed efficiently, in time proportional to the
    degree of the vertex. In an adjacency matrix, this operation takes time proportional to the number of
    vertices in the graph, which may be significantly higher than the degree.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.

    Returns:
        adj_list: adjacency list representation of the graph, mapping an image index to its neighbors
    """
    adj_list = defaultdict(set)

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        adj_list[i1].add(i2)
        adj_list[i2].add(i1)

    return adj_list


def compute_rot_cycle_error(
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    cycle_nodes: Tuple[int, int, int],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    verbose: bool = True,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute the cycle error by the magnitude of the axis-angle rotation after composing 3 rotations.

    Note: i1 < i2 for every valid edge, by construction.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.
        cycle_nodes: 3-tuples of nodes that form a cycle. Nodes of are provided in sorted order.
        two_view_reports_dict:
        verbose: whether to dump to logger information about error in each Euler angle

    Returns:
        cycle_error: deviation from 3x3 identity matrix, in degrees. In other words,
            it is defined as the magnitude of the axis-angle rotation of the composed transformations.
        max_rot_error: maximum rotation error w.r.t. GT across triplet edges, in degrees.
            If ground truth is not known for a scene, None will be returned instead.
        max_trans_error: maximum translation error w.r.t. GT across triplet edges, in degrees.
            If ground truth is not known for a scene, None will be returned instead.
    """
    cycle_nodes = list(cycle_nodes)
    cycle_nodes.sort()

    i0, i1, i2 = cycle_nodes

    i1Ri0 = i2Ri1_dict[(i0, i1)]
    i2Ri1 = i2Ri1_dict[(i1, i2)]
    i0Ri2 = i2Ri1_dict[(i0, i2)].T

    # should compose to identity, with ideal measurements
    i0Ri0 = i0Ri2 @ i2Ri1 @ i1Ri0

    cycle_error = np.abs(rotmat2theta_deg(i0Ri0))

    # form 3 edges between fully connected subgraph (nodes i,j,k)
    e_i = (i0, i1)
    e_j = (i1, i2)
    e_k = (i0, i2)

    rot_errors = [two_view_reports_dict[e].R_error_deg for e in [e_i, e_j, e_k]]
    trans_errors = [two_view_reports_dict[e].U_error_deg for e in [e_i, e_j, e_k]]

    gt_known = all([err is not None for err in rot_errors])
    if gt_known:
        max_rot_error = np.max(rot_errors)
        max_trans_error = np.max(trans_errors)
    else:
        # ground truth unknown, so cannot estimate error w.r.t. GT
        max_rot_error = None
        max_trans_error = None

    gt_classes = [two_view_reports_dict[e].gt_class for e in [e_i, e_j, e_k]]

    if verbose:
        i1Ri0_theta = rotmat2theta_deg(i1Ri0)
        i2Ri1_theta = rotmat2theta_deg(i2Ri1)
        i0Ri2_theta = rotmat2theta_deg(i0Ri2)

        logger.info("\n")
        logger.info(f"{i0},{i1},{i2} --> Cycle error is: {cycle_error:.1f}")
        if gt_known:
            logger.info(f"Triplet: w/ max. R err {max_rot_error:.1f}, and w/ max. t err {max_trans_error:.1f}")

        logger.info(
            "GT %s Theta: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.",
            str(gt_classes),
            i1Ri0_theta,
            i2Ri1_theta,
            i0Ri2_theta,
        )

    return cycle_error, max_rot_error, max_trans_error


def compute_SE2_cycle_error(
    i2Si1_dict: Dict[Tuple[int, int], np.ndarray],
    cycle_nodes: Tuple[int, int, int],
    two_view_reports_dict,
    verbose: bool = True,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute the cycle error over chained Similarity(2) transformations.

    Cycle error here is defined by the magnitude of the deviation from identity (as axis-angle magnitude
    of rotation and separately as L2 norm of translation) after composing the three Similirity(2) objects.

    Note: i1 < i2 for every valid edge, by construction.

    Args:
        i2Si1_dict: mapping from image pair indices to relative Similarity(2) transformation.
        cycle_nodes: 3-tuples of nodes that form a cycle. Nodes of are provided in sorted order.
        two_view_reports_dict:
        verbose: whether to dump to logger information about error in each Euler angle

    Returns:
        rot_cycle_error: deviation from 3x3 identity matrix, in degrees. In other words,
            it is defined as the magnitude of the axis-angle rotation of the composed transformations.
        trans_cycle_error: deviation from zero translation via L2 norm of translation, of composed transformations
            in cycle.
    """
    cycle_nodes = list(cycle_nodes)
    cycle_nodes.sort()

    i0, i1, i2 = cycle_nodes

    i1Si0 = i2Si1_dict[(i0, i1)]
    i2Si1 = i2Si1_dict[(i1, i2)]
    i0Si2 = i2Si1_dict[(i0, i2)].inverse()

    # should compose to identity, with ideal measurements
    i0Si0 = i0Si2.compose(i2Si1).compose(i1Si0)

    rot_cycle_error = np.abs(rotmat2theta_deg(i0Si0.rotation))
    trans_cycle_error = np.linalg.norm(i0Si0.translation)

    if verbose:
        edges = [(i0, i1), (i1, i2), (i0, i2)]
        num_outliers = sum([int(two_view_reports_dict[edge].gt_class != 1) for edge in edges])

        logger.info("\n")
        logger.info(
            f"{i0},{i1},{i2} --> Rot cycle error {rot_cycle_error:.2f}, "
            f"Trans. Cycle error is: {trans_cycle_error:.2f} w/ {num_outliers} outliers."
        )

    return rot_cycle_error, trans_cycle_error


def filter_to_SE2_cycle_consistent_edges(
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    two_view_reports_dict,
    SE2_cycle_rot_threshold_deg: float = 0.5,
    SE2_cycle_trans_threshold: float = 0.01,
    visualize: bool = True,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Sim2]:
    """Filter edges using both rotation and translation error between cycle start and endpoint.

    TODO: change SE2 references to Sim2

    Args:
        i2Si1_dict: dictionary containing estimated Sim(2) relative pose per pano-pano edge.
        two_view_reports_dict
        SE2_cycle_rot_threshold_deg: angular rotation threshold (in degrees).
        SE2_cycle_trans_threshold: translation threshold (in meters).
        visualize:

    Returns:
        i2Si1_dict_consistent: dictionary of filtered Sim(2) relative poses (one per edge).
    """
    rot_cycle_errors = []
    trans_cycle_errors = []
    num_outliers_per_cycle = []

    n_valid_edges = len([i2Si1 for (i1, i2), i2Si1 in i2Si1_dict.items() if i2Si1 is not None])

    # (i1,i2) pairs
    cycle_consistent_keys = set()

    triplets = extract_triplets(i2Si1_dict)
    for (i0, i1, i2) in triplets:
        rot_cycle_error, trans_cycle_error = compute_SE2_cycle_error(
            i2Si1_dict, cycle_nodes=(i0, i1, i2), two_view_reports_dict=two_view_reports_dict, verbose=False
        )
        if rot_cycle_error < SE2_cycle_rot_threshold_deg and trans_cycle_error < SE2_cycle_trans_threshold:

            cycle_consistent_keys.add((i0, i1))
            cycle_consistent_keys.add((i1, i2))
            cycle_consistent_keys.add((i0, i2))

        rot_cycle_errors.append(rot_cycle_error)
        trans_cycle_errors.append(trans_cycle_error)

        if two_view_reports_dict is not None and visualize:
            num_outliers = 0
            edges = [(i0, i1), (i1, i2), (i0, i2)]
            for edge in edges:
                if two_view_reports_dict[edge].gt_class != 1:
                    num_outliers += 1
            num_outliers_per_cycle.append(num_outliers)

            if verbose:
                print(
                    f"Rot. cycle error: {rot_cycle_error:.2f}, " \
                    f"trans. cycle error: {trans_cycle_error:.2f} -> {num_outliers}"
                )

    if two_view_reports_dict is not None and visualize:
        num_outliers_per_cycle = np.array(num_outliers_per_cycle)
        rot_cycle_errors = np.array(rot_cycle_errors)
        render_binned_cycle_errors(
            num_outliers_per_cycle, rot_cycle_errors, max_error_bin_edge=30, error_type="rotation"
        )

        num_outliers_per_cycle = np.array(num_outliers_per_cycle)
        trans_cycle_errors = np.array(trans_cycle_errors)
        render_binned_cycle_errors(
            num_outliers_per_cycle, trans_cycle_errors, max_error_bin_edge=2.0, error_type="translation"
        )

    i2Si1_dict_consistent = {}
    for (i1, i2) in cycle_consistent_keys:
        i2Si1_dict_consistent[(i1, i2)] = i2Si1_dict[(i1, i2)]

    num_consistent_rotations = len(i2Si1_dict_consistent)
    logger.info("Found %d consistent rel. rotations from %d original edges.", num_consistent_rotations, n_valid_edges)
    return i2Si1_dict_consistent


def filter_to_rotation_cycle_consistent_edges(
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    i2Ui1_dict: Dict[Tuple[int, int], np.ndarray],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    visualize: bool = False,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """Remove edges in a graph where concatenated transformations along a 3-cycle does not compose to identity.

    Note: will return only a subset of these two dictionaries.

    Concatenating the transformations along a loop in the graph should return the identity function in an
    ideal, noise-free setting.

    Based off of:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_graph_cycles_by_rotation.cc

    See also:
        C. Zach, M. Klopschitz, and M. Pollefeys. Disambiguating visual relations using loop constraints. In CVPR, 2010
        http://people.inf.ethz.ch/pomarc/pubs/ZachCVPR10.pdf

        Enqvist, Olof; Kahl, Fredrik; Olsson, Carl. Non-Sequential Structure from Motion. ICCVW, 2011.
        https://portal.research.lu.se/ws/files/6239297/2255278.pdf

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.
        i2Ui1_dict: mapping from image pair indices to relative translation direction.
            Should have same keys as i2Ri1_dict.
        two_view_reports_dict
        visualize: boolean indicating whether to plot cycle error vs. pose error w.r.t. GT

    Returns:
        i2Ri1_dict_consistent: subset of i2Ri1_dict, i.e. only including edges that belonged to some triplet
            and had cycle error below the predefined threshold.
        i2Ui1_dict_consistent: subset of i2Ui1_dict, as above.
    """
    # Check the cumulative translation/rotation errors between triplets to throw away cameras.
    cycle_errors = []
    max_rot_errors = []
    max_trans_errors = []

    n_valid_edges = len([i2Ri1 for (i1, i2), i2Ri1 in i2Ri1_dict.items() if i2Ri1 is not None])

    # (i1,i2) pairs
    cycle_consistent_keys = set()

    triplets = extract_triplets(i2Ri1_dict)

    for (i0, i1, i2) in triplets:
        cycle_error, max_rot_error, max_trans_error = compute_rot_cycle_error(
            i2Ri1_dict, [i0, i1, i2], two_view_reports_dict, verbose=False
        )

        if cycle_error < ROT_CYCLE_ERROR_THRESHOLD:

            cycle_consistent_keys.add((i0, i1))
            cycle_consistent_keys.add((i1, i2))
            cycle_consistent_keys.add((i0, i2))

        cycle_errors.append(cycle_error)
        max_rot_errors.append(max_rot_error)
        max_trans_errors.append(max_trans_error)

    if visualize:
        os.makedirs("plots", exist_ok=True)
        plt.scatter(cycle_errors, max_rot_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Rot3 error over cycle triplet")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_rot_error.jpg"), dpi=200)

        plt.scatter(cycle_errors, max_trans_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Unit3 error over cycle triplet")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_trans_error.jpg"), dpi=200)

    # logger.info("cycle_consistent_keys: " + str(cycle_consistent_keys))

    i2Ri1_dict_consistent, i2Ui1_dict_consistent = {}, {}
    for (i1, i2) in cycle_consistent_keys:
        i2Ri1_dict_consistent[(i1, i2)] = i2Ri1_dict[(i1, i2)]
        i2Ui1_dict_consistent[(i1, i2)] = i2Ui1_dict[(i1, i2)]

    num_consistent_rotations = len(i2Ri1_dict_consistent)
    logger.info("Found %d consistent rel. rotations from %d original edges.", num_consistent_rotations, n_valid_edges)
    assert len(i2Ui1_dict_consistent) == num_consistent_rotations
    return i2Ri1_dict_consistent, i2Ui1_dict_consistent


def filter_to_translation_cycle_consistent_edges(
    wRi_list: List[np.ndarray],
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    translation_cycle_thresh: 0.5,
    two_view_reports_dict=None,
    visualize: bool = False,
) -> Dict[Tuple[int, int], Sim2]:
    """ """

    # translation measurement not useful if missing rotation
    for (i1, i2) in i2Si1_dict:
        if wRi_list[i1] is None or wRi_list[i2] is None:
            # effectively remove such measurements
            i2Si1_dict[(i1, i2)] = None

    num_outliers_per_cycle = []
    cycle_errors = []
    n_valid_edges = len([i2Si1 for (i1, i2), i2Si1 in i2Si1_dict.items() if i2Si1 is not None])

    cycle_consistent_keys = set()
    triplets = extract_triplets(i2Si1_dict)

    for (i0, i1, i2) in triplets:
        cycle_error = compute_translation_cycle_error(wRi_list, i2Si1_dict, cycle_nodes=(i0, i1, i2), verbose=False)

        if cycle_error < translation_cycle_thresh:

            cycle_consistent_keys.add((i0, i1))
            cycle_consistent_keys.add((i1, i2))
            cycle_consistent_keys.add((i0, i2))

        cycle_errors.append(cycle_error)
        # max_rot_errors.append(max_rot_error)
        # max_trans_errors.append(max_trans_error)

        # if (i0,i1,i2) == (15, 17, 18):
        #     import pdb; pdb.set_trace()

        if two_view_reports_dict is not None and visualize:
            num_outliers = 0
            edges = [(i0, i1), (i1, i2), (i0, i2)]
            for edge in edges:
                if two_view_reports_dict[edge].gt_class != 1:
                    num_outliers += 1
            num_outliers_per_cycle.append(num_outliers)

    if two_view_reports_dict is not None and visualize:
        num_outliers_per_cycle = np.array(num_outliers_per_cycle)
        cycle_errors = np.array(cycle_errors)
        render_binned_cycle_errors(num_outliers_per_cycle, cycle_errors)

    # if visualize:
    #     plt.scatter(cycle_errors, max_rot_errors)
    #     plt.xlabel("Cycle error")
    #     plt.ylabel("Avg. Rot3 error over cycle triplet")
    #     plt.savefig(os.path.join("plots", "cycle_error_vs_GT_rot_error.jpg"), dpi=200)

    #     plt.scatter(cycle_errors, max_trans_errors)
    #     plt.xlabel("Cycle error")
    #     plt.ylabel("Avg. Unit3 error over cycle triplet")
    #     plt.savefig(os.path.join("plots", "cycle_error_vs_GT_trans_error.jpg"), dpi=200)

    # logger.info("cycle_consistent_keys: " + str(cycle_consistent_keys))

    i2Si1_dict_consistent = {}
    for (i1, i2) in cycle_consistent_keys:
        i2Si1_dict_consistent[(i1, i2)] = i2Si1_dict[(i1, i2)]

    num_consistent_rotations = len(i2Si1_dict_consistent)
    logger.info("Found %d consistent rel. rotations from %d original edges.", num_consistent_rotations, n_valid_edges)
    return i2Si1_dict_consistent


def render_binned_cycle_errors(
    num_outliers_per_cycle: np.ndarray,
    cycle_errors: np.ndarray,
    max_error_bin_edge: float,
    error_type: str,
    num_bins: int = 10,
) -> None:
    """Render a histogram of cycle errors associated with all edges with a specific number of outliers.

    Consider how many edges of the 3 triplet edges are corrupted.
    For example, for translations, we might plot histograms in the range [0,2]. For rotations, we might
    show a histogram of angular error in the range of [0,180].

    Args:
        num_outliers_per_cycle: array of shape (K,) representing number of outliers for each of K cycles.
           No more than 3 outliers can be found for the 3 edges of a triplet cycle.
        cycle_errors: array of shape (K,) representing cycle error for each cycle.
        max_error_bin_edge: far right edge of histogram in plot.
        error_type: string indicating type of error computed ("rotation" or "translation").
        num_bins: number of bins for cycle error histograms.
    """
    if len(num_outliers_per_cycle) != len(cycle_errors):
        raise RuntimeError("Each cycle must have a single associated scalar error and a single # of outliers.")

    plt.figure(figsize=(16, 5))
    outlier_bins = np.unique(num_outliers_per_cycle)
    num_outlier_bins = len(outlier_bins)
    min_error_bin_edge = 0
    bin_edges = np.linspace(min_error_bin_edge, max_error_bin_edge, num_bins)
    num_error_bins = len(bin_edges) - 1

    # Two passes through the data are required. The first pass identifies fixed y-axis limits for a shared y-axis.
    max_bin_count = compute_max_bin_count(
        num_outliers_per_cycle=num_outliers_per_cycle,
        cycle_errors=cycle_errors,
        min_error_bin_edge=min_error_bin_edge,
        max_error_bin_edge=max_error_bin_edge,
        bin_edges=bin_edges,
    )

    ylim_offset = 5
    for i, num_outliers in enumerate(outlier_bins):
        plt.subplot(1, num_outlier_bins, i + 1)
        idxs = num_outliers_per_cycle == num_outliers
        bin_errors = cycle_errors[idxs]
        plt.hist(bin_errors, bins=bin_edges)
        plt.title(f"Num. outliers = {num_outliers}")

        plt.ylim(0, max_bin_count + ylim_offset)
        plt.xlabel(f"Cycle {error_type} error")

        if i == 0:
            plt.ylabel("Counts")

    plt.show()
    plt.close("all")


def compute_max_bin_count(
    num_outliers_per_cycle: np.ndarray,
    cycle_errors: np.ndarray,
    min_error_bin_edge: float,
    max_error_bin_edge: float,
    bin_edges: np.ndarray,
) -> int:
    """Render a histogram of cycle errors associated with all edges with a specific number of outliers.

    Consider how many edges of the 3 triplet edges are corrupted.
    For example, for translations, we might plot histograms in the range [0,2]. For rotations, we might
    show a histogram of angular error in the range of [0,180].

    Args:
        num_outliers_per_cycle: array of shape (K,) representing number of outliers for each of K cycles.
           No more than 3 outliers can be found for the 3 edges of a triplet cycle.
        cycle_errors: array of shape (K,) representing cycle error for each cycle.
        min_error_bin_edge:
        max_error_bin_edge: far right edge of histogram in plot.
        bin_edges: error histogram bin edges.

    Returns:
        Maximum number of errors assigned to any outlier bin, used to set fixed y-axis limits.
    """
    num_error_bins = len(bin_edges) - 1
    outlier_bins = np.unique(num_outliers_per_cycle)

    max_bin_count = 0
    for num_outliers in outlier_bins:
        is_valid = num_outliers_per_cycle == num_outliers
        # Extract errors that fall into the bucket for this # of outliers.
        bin_errors = cycle_errors[is_valid]
        # Digitize provides accounts incremented by 1, so must decrement.
        assigned_bins = np.digitize(x=bin_errors, bins=bin_edges) - 1

        bin_counts = np.bincount(assigned_bins, minlength=num_error_bins)
        if np.argmax(bin_counts) >= num_error_bins:
            # If values in x are beyond the bounds of bins, assigned bin will be outside range.
            continue

        # Find the maximum count in any error bin.
        max_errors_single_bin = bin_counts.max()
        max_bin_count = max(max_bin_count, max_errors_single_bin)
    return max_bin_count


def compute_translation_cycle_error(
    wRi_list: List[np.ndarray],
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    cycle_nodes: Tuple[int, int, int],
    verbose: bool,
) -> float:
    """ """
    cycle_nodes = list(cycle_nodes)
    cycle_nodes.sort()

    i0, i1, i2 = cycle_nodes

    # translations must be in the world frame to make sense
    i1ti0 = wRi_list[i1] @ i2Si1_dict[(i0, i1)].translation * i2Si1_dict[(i0, i1)].scale
    i2ti1 = wRi_list[i2] @ i2Si1_dict[(i1, i2)].translation * i2Si1_dict[(i1, i2)].scale
    i0ti2 = wRi_list[i0] @ i2Si1_dict[(i0, i2)].inverse().translation * i2Si1_dict[(i0, i2)].inverse().scale

    # should add to zero translation, with ideal measurements
    i0ti0 = i0ti2 + i2ti1 + i1ti0

    cycle_error = np.linalg.norm(i0ti0)
    if verbose:
        logger.info(f"Cycle {i0}->{i1}->{i2} had cycle error {cycle_error:.2f}")
    return cycle_error


def estimate_rot_cycle_filtering_classification_acc(i2Ri1_dict, i2Ri1_dict_consistent, two_view_reports_dict) -> float:
    """
            Predicted
    Actual   ...

    Args:
        i2Ri1_dict:
        i2Ri1_dict_consistent:
        two_view_reports_dict:

    Returns:
        mean accuracy:
    """
    keys = list(i2Ri1_dict.keys())
    num_keys = len(keys)

    # Create confusion matrix.
    gt_idxs = np.zeros(num_keys, dtype=np.uint32)
    pred_idxs = np.zeros(num_keys, dtype=np.uint32)

    for i, key in enumerate(keys):
        gt_idxs[i] = two_view_reports_dict[key].gt_class
        pred_idxs[i] = 1 if key in i2Ri1_dict_consistent else 0

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true=gt_idxs, y_pred=pred_idxs)
    return prec, rec, mAcc
