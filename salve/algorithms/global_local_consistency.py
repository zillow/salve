"""Algorithms and utilities for comparing global and local consistency of poses."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import salve.algorithms.rotation_averaging as rotation_averaging
import salve.utils.rotation_utils as rotation_utils
from salve.algorithms.cycle_consistency import TwoViewEstimationReport
from salve.common.sim2 import Sim2
from salve.common.edge_classification import EdgeClassification


def compute_edge_consistency_fraction(
    wSi_list: List[Optional[Sim2]],
    i2Si1_dict: Dict[Tuple[int,int], Sim2],
    max_allowed_deviation_deg: float,
    two_view_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]] = None,
) -> float:
    """Compute the fraction of rotation-consistent edges, given input edges and estimated global poses.

    Args:
        wSi_list: estimated global poses.
        i2Si1_dict: relative pose i2Si1 as Sim(2) for each pano pair (i1,i2).
        max_allowed_deviation_deg: max allowed rotation angular deviation.

    Returns:
        Fraction of rotation consistent edges, in the range [0,1].
    """
    i2Ri1_dict = { (i1,i2): i2Si1.rotation for (i1,i2), i2Si1 in i2Si1_dict.items()}
    wRi_list = [wSi.rotation if wSi is not None else None for wSi in wSi_list]

    i2Ri1_dict_consistent = filter_measurements_to_absolute_rotations(
        wRi_list=wRi_list,
        i2Ri1_dict = i2Ri1_dict,
        max_allowed_deviation_deg=max_allowed_deviation_deg,
        verbose=True,
        two_view_reports_dict=two_view_reports_dict,
        visualize = False,
    )
    return len(i2Ri1_dict_consistent) / len(i2Si1_dict)



def convert_to_i2Ri1_dict(i2Si1_dict: Dict[Tuple[int, int], Sim2]) -> Dict[Tuple[int, int], np.ndarray]:
    """Extract Rot(2)'s from Sim(2) dictionary.

    Args:
        i2Si1_dict: Similarity(2) relative pose for each (i1,i2) pano-pano edge.

    Returns:
        i2Ri1_dict: 2d relative rotation for each edge.
    """
    return {(i1, i2): i2Si1.rotation for (i1, i2), i2Si1 in i2Si1_dict.items()}


def filter_measurements_by_global_local_consistency(
    i2Si1_dict: Dict[Tuple[int, int], Sim2],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    max_allowed_deviation_deg: float,
) -> Dict[Tuple[int, int], Sim2]:
    """Use 2d rotation averaging to estimate global rotations per pano, and then filter by local-global consistency.

    Args:
        i2Si1_dict: Similarity(2) relative pose for each (i1,i2) pano-pano edge. Contains E1 edges.
        two_view_reports_dict: mapping from (i1,i2) pano pair to relative rotation and translation error w.r.t. GT.
        max_allowed_deviation_deg:

    Returns:
        i2Si1_dict: Similarity(2) relative pose for each **FILTERED** (i1,i2) pano-pano edge. Contains E2 edges,
            where E2 <= E1.
    """
    wRi_list = rotation_averaging.globalaveraging2d(convert_to_i2Ri1_dict(i2Si1_dict))
    # Filter to rotations that are consistent with estimated global rotations.
    i2Ri1_dict_consistent = filter_measurements_to_absolute_rotations(
        wRi_list=wRi_list,
        i2Ri1_dict=convert_to_i2Ri1_dict(i2Si1_dict),
        max_allowed_deviation_deg=5.0,
        verbose=False,
        two_view_reports_dict=two_view_reports_dict,
        visualize=False,
    )
    # Remove the edges that we deem to be outliers.
    outlier_edges = set(i2Si1_dict.keys()) - set(i2Ri1_dict_consistent.keys())
    for outlier_edge in outlier_edges:
        del i2Si1_dict[outlier_edge]
    return i2Si1_dict


def filter_measurements_to_absolute_rotations(
    wRi_list: List[Optional[np.ndarray]],
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray],
    max_allowed_deviation_deg: float = 5.0,
    verbose: bool = False,
    two_view_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]] = None,
    visualize: bool = False,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Determine which edges are consistent between actual and synthesized relative rotation (from est. global rot.).

    Simulate the relative pose measurement, given the global rotations:
    wTi0 = (wRi0, wti0). wTi1 = (wRi1, wti1) -> wRi0, wRi1 -> i1Rw * wRi0 -> i1Ri0_ vs. i1Ri0

    Reference: See FilterViewPairsFromOrientation()
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_orientation.h
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_pairs_from_orientation.cc

    Theia also uses a 5 degree threshold:
    https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/reconstruction_estimator_options.h#L122

    Args:
        wRi_list: global rotations per camera (ordered by panorama index).
        i2Ri1_dict: relative rotation for each (i1,i2) pano pair.
        max_allowed_deviation_deg: maximum allowed angular deviation in degrees.
        verbose: whether to log to STDOUT the deviation of synthetic vs. actual measurement.
        two_view_reports_dict:
        visualize:

    Returns:
        i2Ri1_dict_consistent: mapping from (i1,i2) to relative rotation i2Ri1 for each edge
            that has sufficient consistency between global rotations and relative rotation measurement.
    """
    deviations = []
    is_gt_edge = []

    edges = i2Ri1_dict.keys()

    i2Ri1_dict_consistent = {}

    for (i1, i2) in edges:

        if wRi_list[i1] is None or wRi_list[i2] is None:
            continue

        # Find synthetic relative rotation, given estimated absolute rotations.
        wTi1 = wRi_list[i1]
        wTi2 = wRi_list[i2]
        i2Ri1_inferred = wTi2.T @ wTi1

        i2Ri1_measured = i2Ri1_dict[(i1, i2)]

        theta_deg_inferred = rotation_utils.rotmat2theta_deg(i2Ri1_inferred)
        theta_deg_measured = rotation_utils.rotmat2theta_deg(i2Ri1_measured)

        # need to wrap around at 360
        err = rotation_utils.wrap_angle_deg(theta_deg_inferred, theta_deg_measured)

        if verbose:
            print(
                f"\tPano pair ({i1},{i2}): Measured {theta_deg_measured:.1f} vs. Inferred {theta_deg_inferred:.1f}"
                + f", {err:.2f} --> {two_view_reports_dict[(i1,i2)].gt_class}"
            )

        if err < max_allowed_deviation_deg:
            i2Ri1_dict_consistent[(i1, i2)] = i2Ri1_measured

        if two_view_reports_dict is not None:
            is_gt_edge.append(two_view_reports_dict[(i1, i2)].gt_class)
            deviations.append(err)

    if two_view_reports_dict is not None and visualize:
        _visualize_filtering_accuracy(deviations=deviations, is_gt_edge=is_gt_edge)

    print(
        f"\tFound that {len(i2Ri1_dict_consistent)} of {len(i2Ri1_dict)} rotations were consistent w/ global rotations"
    )
    return i2Ri1_dict_consistent


def _visualize_filtering_accuracy(deviations, is_gt_edge) -> None:
    """TODO

    Args:
        deviations:
        is_gt_edge:
    """
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
