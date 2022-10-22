"""Unit tests on relative pose cycle consistency utilities."""

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gtsam import Pose2, Rot2, Rot3, Unit3

import salve.algorithms.cycle_consistency as cycle_consistency
from salve.algorithms.cycle_consistency import TwoViewEstimationReport
from salve.common.sim2 import Sim2
from salve.utils.rotation_utils import rotmat2d


def test_render_binned_cycle_errors() -> None:
    """ """
    num_outliers_per_cycle = np.array([1, 2, 0, 0, 3])
    cycle_errors = np.array([1.5, 2.5, 0, 0.1, 2.9])
    # cycle_consistency.render_binned_cycle_errors(
    #     num_outliers_per_cycle, cycle_errors, max_error_bin_edge=3.0, error_type="rotation"
    # )


def test_compute_max_bin_count_all_inliers() -> None:
    """ """
    num_outliers_per_cycle = np.array([1, 2, 0, 0, 3])
    cycle_errors = np.array([1.5, 2.5, 0, 0.1, 2.9])

    min_error_bin_edge = 0.0
    max_error_bin_edge = 3.0

    bin_edges = [0, 1, 2, 3]
    max_bin_count = cycle_consistency.compute_max_bin_count(
        num_outliers_per_cycle=num_outliers_per_cycle,
        cycle_errors=cycle_errors,
        min_error_bin_edge=min_error_bin_edge,
        max_error_bin_edge=max_error_bin_edge,
        bin_edges=bin_edges,
    )
    assert max_bin_count == 2


def test_compute_max_bin_count_all_outliers() -> None:
    """ """
    num_outliers_per_cycle = np.array([1, 2, 0, 0, 3])
    cycle_errors = np.array([98, 99, 100, 101, 102])

    min_error_bin_edge = 0.0
    max_error_bin_edge = 3.0

    bin_edges = [0, 1, 2, 3, 4]

    max_bin_count = cycle_consistency.compute_max_bin_count(
        num_outliers_per_cycle=num_outliers_per_cycle,
        cycle_errors=cycle_errors,
        min_error_bin_edge=min_error_bin_edge,
        max_error_bin_edge=max_error_bin_edge,
        bin_edges=bin_edges,
    )
    assert max_bin_count == 0


def test_compute_max_bin_count_not_all_outlier_counts_represented() -> None:
    """Note that 0-outliers and 3-outliers are not represented here."""

    num_outliers_per_cycle = np.array([2, 1])
    cycle_errors = np.array([0.1, 5.1])

    min_error_bin_edge = 0.0
    max_error_bin_edge = 2.0
    # Two bins only.
    bin_edges = [0, 1, 2]

    max_bin_count = cycle_consistency.compute_max_bin_count(
        num_outliers_per_cycle=num_outliers_per_cycle,
        cycle_errors=cycle_errors,
        min_error_bin_edge=min_error_bin_edge,
        max_error_bin_edge=max_error_bin_edge,
        bin_edges=bin_edges,
    )

    assert max_bin_count == 1


def test_compute_translation_cycle_error() -> None:
    """Ensures that translation cycle-error computation is correct."""
    wTi_list = [
        Pose2(Rot2.fromDegrees(0), np.array([0, 0])),
        Pose2(Rot2.fromDegrees(90), np.array([0, 4])),
        Pose2(Rot2.fromDegrees(0), np.array([4, 0])),
        Pose2(Rot2.fromDegrees(0), np.array([8, 0])),
    ]
    wRi_list = [wTi.rotation().matrix() for wTi in wTi_list]

    i2Si1_dict = {
        (0, 1): Sim2(R=rotmat2d(-90), t=np.array([-4, 0]), s=1.0),
        (1, 2): Sim2(R=rotmat2d(90), t=np.array([-4, 4]), s=1.0),
        (0, 2): Sim2(R=np.eye(2), t=np.array([-4, 0]), s=1.0),
        (1, 3): Sim2(R=-rotmat2d(90), t=np.array([-8.2, 4]), s=1.0),  # add 0.2 noise
        (0, 3): Sim2(R=np.eye(2), t=np.array([-8, -0.2]), s=1.0),  # add 0.2 noise
    }

    cycle_nodes = (0, 1, 2)
    cycle_error = cycle_consistency.compute_translation_cycle_error(
        wRi_list, i2Si1_dict, cycle_nodes=cycle_nodes, verbose=True
    )
    # should be approximately zero error on this triplet
    assert np.isclose(cycle_error, 0.0, atol=1e-4)

    cycle_nodes = (0, 1, 3)
    cycle_error = cycle_consistency.compute_translation_cycle_error(
        wRi_list, i2Si1_dict, cycle_nodes=cycle_nodes, verbose=True
    )
    expected_cycle_error = np.sqrt(0.2**2 + 0.2**2)
    assert np.isclose(cycle_error, expected_cycle_error)


def test_estimate_rot_cycle_filtering_classification_acc() -> None:
    """Ensures that cycle-consistency based filtering by rotation measurements alone is correct."""
    i2Ri1 = np.eye(2)  # dummy value

    i2Ri1_dict = {
        (0, 1): i2Ri1,  # model TP, cycle TP
        (1, 2): i2Ri1,  # model TP, cycle TP
        (2, 3): i2Ri1,  # model FP, cycle TN
        (3, 4): i2Ri1,  # model FP, cycle TP
        (4, 5): i2Ri1,  # model TP, cycle FN
    }

    # only predicted positives go into the initial graph, and are eligible for cycle estimation.
    # 1 indicates a match
    two_view_reports_dict = {
        (0, 1): TwoViewEstimationReport(gt_class=1),  # model TP, cycle TP
        (1, 2): TwoViewEstimationReport(gt_class=1),  # model TP, cycle TP
        (2, 3): TwoViewEstimationReport(gt_class=0),  # model FP, cycle TN
        (3, 4): TwoViewEstimationReport(gt_class=0),  # model FP, cycle TP
        (4, 5): TwoViewEstimationReport(gt_class=1),  # model TP, cycle FN
    }

    # Note: (4,5) was erroneously removed by the cycle-based filtering, and (2,3) was correctly filtered out.
    i2Ri1_dict_consistent = {
        (0, 1): i2Ri1,
        (1, 2): i2Ri1,
        (3, 4): i2Ri1,
    }
    prec, rec, mAcc = cycle_consistency.estimate_rot_cycle_filtering_classification_acc(
        i2Ri1_dict, i2Ri1_dict_consistent, two_view_reports_dict
    )
    # 1 / 2 false positives are discarded -> 0.5 for class 0
    # 2 / 3 true positives were kept -> 0.67 for class 1
    expected_mAcc = np.mean(np.array([1 / 2, 2 / 3]))
    assert np.isclose(mAcc, expected_mAcc, atol=1e-4)


def test_filter_to_translation_cycle_consistent_edges() -> None:
    """Ensures that cycle-consistency based filtering by translation measurements alone is correct.

    GT pose graph:

          | pano 1 = (0,4)
        --o .
          | .    .
          .    .     .
          .       .       .
          |         |         . |
          o--  ...  o--  ...    o---
    pano 0     pano 2 = (4,0)  pano 3 = (8,0)
      (0,0)
    """
    wTi_list = [
        Pose2(Rot2.fromDegrees(0), np.array([0, 0])),
        Pose2(Rot2.fromDegrees(90), np.array([0, 4])),
        Pose2(Rot2.fromDegrees(0), np.array([4, 0])),
        Pose2(Rot2.fromDegrees(0), np.array([8, 0])),
    ]
    wRi_list = [wTi.rotation().matrix() for wTi in wTi_list]

    i2Si1_dict = {
        (0, 1): Sim2(R=rotmat2d(-90), t=np.array([-4.6, 0]), s=1.0),  # add 0.6 noise
        (1, 2): Sim2(R=rotmat2d(90), t=np.array([-4, 4]), s=1.0),
        (0, 2): Sim2(R=np.eye(2), t=np.array([-4, 0]), s=1.0),
        (1, 3): Sim2(R=-rotmat2d(90), t=np.array([-8.2, 4]), s=1.0),  # add 0.2 noise
        (0, 3): Sim2(R=np.eye(2), t=np.array([-8, -0.2]), s=1.0),  # add 0.2 noise
    }

    # with a 0.5 meter thresh
    i2Si1_cycle_consistent = cycle_consistency.filter_to_translation_cycle_consistent_edges(
        wRi_list, i2Si1_dict, translation_cycle_thresh=0.5
    )

    # one triplet was too noisy, and is filtered out.
    assert set(list(i2Si1_cycle_consistent.keys())) == {(0, 1), (0, 3), (1, 3)}
