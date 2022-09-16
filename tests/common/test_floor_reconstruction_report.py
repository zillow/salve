"""Ensures that pose graph evaluation is correct."""

import tempfile
from pathlib import Path

import numpy as np

import salve.common.posegraph2d as posegraph2d
import salve.common.floor_reconstruction_report as floor_reconstruction_report
from salve.common.floor_reconstruction_report import FloorReconstructionReport
from salve.common.posegraph2d import PoseGraph2d
from salve.common.sim2 import Sim2


_ZIND_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "test_data" / "ZInD"


def test_from_est_floor_pose_graph() -> None:
    """Ensure we can correctly create a FloorReconstructionReport from pose graph optimization output.

    Data taken from Building 1210, Floor 02 of ZinD.
    """
    np.random.seed(0)

    # 0-indexed list of estimated poses for floor 2.
    wSi_list = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Sim2(
            R=np.array([[1.0000000e00, 1.4511669e-13], [-1.4511669e-13, 1.0000000e00]], dtype=np.float32),
            t=np.array([3.1663807e-13, 4.0534674e-13], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9963625, 0.08521605], [-0.08521605, -0.9963625]], dtype=np.float32),
            t=np.array([-0.05208764, -0.657844], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.8538526, 0.5205148], [-0.5205148, -0.8538526]], dtype=np.float32),
            t=np.array([0.77260476, -1.6241723], dtype=np.float32),
            s=1.0,
        ),
        None,
        Sim2(
            R=np.array([[0.007844, -0.99996924], [0.99996924, 0.007844]], dtype=np.float32),
            t=np.array([-0.743632, 0.03829836], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.8644665, -0.50269043], [0.50269043, -0.8644665]], dtype=np.float32),
            t=np.array([-1.3128754, -0.0555355], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9977786, -0.06661703], [0.06661703, -0.9977786]], dtype=np.float32),
            t=np.array([-2.2001665, -1.263223], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9995646, -0.02950616], [0.02950616, -0.9995646]], dtype=np.float32),
            t=np.array([-0.79566294, -0.76166594], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.00257046, -0.9999967], [0.9999967, -0.00257046]], dtype=np.float32),
            t=np.array([-0.6911983, 0.80846286], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[0.00632679, -0.99998], [0.99998, 0.00632679]], dtype=np.float32),
            t=np.array([-1.3925239, 0.91490793], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.01266379, -0.99991983], [0.99991983, -0.01266379]], dtype=np.float32),
            t=np.array([-2.4355152, 1.7160583], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.01020425, -0.9999479], [0.9999479, -0.01020425]], dtype=np.float32),
            t=np.array([-2.3332891, 0.30607823], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.10058811, 0.9949282], [-0.9949282, -0.10058811]], dtype=np.float32),
            t=np.array([-1.3064604, 2.2962294], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[0.02900542, 0.99957925], [-0.99957925, 0.02900542]], dtype=np.float32),
            t=np.array([-0.8010526, 2.38649], dtype=np.float32),
            s=1.0,
        ),
        None,
        None,
        None,
        None,
        None,
    ]

    raw_dataset_dir = _ZIND_SAMPLE_ROOT
    building_id = "1210"
    floor_id = "floor_02"
    gt_floor_pg = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    plot_save_dir = str(tempfile.TemporaryDirectory())
    est_floor_pose_graph = PoseGraph2d.from_wSi_list(wSi_list, gt_floor_pg)
    report = FloorReconstructionReport.from_est_floor_pose_graph(
        est_floor_pose_graph, gt_floor_pg, plot_save_dir=plot_save_dir
    )

    assert np.isclose(report.avg_abs_rot_err, 1.37, atol=1e-2)
    assert np.isclose(report.avg_abs_trans_err, 0.19, atol=2e-2)
    # 13/19 localized above.
    assert np.isclose(report.percent_panos_localized, (13 / 19) * 100, atol=1e-2)


def test_compute_translation_errors_against_threshold() -> None:
    """Ensure that translation localization success rate is computed correctly."""
    reconstruction_reports = [
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([0.0, 0.1, 0.19, 0.3, 0.4, 900]),  # 3/6 are under threshold.
        ),
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([0.0, 0.1, 0.18, 0.19, 0.21]),  # 4/5 are under threshold
        ),
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([800, 900, 1000]),  # 0/3 are under threshold.
        ),
    ]
    threshold = 0.2
    avg_success_rate = floor_reconstruction_report.compute_translation_errors_against_threshold(
        reconstruction_reports, threshold
    )
    expected_avg_success_rate = np.mean([3 / 6, 4 / 5, 0 / 3])
    assert np.isclose(avg_success_rate, expected_avg_success_rate)
