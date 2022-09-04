"""Unit tests for 2d pose graph utilities (including accuracy evaluation)."""

from unittest.mock import MagicMock

import numpy as np
from gtsam import Pose3, Rot3, Similarity3

import salve.common.posegraph2d as posegraph2d
import salve.utils.rotation_utils as rotation_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.common.sim2 import Sim2


def test_convert_Sim3_to_Sim2() -> None:
    """Ensure that Similarity(3) to Similarity(2) conversion works by projection (x,y,z) to (x,y)."""
    a_Sim3_b = Similarity3(
        R=Rot3(np.array([[0.999997, 0.00256117, 0], [-0.00256117, 0.999997, 0], [0, 0, 1]])),
        t=np.array([0.02309136, -0.00173048, 0.0]),
        s=1.0653604360576439,
    )

    a_Sim2_b = posegraph2d.convert_Sim3_to_Sim2(a_Sim3_b)

    expected_aRb = np.array([[0.999997, 0.00256117], [-0.00256117, 0.999997]], dtype=np.float32)
    expected_atb = np.array([0.02309136, -0.00173048], dtype=np.float32)
    expected_scale = 1.0653604360576439
    assert np.allclose(a_Sim2_b.rotation, expected_aRb)
    assert np.allclose(a_Sim2_b.translation, expected_atb)
    assert np.isclose(a_Sim2_b.scale, expected_scale)


def test_measure_avg_rel_rotation_err() -> None:
    """Ensures that average relative pose (rotation) error is estimated when all nodes' poses are estimated.

    GT rotation graph: (dummy scenario)

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotation_utils.rotmat2d(5), rotation_utils.rotmat2d(95), rotation_utils.rotmat2d(0)]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    # First, consider 1-edge only case.
    gt_edges = [(0, 1)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    # Pano 0 and Pano 1 have absolute angles that are shifted (incorrectly) by the same amount, cancelling out to zero error
    assert mean_rel_rot_err == 0

    # Now, consider 3-edge case.
    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    # 10 degrees of total error spread across 3 edges.
    # 5 degrees of error come from est. edge (0,2)'s 5 degree angle, while GT dictates edge (0,2) has 0 deg. angle.
    # Other 5 degrees of error come from edge (1,2) estimate vs. GT -- 95 vs. 90 degrees.
    assert np.isclose(mean_rel_rot_err, 10 / 3, atol=1e-3)


def test_measure_avg_rel_rotation_err_unestimated() -> None:
    """Ensures that average relative pose (rotation) error is estimated when some nodes are unestimated.

    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

    GT rotation graph: (dummy scenario)

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    # Because pano 2's pose is unestimated, only 1 edge here can be measured for correctness -- the edge (0,1).
    wRi_list = [rotation_utils.rotmat2d(105), rotation_utils.rotmat2d(190), None]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    assert np.isclose(mean_rel_rot_err, 5.0, atol=1e-5)


def test_measure_avg_abs_rotation_err() -> None:
    """Ensures that **absolute** rotation errors are evaluated correctly.

    GT rotation graph: (dummy scenario)

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotation_utils.rotmat2d(105), rotation_utils.rotmat2d(195), rotation_utils.rotmat2d(100)]

    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)
    
    # Large absolute differences become tiny relative differences after alignment (1.7, 1.7, and -3.3 degree errors).
    assert np.isclose(mean_abs_rot_err, 2.222, atol=1e-3)


def test_measure_abs_pose_error_shifted() -> None:
    """Ensures that error is zero between two 2d pose graphs identical besides a (-1,0) translation shift.
    
    Pose graph is shifted to the left by 1 meter, but Sim(3) alignment resolves this, yielding zero error.

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

    Estimated PG:
       | pano 1 = (-1,4)
     --o
       | .
       .   .
       .     .
       |       |
       o-- ... o--
    pano 0          pano 2 = (3,0)
      (-1,0)
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(90), rotation_utils.rotmat2d(0)]
    wti_list = [np.array([-1, 0]), np.array([-1, 4]), np.array([3, 0])]

    gt_floor_pg = MagicMock()
    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pg=gt_floor_pg)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(90), rotation_utils.rotmat2d(0)]
    wti_list_gt = [np.array([0, 0]), np.array([0, 4]), np.array([4, 0])]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list_gt, wti_list, gt_floor_pg=gt_floor_pg)

    avg_rot_error, avg_trans_error, _, _ = est_floor_pose_graph.measure_unaligned_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(avg_rot_error, 0.0, atol=1e-3)
    assert np.isclose(avg_trans_error, 0.0, atol=1e-3)
