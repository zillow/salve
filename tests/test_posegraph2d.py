
"""Ensure that 2d pose graph evaluation is correct."""


import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Pose3, Rot3, Similarity3

import salve.common.posegraph2d as posegraph2d
import salve.utils.rotation_utils as rotation_utils
from salve.common.posegraph2d import PoseGraph2d


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

    """
    #bTi_list_est[16]
    bTi = Pose3(
        Rot3(
            np.array(
                [
                    [1, 1.45117e-13, 0],
                    [-1.45117e-13, 1, 0],
                    [0, 0, 1]
                ]
            )
        ),
        np.array([ 3.16638e-13, 4.05347e-13,           0 ])
    )
    """

    a_Sim2_b = Sim2(
        R=expected_aRb,
        t=expected_atb,
        s=expected_scale
    )
    b_Sim2_i = Sim2(
        R=np.array(
            [
                [1, 1.45117e-13],
                [-1.45117e-13, 1]
            ]
        ),
        t=np.array([ 3.16638e-13, 4.05347e-13]),
        s=1.0
    )
    a_Sim2_i = a_Sim2_b.compose(b_Sim2_i)
    import pdb; pdb.set_trace()


def test_Sim2_compose():
    """ """


"""

(Pdb) p bTi_list_est[16]
R: [
	1, 1.45117e-13, 0;
	-1.45117e-13, 1, 0;
	0, 0, 1
]
t: 3.16638e-13 4.05347e-13           0



(Pdb) p aligned_bTi_list_est[16]
R: [
	0.999997, 0.00256117, 0;
	-0.00256117, 0.999997, 0;
	0, 0, 1
]
t:   0.0246006 -0.00184358           0





(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.rotation
array([[ 0.9999967 ,  0.00256117],
       [-0.00256117,  0.9999967 ]], dtype=float32)
(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.translation
array([ 0.02309136, -0.00173048], dtype=float32)
(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.scale
1.0

"""



def test_measure_avg_rel_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

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

    wRi_list = [rotation_utils.rotmat2d(-5), rotation_utils.rotmat2d(-95), rotation_utils.rotmat2d(0)]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(-90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0, 1)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    # both are incorrect by the same amount, cancelling out to zero error
    assert mean_rel_rot_err == 0

    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    assert np.isclose(mean_rel_rot_err, 10 / 3, atol=1e-3)


def test_measure_avg_rel_rotation_err_unestimated() -> None:
    """Estimate average relative pose (rotation) error when some nodes are unestimated.

    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

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

    # only 1 edge can be measured for correctness
    wRi_list = [rotation_utils.rotmat2d(-5), rotation_utils.rotmat2d(-90), None]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(-90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    assert mean_rel_rot_err == 5.0


def test_measure_avg_abs_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure absolute rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

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

    wRi_list = [rotation_utils.rotmat2d(-5), rotation_utils.rotmat2d(-95), rotation_utils.rotmat2d(0)]

    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(-90), rotation_utils.rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(mean_abs_rot_err, 10 / 3, atol=1e-3)


def test_measure_abs_pose_error_shifted() -> None:
    """Pose graph is shifted to the left by 1 meter, but Sim(3) alignment should fix this. Should have zero error.

    TODO: fix rotations to be +90

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

    wRi_list = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(-90), rotation_utils.rotmat2d(0)]
    wti_list = [np.array([-1, 0]), np.array([-1, 4]), np.array([3, 0])]

    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, building_id, floor_id)

    wRi_list_gt = [rotation_utils.rotmat2d(0), rotation_utils.rotmat2d(-90), rotation_utils.rotmat2d(0)]
    wti_list_gt = [np.array([0, 0]), np.array([0, 4]), np.array([4, 0])]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list_gt, wti_list, building_id, floor_id)

    avg_rot_error, avg_trans_error = est_floor_pose_graph.measure_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(avg_rot_error, 0.0, atol=1e-3)
    assert np.isclose(avg_trans_error, 0.0, atol=1e-3)



if __name__ == "__main__":
    test_convert_Sim3_to_Sim2()

    # test_measure_avg_rel_rotation_err()
    # test_measure_avg_abs_rotation_err()
    # test_measure_avg_rel_rotation_err_unestimated()
    # test_measure_abs_pose_error()

    # test_measure_abs_pose_error_shifted()
