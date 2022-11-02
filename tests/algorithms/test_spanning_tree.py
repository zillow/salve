"""Unit tests for Spanning Tree creation utilities."""

from typing import Dict, List, Tuple

import numpy as np
from gtsam import Rot2, Pose2

import salve.algorithms.spanning_tree as spanning_tree
import salve.utils.rotation_utils as rotation_utils
from salve.algorithms.rotation_averaging import globalaveraging2d
from salve.common.edge_classification import EdgeClassification
from salve.common.sim2 import Sim2


RELATIVE_ROTATION_DICT = Dict[Tuple[int, int], np.ndarray]


def _get_ordered_chain_pose_data() -> Tuple[RELATIVE_ROTATION_DICT, List[float]]:
    """Return data for a scenario with 5 camera poses, with ordering that follows their connectivity.

    Accordingly, we specify i1 < i2 for all edges (i1,i2).

    Graph topology:

              | 2     | 3
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    0         1       4

    Returns:
        Tuple of mapping from image index pair to relative rotations, and expected global rotation angles.
    """
    # Ground truth 2d rotations for 5 ordered poses (0,1,2,3,4)
    wRi_list_gt = [
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(90),
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(90),
    ]

    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    i2Ri1_dict = _create_synthetic_relative_pose_measurements(wRi_list_gt, edges=edges)

    # expected angles
    wRi_list_euler_deg_expected = [
        0,
        90,
        0,
        0,
        90,
    ]
    return i2Ri1_dict, wRi_list_euler_deg_expected


def _get_mixed_order_chain_pose_data() -> Tuple[RELATIVE_ROTATION_DICT, List[float]]:
    """Return data for a scenario with 5 camera poses, with ordering that does NOT follow their connectivity.

    Below, we do NOT specify i1 < i2 for all edges (i1,i2).

    Graph topology:

              | 3     | 0
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    4         1       2

    """
    # Ground truth 2d rotations for 5 ordered poses (0,1,2,3,4).
    wRi_list_gt = [
        rotation_utils.rotmat2d(0),  # 0
        rotation_utils.rotmat2d(90),  # 1
        rotation_utils.rotmat2d(90),  # 2
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(0),
    ]

    edges = [(1, 4), (1, 3), (0, 3), (0, 2)]
    i2Ri1_dict = _create_synthetic_relative_pose_measurements(wRi_list_gt=wRi_list_gt, edges=edges)

    # expected angles
    wRi_list_euler_deg_expected = [
        0,
        90,
        90,
        0,
        0,
    ]
    return i2Ri1_dict, wRi_list_euler_deg_expected


def _create_synthetic_relative_pose_measurements(
    wRi_list_gt: List[np.ndarray], edges: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], np.ndarray]:
    """Generate synthetic relative rotation measurements, from ground truth global rotations.

    Args:
        wRi_list_gt: List of (2,2) rotation matrices.
        edges: edges as pairs of image indices.

    Returns:
        i2Ri1_dict: Relative rotation measurements.
    """
    i2Ri1_dict = {}
    for (i1, i2) in edges:
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.T @ wRi1

    return i2Ri1_dict


def test_greedily_construct_st_ordered_chain() -> None:
    """Ensures that we can greedily construct a Spanning Tree for an ordered chain."""

    i2Ri1_dict, wRi_list_euler_deg_expected = _get_ordered_chain_pose_data()

    wRi_list_greedy = spanning_tree.greedily_construct_st(i2Ri1_dict)

    wRi_list_euler_deg_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert np.allclose(wRi_list_euler_deg_est, wRi_list_euler_deg_expected)


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Map angle (in degrees) from domain [-π, π] to [0, 360).

    An alternative method can be found here:
    https://github.com/argoai/argoverse-api/blob/469bb466a8df04d48cc1c1c840c4695fd3d97e74/argoverse/evaluation/detection/utils.py#L465

    Args:
        angles: array of shape (N,) representing angles (in degrees) in any interval.

    Returns:
        Array of shape (N,) representing the angles (in degrees) mapped to the interval [0, 360].
    """
    period = 360
    phase_shift = 180
    return (angles + 180) % period - phase_shift


def test_globalaveraging2d_ordered_chain() -> None:
    """Ensures that we can perform 2d global rotation averaging, when pano ordering follows connectivity."""

    i2Ri1_dict, wRi_list_euler_deg_expected = _get_ordered_chain_pose_data()

    wRi_list_shonan = globalaveraging2d(i2Ri1_dict)
    wRi_list_shonan_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # Set first angle equal to 0 degrees.
    wRi_list_shonan_est_shifted = np.array(wRi_list_shonan_est) - np.array(wRi_list_shonan_est)[0]

    # Wrap angles to [0,360) degrees.
    wRi_list_shonan_est_wrapped = _wrap_angles(wRi_list_shonan_est_shifted)

    # The angles below should be equivalent to the spanning tree result above.
    wRi_list_shonan_expected = [0, 90, 0, 0, 90]
    assert np.allclose(wRi_list_shonan_est_wrapped, wRi_list_shonan_expected, atol=0.01)


def test_greedily_construct_st_mixed_order_chain() -> None:
    """Ensures that we can greedily construct a Spanning Tree for an ordered chain."""
    i2Ri1_dict, wRi_list_euler_deg_expected = _get_mixed_order_chain_pose_data()

    wRi_list_greedy = spanning_tree.greedily_construct_st(i2Ri1_dict)

    wRi_list_euler_deg_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert np.allclose(wRi_list_euler_deg_est, wRi_list_euler_deg_expected)


def test_ransac_spanning_trees() -> None:
    """Test that we can fit spanning trees in a RANSAC loop to eliminate noisy measurements.

    We use a toy scenario with 3 nodes (3 accurate edges, and 1 noisy edge).
    """
    np.random.seed(0)

    # Define ground truth global poses for 3 panoramas.
    wT0 = Pose2(Rot2(), np.array([0, 0]))
    wT1 = Pose2(Rot2(), np.array([2, 0]))
    wT2 = Pose2(Rot2(), np.array([2, 2]))

    # Define one noisy global pose.
    wT2_noisy = Pose2(Rot2(), np.array([3, 3]))

    # Generate synthetic relative poses.
    i1Ti0 = wT1.between(wT0)
    i2Ti1 = wT2.between(wT1)
    i2Ti0 = wT2.between(wT0)

    i2Ti0_noisy = wT2_noisy.between(wT0)

    i1Si0 = _convert_Pose2_to_Sim2(i1Ti0)
    i2Si1 = _convert_Pose2_to_Sim2(i2Ti1)
    i2Si0 = _convert_Pose2_to_Sim2(i2Ti0)
    i2Si0_noisy = _convert_Pose2_to_Sim2(i2Ti0_noisy)

    def _construct_edge_classification(i1: int, i2: int, i2Si1_: Sim2) -> EdgeClassification:
        # Assign dummy attributes for rest of values
        prob = 0.99
        y_hat = 1
        y_true = 1
        pair_idx = 9999
        wdo_pair_uuid = ""
        configuration = "identity"
        building_id = "9999"
        floor_id = "floor_00"
        return EdgeClassification(
            i1=i1,
            i2=i2,
            i2Si1=i2Si1_,
            prob=prob,
            y_hat=y_hat,
            y_true=y_true,
            pair_idx=pair_idx,
            wdo_pair_uuid=wdo_pair_uuid,
            configuration=configuration,
            building_id=building_id,
            floor_id=floor_id,
        )

    # fmt: off
    high_conf_measurements = [
        _construct_edge_classification(i1=0, i2=1, i2Si1_=i1Si0), # prob, y_hat, y_true, pair_idx, wdo_pair_uuid, configuration, buiiling_id, floor_id
        _construct_edge_classification(i1=1, i2=2, i2Si1_=i2Si1),
        _construct_edge_classification(i1=0, i2=2, i2Si1_=i2Si0),
        _construct_edge_classification(i1=0, i2=2, i2Si1_=i2Si0_noisy)
    ]
    # fmt: on

    # RANSAC module for Spanning Tree fitting produces Sim(2) objects, so conversion will be required for comparison.
    wSi_list = spanning_tree.ransac_spanning_trees(
        high_conf_measurements, num_hypotheses=10, min_num_edges_for_hypothesis=3, visualize=False
    )

    # 3 global poses should be clean, accurate versions (noisy variant should have disappeared).
    assert len(wSi_list) == 3
    assert wSi_list[0] == _convert_Pose2_to_Sim2(wT0)
    assert wSi_list[1] == _convert_Pose2_to_Sim2(wT1)
    assert wSi_list[2] == _convert_Pose2_to_Sim2(wT2)


def _convert_Pose2_to_Sim2(i2Ti1: Pose2) -> Sim2:
    """Convert a Pose(2) object to a Sim(2) object, fixing scale to 1."""
    return Sim2(R=i2Ti1.rotation().matrix(), t=i2Ti1.translation(), s=1.0)


def test_compute_objective_function_improvement() -> None:
    """ """

    dict_best =     {"rot_mean": 16.4,  "rot_med": 1.4, "trans_mean" : 0.44, "trans_med": 0.11, "#poses": 38}
    dict_proposed = {"rot_mean": 13.0,  "rot_med": 1.7, "trans_mean" : 0.58, "trans_med": 0.14, "#poses": 104}


    # dict_best =         {"rot_mean": 10.9,  "rot_med": 1.2, "trans_mean" : 0.23, "trans_med": 0.08, "#poses": 38}
    # dict_proposed =     {"rot_mean": 3.4,   "rot_med": 1.7, "trans_mean" : 0.18, "trans_med": 0.14, "#poses": 66}
    # # win of 2.53

    # dict_best =     {"rot_mean": 3.4,   "rot_med": 1.7, "trans_mean" : 0.18, "trans_med": 0.14, "#poses": 66}
    # dict_proposed = {"rot_mean":  3.1,  "rot_med": 1.7, "trans_mean" : 0.17, "trans_med": 0.10, "#poses": 66}


    win_ratio = spanning_tree.compute_objective_function_improvement(
        avg_rot_error=dict_proposed["rot_mean"],
        avg_rot_error_best=dict_best["rot_mean"],
        avg_trans_error=dict_proposed["trans_mean"],
        avg_trans_error_best=dict_best["trans_mean"],
        num_poses_estimated=dict_proposed["#poses"],
        num_poses_estimated_best=dict_best["#poses"]
    )
    print("Win ratio: ", win_ratio)


if __name__ == "__main__":

    test_compute_objective_function_improvement()






