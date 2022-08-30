"""Unit tests for Spanning Tree creation utilities."""

import numpy as np
from gtsam import Rot2, Pose2

import salve.algorithms.spanning_tree as spanning_tree
from salve.common.sim2 import Sim2
import salve.utils.rotation_utils as rotation_utils
from salve.algorithms.rotation_averaging import globalaveraging2d
from salve.algorithms.spanning_tree import RelativePoseMeasurement


def test_greedily_construct_st():
    """
    Below, we specify i1 < i2 for all edges (i1,i2)

    Graph topology:

              | 2     | 3
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    0         1       4

    """
    # ground truth 2d rotations
    wRi_list_gt = [
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(90),
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(90)
        # Rot3(), # 0
        # Rot3.Rz(np.deg2rad(90)), # 1
        # Rot3(), # 2
        # Rot3(), # 3
        # Rot3.Rz(np.deg2rad(90))# 4
    ]

    i2Ri1_dict = {}
    for i1 in range(4):
        i2 = i1 + 1
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.T @ wRi1

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        print(f"EDGE_SE2 {i1} {i2} 0 0 {rotation_utils.rotmat2theta_deg(i2Ri1)}")

    wRi_list_greedy = spanning_tree.greedily_construct_st(i2Ri1_dict)

    # expected angles
    wRi_list_euler_deg_exp = [
        0,
        90,
        0,
        0,
        90,
    ]
    # wRi_list_euler_deg_est = [ np.rad2deg(wRi.xyz()).tolist() for wRi in wRi_list_greedy]
    wRi_list_euler_deg_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est

    wRi_list_shonan = globalaveraging2d(i2Ri1_dict)

    wRi_list_shonan_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # Note that:
    # 360 - 125.812 =  234.188
    # 234.188 - 144.188 = 90.0
    wRi_list_shonan_exp = [-125.81, 144.18, -125.81, -125.81, 144.18]
    assert np.allclose(wRi_list_shonan_exp, wRi_list_shonan_est, atol=0.01)

    # # cast to a 2d problem
    # wRi_list_Rot3_shonan = global_averaging(i2Ri1_dict)
    # wRi_list_shonan = posegraph3d_to_posegraph2d(wRi_list_Rot3_shonan)

    # wRi_list_shonan_est = [ rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_shonan]

    # # corresponds to 110.5 and 200.4 degrees (as if 0 and 90 degrees)
    # wRi_list_shonan_exp = [110.52, -159.61, 110.52, 110.52, -159.61]
    # assert np.allclose(wRi_list_shonan_exp, wRi_list_shonan_est, atol=0.01)


def test_greedily_construct_st2():
    """
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
    # ground truth 2d rotations
    wRi_list_gt = [
        rotation_utils.rotmat2d(0),  # 0
        rotation_utils.rotmat2d(90),  # 1
        rotation_utils.rotmat2d(90),  # 2
        rotation_utils.rotmat2d(0),
        rotation_utils.rotmat2d(0)
        # Rot3(), # 0
        # Rot3.Rz(np.deg2rad(90)), # 1
        # Rot3(), # 2
        # Rot3(), # 3
        # Rot3.Rz(np.deg2rad(90))# 4
    ]

    edges = [(1, 4), (1, 3), (0, 3), (0, 2)]

    i2Ri1_dict = {}
    for (i1, i2) in edges:
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.T @ wRi1

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        print(f"EDGE_SE2 {i1} {i2} 0 0 {rotation_utils.rotmat2theta_deg(i2Ri1)}")

    import pdb

    pdb.set_trace()
    wRi_list_greedy = spanning_tree.greedily_construct_st(i2Ri1_dict)

    # expected angles
    wRi_list_euler_deg_exp = [
        0,
        90,
        90,
        0,
        0,
    ]
    # wRi_list_euler_deg_est = [ np.rad2deg(wRi.xyz()).tolist() for wRi in wRi_list_greedy]
    wRi_list_euler_deg_est = [rotation_utils.rotmat2theta_deg(wRi) for wRi in wRi_list_greedy]
    assert wRi_list_euler_deg_exp == wRi_list_euler_deg_est



def test_ransac_spanning_trees() -> None:
    """Toy scenario with 3 nodes (3 accurate edges, and 1 noisy edge)."""

    np.random.seed(0)

    wT0 = Pose2(Rot2(), np.array([0, 0]))
    wT1 = Pose2(Rot2(), np.array([2, 0]))
    wT2 = Pose2(Rot2(), np.array([2, 2]))

    wT2_noisy = Pose2(Rot2(), np.array([3, 3]))

    i1Ti0 = wT1.between(wT0)
    i2Ti1 = wT2.between(wT1)
    i2Ti0 = wT2.between(wT0)

    i2Ti0_noisy = wT2_noisy.between(wT0)

    i1Si0 = convert_Pose2_to_Sim2(i1Ti0)
    i2Si1 = convert_Pose2_to_Sim2(i2Ti1)
    i2Si0 = convert_Pose2_to_Sim2(i2Ti0)
    i2Si0_noisy = convert_Pose2_to_Sim2(i2Ti0_noisy)

    # fmt: off
    high_conf_measurements = [
        RelativePoseMeasurement(i1=0, i2=1, i2Si1=i1Si0),
        RelativePoseMeasurement(i1=1, i2=2, i2Si1=i2Si1),
        RelativePoseMeasurement(i1=0, i2=2, i2Si1=i2Si0),
        RelativePoseMeasurement(i1=0, i2=2, i2Si1=i2Si0_noisy)
    ]
    # fmt: on
    wSi_list = spanning_tree.ransac_spanning_trees(high_conf_measurements, num_hypotheses=10, min_num_edges_for_hypothesis=3)

    assert len(wSi_list) == 3
    assert wSi_list[0] == convert_Pose2_to_Sim2(wT0)
    assert wSi_list[1] == convert_Pose2_to_Sim2(wT1)
    assert wSi_list[2] == convert_Pose2_to_Sim2(wT2)


def convert_Pose2_to_Sim2(i2Ti1: Pose2) -> Sim2:
    """ """
    return Sim2(R=i2Ti1.rotation().matrix(), t=i2Ti1.translation(), s=1.0)





if __name__ == "__main__":
    test_ransac_spanning_trees()



