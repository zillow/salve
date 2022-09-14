"""Unit tests on colored ICP utilities."""

import copy
import os
from pathlib import Path

import numpy as np
import open3d
from gtsam import Pose3, Rot3

import salve.baselines.open3d_icp as open3d_icp


TEST_DATA_ROOT = Path(__file__).resolve().parent / "test_data"


# def test_register_point_cloud() -> None:
#     """
#     test data from https://github.com/isl-org/Open3D/tree/master/examples/test_data/ColoredICP
#     """

#     # Load two point clouds and show initial pose
#     source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
#     target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

#     # draw initial alignment
#     tTs = np.identity(4)
#     # draw_registration_result_original_color(source, target, tTs)

#     tTs = open3d_icp.register_point_clouds(source, target)
#     # draw_registration_result_original_color(source, target, tTs)

#     assert isinstance(tTs, np.ndarray)
#     assert tTs.shape == (4, 4)


# def test_register_colored_point_clouds() -> None:
#     """ """
#     source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
#     target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

#     tTs = open3d_icp.register_colored_point_clouds(source, target)
#     # draw_registration_result_original_color(source, target, tTs)
#     assert isinstance(tTs, np.ndarray)
#     assert tTs.shape == (4, 4)


def test_register_colored_point_clouds() -> None:
    """ """
    # # fmt: off
    # source_xyz = np.array(
    #     [
    #         [-4,5,0],
    #         [-3,5,0],
    #         [-2,5,0],
    #         [-3,4,0],
    #         [-3,3,0],
    #         [-3,2,0]
    #     ]
    # ).astype(np.float32)
    # source_rgb = np.array(
    #     [
    #         [0,0,0],
    #         [1,1,1],
    #         [2,2,2],
    #         [3,3,3],
    #         [4,4,4],
    #         [255,255,255]
    #     ]
    # )
    # target_xyz = np.array(
    #     [
    #         [4,-4, 4],
    #         [3,-4, 4],
    #         [2,-4, 4],
    #         [3,-3, 4],
    #         [3,-2, 4],
    #         [3,-1, 4]
    #     ]
    # ).astype(np.float32)
    # target_rgb = np.array(
    #     [
    #         [0,0,0],
    #         [1,1,1],
    #         [2,2,2],
    #         [3,3,3],
    #         [4,4,4],
    #         [255,255,255]
    #     ]
    # )
    # xyzrgb_source = np.hstack([source_xyz, source_rgb])
    # xyzrgb_target = np.hstack([target_xyz, target_rgb])
    # # fmt: on

    # source = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb_source)
    # target = open3d_icp.xyzrgb_to_open3d_point_cloud(xyzrgb_target)

    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))

    R = Rot3.RzRyRx(x=0, y=0, z=np.pi/10)
    t = np.array([0.1, 0.2, 0.3])
    tTs = Pose3(R, t)

    source = copy.deepcopy(target)
    source_xyz = np.asarray(source.points)
    sTt = tTs.inverse()
    # fill up source with the real source points now.
    source_xyz = np.array([sTt.transformFrom(pt) for pt in source_xyz])
    source.points = open3d.utility.Vector3dVector(source_xyz)

    tTs = np.eye(4)
    draw_registration_result_original_color(source, target, tTs)

    tTs = open3d_icp.register_colored_point_clouds(source, target)
    draw_registration_result_original_color(source, target, tTs)

    #import pdb; pdb.set_trace()
    

def draw_registration_result_original_color(
    source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud, transformation: np.ndarray
) -> None:
    """ """
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries(
        [source_temp, target],
        zoom=0.5,
        front=[-0.2458, -0.8088, 0.5342],
        lookat=[1.7745, 2.2305, 0.9787],
        up=[0.3109, -0.5878, -0.7468],
    )


def test_transform() -> None:
    """ """
    # fmt: off
    points_src = np.array(
        [
            [3.,0,0],
            [0,3.,0]
        ]
    )
    # fmt: on
    R = Rot3.RzRyRx(x=np.pi / 2, y=0, z=0)
    t = np.array([1, 2, 3])
    tTs = Pose3(R, t)

    point_t0 = tTs.transformFrom(points_src[0])
    point_t1 = tTs.transformFrom(points_src[1])

    pcd_s = open3d.geometry.PointCloud()
    pcd_s.points = open3d.utility.Vector3dVector(points_src)

    # identical to tTs.transformFrom(pcd_s)
    pcd_t = pcd_s.transform(tTs.matrix())

    # [4,2,3]
    # [1,2,6]
    assert np.allclose(np.asarray(pcd_t.points)[0], point_t0)
    assert np.allclose(np.asarray(pcd_t.points)[1], point_t1)



if __name__ == "__main__":
    test_transform()
    

