"""Unit tests on ICP and colored ICP utilities."""

import copy
import os
from pathlib import Path

import numpy as np
import open3d
from gtsam import Pose3, Rot3

import salve.baselines.open3d_icp as open3d_icp


TEST_DATA_ROOT = Path(__file__).resolve().parent / "test_data"


def test_register_point_cloud() -> None:
    """Test that vanilla, non-colored ICP successfully registers two separate overlapping point clouds.

    Test data from https://github.com/isl-org/Open3D/tree/master/examples/test_data/ColoredICP
    """
    # Load two point clouds and show initial pose.
    source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

    # Draw initial alignment.
    tTs = np.identity(4)
    # _draw_registration_result_original_color(source, target, tTs)

    tTs = open3d_icp.register_point_clouds(source, target)
    # _draw_registration_result_original_color(source, target, tTs)

    assert isinstance(tTs, np.ndarray)
    assert tTs.shape == (4, 4)


def test_register_colored_point_clouds() -> None:
    """Test that colored ICP registration works on point clouds representing walls with triangles.

    Two separate, overlapping point clouds are used. Result should be better than vanilla ICP above.
    """
    source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

    # open3d.visualization.draw_geometries([source])
    # open3d.visualization.draw_geometries([target])

    tTs = open3d_icp.register_colored_point_clouds(source, target)
    # _draw_registration_result_original_color(source, target, tTs)
    assert isinstance(tTs, np.ndarray)
    assert tTs.shape == (4, 4)


def test_register_colored_point_clouds_jitter_wall_point_cloud() -> None:
    """Test that colored ICP registration works on point clouds representing walls with triangles.

    Second point cloud is obtained by jittering the first.
    """
    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))

    # Obtain second point cloud by jittering the first.
    # Rotate original point cloud by 18 degrees, and then translate in each dimension.
    R = Rot3.RzRyRx(x=0, y=0, z=np.pi/10)
    t = np.array([0.1, 0.2, 0.3])
    tTs = Pose3(R, t)

    source = copy.deepcopy(target)
    source_xyz = np.asarray(source.points)
    sTt = tTs.inverse()

    # Generate perturbed version of `target` point cloud.
    source_xyz = np.array([sTt.transformFrom(pt) for pt in np.asarray(target.points)])
    source.points = open3d.utility.Vector3dVector(source_xyz)

    tTs = np.eye(4)
    # _draw_registration_result_original_color(source, target, tTs=tTs)

    tTs_ = open3d_icp.register_colored_point_clouds(source, target)
    # _draw_registration_result_original_color(source, target, tTs=tTs_)
    

def _draw_registration_result_original_color(
    source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud, tTs: np.ndarray
) -> None:
    """Visualize with Open3d transformed source point cloud to target frame (according to some transformation).

    Args:
        source: source point cloud.
        target: target point cloud.
        tTs: relative pose tTs, i.e. target_T_source.
    """
    source_temp = copy.deepcopy(source)
    # Move `source` points to `target` frame.
    source_temp.transform(tTs)
    open3d.visualization.draw_geometries(
        [source_temp, target],
        zoom=0.5,
        front=[-0.2458, -0.8088, 0.5342],
        lookat=[1.7745, 2.2305, 0.9787],
        up=[0.3109, -0.5878, -0.7468],
    )


def test_transform() -> None:
    """Ensures that Open3d transform() method equates to GTSAM's transformFrom() method."""
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

    # Open3d operation identical to tTs.transformFrom(pcd_s)
    pcd_t = pcd_s.transform(tTs.matrix())

    # [4,2,3]
    # [1,2,6]
    assert np.allclose(np.asarray(pcd_t.points)[0], point_t0)
    assert np.allclose(np.asarray(pcd_t.points)[1], point_t1)
