"""Utilities for ICP and colored ICP registration.

Code adapted from:
http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html#id1
https://github.com/isl-org/Open3D/blob/master/examples/python/pipelines/colored_icp_registration.py
"""

import glob
from pathlib import Path
from types import SimpleNamespace

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d
from gtsam import Rot3, Pose3

import salve.utils.bev_rendering_utils as bev_rendering_utils


def xyzrgb_to_open3d_point_cloud(xyzrgb: np.ndarray) -> open3d.geometry.PointCloud:
    """Convert Numpy array representing a colored point cloud to an Open3d PointCloud object.

    Args:
        xyzrgb: array of shape (N,6) representing (x,y,z,r,g,b) values for each point in a point cloud.

    Returns:
        pcd: Open3d point cloud object
    """
    colors = xyzrgb[:, 3:].astype(np.float64)  # / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


def register_colored_point_clouds(source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud) -> np.ndarray:
    """Register source point cloud to the target point cloud, using Colored Point Cloud Registration.

    Note: 3 layers of multi-resolution point clouds are used in registration.

    See J. Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017.
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf

    Args:
        source: source point cloud with (x,y,z) and (r,g,b).
        target: target point cloud with (x,y,z) and (r,g,b).

    Returns:
        tTs: transforms points from the source frame to the target frame.
    """
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    # Initialize transformation with identity.
    current_transformation = np.identity(4)
    print("Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # Downsample with a voxel size %.2f" % radius
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # Estimate normals.
        source_down.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Apply colored point cloud registration.
        result_icp = open3d.pipelines.registration.registration_colored_icp(
            source=source_down,
            target=target_down,
            max_correspondence_distance=radius,
            init=current_transformation,
            estimation_method=open3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
            ),
        )
        current_transformation = result_icp.transformation
        print(result_icp)
    return result_icp.transformation


def register_point_clouds(source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud) -> np.ndarray:
    """Register source point cloud to the target point cloud, using point to plane ICP (color is not used).

    Args:
        source: source point cloud with (x,y,z) and no color information.
        target: target point cloud with (x,y,z) and no color information.

    Returns:
        tTs: transforms points from the souce frame to the target frame.
    """
    current_transformation = np.identity(4)
    print("Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = open3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=0.02,
        init=current_transformation,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(result_icp)
    return result_icp.transformation

