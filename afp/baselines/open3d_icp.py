# try just ICP between two point clouds

"""
See J. Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017.
https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf

http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html#id1
"""

import numpy as np
import open3d as o3d


def register_colored_point_clouds(source, target):
    """
    3 layers of multi-resolution point clouds
    """
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # Downsample with a voxel size %.2f" % radius
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # Estimate normal.")
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Applying colored point cloud registration
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source=source_down,
            target=target_down,
            max_correspondence_distance=radius,
            init=current_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
            ),
        )
        current_transformation = result_icp.transformation
        print(result_icp)
    return result_icp.transformation


def register_point_clouds(source, target):
    """
    point to plane ICP (color is not used).
    """
    current_transformation = np.identity(4)
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=0.02,
        init=current_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(result_icp)
    return result_icp.transformation


