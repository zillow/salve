# try just ICP between two point clouds

"""

Code adapted from:
http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html#id1

Floorplan-jigsaw:  Jointly estimating scene layout and aligning partialscans
https://arxiv.org/abs/1812.06677
"""

import glob
from pathlib import Path
from types import SimpleNamespace

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d
from gtsam import Rot3, Pose3

import afp.utils.bev_rendering_utils as bev_rendering_utils


def xyzrgb_to_open3d_point_cloud(xyzrgb: np.ndarray) -> open3d.geometry.PointCloud:
    """

    Args:
        xyzrgb: Nx6 array x,y,z,r,g,b

    Returns:
        pcd
    """
    colors = xyzrgb[:, 3:].astype(np.float64)  # / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


def register_colored_point_clouds(source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud) -> np.ndarray:
    """Register source point cloud to the target point cloud, using Colored Point Cloud Registration.

    Note: 3 layers of multi-resolution point clouds are used.

    See J. Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017.
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf

    Args:
        source: point cloud
        target: point

    Returns:
        tTs: transforms points from the souce frame to the target frame.
    """
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # Downsample with a voxel size %.2f" % radius
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # Estimate normal.
        source_down.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Applying colored point cloud registration
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
        source: point cloud
        target: point

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


def register_all_scenes() -> None:
    """

    Look for pairs, e.g.
        floor_01_partial_room_01_pano_3.depth.png
        floor_01_partial_room_01_pano_3.jpg
    """

    args = SimpleNamespace(
        **{
            "scale": 0.001,
            "crop_ratio": 80 / 512,  # throw away top 80 and bottom 80 rows of pixel (too noisy of estimates)
            "crop_z_range": [-10, 10],  # crop_z_range #0.3 # -1.0 # -0.5 # 0.3 # 1.2
            # "crop_z_above": 2
        }
    )

    # depthmap_dirpath = "/Users/johnlam/Downloads/HoHoNet_Depth_Maps/000"
    # pano_dirpath = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw/000/panos"

    depthmap_dirpath = "/Users/johnlam/Downloads/HoHoNet_Depth_Maps/267"
    pano_dirpath = "/Users/johnlam/Downloads/complete_07_10_new/267/panos"

    for floor_id in ["floor_00"]:  # , "floor_01", "floor_02", "floor_03"]:

        depthmap_fpaths = glob.glob(f"{depthmap_dirpath}/*{floor_id}*.png")
        depthmap_fpaths.sort()
        for i1, depthmap_fpath1 in enumerate(depthmap_fpaths):

            depthmap_img1 = imageio.imread(depthmap_fpath1)
            pano_fname1 = Path(depthmap_fpath1).stem.replace(".depth", "") + ".jpg"
            pano_fpath1 = f"{pano_dirpath}/{pano_fname1}"

            if int(Path(pano_fpath1).stem.split("_")[-1]) != 42:
                continue

            if not Path(pano_fpath1).exists():
                continue

            pano_img1 = imageio.imread(pano_fpath1)
            xyzrgb1 = bev_rendering_utils.get_xyzrgb_from_depth(
                args=args, depth_fpath=depthmap_fpath1, rgb_fpath=pano_fpath1, is_semantics=False
            )

            pcd1 = xyzrgb_to_open3d_point_cloud(xyzrgb1)
            # pcd1.estimate_normals(
            #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            # )
            open3d.visualization.draw_geometries([pcd1])

            for i2, depthmap_fpath2 in enumerate(depthmap_fpaths):

                if i2 <= i1:
                    continue

                depthmap_img2 = imageio.imread(depthmap_fpath2)
                pano_fname2 = Path(depthmap_fpath2).stem.replace(".depth", "") + ".jpg"
                pano_fpath2 = f"{pano_dirpath}/{pano_fname2}"

                if int(Path(pano_fpath2).stem.split("_")[-1]) != 43:
                    continue

                if not Path(pano_fpath2).exists():
                    continue

                pano_img2 = imageio.imread(pano_fpath2)

                plt.subplot(2, 2, 1)
                plt.imshow(pano_img1)
                plt.subplot(2, 2, 2)
                plt.imshow(depthmap_img1)

                plt.subplot(2, 2, 3)
                plt.imshow(pano_img2)
                plt.subplot(2, 2, 4)
                plt.imshow(depthmap_img2)

                plt.show()

                xyzrgb2 = bev_rendering_utils.get_xyzrgb_from_depth(
                    args=args, depth_fpath=depthmap_fpath2, rgb_fpath=pano_fpath2, is_semantics=False
                )

                pcd2 = xyzrgb_to_open3d_point_cloud(xyzrgb2)
                # pcd2.estimate_normals(
                #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                # )
                open3d.visualization.draw_geometries([pcd2])

                print(f"Register {i1} {i2}")
                import pdb

                pdb.set_trace()
                i2Ti1 = register_colored_point_clouds(source=pcd1, target=pcd2)
                # i2Ti1 = register_point_clouds(source=pcd1, target=pcd2)

                open3d.visualization.draw_geometries([pcd2, pcd1.transform(i2Ti1)])


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
    register_all_scenes()
