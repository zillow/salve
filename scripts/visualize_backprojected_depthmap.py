"""Visualize with Open3d backprojected RGB values (using depth map) as colored point cloud.

Adapted from HoHoNet: https://github.com/sunset1995/HoHoNet/blob/master/vis_depth.py#L19
"""

import argparse
from argparse import Namespace
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import salve.utils.hohonet_pano_utils as hohonet_pano_utils
import salve.utils.bev_rendering_utils as bev_rendering_utils
from salve.common.bevparams import BEVParams


def vis_depth_and_render(args: Namespace, is_semantics: bool) -> None:
    """Visualize point cloud, and then render texture map."""
    xyzrgb = bev_rendering_utils.get_xyzrgb_from_depth(
        args, depth_fpath=args.depth, rgb_fpath=args.img, is_semantics=is_semantics
    )

    # Visualize
    visualize_3d = True
    if visualize_3d:

        invalid = np.isnan(xyzrgb[:, 0])
        xyzrgb = xyzrgb[~invalid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

        o3d.visualization.draw_geometries(
            geometry_list=[
                pcd,
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0]),
            ],
            window_name=Path(args.img).stem,
        )

    # Render and display depth map.
    bev_params = BEVParams()
    bev_img = bev_rendering_utils.render_bev_image(bev_params, xyzrgb, is_semantics)
    plt.imshow(bev_img)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img", required=True, help="Path to panorama image in equirectangular format.")
    parser.add_argument("--depth", required=True, help="Path to depth map file (as .depth.png)")
    parser.add_argument("--scale", default=0.001, type=float, help="Scale factor to use when rescaling the depth map.")
    parser.add_argument(
        "--crop_ratio", default=80 / 512, type=float, help="Crop ratio for upper and lower part of the image"
    )
    parser.add_argument("--crop_z_range", default=1.2, type=float, help="Filter 3D point with z coordinate above")
    args = parser.parse_args()

    surface_type = "ceiling"  # "floor" # "all"
    if surface_type == "floor":
        # everything 1 meter and below the camera
        args.crop_z_range = [-float("inf"), -1.0]

    elif surface_type == "ceiling":
        # everything 50 cm and above camera
        args.crop_z_range = [0.5, float("inf")]

    else:
        args.crop_z_range = [-float("inf"), float("inf")]

    vis_depth_and_render(args, is_semantics=False)
