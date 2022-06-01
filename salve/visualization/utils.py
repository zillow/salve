
"""Visualization utilities that use the Open3d library.
"""

from typing import List, Optional

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
import numpy as np
import open3d
from colour import Color
from gtsam import Pose3


def get_colormap(N: int) -> np.ndarray:
    """Obtain an RGB colormap from red to green, with N unique colors.

    Args:
        N: number of unique colors to generate.

    Returns:
        colormap: uint8 array of shape (N,3)
    """
    colormap = np.array([[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), N)]).squeeze()
    colormap = (colormap * 255).astype(np.uint8)
    return colormap


def get_colormapped_spheres(wTi_list: List[Optional[Pose3]]) -> np.ndarray:
    """
    Args:
        wTi_list

    Returns:
        point_cloud
        rgb
    """
    num_valid_poses = sum([1 if wTi is not None else 0 for wTi in wTi_list])
    colormap = get_colormap(N=num_valid_poses)

    curr_color_idx = 0
    point_cloud = []
    rgb = []
    for i, wTi in enumerate(wTi_list):
        if wTi is None:
            continue
        point_cloud += [wTi.translation()]
        rgb += [colormap[curr_color_idx]]
        curr_color_idx += 1

    point_cloud = np.array(point_cloud)
    rgb = np.array(rgb)
    return point_cloud, rgb


def draw_coordinate_frame(wTc: Pose3, axis_length: float = 1.0) -> List[open3d.geometry.LineSet]:
    """Draw 3 orthogonal axes representing a camera coordinate frame.

    Note: x,y,z axes correspond to red, green, blue colors.

    Args:
        wTc: Pose of any camera in the world frame.
        axis_length:

    Returns:
        line_sets: list of Open3D LineSet objects
    """
    RED = np.array([1, 0, 0])
    GREEN = np.array([0, 1, 0])
    BLUE = np.array([0, 0, 1])
    colors = (RED, GREEN, BLUE)

    line_sets = []
    for axis, color in zip([0, 1, 2], colors):

        lines = [[0, 1]]
        verts_worldfr = np.zeros((2, 3))

        verts_camfr = np.zeros((2, 3))
        verts_camfr[0, axis] = axis_length

        for i in range(2):
            verts_worldfr[i] = wTc.transformFrom(verts_camfr[i])

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(verts_worldfr),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(color.reshape(1, 3))
        line_sets.append(line_set)

    return line_sets


def plot_3d_poses(aTi_list_gt: List[Optional[Pose3]], bTi_list_est: List[Optional[Pose3]]) -> None:
    """

    Ground truth poses are rendered large (sphere of radius 0.5)
    Estimated poses are rendered small (spehere of radius 0.2)

    Args:
        aTi_list_gt: list of ground truth camera poses.
        bTi_list_est: list of estimated camera poses.
    """
    point_cloud_est, rgb_est = get_colormapped_spheres(bTi_list_est)
    point_cloud_gt, rgb_gt = get_colormapped_spheres(aTi_list_gt)
    geo1 = open3d_vis_utils.create_colored_spheres_open3d(point_cloud_est, rgb_est, sphere_radius=0.2)
    geo2 = open3d_vis_utils.create_colored_spheres_open3d(point_cloud_gt, rgb_gt, sphere_radius=0.5)

    def get_coordinate_frames(wTi_list: List[Optional[Pose3]]) -> List[open3d.geometry.LineSet]:
        frames = []
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            frames.extend(draw_coordinate_frame(wTi))
        return frames

    frames1 = get_coordinate_frames(aTi_list_gt)
    frames2 = get_coordinate_frames(bTi_list_est)

    open3d.visualization.draw_geometries(geo1 + geo2 + frames1 + frames2)

