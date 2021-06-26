import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud
from argoverse.utils.se2 import SE2
from argoverse.utils.sim2 import Sim2
from imageio import imread
from scipy.interpolate import griddata  # not quite the same as `matplotlib.mlab.griddata`

# from vis_zind_annotations import rotmat2d

from zorder_utils import choose_elevated_repeated_vals
from interp_artifact_removal import remove_hallucinated_content


def get_uni_sphere_xyz(H, W):
    """Make spherical system match the world system"""
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = -(i + 0.5) / W * 2 * np.pi
    v = ((j + 0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz


def prune_to_2d_bbox(
    pts: np.ndarray, rgb: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float
) -> np.ndarray:
    """"""
    x = pts[:, 0]
    y = pts[:, 1]
    is_valid = np.logical_and.reduce([xmin <= x, x <= xmax, ymin <= y, y <= ymax])
    return pts[is_valid], rgb[is_valid]


def test_prune_to_2d_bbox():
    """ """
    pts = np.array([[-2, 2], [2, 0], [1, 2], [0, 1]])  # will be discarded  # will be discarded
    xmin = -1
    ymin = -1
    xmax = 1
    ymax = 2

    pts = prune_to_2d_bbox(pts, xmin, ymin, xmax, ymax)
    gt_pts = np.array([[1, 2], [0, 1]])
    assert np.allclose(pts, gt_pts)


class BEVParams:
    def __init__(
        self, img_h: int = 2000, img_w: int = 2000, meters_per_px: float = 0.005, accumulate_sweeps: bool = True
    ) -> None:
        """meters_per_px is resolution

        1000 pixels * (0.005 m / px) = 5 meters in each direction
        """
        self.img_h = img_h
        self.img_w = img_w
        self.meters_per_px = meters_per_px
        self.accumulate_sweeps = accumulate_sweeps

        # num px in horizontal direction
        h_px = img_w / 2

        # num px in vertical direction
        v_px = img_h / 2

        # get grid boundaries in meters
        xmin_m = -int(h_px * meters_per_px)
        xmax_m = int(h_px * meters_per_px)
        ymin_m = -int(v_px * meters_per_px)
        ymax_m = int(v_px * meters_per_px)

        xlims = [xmin_m, xmax_m]
        ylims = [ymin_m, ymax_m]

        self.xlims = xlims
        self.ylims = ylims


def interp_dense_grid_from_sparse(
    bev_img: np.ndarray, points: np.ndarray, rgb_values: np.ndarray, grid_h: int, grid_w: int, is_semantics: bool
) -> np.ndarray:
    """
    Args:
       points: (N,2) or (N,3) array of (x,y,z) or (x,y)
    """
    grid_coords = get_mesh_grid_as_point_cloud(min_x=0, max_x=grid_w - 1, min_y=0, max_y=grid_h - 1)
    # Note: `xi` -- Points at which to interpolate data.
    interp_rgb_vals = griddata(
        points=points[:, :2], values=rgb_values, xi=grid_coords, method="nearest" if is_semantics else "linear"
    )  # ) # # or method='linear', method='cubic'

    # can swap axes arbitrarily
    Y = grid_coords[:, 1].astype(np.int32)
    X = grid_coords[:, 0].astype(np.int32)
    bev_img[Y, X, :] = interp_rgb_vals
    return bev_img


def render_bev_image(bev_params: BEVParams, xyzrgb: np.ndarray, is_semantics: bool) -> Optional[np.ndarray]:
    """
    Args:
        xyz: should be inside the world coordinate frame
        bev_params: parameters for rendering
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:] * 255

    # in meters
    grid_xmin, grid_xmax = bev_params.xlims
    grid_ymin, grid_ymax = bev_params.ylims

    # in meters
    xyz, rgb = prune_to_2d_bbox(xyz, rgb, grid_xmin, grid_ymin, grid_xmax, grid_ymax)

    num_pts = xyz.shape[0]
    print(f"Rendering {num_pts/1e6} million points")

    if num_pts == 0:
        return None

    # # m/px -> px/m, then px/m * #meters = #pixels
    bevimg_Sim2_world = Sim2(R=np.eye(2), t=np.array([-grid_xmin, -grid_ymin]), s=1 / bev_params.meters_per_px)

    xy = xyz[:, :2]
    z = xyz[:, 2]
    img_xy = bevimg_Sim2_world.transform_from(xy)
    img_xy = np.round(img_xy).astype(np.int64)
    # xmax, ymax = np.amax(img_xy, axis=0)
    # img_h = ymax + 1
    # img_w = xmax + 1

    #import pdb; pdb.set_trace()
    img_h = bev_params.img_h + 1
    img_w = bev_params.img_w + 1

    x = img_xy[:, 0]
    y = img_xy[:, 1]

    prioritize_elevated = True
    if prioritize_elevated:
        # import pdb; pdb.set_trace()
        valid = choose_elevated_repeated_vals(x, y, z)
        img_xy = img_xy[valid]
        rgb = rgb[valid]

        x = x[valid]
        y = y[valid]

    sparse_bev_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    sparse_bev_img[y, x] = rgb

    interp_bev_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # now, apply interpolation to it
    interp_bev_img = interp_dense_grid_from_sparse(
        interp_bev_img, img_xy, rgb, grid_h=img_h, grid_w=img_w, is_semantics=is_semantics
    )

    # apply filter to it, to remove hallicuniated parts in all black regions
    bev_img = remove_hallucinated_content(sparse_bev_img, interp_bev_img)
    bev_img = np.flipud(bev_img)

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(bev_img)
        plt.show()

    return bev_img


def colormap(rgb: bool = True) -> np.ndarray:
    """
    Create an array of visually distinctive RGB values.
    Args:
    -     rgb: boolean, whether to return in RGB or BGR order. BGR corresponds to OpenCV default.
    Returns:
    -     color_list: Numpy array of dtype uin8 representing RGB color palette.
    """
    color_list = np.array(
        [
            [252, 233, 79],
            # [237, 212, 0],
            [196, 160, 0],
            [252, 175, 62],
            # [245, 121, 0],
            [206, 92, 0],
            [233, 185, 110],
            [193, 125, 17],
            [143, 89, 2],
            [138, 226, 52],
            # [115, 210, 22],
            [78, 154, 6],
            [114, 159, 207],
            # [52, 101, 164],
            [32, 74, 135],
            [173, 127, 168],
            # [117, 80, 123],
            [92, 53, 102],
            [239, 41, 41],
            # [204, 0, 0],
            [164, 0, 0],
            [238, 238, 236],
            # [211, 215, 207],
            # [186, 189, 182],
            [136, 138, 133],
            # [85, 87, 83],
            [46, 52, 54],
        ]
    ).astype(np.uint8)
    assert color_list.shape[1] == 3
    assert color_list.ndim == 2

    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def grayscale_to_color(gray_img: np.ndarray) -> np.ndarray:
    """Duplicate the grayscale channel 3 times.

    Args:
    gray_img: Array with shape (M,N)

    Returns:
    rgb_img: Array with shape (M,N,3)
    """
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb_img[:, :, i] = gray_img
    return rgb_img


CEILING_CLASS_IDX = 36
MIRROR_CLASS_IDX = 85
WALL_CLASS_IDX = 191


def get_xyzrgb_from_depth(args, depth_fpath: str, rgb_fpath: str, is_semantics: bool) -> np.ndarray:
    """
    Args:
        xyzrgb, with rgb in [0,1] as floats
    """
    depth = imread(depth_fpath)[..., None].astype(np.float32) * args.scale

    # Reading rgb-d
    rgb = imread(rgb_fpath)

    width = 1024
    height = 512
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST if is_semantics else cv2.INTER_LINEAR)

    if is_semantics:
        # remove ceiling and mirror points
        invalid = np.logical_or(rgb == CEILING_CLASS_IDX, rgb == MIRROR_CLASS_IDX)
        depth[invalid] = np.nan

        rgb_colormap = colormap()
        num_colors = rgb_colormap.shape[0]
        rgb = rgb_colormap[rgb % num_colors]
    else:

        if rgb.ndim == 2:
            rgb = grayscale_to_color(rgb)

    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)

    xyzrgb = np.concatenate([xyz, rgb / 255.0], 2)

    # Crop the image and flatten
    if args.crop_ratio > 0:
        assert args.crop_ratio < 1
        crop = int(H * args.crop_ratio)
        # remove the bottom rows of pano, and remove the top rows of pano
        xyzrgb = xyzrgb[crop:-crop]

    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    within_crop_range = np.logical_and(
        xyzrgb[:, 2] > args.crop_z_range[0],
        xyzrgb[:, 2] <= args.crop_z_range[1]
    )
    xyzrgb = xyzrgb[within_crop_range]

    return xyzrgb


def vis_depth_and_render(args, is_semantics: bool):
    """ """
    xyzrgb = get_xyzrgb_from_depth(args, depth_fpath=args.depth, rgb_fpath=args.img, is_semantics=is_semantics)

    # Visualize
    visualize = True
    if visualize:

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

    bev_params = BEVParams()
    bev_img = render_bev_image(bev_params, xyzrgb, is_semantics)
    return bev_img


def vis_depth(args):
    """ """
    # Reading rgb-d
    rgb = imread(args.img)
    depth = imread(args.depth)[..., None].astype(np.float32) * args.scale

    width = 1024
    height = 512
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb / 255.0], 2)

    # Crop the image and flatten
    if args.crop_ratio > 0:
        assert args.crop_ratio < 1
        crop = int(H * args.crop_ratio)
        # remove the bottom rows of pano, and remove the top rows of pano
        xyzrgb = xyzrgb[crop:-crop]

    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    xyzrgb = xyzrgb[xyzrgb[:, 2] <= args.crop_z_above]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])]
    )


def rotmat2d(theta_deg: float) -> np.ndarray:
    """Generate 2x2 rotation matrix, given a rotation angle in degrees."""
    theta_rad = np.deg2rad(theta_deg)

    s = np.sin(theta_rad)
    c = np.cos(theta_rad)

    # fmt: off
    R = np.array(
        [
            [c, -s],
            [s, c]
        ]
    )
    # fmt: on
    return R


def render_bev_pair(args, building_id: str, floor_id: str, i1: int, i2: int, i2Ti1: Sim2, is_semantics: bool) -> Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
    """ """

    xyzrgb1 = get_xyzrgb_from_depth(args, depth_fpath=args.depth_i1, rgb_fpath=args.img_i1, is_semantics=is_semantics)
    xyzrgb2 = get_xyzrgb_from_depth(args, depth_fpath=args.depth_i2, rgb_fpath=args.img_i2, is_semantics=is_semantics)

    # floor_map_json['scale_meters_per_coordinate']
    scale_meters_per_coordinate = 3.7066488344243465
    print(i2Ti1)

    # because of the reflection!
    xyzrgb1[:, 0] *= -1
    xyzrgb2[:, 0] *= -1

    # HoHoNet center of pano is to -x, but in ZinD center of pano is +y
    R = rotmat2d(90)

    xyzrgb1[:, :2] = xyzrgb1[:, :2] @ R.T
    xyzrgb2[:, :2] = xyzrgb2[:, :2] @ R.T

    HOHO_S_ZIND_SCALE_FACTOR = 1.5

    xyzrgb1[:, :2] = (xyzrgb1[:, :2] @ i2Ti1.rotation.T) + (
        i2Ti1.translation * HOHO_S_ZIND_SCALE_FACTOR
    )  # * scale_meters_per_coordinate # * np.array([-1,1]))
    # xyzrgb1[:,:2] = i2Ti1.transform_from(xyzrgb1[:,:2]) #

    # reflect back to Cartesian space
    xyzrgb1[:, 0] *= -1
    xyzrgb2[:, 0] *= -1

    bev_params = BEVParams()
    img1 = render_bev_image(bev_params, xyzrgb1, is_semantics=is_semantics)
    img2 = render_bev_image(bev_params, xyzrgb2, is_semantics=is_semantics)

    if img1 is None or img2 is None:
        return None, None

    visualize = False
    if visualize:
        # plt.scatter(xyzrgb1[:,0], xyzrgb1[:,1], 10, color='r', marker='.', alpha=0.1)
        plt.scatter(xyzrgb1[:, 0], xyzrgb1[:, 1], 10, c=xyzrgb1[:, 3:], marker=".", alpha=0.1)
        # plt.axis("equal")
        # plt.show()

        # plt.scatter(xyzrgb2[:,0], xyzrgb2[:,1], 10, color='b', marker='.', alpha=0.1)
        plt.scatter(xyzrgb2[:, 0], xyzrgb2[:, 1], 10, c=xyzrgb2[:, 3:], marker=".", alpha=0.1)

        plt.title("")
        plt.axis("equal")
        # plt.show()
        save_fpath = f"aligned_examples_2021_06_22/gt_aligned_approx/{building_id}/{floor_id}/{i1}_{i2}.jpg"
        os.makedirs(Path(save_fpath).parent, exist_ok=True)
        plt.savefig(save_fpath, dpi=1000)
        plt.close("all")

        # 30,35 on floor2 bug

    return img1, img2

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--img', required=True,
    #                     help='Image texture in equirectangular format')
    # parser.add_argument('--depth', required=True,
    #                     help='Depth map')
    # parser.add_argument('--scale', default=0.001, type=float,
    #                     help='Rescale the depth map')
    # parser.add_argument('--crop_ratio', default=80/512, type=float,
    #                     help='Crop ratio for upper and lower part of the image')
    # parser.add_argument('--crop_z_above', default=1.2, type=float,
    #                     help='Filter 3D point with z coordinate above')
    # args = parser.parse_args()

    # test_find_duplicates_1()
    # test_find_duplicates_2()
    # test_find_duplicates_3()
    pass
