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
        """meters_per_px is resolution"""
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


def find_duplicates(a: np.ndarray) -> np.ndarray:
    """
    Return the indices where values are repeated

    e.g. if the input array is [5,2,6,2,0,1,6,6]
    then sorted, it is [0,1,2,2,5,6,6,6]

    the question is where am i equal to my neighbor in front?

    [1,2,2,5,6,6,6,$]
    [0,1,2,2,5,6,6,6]

    and where am I equal to my neighbor behind?

    [$,0,1,2,2,5,6,6,6]
    [0,1,2,2,5,6,6,6,$]

    TODO: verify why this logic works
    """
    idxs = np.argsort(a, axis=None)
    sorted_a = a[idxs]
    is_dup = sorted_a[1:] == sorted_a[:-1]

    dup1 = idxs[:-1][is_dup]

    DUMMY_VAL = 9999999
    arr1 = np.array([DUMMY_VAL] + list(sorted_a))
    arr2 = np.array(list(sorted_a) + [DUMMY_VAL])

    dup2 = idxs[np.where(arr1 == arr2)[0]]

    all_dup_idxs = np.concatenate([dup1, dup2])
    # remove redundancy

    return np.unique(all_dup_idxs)


def test_find_duplicates_1() -> None:
    """ """
    arr = np.array([5, 2, 6, 2, 0, 1, 6, 6])
    dup_idxs = find_duplicates(arr)
    gt_dup_idxs = np.array([1, 2, 3, 6, 7])
    assert np.allclose(dup_idxs, gt_dup_idxs)

    # now try in a different order
    arr = np.array([6, 6, 6, 5, 2, 2, 1, 0])
    dup_idxs = find_duplicates(arr)
    gt_dup_idxs = np.array([0, 1, 2, 4, 5])
    assert np.allclose(dup_idxs, gt_dup_idxs)


def test_find_duplicates_2() -> None:
    """no duplicates"""
    arr = np.array([3, 1, 4, 0])
    dup_idxs = find_duplicates(arr)
    assert dup_idxs.size == 0
    assert isinstance(dup_idxs, np.ndarray)


def test_find_duplicates_3() -> None:
    """all duplicates"""
    arr = np.array([3, 3, 3, 3, 3])
    dup_idxs = find_duplicates(arr)
    dup_idxs_gt = np.arange(5)
    assert np.allclose(dup_idxs, dup_idxs_gt)


# def choose_elevated_repeated_vals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
#     """Use lights, instead of floor, when available for salient features.

#     Args:
#        valid: logicals, for where at this location, we are the highest z value
#     """
#     dup_x_idxs = find_duplicates(x)
#     dup_y_idxs = find_duplicates(y)
#     dup_xy_idxs = np.intersect1d(dup_x_idxs, dup_y_idxs)

#     N = x.shape[0]
#     # set all initially to true
#     valid = np.ones(N, dtype=bool)

#     seen = set()

#     for loop_idx, i in enumerate(dup_xy_idxs):

#         if loop_idx % 5000 == 0:
#             print(f"On {loop_idx}/{len(dup_xy_idxs)}")

#         if i in seen:
#             continue

#         # indices at the same (x,y) location
#         loc_idxs = np.argwhere(np.logical_and(x == x[i], y == y[i]))
#         seen = seen.union(set(list(loc_idxs.squeeze(axis=1))))

#         # keep the one with the highest z val
#         keep_idx = loc_idxs[ np.argmax(z[loc_idxs]) ]

#         # Return the unique values in ar1 that are not in ar2.
#         discard_idxs = np.setdiff1d(loc_idxs, np.array([keep_idx]))
#         valid[discard_idxs] = False

#     return valid


def choose_elevated_repeated_vals(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, zmin: float = -2, zmax: float = 2, num_slices: int = 4
) -> np.ndarray:
    """fill in the image, from top to bottom, after histogramming by z-value

    Guarantee: x and y are integers between 0 and some large value
    Since want to have corresponding indices for semantics, we dont just immediately return new (x,y,z)
    although this is easier (save z instead of index into array)

    Note:Use lights, instead of floor, when available for salient features.

    Args:
       valid: logicals, for where at this location, we are the highest z value
    """
    num_pts = x.shape[0]
    global_idxs = np.arange(num_pts)

    # find min x, max x
    xmin, xmax = 0, x.max()
    # find min y, max y
    ymin, ymax = 0, y.max()

    img_h = ymax - ymin + 1
    img_w = xmax - xmin + 1

    DUMMY_VAL = np.iinfo(np.uint64).max

    # make 2d grid, fill it up with dummy values
    img = np.ones((img_h, img_w), dtype=np.uint64) * DUMMY_VAL

    # only bottom to top is supported currently
    z_planes = np.linspace(zmin, zmax, num_slices + 1)

    # z_plane_indices = np.digitize(z, z_planes)

    for z_plane_idx in range(len(z_planes) - 1):

        z_start = z_planes[z_plane_idx]
        z_end = z_planes[z_plane_idx + 1]

        # within the slice
        ws = np.logical_and(z >= z_start, z < z_end)
        x_ws = x[ws]
        y_ws = y[ws]
        idxs_ws = global_idxs[ws]

        # place into the grid, the global index of the point
        img[y_ws, x_ws] = idxs_ws

    # use np.meshgrid to index into array
    pts = get_mesh_grid_as_point_cloud(xmin, xmax, ymin, ymax)
    pts = np.round(pts).astype(np.uint64)

    idxs = img[ pts[:,1], pts[:,0] ]

    # throw away all indices where they are dummy values
    is_valid = np.where(idxs != DUMMY_VAL)[0]

    # get back 2d coordinates that are not fully of dummy values
    # look up the global point index, for those 2d coordinates
    valid_coords = pts[is_valid]
    global_idxs = img[ valid_coords[:,1], valid_coords[:,0] ]

    is_valid_ = np.zeros(num_pts, dtype=bool)

    is_valid_[global_idxs] = 1

    return is_valid_


def test_choose_elevated_repeated_vals_1() -> None:
    """Single location is repeated"""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [0, 1, 5], [5, 6, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    import pdb; pdb.set_trace()
    valid = choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_2() -> None:
    """No location is repeated"""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [2, 3, 5], [3, 4, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([True, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_3() -> None:
    """Every location is repeated"""
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_4() -> None:
    """
    Some z-values are outside of the specified range
    """
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 10], [0, 1, 11]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    import pdb; pdb.set_trace()
    valid = choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, False, False])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_5() -> None:
    """
    Try for just 2 slices
    """
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=4, num_slices=2)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)


def render_bev_image(bev_params: BEVParams, xyzrgb: np.ndarray, is_semantics: bool) -> np.ndarray:
    """
    Args:
        xyz: should be inside the world coordinate frame
        bev_params: parameters for rendering
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:] * 255

    grid_xmin, grid_xmax = bev_params.xlims
    grid_ymin, grid_ymax = bev_params.ylims

    xyz, rgb = prune_to_2d_bbox(xyz, rgb, grid_xmin, grid_ymin, grid_xmax, grid_ymax)

    num_pts = xyz.shape[0]
    print(f"Rendering {num_pts/1e6} million points")

    # # m/px -> px/m, then px/m * #meters = #pixels
    bevimg_Sim2_world = Sim2(R=np.eye(2), t=np.array([-grid_xmin, -grid_ymin]), s=1 / bev_params.meters_per_px)

    xy = xyz[:, :2]
    z = xyz[:, 2]
    img_xy = bevimg_Sim2_world.transform_from(xy)
    img_xy = np.round(img_xy).astype(np.int64)
    xmax, ymax = np.amax(img_xy, axis=0)
    img_h = ymax + 1
    img_w = xmax + 1
    x = img_xy[:, 0]
    y = img_xy[:, 1]

    prioritize_elevated = False
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

    interp_bev_img = interp_dense_grid_from_sparse(
        interp_bev_img, img_xy, rgb, grid_h=img_h, grid_w=img_w, is_semantics=is_semantics
    )

    # apply filter to it, to remove hallicuniated parts in all black regions
    bev_img = remove_hallucinated_content(sparse_bev_img, interp_bev_img)
    bev_img = np.flipud(bev_img)

    # now, apply interpolation to it

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(bev_img)
        plt.show()

    return bev_img


def remove_hallucinated_content(sparse_bev_img: np.ndarray, interp_bev_img: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Args:
        K: kernel size, e.g. 3 for 3x3, 5 for 5x5
    """
    import copy

    bev_img = copy.deepcopy(interp_bev_img)

    H, W, _ = interp_bev_img.shape

    for i in range(H - K):
        for j in range(W - K):
            if sparse_bev_img[i : i + K, j : j + K, :].sum() == 0:
                bev_img[i : i + K, j : j + K, :] = 0

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
    xyzrgb = xyzrgb[xyzrgb[:, 2] <= args.crop_z_above]

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


def render_bev_pair(args, building_id: str, floor_id: str, i1: int, i2: int, i2Ti1: Sim2, is_semantics: bool) -> None:
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

    # plt.scatter(xyzrgb1[:,0], xyzrgb1[:,1], 10, color='r', marker='.', alpha=0.1)
    plt.scatter(-xyzrgb1[:, 0], xyzrgb1[:, 1], 10, c=xyzrgb1[:, 3:], marker=".", alpha=0.1)
    # plt.axis("equal")
    # plt.show()

    # xyzrgb2[:,:2] = xyzrgb2[:,:2] @ i2Ti1.rotation.T
    # xyzrgb2[:,:2] = i2Ti1.transform_from(xyzrgb2[:,:2]) #* scale_meters_per_coordinate

    # plt.scatter(xyzrgb2[:,0], xyzrgb2[:,1], 10, color='b', marker='.', alpha=0.1)
    plt.scatter(-xyzrgb2[:, 0], xyzrgb2[:, 1], 10, c=xyzrgb2[:, 3:], marker=".", alpha=0.1)

    # plt.plot([0, i2Ti1.translation[0]] * 10, [0,i2Ti1.translation[1]] * 10, color='k')

    plt.title("")
    plt.axis("equal")
    # plt.show()
    save_fpath = f"aligned_examples_2021_06_22/gt_aligned_approx/{building_id}/{floor_id}/{i1}_{i2}.jpg"
    os.makedirs(Path(save_fpath).parent, exist_ok=True)
    plt.savefig(save_fpath, dpi=1000)
    plt.close("all")

    # import pdb; pdb.set_trace()

    # 30,35 on floor2 bug


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
    # test_choose_elevated_repeated_vals_1()
    # test_choose_elevated_repeated_vals_2()
    # test_choose_elevated_repeated_vals_3()

    test_choose_elevated_repeated_vals_4()
    test_choose_elevated_repeated_vals_5()
