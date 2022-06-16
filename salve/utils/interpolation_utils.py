"""
Utilities for interpolating a regular grid of values from sparse points, and for removing
interpolation artifacts from areas where there was no signal.
"""

import numpy as np
import torch
import torch.nn.functional as F

import scipy.interpolate  # not quite the same as `matplotlib.mlab.griddata`

from salve.utils.mesh_grid import get_mesh_grid_as_point_cloud


# DEFAULT_KERNEL_SZ = 41
# DEFAULT_KERNEL_SZ = 21
DEFAULT_KERNEL_SZ = 11

# minimum number of points required by QHull to construct an initial simplex, for interpolation.
MIN_REQUIRED_POINTS_SIMPLEX = 4


def interp_dense_grid_from_sparse(
    bev_img: np.ndarray, points: np.ndarray, rgb_values: np.ndarray, grid_h: int, grid_w: int, is_semantics: bool
) -> np.ndarray:
    """
    Args:
        bev_img: dense grid of shape (grid_h, grid_w, 3) to be populated with interpolated values
        points: (N,2) or (N,3) array of (x,y,z) or (x,y) coordinates
        rgb_values:
        grid_h:
        grid_w:
        is_semantics: whether or not the input RGB values represent a semantic colormap, in which case
            only `nearest` interpolation makes sense.

    Returns:
        bev_img: dense grid of shape (grid_h, grid_w, 3) populated with interpolated values
    """
    if points.shape[0] < MIN_REQUIRED_POINTS_SIMPLEX:
        # return the empty grid, since we can't interpolate.
        return bev_img

    if is_collinear(points):
        return bev_img

    grid_coords = get_mesh_grid_as_point_cloud(min_x=0, max_x=grid_w - 1, min_y=0, max_y=grid_h - 1)
    # Note: `xi` -- Points at which to interpolate data.
    interp_rgb_vals = scipy.interpolate.griddata(
        points=points[:, :2], values=rgb_values, xi=grid_coords, method="nearest" if is_semantics else "linear"
    )  # ) # # or method='linear', method='cubic'

    # can swap axes arbitrarily
    Y = grid_coords[:, 1].astype(np.int32)
    X = grid_coords[:, 0].astype(np.int32)
    bev_img[Y, X, :] = interp_rgb_vals
    return bev_img


def is_collinear(points: np.ndarray) -> bool:
    """
    Args:
        points: (N,2) array.
    """
    # all share the same x coordinate?
    if np.allclose(points[:,0], points[0,0]):
        return True

    if np.allclose(points[:,1], points[0,1]):
        return True

    return False


def remove_hallucinated_content(
    sparse_bev_img: np.ndarray, interp_bev_img: np.ndarray, K: int = DEFAULT_KERNEL_SZ
) -> np.ndarray:
    """Zero-out portions of an interpolated image where the signal is unreliable due to no measurements.

        If a KxK subgrid of an image has no sparse signals within it, and is initialized to a default value of zero,
        then convolution of the subgrid with a box filter of all 1's will yield zero back. These are the `counts`
        variable. In short, if the convolved output is zero in any ij cell, then we know that there was no true
        support for interpolation in this region, and we should mask out this interpolated value to zero.

    Args:
        sparse_bev_img: array of shape (H,W,C) representing a sparse bird's-eye-view image
        interp_bev_img: array of shape (H,W,C) representing an interpolated bird's-eye-view image
        K: integer representing kernel size, e.g. 3 for 3x3, 5 for 5x5

    Returns:
        unhalluc_img: array of shape (H,W,C)
    """
    H, W, _ = interp_bev_img.shape

    # check if any channel is populated
    mul_bev_img = sparse_bev_img[:, :, 0] * sparse_bev_img[:, :, 1] * sparse_bev_img[:, :, 2]

    mul_bev_img = torch.from_numpy(mul_bev_img).reshape(1, 1, H, W)
    nonempty = (mul_bev_img > 0).type(torch.float32)

    # box filter to sum neighbors
    weight = torch.ones(1, 1, K, K).type(torch.float32)

    # Use GPU whenever is possible, as convolution with a large kernel on the CPU is extremely slow
    if torch.cuda.is_available():
        weight = weight.cuda()
        nonempty = nonempty.cuda()

    # check counts of valid sparse pixel signals in each cell's KxK neighborhood
    counts = F.conv2d(input=nonempty, weight=weight, bias=None, stride=1, padding=K // 2)

    if torch.cuda.is_available():
        counts = counts.cpu()

    mask = counts > 0
    mask = mask.numpy().reshape(H, W).astype(np.float32)

    # CHW -> HWC
    mask = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)

    # multiply interpolated image with binary unreliability mask to zero-out unreliable values
    unhalluc_img = (mask * interp_bev_img).astype(np.uint8)
    return unhalluc_img

