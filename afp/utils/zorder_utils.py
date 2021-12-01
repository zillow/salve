
"""
Utilities for filling a raster canvas from top-to-bottom, or bottom-to-top.
"""

import numpy as np
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud


def choose_elevated_repeated_vals(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, zmin: float = -2, zmax: float = 2, num_slices: int = 4
) -> np.ndarray:
    """fill in the image, from top to bottom, after histogramming by z-value.

    Guarantee: x and y are integers between 0 and some large value
    Since want to have corresponding indices for semantics, we don't just immediately return new (x,y,z)
    although this is easier (save z instead of index into array)

    Note: Use lights, instead of floor, when available for salient features.

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
