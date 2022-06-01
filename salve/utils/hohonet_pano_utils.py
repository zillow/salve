"""
Utility for converting equirectangular panorama coordinates to spherical coordinates.
Specifically for the HoHoNet coordinate system (not for ZinD coordinate systems).

see https://github.com/sunset1995/PanoPlane360
"""

import numpy as np


def get_uni_sphere_xyz(H: int, W: int) -> np.ndarray:
    """Equirectangular proj. pixel coordinates to spherical coordinates (theta, phi).
    Then convert spherical to rectangular/cartesian.

    Adapted from HoHoNet: https://github.com/sunset1995/HoHoNet/blob/master/vis_depth.py#L7
    The -x axis points towards the center pixel of the equirectangular projection.

    Args:
        H: integer representing height of equirectangular panorama image. An (H,W) grid will be created.
        W: integer representing width of equirectangular panorama image.

    Returns:
        sphere_xyz: array of shape (H,W,3) representing x,y,z coordinates on the unit sphere
            for each pixel location. Note that rho = 1 for all spherical coordinate points.
            Note that a spherical depth map (containing rho values) could be elementwise multiplied
            with the output of this function (rho is a constant multipler on each dimension below).
    """
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # horizontal dir -- go through the center of the pixel (add 0.5) and account for image left to right unwrapping
    theta = -(u + 0.5) / W
    # scale theta to [0,2pi]
    theta *= 2 * np.pi
    # now vertical direction -- go through the center of the pixel
    phi = (v + 0.5) / H
    # scale to [-0.5,0.5]
    phi -= 0.5
    # scale to [0,pi]
    phi *= np.pi

    z = -np.sin(phi)
    r = np.cos(phi)  # TODO: explain why this is not sin(phi)
    y = r * np.sin(theta)
    x = r * np.cos(theta)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz
