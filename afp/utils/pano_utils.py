
"""
Utility for converting equirectangular panorama coordinates to spherical coordinates.
"""

import numpy as np

def get_uni_sphere_xyz(H: int, W: int) -> np.ndarray:
    """Equirectangular proj. pixel coordinates to spherical coordinates (theta, phi).
    Then convert spherical to rectangular/cartesian.

    Adapted from HoHoNet: https://github.com/sunset1995/HoHoNet/blob/master/vis_depth.py#L7

    Args:
        H: integer representing height of equirectangular panorama image.
        W: integer representing width of equirectangular panorama image.

    Returns:
        sphere_xyz: array of shape (H,W,3) representing x,y,z coordinates on the unit sphere
            for each pixel location.
    """
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # `u` is theta
    u = -(i + 0.5) / W * 2 * np.pi
    # `v` is phi
    v = ((j + 0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz




"""

https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py#L3853:31
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/master/evaluate_horizonnet_output.py#L734


"""