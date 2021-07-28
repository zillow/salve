
"""
Utility for converting equirectangular panorama coordinates to spherical coordinates.
"""

import numpy as np

def get_uni_sphere_xyz(H: int, W: int) -> np.ndarray:
    """Make spherical system match the world system.

    Adapted from HoHoNet: https://github.com/sunset1995/HoHoNet/blob/master/vis_depth.py#L7

    Args:
        H: integer representing height of equirectangular panorama image.
        W: integer representing width of equirectangular panorama image.

    Returns:
        sphere_xyz
    """
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = -(i + 0.5) / W * 2 * np.pi
    v = ((j + 0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz