

import numpy as np


def convert_points_worldmetric_to_px(points_worldmetric: np.ndarray, image_width: int, camera_height_m: float) -> np.ndarray:
    """TODO

    Only for points on the ground.

    Args:
        points_worldmetric: array of shape (N,3)
        image_width: image width (in pixels)
        camera_height_m: height of camera during capture (in meters)

    Returns:
       Array of shape (N,2) representing points in pixel coordinates
    """
    points_roomcartesian = zind_worldmetric_to_room_cartesian(points_worldmetric, camera_height_m=camera_height_m)
    points_sph = zind_cartesian_to_sphere(points_roomcartesian)
    return zind_sphere_to_pixel(points_sph, width=image_width)


def zind_worldmetric_to_room_cartesian(worldmetric_coordinates: np.ndarray, camera_height_m: float) -> np.ndarray:
    """Convert world-metric cartesian coordinates to room cartesian coordinates.

    Args:
        worldmetric_coordinates:
        camera_height_m:

    Returns:
        room_cartesian_coords:
    """
    # Undo reflection.
    worldmetric_coordinates[:, 0] *= -1

    # Re-shuffle the axes, so that y is the upright axis instead of z.
    room_cartesian_coords = worldmetric_coordinates[:, np.array([0, 2, 1])]

    room_cartesian_coords /= camera_height_m
    # TODO: fix bug
    import pdb; pdb.set_trace()
    # NOTE: the step below loses the sign, making this transformation unrecoverable.
    room_cartesian_coords *= room_cartesian_coords[:, 1].reshape(-1,1)

    # Flip Z to go from right-handed to left-handed system.
    room_cartesian_coords[:, 2] *= -1

    rho = np.linalg.norm(room_cartesian_coords, axis=1).reshape(-1,1)
    room_cartesian_coords /= rho
    return room_cartesian_coords
