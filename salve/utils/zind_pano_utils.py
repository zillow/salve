"""
There are at least 4 coordinate systems we work with in ZInD.
We have these 4 CSs and they have different scales.

1. Spherical coordinate system (we are still working on defining
the axes configuration w.r.t. the equirectangular projection). 

2. Ego-normalized coordinate system. (a Cartesian conversion of the spherical coordinate system.
If this cartesian system was scaled so that the camera was 1 unit above the floor,
this is our ego-normalized coordinate system.
(ego-normalized CS = room CS)

3. World-normalized coordinate system. We know how the ego-normalized coordinate system fits into
the world-normalized coordinate system, by (R,t,s).
(world-normalized = floor)

4.World-metric coordinate system, where units are in meters. (`world`)
The constant scale_meters_per_coordinate is providing the scaling to go from world-normalized to world-metric.


We used to name ego-normalized the room CS and world-normalized the floor CS

there is a reflection between one of these coordinate systems
(specifically between ego-normalized and world-normalized.

See for reference: 
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py#L3853:31
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/master/evaluate_horizonnet_output.py#L734
"""

import math

import numpy as np

EPS_RAD = 1e-10

EPS = np.deg2rad(1)

# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_pixel_to_sphere(points_pix: np.ndarray, width: int) -> np.ndarray:
    """Convert pixel coordinates into spherical coordinates from a 360 pano with a given width.

    Inverts: https://github.com/zillow/zind/blob/main/code/transformations.py#L150

    Note:
        We assume the width covers the full 360 degrees horizontally, and the height is derived
        as width/2 and covers the full 180 degrees vertical, i.e. we support mapping only on full FoV panos.

    Args:
        points_pix: array of shape (N,2) representing N points given in pano image coordinates [x, y] in range [0,width-1].
        width: The width of the pano image (defines the azimuth scale).

    Return:
        array of shape (N,2) representing points in spherical coordinates [theta, phi], where the
        spherical point [theta=0, phi=0] maps to the image center. We assume rho=1.0.

        theta (horizontal) is far left of image (-pi) to far right of image (pi)
        phi (vertical) is bottom of image (-pi/2) to top of image (pi/2)
    """
    if not isinstance(points_pix, np.ndarray) or points_pix.ndim != 2:
        raise RuntimeError(f"Input shape should have been (N,2), but received {points_pix.shape}")

    if points_pix.shape[1] != 2:
        raise RuntimeError(f"Input shape should have been (N,2), but received {points_pix.shape}")

    num_points, num_coords = points_pix.shape
    output_shape = (points_pix.shape[0], 2)  # type: ignore

    assert num_points > 0

    height = width / 2
    assert width > 1 and height > 1

    # We only consider the azimuth and elevation angles.
    x_arr = points_pix[:, 0]
    assert np.all(np.greater_equal(x_arr, 0.0))
    assert np.all(np.less(x_arr, width))

    y_arr = points_pix[:, 1]
    assert np.all(np.greater_equal(y_arr, 0.0))
    if not np.all(np.less(y_arr, height)):
        # TODO: this is weird that sometimes the predictions lie outside the height (514 vs. 512 height). need to debug why.
        print(f"Found coords {np.amax(y_arr)} > height {height}, clipping to [0, {height-1}]")
        y_arr = np.clip(y_arr, a_min=0, a_max=height-1)
        #import pdb; pdb.set_trace()
    #assert np.all(np.less(y_arr, height))

    # Convert the x-coordinates to azimuth spherical coordinates, where
    # theta=0 maps to the horizontal center.
    theta = x_arr / (width - 1)  # Map to [0, 1]
    theta *= 2.0 * math.pi  # Map to [0, 2*pi]
    theta -= math.pi  # Map to [-pi, pi]

    # Convert the y-coordinates to elevation spherical coordinates, where
    # phi=0 maps to the vertical center.
    phi = y_arr / (height - 1)  # Map to [0, 1]
    phi = 1.0 - phi  # Flip so that y=0 corresponds to pi/2
    phi *= math.pi  # Map to [0, pi]
    phi -= math.pi / 2.0  # Map to [-pi/2, pi/2]

    return np.column_stack((theta, phi)).reshape(output_shape)


# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_sphere_to_cartesian(points_sph: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to (room) cartesian coordinate system.

    Center pixel of equirectangular projection corresponds to -z axis in Cartesian.
    Inverts: https://github.com/zillow/zind/blob/main/code/transformations.py#L124

    Args:
        points_sph: array of shape (N,2) representing points in spherical coordinates [theta, phi],
            where the spherical point [theta=0, phi=0] maps to the image center.
            We assume all points lie on the unit sphere, i.e. rho = 1.0 for all points.
            theta (horizontal) is far left of image (-pi) to far right of image (pi)
            phi (vertical) is bottom of image (-pi/2) to top of image (pi/2)
            theta is the azimuthal angle in [-pi, pi],
            phi is the elevation angle in [-pi/2, pi/2]
            rho is the radial distance in (0, inf)

    Return:
        Array of shape (num_points, 3) representing points in cartesian coordinates [x, y, z].
    """
    output_shape = (points_sph.shape[0], 3)  # type: ignore

    num_points, num_coords = points_sph.shape
    assert num_points > 0

    assert num_coords == 2 or num_coords == 3

    theta = points_sph[:, 0]

    # Validate the azimuthal angles.
    assert np.all(np.greater_equal(theta, -math.pi - EPS_RAD))
    assert np.all(np.less_equal(theta, math.pi + EPS_RAD))

    phi = points_sph[:, 1]

    # Validate the elevation angles.
    if not np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS_RAD)):
        print(f"Found phi {np.amin(phi):.3f} vs [-pi/2,pi/2]=[{-np.pi/2:.3f},{np.pi/2:.3f}]. Clipping...")
        phi = np.clip(phi, a_min=-np.pi/2, a_max=np.pi/2)

    assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS_RAD))

    if num_coords == 2:
        rho = np.ones_like(theta)
    else:
        rho = points_sph[:, 2]

    # Validate the radial distances.
    assert np.all(np.greater(rho, 0.0))

    rho_cos_phi = rho * np.cos(phi) # equivalent to 'r' in the x-y plane.

    x_arr = rho_cos_phi * np.sin(theta)
    y_arr = rho * np.sin(phi)
    z_arr = rho_cos_phi * np.cos(theta)

    return np.column_stack((x_arr, y_arr, z_arr)).reshape(output_shape)


# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_intersect_cartesian_with_floor_plane(cartesian_coordinates: np.ndarray, camera_height: float) -> np.ndarray:
    """Obtain floor coordinates by intersecting with the floor plane.

    To get unit-norm rays, then scale so that y has unit norm.

    Args:
        cartesian_coordinates: TODO
        camera_height: TODO

    Returns:
        Coordinates in world metric space ... TODO
    """
    # flip Z
    cartesian_coordinates[:, 2] *= -1

    y = cartesian_coordinates[:, 1]
    return cartesian_coordinates * camera_height / y.reshape(-1, 1)


def convert_points_px_to_worldmetric(points_px: np.ndarray, image_width: int, camera_height_m: float) -> np.ndarray:
    """Convert pixel coordinates to Cartesian coordinates with a known scale (i.e. the units are meters).

    Args:
        points_px: 2d points in pixel coordinates
        image_width: width of image, in pixels.
        camera_height_m: height of camera during panorama capture (in meters).

    Returns:
        points_worldmetric: 
    """
    points_sph = zind_pixel_to_sphere(points_px, width=image_width)
    points_cartesian = zind_sphere_to_cartesian(points_sph)
    points_worldmetric = zind_intersect_cartesian_with_floor_plane(points_cartesian, camera_height_m)
    return points_worldmetric


def convert_points_worldmetric_to_px(points_worldmetric: np.ndarray, image_width: int, camera_height_m: float) -> np.ndarray:
    """

    Args:
        points_worldmetric
        image_width
        camera_height_m

    Returns:
       Points in pixel coordinates
    """
    points_cart = points_worldmetric
    # worldnormalized_to_room_cartesian(), swap the signs, and the axes
    points_sph = cartesian_to_sphere(points_cart)
    return zind_sphere_to_pixel(points_sph, width), 

    cartesian_coordinates = points_worldmetric / camera_height


def zind_cartesian_to_sphere(points_cart: np.ndarray) -> np.ndarray:
    """Convert cartesian to spherical coordinates.

    Note: conflicts with all other methods in this file.
    See: https://github.com/zillow/zind/blob/main/code/transformations.py#L124

    Args:
        points_cart: (N,3) points in room cartesian (ego) coordinate system

    Returns:
        points_sph: (N,3) points in spherical coordinate system, as (theta, phi)
          where theta in [-pi, pi] and phi in [-pi/2, pi/2]
    """
    output_shape = (points_cart.shape[0], 3)  # type: ignore

    num_points = points_cart.shape[0]
    assert num_points > 0

    num_coords = points_cart.shape[1]
    assert num_coords == 3

    x_arr = points_cart[:, 0]
    y_arr = points_cart[:, 1]
    z_arr = points_cart[:, 2]

    # Azimuth angle is in [-pi, pi].
    # Note the x-axis flip to align the handedness of the pano and room shape coordinate systems.
    theta = np.arctan2(x_arr, z_arr)

    # Radius can be anything between (0, inf)
    rho = np.linalg.norm(points_cart, axis=1)
    phi = np.arcsin(y_arr / rho)  # Map elevation to [-pi/2, pi/2]
    return np.column_stack((theta, phi, rho)).reshape(output_shape)


def zind_sphere_to_pixel(points_sph: np.ndarray, width: int) -> np.ndarray:
    """Convert spherical coordinates to pixel coordinates inside a 360 pano image with a given width.

    Args:
        points_sph: (N,3) points in spherical coordinates
        width: panorama image width (in pixels). Assumed to be in equirectangular projection format.

    Returns:
        Array of shape (N,3) representing points in pixel coordinates.
    """
    output_shape = (points_sph.shape[0], 2)  # type: ignore

    num_points = points_sph.shape[0]
    assert num_points > 0

    num_coords = points_sph.shape[1]
    assert num_coords == 2 or num_coords == 3

    height = width / 2
    assert width > 1 and height > 1

    # We only consider the azimuth and elevation angles.
    theta = points_sph[:, 0]
    assert np.all(np.greater_equal(theta, -math.pi - EPS))
    assert np.all(np.less_equal(theta, math.pi + EPS))

    phi = points_sph[:, 1]
    assert np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS))
    assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS))

    # Convert the azimuth to x-coordinates in the pano image, where
    # theta = 0 maps to the horizontal center.
    x_arr = theta + math.pi  # Map to [0, 2*pi]
    x_arr /= 2.0 * math.pi  # Map to [0, 1]
    x_arr *= width - 1  # Map to [0, width)

    # Convert the elevation to y-coordinates in the pano image, where
    # phi = 0 maps to the vertical center.
    y_arr = phi + math.pi / 2.0  # Map to [0, pi]
    y_arr /= math.pi  # Map to [0, 1]
    y_arr = 1.0 - y_arr  # Flip so that y goes up.
    y_arr *= height - 1  # Map to [0, height)

    return np.column_stack((x_arr, y_arr)).reshape(output_shape)

