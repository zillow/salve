
"""
There are at least 4 coordinate systems we work with in ZinD.
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
"""

import math

import numpy as np

EPS_RAD = 1e-10

# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_intersect_cartesian_with_floor_plane(cartesian_coordinates: np.ndarray, camera_height: float) -> np.ndarray:
    """
    In order to get the floor coordinates, intersect with the floor plane

    get unit-norm rays, then scale so that y has unit norm
    """
    return cartesian_coordinates * camera_height / cartesian_coordinates[:, 1].reshape(-1, 1)



def zind_cartesian_to_sphere(points_cart: np.ndarray) -> np.ndarray:
    """Convert cartesian to spherical coordinates.

    Note: conflicts with all other methods in this file.
    See: https://github.com/zillow/zind/blob/main/code/transformations.py#L124

    Args:
        points_cart:

    Returns:
        points_sph:
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
    theta = np.arctan2(-x_arr, y_arr)

    # Radius can be anything between (0, inf)
    rho = np.sqrt(np.sum(np.square(points_cart), axis=1))
    phi = np.arcsin(z_arr / rho)  # Map elevation to [-pi/2, pi/2]
    return np.column_stack((theta, phi, rho)).reshape(output_shape)


def test_zinc_cartesian_to_sphere() -> None:
    """ """
    pass
    # # fmt: off
    # points_cart = np.array(
    #     [
    #         [0,  1, 0],
    #         [0, -1, 0],
    #         [0, -1, 0],
    #         [0,  1, 0],
    #         [0,  0,-1],
    #         [1, 0, 0],
    #         [0, 0, 1]
    #    ]
    # )
    # # fmt: on

    # # provided as (u,v) coordinates.
    # expected_points_sph = np.array(
    #     [
    #         [-np.pi, np.pi/2], # corresponds to (u,v) = (0,0)
    #         [-np.pi, -np.pi/2], # corresponds to (u,v) = (0,511)
    #         [np.pi, -np.pi/2], # corresponds to (u,v) = (1023,511)
    #         [np.pi, np.pi/2], # corresponds to (u,v) = (1023,0)
    #         [0, 0], # corresponds to (u,v) = (512,256)
    #         [np.pi/2, 0], # 1/4 way from left edge of pano, midway up pano (u,v)=(256,0)
    #         [-np.pi,0]
    #     ])

    # points_sph = zind_cartesian_to_sphere(points_cart)
    # import pdb; pdb.set_trace()


def test_zind_sphere_to_cartesian() -> None:
    """
    Imagine (H,W)=(512,1024) image
    """
    # provided as (u,v) coordinates.
    points_sph = np.array(
        [
            [-np.pi, np.pi/2], # corresponds to (u,v) = (0,0)
            [-np.pi, -np.pi/2], # corresponds to (u,v) = (0,511)
            [np.pi, -np.pi/2], # corresponds to (u,v) = (1023,511)
            [np.pi, np.pi/2], # corresponds to (u,v) = (1023,0)
            [0, 0], # corresponds to (u,v) = (512,256)
            [np.pi/2, 0], # 1/4 way from left edge of pano, midway up pano (u,v)=(256,0)
            [-np.pi,0]
        ])

    points_cart = zind_sphere_to_cartesian(points_sph)

    # fmt: off
    expected_points_cart = np.array(
        [
            [0,  1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0,  1, 0],
            [0,  0,-1],
            [1, 0, 0],
            [0, 0, 1]
       ]
    )
    # fmt: on
    assert np.allclose(points_cart, expected_points_cart)


# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_sphere_to_cartesian(points_sph: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to cartesian.

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
        List of points in cartesian coordinates [x, y, z], where the shape
        is (num_points, 3)
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
        print("Phi out of bounds")
        import pdb; pdb.set_trace()
    assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS_RAD))

    if num_coords == 2:
        rho = np.ones_like(theta)
    else:
        rho = points_sph[:, 2]

    # Validate the radial distances.
    assert np.all(np.greater(rho, 0.0))

    rho_cos_phi = rho * np.cos(phi)

    x_arr = rho_cos_phi * np.sin(theta)
    y_arr = rho * np.sin(phi)
    z_arr = -rho_cos_phi * np.cos(theta)

    return np.column_stack((x_arr, y_arr, z_arr)).reshape(output_shape)


# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def zind_pixel_to_sphere(points_pix: np.ndarray, width: int) -> np.ndarray:
    """Convert pixel coordinates into spherical coordinates from a 360 pano with a given width.

    Inverts: https://github.com/zillow/zind/blob/main/code/transformations.py#L150

    Note:
        We assume the width covers the full 360 degrees horizontally, and the height is derived
        as width/2 and covers the full 180 degrees vertical, i.e. we support mapping only on full FoV panos.

    Args:
        points_pix: array of shape (N,2) represenenting N points given in pano image coordinates [x, y],
        width: The width of the pano image (defines the azimuth scale).

    Return:
        array of shape (N,2) representing points in spherical coordinates [theta, phi], where the
        spherical point [theta=0, phi=0] maps to the image center.

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
        print(f"Found coords {np.amax(y_arr)} vs {height}")
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


def test_zind_pixel_to_sphere() -> None:
    """
    Assume (512,1024) for (H,W) of equirectangular projection of panorama.
    """
    # fmt: off
    # provided as (u,v) in image
    points_pix = np.array(
        [
            [0,0], # should be (-pi,pi/2)
            [0,511], # should be (-pi, -pi/2)
            [1023,511], # should be (pi, -pi/2)
            [1023,0] # should be (pi, pi/2)
        ])
    
    # provided as (u,v) coordinates.
    expected_points_sph = np.array(
        [
            [-np.pi, np.pi/2],
            [-np.pi, -np.pi/2],
            [np.pi, -np.pi/2],
            [np.pi, np.pi/2]
        ])
    # fmt: on
    points_sph = zind_pixel_to_sphere(points_pix, width=1024)
    assert np.allclose(points_sph, expected_points_sph)


def convert_points_px_to_worldmetric(points_px: np.ndarray, image_width: int, camera_height_m: int) -> np.ndarray:
    """Convert pixel coordinates to Cartesian coordinates with a known scale (i.e. the units are meters).

    Args:
        points_px: 2d points in pixel coordaintes

    Returns:
        points_worldmetric: 
    """

    points_sph = zind_pixel_to_sphere(points_px, width=image_width)
    points_cartesian = zind_sphere_to_cartesian(points_sph)
    points_worldmetric = zind_intersect_cartesian_with_floor_plane(points_cartesian, camera_height_m)
    return points_worldmetric



"""
See for reference: 
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/train_on_zind/convert_zind_to_horizonnet_annotations.py#L3853:31
https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/hnet_confidence/-/blob/master/evaluate_horizonnet_output.py#L734
"""

    