"""
For Orthographic camera. Pixel uncertainty is related to the cosine of the angle between a W/D/O and the
ray from camera to W/D/O midpoint.
"""

import numpy as np

from salve.common.pano_data import WDO


def compute_width_uncertainty(pano_wd: WDO) -> float:
    """Compute uncertainty scaling factor on measurement.

    Note: accurate only for a orthographic camera (subject to depth and focal length for a perspective camera).
    """
    cam_center = np.zeros(2)
    ray_to_camera = cam_center - pano_wd.centroid

    # pointing CW from pt 1 to pt2
    wdo_normal = -pano_wd.get_wd_normal_2d()

    theta_deg = compute_relative_angle(ray_to_camera, wdo_normal)
    uncertainty_factor = 1 / np.cos(np.deg2rad(theta_deg))
    return np.absolute(uncertainty_factor)


def test_compute_width_uncertainty_no_uncertainty1():
    """Fronto parallel, towards (0,1)."""
    pano_wd = WDO(global_Sim2_local=None, pt1=(-2, 4), pt2=(2, 4), bottom_z=-np.nan, top_z=np.nan, type="opening")
    uncertainty_factor = compute_width_uncertainty(pano_wd)
    assert np.isclose(uncertainty_factor, 1.0)


def test_compute_width_uncertainty_no_uncertainty2():
    """Fronto parallel, but towards (1,1)."""
    pano_wd = WDO(global_Sim2_local=None, pt1=(2, 3), pt2=(3, 2), bottom_z=-np.nan, top_z=np.nan, type="opening")
    uncertainty_factor = compute_width_uncertainty(pano_wd)
    assert np.isclose(uncertainty_factor, 1.0)


def test_compute_width_uncertainty_some_uncertainty():
    """Fronto parallel, but towards (1,1)."""
    pano_wd = WDO(global_Sim2_local=None, pt1=(-3, 2), pt2=(-3, 3), bottom_z=-np.nan, top_z=np.nan, type="opening")
    uncertainty_factor = compute_width_uncertainty(pano_wd)
    assert np.isclose(uncertainty_factor, 1.3017, atol=1e-3)

    # now, provide slightly less tilt on the WDO
    pano_wd = WDO(global_Sim2_local=None, pt1=(-3.0, 2), pt2=(-2.9, 3), bottom_z=-np.nan, top_z=np.nan, type="opening")
    uncertainty_factor = compute_width_uncertainty(pano_wd)
    # should be slightly less uncertainty now
    assert np.isclose(uncertainty_factor, 1.2144, atol=1e-3)


def compute_relative_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in degrees between two 2d vectors. Both vectors must be of float type."""
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def test_compute_relative_angle() -> None:
    """ """
    v1 = np.array([1.0, 1])
    v2 = np.array([1.0, 0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 45)

    v1 = np.array([1.0, -1])
    v2 = np.array([1.0, 0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 45)

    v1 = np.array([0, 1.0])
    v2 = np.array([1, 0.0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 90)

    v1 = np.array([0, -1.0])
    v2 = np.array([1, 0.0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 90)

    v1 = np.array([1.0, 0])
    v2 = np.array([-1.0, 0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 180)

    v1 = np.array([1.0, 0])
    v2 = np.array([1.0, 0])
    angle_deg = compute_relative_angle(v1, v2)
    assert np.isclose(angle_deg, 0)


def test_compute_relative_angle2() -> None:
    """ """
    vec1 = np.array([5.0, 0])
    vec2 = np.array([0.0, 9])

    angle_deg = compute_relative_angle(vec1, vec2)
    assert angle_deg == 90.0
