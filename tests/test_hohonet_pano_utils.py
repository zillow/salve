"""Unit tests for HoHoNet's coordinate system transformations."""

import numpy as np

import salve.utils.hohonet_pano_utils as hohonet_pano_utils


def test_get_uni_sphere_xyz() -> None:
    """
    the -x axis is pointing towards the center pixel
    """
    sphere_xyz = hohonet_pano_utils.get_uni_sphere_xyz(H=512, W=1024)

    u, v = 0,0
    assert np.allclose(sphere_xyz[v,u], np.array([0,0,1]), atol=4e-3) # top-left pixel should point upwards @x=0

    u, v = 1023, 0
    assert np.allclose(sphere_xyz[v,u], np.array([0,0,1]), atol=4e-3) # top-right pixel should point upwards @x=0 (wrapped around)

    u, v = 0, 511
    assert np.allclose(sphere_xyz[v,u], np.array([0,0,-1]), atol=4e-3) # bottom-left pixel should point downwards @x=0
    
    u, v = 512, 256
    assert np.allclose(sphere_xyz[v,u], np.array([-1,0,0]), atol=4e-3) # center pixel of panorama points towards -x direction


