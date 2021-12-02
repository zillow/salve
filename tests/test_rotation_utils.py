
"""Unit tests on 2d/3d rotation matrix utilities."""

import numpy as np

import afp.utils.rotation_utils as rotation_utils


def test_rotmat2d() -> None:
    """ """
    for _ in range(1000):

        theta = np.random.rand() * 360
        R = rotation_utils.rotmat2d(theta)
        computed = R.T @ R
        #print(np.round(computed, 1))
        expected = np.eye(2)
        assert np.allclose(computed, expected)


def test_angle_is_equal() -> None:
    """
    """
    angle1 = -177.8
    angle2 =  179.5
    assert rotation_utils.angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -170
    angle2 = 170
    assert not rotation_utils.angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -170
    angle2 = 180
    assert rotation_utils.angle_is_equal(angle1, angle2, atol=10.0)

    angle1 = 5
    angle2 = 11
    assert not rotation_utils.angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -5
    angle2 = -11
    assert not rotation_utils.angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -5
    angle2 = -9
    assert rotation_utils.angle_is_equal(angle1, angle2, atol=5.0)


def test_wrap_angle_deg() -> None:
    """ """
    angle1 = 180
    angle2 = -180
    orientation_err = rotation_utils.wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -180
    angle2 = 180
    orientation_err = rotation_utils.wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -45
    angle2 = -47
    orientation_err = rotation_utils.wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 1
    angle2 = -1
    orientation_err = rotation_utils.wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 10
    angle2 = 11.5
    orientation_err = rotation_utils.wrap_angle_deg(angle1, angle2)
    assert orientation_err == 1.5
