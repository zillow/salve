

import numpy as np


def rotmat2d(theta_deg: float) -> np.ndarray:
    """Generate 2x2 rotation matrix, given a rotation angle in degrees."""
    theta_rad = np.deg2rad(theta_deg)

    s = np.sin(theta_rad)
    c = np.cos(theta_rad)

    # fmt: off
    R = np.array(
        [
            [c, -s],
            [s, c]
        ]
    )
    # fmt: on
    return R


def test_rotmat2d() -> None:
    """ """
    for _ in range(1000):

        theta = np.random.rand() * 360
        R = rotmat2d(theta)
        computed = R.T @ R
        #print(np.round(computed, 1))
        expected = np.eye(2)
        assert np.allclose(computed, expected)


def rotmat2theta_deg(R: np.ndarray) -> float:
    """Recover the rotation angle `theta` (in degrees) from the 2d rotation matrix.
    Note: the first column of the rotation matrix R provides sine and cosine of theta,
        since R is encoded as [c,-s]
                              [s, c]
    We use the following identity: tan(theta) = s/c = (opp/hyp) / (adj/hyp) = opp/adj
    """
    c, s = R[0, 0], R[1, 0]
    theta_rad = np.arctan2(s, c)
    return float(np.rad2deg(theta_rad))


def wrap_angle_deg(angle1: float, angle2: float):
    """
    https://stackoverflow.com/questions/28036652/finding-the-shortest-distance-between-two-angles/28037434
    """
    # mod n will wrap x to [0,n)
    diff = (angle2 - angle1 + 180) % 360 - 180
    if diff < -180:
        return np.absolute(diff + 360)
    else:
        return np.absolute(diff)


def angle_is_equal(angle1: float, angle2: float, atol: float) -> bool:
    """Calculate shortest distance betwen two angles, provided in degrees.

    See: https://stackoverflow.com/questions/28036652/finding-the-shortest-distance-between-two-angles/28037434

    Works for angles in the range [-360,360], but we use this only for Sim(2) angles that are spit out of np.arctan2
    thus range is limited to [-180,180]

    Args:
        angle1: angle 1 (in degrees), in [-360,360]
        angle2: angle 2 (in degrees), in [-360,360]
    """
    #wrap that result to the range [-180, 179)
    diff = ( angle2 - angle1 + 180 ) % 360 - 180
    if diff < -180:
        # do nothing
        diff = diff + 360

    return np.absolute(diff) <= atol
  

def test_angle_is_equal() -> None:
    """
    """
    angle1 = -177.8
    angle2 =  179.5
    import pdb; pdb.set_trace()
    assert angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -170
    angle2 = 170
    assert not angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -170
    angle2 = 180
    assert angle_is_equal(angle1, angle2, atol=10.0)

    angle1 = 5
    angle2 = 11
    assert not angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -5
    angle2 = -11
    assert not angle_is_equal(angle1, angle2, atol=5.0)

    angle1 = -5
    angle2 = -9
    assert angle_is_equal(angle1, angle2, atol=5.0)


def test_wrap_angle_deg() -> None:
    """ """
    angle1 = 180
    angle2 = -180
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -180
    angle2 = 180
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -45
    angle2 = -47
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 1
    angle2 = -1
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 10
    angle2 = 11.5
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 1.5
