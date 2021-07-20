

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
