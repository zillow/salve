

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


def rotmat2theta_deg(R) -> float:
    """Recover the rotation angle `theta` (in degrees) from the 2d rotation matrix.
    Note: the first column of the rotation matrix R provides sine and cosine of theta,
        since R is encoded as [c,-s]
                              [s, c]
    We use the following identity: tan(theta) = s/c = (opp/hyp) / (adj/hyp) = opp/adj
    """
    c, s = R[0, 0], R[1, 0]
    theta_rad = np.arctan2(s, c)
    return float(np.rad2deg(theta_rad))
