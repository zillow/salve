
"""Utilities for working with 2d and 3d rotation matrices."""

import numpy as np
from gtsam import Rot3


def rot2x2_to_Rot3(R: np.ndarray) -> Rot3:
    """
    2x2 rotation matrix to Rot3 object
    """
    R_Rot3 = np.eye(3)
    R_Rot3[:2, :2] = R
    return Rot3(R_Rot3)


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


def wrap_angle_deg(angle1: float, angle2: float) -> float:
    """Find the minimum angular difference between two input angles.

    We must wrap around at 0 degrees and 360 degrees.

    https://stackoverflow.com/questions/28036652/finding-the-shortest-distance-between-two-angles/28037434

    Args:
        angle1: angle 1 (in degrees)
        angle2: angle 2 (in degrees)

    Returns:
        minimum angular difference (in degrees)
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
  

def rotate_polygon_about_pt(pts: np.ndarray, rotmat: np.ndarray, center_pt: np.ndarray) -> np.ndarray:
    """Rotate a polygon about a point with a given rotation matrix.

    Reference: https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/geometry.py#L13

    Args:
        pts: Array of shape (N, 3) representing a polygon or point cloud
        rotmat: Array of shape (3, 3) representing a rotation matrix
        center_pt: Array of shape (3,) representing point about which we rotate the polygon
    
    Returns:
        rot_pts: Array of shape (N, 3) representing a ROTATED polygon or point cloud
    """
    pts -= center_pt
    rot_pts = pts.dot(rotmat.T)
    rot_pts += center_pt
    return rot_pts

