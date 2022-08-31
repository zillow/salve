"""Utility for Sim(3) estimation between two point clouds."""

from typing import Optional, Tuple

import numpy as np
from gtsam import Point3Pairs, Similarity3

import salve.utils.rotation_utils as rotation_utils
from salve.common.sim2 import Sim2


def align_points_sim3(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[Optional[Sim2], np.ndarray]:
    """
    Args:
        pts_a: target/reference
        pts_b: source/query
    """
    if pts_a.shape != pts_b.shape:
        return None, np.zeros_like(pts_a)

    if pts_a.shape[1] != 3 or pts_b.shape[1] != 3:
        raise RuntimeError(f"Input point clouds were of shape {pts_a.shape}, but should have been (N,3)")

    ab_pairs = Point3Pairs(list(zip(pts_a, pts_b)))

    aSb = Similarity3.Align(ab_pairs)
    # TODO: in latest wheel
    # pts_a_ = aSb.transformFrom(pts_b)

    aRb = aSb.rotation().matrix()
    atb = aSb.translation()
    asb = aSb.scale()
    pts_a_ = asb * (pts_b @ aRb.T + atb)

    # Convert Similarity3 to Similarity2
    aSb = Sim2(R=aRb[:2, :2], t=atb[:2], s=asb)

    computed = aSb.rotation.T @ aSb.rotation
    expected = np.eye(2)
    # print(computed, expected)

    if not np.allclose(computed, expected, atol=0.05):
        print("Renormalizing Sim2:", aSb)
        aSb = reorthonormalize_sim2(aSb)

    # assert np.allclose(computed, expected, atol=0.05)

    return aSb, pts_a_


def reorthonormalize_sim2(i2Ti1: Sim2) -> Sim2:
    """ """
    R = i2Ti1.rotation

    # import pdb; pdb.set_trace()

    print("(0,0) entry said: ", np.rad2deg(np.arccos(R[0, 0])))
    print("(1,0) entry said: ", np.rad2deg(np.arcsin(R[1, 0])))

    theta_rad = np.arctan2(R[1, 0], R[0, 0])
    theta_deg = np.rad2deg(theta_rad)

    print("Combination said: ", theta_deg)

    R_ = rotation_utils.rotmat2d(theta_deg)

    i2Ti1_ = Sim2(R_, i2Ti1.translation, i2Ti1.scale)

    expected = np.eye(2)
    computed = i2Ti1_.rotation.T @ i2Ti1_.rotation
    assert np.allclose(expected, computed, atol=1e-5)

    return i2Ti1_

