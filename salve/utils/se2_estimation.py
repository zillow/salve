"""Utility for SE(2) estimation between two point clouds."""

from typing import Optional, Tuple

import numpy as np
from gtsam import Rot2, Pose2

from salve.common.sim2 import Sim2


def align_points_SE2(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[Optional[Sim2], np.ndarray]:
    r"""
    TODO: move this into GTSAM outright.

        It finds the angle using a linear method:
        q = Pose2::transformFrom(p) = t + R*p
        We need to remove the centroids from the data to find the rotation
        using dp=[dpx;dpy] and q=[dqx;dqy] we have
         |dqx|   |c  -s|     |dpx|     |dpx -dpy|     |c|
         |   | = |     |  *  |   |  =  |        |  *  | | = H_i*cs
         |dqy|   |s   c|     |dpy|     |dpy  dpx|     |s|
        where the Hi are the 2*2 matrices. Then we will minimize the criterion
        J = \sum_i norm(q_i - H_i * cs)
        Taking the derivative with respect to cs and setting to zero we have
        cs = (\sum_i H_i' * q_i)/(\sum H_i'*H_i)
        The hessian is diagonal and just divides by a constant, but this
        normalization constant is irrelevant, since we take atan2.
        i.e., cos ~ sum(dpx*dqx + dpy*dqy) and sin ~ sum(-dpy*dqx + dpx*dqy)
        The translation is then found from the centroids
        as they also satisfy cq = t + R*cp, hence t = cq - R*cp

    Args:
        pts_a: target/reference
        pts_b: source/query

    Returns:
        aSb: estimated SE(2) transformation.
        pts_a_: source points, now aligned to reference.
    """
    # Point2Pair ab_pt_pairs
    # aTb = gtsam.Pose2.align(ab_pts_pairs)

    n = pts_a.shape[0]
    assert n == pts_b.shape[0]

    # We need at least 2 pairs.
    if n < 2:
        return None

    if pts_a.shape[1] != 2 or pts_b.shape[1] != 2:
        raise RuntimeError(f"Input point clouds were of shape {pts_a.shape}, but should have been (N,2)")

    # Calculate centroids.
    cp = np.zeros(2)
    cq = np.zeros(2)

    for pt_a, pt_b in zip(pts_a, pts_b):
        cp += pt_a
        cq += pt_b

    f = 1.0 / n
    cp *= f
    cq *= f

    # Calculate cos and sin.
    c, s = 0, 0

    for pt_a, pt_b in zip(pts_a, pts_b):
        dp = pt_a - cp
        dq = pt_b - cq
        c += dp[0] * dq[0] + dp[1] * dq[1]
        s += -dp[1] * dq[0] + dp[0] * dq[1]

    # Calculate angle and translation.
    theta = np.arctan2(s, c)
    R = Rot2.fromAngle(theta)
    t = cq - R.matrix() @ cp

    bTa = Pose2(R, t)
    aTb = bTa.inverse()

    aSb = Sim2(R=aTb.rotation().matrix(), t=aTb.translation(), s=1.0)

    pts_a_ = (pts_b @ aTb.rotation().matrix().T + aTb.translation())
    return aSb, pts_a_
