"""Utility for SE(2) estimation between two point clouds."""

from typing import Optional, Tuple

import numpy as np
from gtsam import Rot2, Point2Pairs, Pose2

from salve.common.sim2 import Sim2


def align_points_SE2(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[Optional[Sim2], np.ndarray]:
    r"""
    See https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Pose2.cpp#L311-L330
    for a derivation.

    Args:
        pts_a: array of shape (N,2) as target/reference.
        pts_b: array of shape (N,2) as source/query.

    Returns:
        aSb: estimated SE(2) transformation.
        pts_a_: array of shape (N,2) representing source points, now aligned to reference.
    """
    n = pts_a.shape[0]
    assert n == pts_b.shape[0]

    # We need at least 2 pairs.
    if n < 2:
        return None

    if pts_a.shape[1] != 2 or pts_b.shape[1] != 2:
        raise RuntimeError(f"Input point clouds were of shape {pts_a.shape}, but should have been (N,2)")

    ab_pairs = Point2Pairs(list(zip(pts_a, pts_b)))
    aTb = Pose2.Align(ab_pairs)

    aSb = Sim2(R=aTb.rotation().matrix(), t=aTb.translation(), s=1.0)

    pts_a_ = (pts_b @ aTb.rotation().matrix().T + aTb.translation())
    return aSb, pts_a_
