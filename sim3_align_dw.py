

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Point3Pairs, Rot3, Similarity3, Unit3


def align_points_sim3(pts_a: np.ndarray, pts_b: np.ndarray):
    """
    Args:
        pts_a: target/reference
        pts_b: source/query
    """
    ab_pairs = Point3Pairs(list(zip(pts_a, pts_b)))

    aSb = Similarity3.Align(ab_pairs)
    pts_a_ = aSb.transformFrom(pts_b)

    return aSb, pts_a_


def test_align_horseshoe() -> None:
    """ """

    # small horseshoe
    pts_a = np.array(
        [
            [3,1,0],
            [1,1,0],
            [1,3,0],
            [3,3,0]
        ]).tolist()

    # large horseshoe
    pts_b = np.array(
        [
            [ 3,1,10],
            [-1,1,10],
            [-1,5,10],
            [ 3,5,10]
        ]).tolist()

    
    aSb, pts_a_ = align_points_sim3(pts_a, pts_b)



if __name__ == '__main__':
    test_align_horseshoe()