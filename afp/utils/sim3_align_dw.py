
from typing import Tuple

import gtsam
import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Point3, Pose3, Point3Pairs, Rot3, Similarity3, Unit3

from afp.utils.rotation_utils import rotmat2d


def align_points_sim3(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[Sim2, np.ndarray]:
    """
    Args:
        pts_a: target/reference
        pts_b: source/query
    """
    ab_pairs = Point3Pairs(list(zip(pts_a, pts_b)))

    aSb = Similarity3.Align(ab_pairs)
    # TODO: in latest wheel
    #pts_a_ = aSb.transformFrom(pts_b)

    aRb = aSb.rotation().matrix()
    atb = aSb.translation()
    asb = aSb.scale()
    pts_a_ = asb * (pts_b @ aRb.T + atb)

    # Convert Similarity3 to Similarity2
    aSb = Sim2(R=aRb[:2,:2], t=atb[:2], s=asb)

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

    #import pdb; pdb.set_trace()

    print("(0,0) entry said: ", np.rad2deg(np.arccos(R[0,0])))
    print("(1,0) entry said: ", np.rad2deg(np.arcsin(R[1,0])))

    theta_rad = np.arctan2(R[1,0], R[0,0])
    theta_deg = np.rad2deg(theta_rad)

    print("Combination said: ", theta_deg)

    R_ = rotmat2d(theta_deg)

    i2Ti1_ = Sim2(R_, i2Ti1.translation, i2Ti1.scale)

    expected = np.eye(2)
    computed = i2Ti1_.rotation.T @ i2Ti1_.rotation
    assert np.allclose(expected, computed, atol=1e-5)

    return i2Ti1_


def test_reorthonormalize():
    """ """
    pano3_data = {'translation': [0.01553549307166846, -0.002272521859178478], 'rotation': -352.5305535406924, 'scale': 0.4042260417272217}
    pano4_data = {'translation': [0.0, 0.0], 'rotation': 0.0, 'scale': 0.4042260417272217}

    global_SIM2_i3 = generate_Sim2_from_floorplan_transform(pano3_data)
    global_SIM2_i4 = generate_Sim2_from_floorplan_transform(pano4_data)

    import pdb; pdb.set_trace()

    i2Ti1_gt = global_SIM2_i4.inverse().compose(global_SIM2_i3)

    reorthonormalize_sim2(i2Ti1_gt)



def test_align_points_sim3_horseshoe() -> None:
    """Ensure align_points_sim3() works properly.

    """
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

    # a is the reference, and b will be transformed to a_
    aSb, pts_a_ = align_points_sim3(pts_a, pts_b)

    # fmt: off

    # shift:
    # [ 3,1]         = [6,2]       = [3,1]
    # [-1,1] + [3,1] = [2,2] * 0.5 = [1,1]
    # [-1,5]         = [2,6]       = [1,3]
    # [ 3,5]         = [6,6]       = [3,3]

    expected_pts_a_ = np.array([
        [3, 1, 0],
        [1, 1, 0],
        [1, 3, 0],
        [3, 3, 0]
    ])
    # fmt: on

    assert np.allclose(expected_pts_a_, pts_a)
    assert aSb.scale == 0.5
    assert np.allclose(aSb.rotation, np.eye(2))
    assert np.allclose(aSb.translation, np.array([3,1]))



if __name__ == '__main__':
    test_align_points_sim3_horseshoe()

    #test_rotmat2d()
    #test_reorthonormalize()

