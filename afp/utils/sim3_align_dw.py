from typing import Optional, Tuple

import gtsam
import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Point3, Pose2, Pose3, Point3Pairs, Rot2, Rot3, Similarity3, Unit3

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

    R_ = rotmat2d(theta_deg)

    i2Ti1_ = Sim2(R_, i2Ti1.translation, i2Ti1.scale)

    expected = np.eye(2)
    computed = i2Ti1_.rotation.T @ i2Ti1_.rotation
    assert np.allclose(expected, computed, atol=1e-5)

    return i2Ti1_


def test_reorthonormalize():
    """ """
    # from afp.common.pano_data import generate_Sim2_from_floorplan_transform
    pano3_data = {
        "translation": [0.01553549307166846, -0.002272521859178478],
        "rotation": -352.5305535406924,
        "scale": 0.4042260417272217,
    }
    pano4_data = {"translation": [0.0, 0.0], "rotation": 0.0, "scale": 0.4042260417272217}

    global_SIM2_i3 = generate_Sim2_from_floorplan_transform(pano3_data)
    global_SIM2_i4 = generate_Sim2_from_floorplan_transform(pano4_data)

    import pdb

    pdb.set_trace()

    i2Ti1_gt = global_SIM2_i4.inverse().compose(global_SIM2_i3)

    reorthonormalize_sim2(i2Ti1_gt)


def test_align_points_sim3_horseshoe() -> None:
    """Ensure align_points_sim3() works properly."""
    # fmt: off
    # small horseshoe
    pts_a = np.array(
        [
            [3, 1, 0],
            [1, 1, 0],
            [1, 3, 0],
            [3, 3, 0]
        ]
    ).tolist()

    # large horseshoe
    pts_b = np.array(
        [
            [3, 1, 10],
            [-1, 1, 10],
            [-1, 5, 10],
            [3, 5, 10]
        ]
    ).tolist()

    # fmt: on
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
    assert np.allclose(aSb.translation, np.array([3, 1]))


def align_points_SE2(pts_a: np.ndarray, pts_b: np.ndarray) -> Optional[Pose2]:
    """
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
        pts_a
        pts_b

    Returns:
    """
    # Point2Pair ab_pt_pairs
    # aTb = gtsam.Pose2.align(ab_pts_pairs)

    n = pts_a.shape[0]
    assert n == pts_b.shape[0]

    # we need at least 2 pairs
    if n < 2:
        return None

    # calculate centroids
    cp = np.zeros(2)
    cq = np.zeros(2)

    for pt_a, pt_b in zip(pts_a, pts_b):
        cp += pt_a
        cq += pt_b

    f = 1.0 / n
    cp *= f
    cq *= f

    # calculate cos and sin
    c, s = 0, 0

    for pt_a, pt_b in zip(pts_a, pts_b):
        dp = pt_a - cp
        dq = pt_b - cq
        c += dp[0] * dq[0] + dp[1] * dq[1]
        s += -dp[1] * dq[0] + dp[0] * dq[1]

    # calculate angle and translation
    theta = np.arctan2(s, c)
    R = Rot2.fromAngle(theta)
    t = cq - R.matrix() @ cp

    aTb = Pose2(R, t)
    return aTb


def test_align_points_SE2() -> None:
    """
    Two horseshoes of the same size, just rotated and translated.

    TODO: fix bug in GTSAM where the inverse() object bTa is inappropriately returned for ab pairs.
    """
    # fmt: off
    pts_a = np.array(
        [
            [3, 1],
            [1, 1],
            [1, 3],
            [3, 3]
        ])
    pts_b = np.array(
        [
            [1, -3],
            [1, -5],
            [-1, -5],
            [-1, -3]
        ])

    # fmt: on
    bTa = align_points_SE2(pts_a, pts_b)
    assert bTa is not None

    for pt_a, pt_b in zip(pts_a, pts_b):

        pt_b_ = bTa.transformFrom(pt_a)
        assert np.allclose(pt_b, pt_b_)
        print("match")


def test_align_points_SE2_doorway() -> None:
    """
    Check least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.

    X o -- o X

    """
    # fmt: off
    pts_a = np.array(
        [
            [-4, 2],
            [-2, 2]
        ])
    pts_b = np.array(
        [
            [-5, 2],
            [-1, 2]
        ])

    # fmt: on
    bTa = align_points_SE2(pts_a, pts_b)

    assert bTa.theta() == 0.0
    assert np.allclose(bTa.translation(), np.zeros(2))


def test_align_points_SE2_doorway_rotated() -> None:
    """
    Check least-squares fit, for endpoints of a large doorway and small doorway, in presence of noise.
    
    X
    |
    | o -- o
    X

    """
    # fmt: off
    pts_a = np.array(
        [
            [7, 3],
            [9, 3]
        ])
    pts_b = np.array(
        [
            [5, 2],
            [5, 6]
        ])

    # fmt: on
    bTa = align_points_SE2(pts_a, pts_b)
    
    expected_pta1 = np.array([5., 3., 1.])
    assert np.allclose(expected_pta1, bTa.matrix() @ np.array([7,3,1]))

    expected_pta2 = np.array([5., 5., 1.])
    assert np.allclose(expected_pta2, bTa.matrix() @ np.array([9,3,1]))


if __name__ == "__main__":
    # test_align_points_sim3_horseshoe()

    # test_rotmat2d()
    # test_reorthonormalize()

    #test_align_points_SE2()
    #test_align_points_SE2_doorway()
    test_align_points_SE2_doorway_rotated()
