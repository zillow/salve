""" """
import copy

import numpy as np
from gtsam import Pose3, Rot3

import salve.utils.ransac as ransac_utils


def test_ransac_align_poses_sim3_ignore_missing_pureidentity() -> None:
    """Ensure that for identity poses, and thus identity Similarity(3), we get back exactly what we started with."""

    aTi_list = [
        Pose3(
            Rot3(np.array([[0.771176, -0.636622, 0], [0.636622, 0.771176, 0], [0, 0, 1]])),
            t=np.array([6.94918, 2.4749, 0]),
        ),
        Pose3(
            Rot3(
                np.array([[0.124104, -0.992269, 0], [0.992269, 0.124104, 0], [0, 0, 1]]),
            ),
            t=np.array([6.06848, 4.57841, 0]),
        ),
        Pose3(
            Rot3(
                np.array([[0.914145, 0.405387, 0], [-0.405387, 0.914145, 0], [0, 0, 1]]),
            ),
            t=np.array([6.47869, 5.29594, 0]),
        ),
        Pose3(
            Rot3(np.array([[0.105365, -0.994434, 0], [0.994434, 0.105365, 0], [0, 0, 1]])),
            t=np.array([5.59441, 5.22469, 0]),
        ),
        Pose3(
            Rot3(np.array([[-0.991652, -0.12894, 0], [0.12894, -0.991652, 0], [0, 0, 1]])),
            t=np.array([7.21399, 5.41445, 0]),
        ),
    ]
    # make twice as long
    aTi_list = aTi_list + aTi_list

    bTi_list = copy.deepcopy(aTi_list)

    aligned_bTi_list_est, aSb = ransac_utils.ransac_align_poses_sim3_ignore_missing(aTi_list, bTi_list)

    for aTi, aTi_ in zip(aTi_list, aligned_bTi_list_est):
        assert np.allclose(aTi.rotation().matrix(), aTi_.rotation().matrix(), atol=1e-3)
        assert np.allclose(aTi.translation(), aTi_.translation(), atol=1e-3)


def test_ransac_align_poses_sim3_ignore_missing() -> None:
    """Unit test for simple case of 3 poses (one is an outlier with massive translation error.)"""

    aTi_list = [
        None,
        Pose3(Rot3(), np.array([50, 0, 0])),
        Pose3(Rot3(), np.array([0, 10, 0])),
        Pose3(Rot3(), np.array([0, 0, 20])),
        None,
    ]

    # below was previously in b's frame. Has a bit of noise compared to pose graph above.
    bTi_list = [
        None,
        Pose3(Rot3(), np.array([50.1, 0, 0])),
        Pose3(Rot3(), np.array([0, 9.9, 0])),
        Pose3(Rot3(), np.array([0, 0, 2000])),
        None,
    ]

    aligned_bTi_list_est, aSb = ransac_utils.ransac_align_poses_sim3_ignore_missing(aTi_list, bTi_list)
    assert np.isclose(aSb.scale(), 1.0, atol=1e-2)
    assert np.allclose(aligned_bTi_list_est[1].translation(), np.array([50.0114, 0.0576299, 0]), atol=1e-3)
    assert np.allclose(aligned_bTi_list_est[2].translation(), np.array([-0.0113879, 9.94237, 0]), atol=1e-3)


if __name__ == "__main__":
    test_ransac_align_poses_sim3_ignore_missing()
