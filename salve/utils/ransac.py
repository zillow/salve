"""RANSAC based Sim(3) pose alignment."""

import copy
import math
from typing import List, Optional, Tuple

import numpy as np
import gtsfm.utils.geometry_comparisons as gtsfm_geometry_comparisons
from gtsam import Pose3, Similarity3


DEFAULT_RANSAC_ALIGNMENT_DELETE_FRAC = 0.33


def ransac_align_poses_sim3_ignore_missing(
    aTi_list_ref: List[Optional[Pose3]],
    bTi_list_est: List[Optional[Pose3]],
    num_iters: int = 1000,
    delete_frac: float = DEFAULT_RANSAC_ALIGNMENT_DELETE_FRAC,
    verbose: bool = False,
) -> Tuple[List[Optional[Pose3]], Similarity3]:
    """Align pose graphs by estimating a Similarity(3) transformation, while accounting for outliers in the pose graph.

    Args:
        aTi_list_ref: pose graph 1 (reference/target to align to).
        bTi_list_est: pose graph 2 (to be aligned).
        num_iters: number of RANSAC iterations to execture.
        delete_frac: what percent of data to remove when fitting a single hypothesis.
        verbose: whether to print out information about each iteration.

    Returns:
        best_aligned_bTi_list_est_full: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame). Rather than
            representing an aligned subset, this represents the entire aligned pose graph.
        best_aSb: Similarity(3) object that aligns the two pose graphs (best among all hypotheses).
    """
    best_aligned_bTi_list_est = None
    best_aSb = None
    best_trans_error = float("inf")
    best_rot_error = float("inf")

    valid_idxs = [i for i, bTi in enumerate(bTi_list_est) if bTi is not None]
    num_to_delete = math.ceil(delete_frac * len(valid_idxs))
    if len(valid_idxs) - num_to_delete < 2:
        # run without RANSAC! want at least 2 frame pairs for good alignment.
        aligned_bTi_list_est, aSb = gtsfm_geometry_comparisons.align_poses_sim3_ignore_missing(
            aTi_list_ref, bTi_list_est
        )
        return aligned_bTi_list_est, aSb

    # randomly delete some elements
    for _ in range(num_iters):

        aTi_list_ref_subset = copy.deepcopy(aTi_list_ref)
        bTi_list_est_subset = copy.deepcopy(bTi_list_est)

        # randomly delete 33% of the poses
        delete_idxs = np.random.choice(a=valid_idxs, size=num_to_delete, replace=False)
        for del_idx in delete_idxs:
            bTi_list_est_subset[del_idx] = None

        aligned_bTi_list_est, aSb = gtsfm_geometry_comparisons.align_poses_sim3_ignore_missing(
            aTi_list_ref_subset, bTi_list_est_subset
        )

        # evaluate inliers.
        rot_error, trans_error, _, _ = compute_pose_errors_3d(aTi_list_ref, aligned_bTi_list_est)
        if verbose:
            print("Deleted ", delete_idxs, f" -> trans_error {trans_error:.1f}, rot_error {rot_error:.1f}")

        if trans_error <= best_trans_error and rot_error <= best_rot_error:
            if verbose:
                print(f"\tFound better trans error {trans_error:.2f} < {best_trans_error:.2f}")
                print(f"\tFound better rot error {rot_error:.2f} < {best_rot_error:.2f}")
            best_aligned_bTi_list_est = aligned_bTi_list_est
            best_aSb = aSb
            best_trans_error = trans_error
            best_rot_error = rot_error

    # now go back and transform the full, original list (not just a subset).
    best_aligned_bTi_list_est_full = [None] * len(bTi_list_est)
    for i, bTi_ in enumerate(bTi_list_est):
        if bTi_ is None:
            continue
        best_aligned_bTi_list_est_full[i] = best_aSb.transformFrom(bTi_)
    return best_aligned_bTi_list_est_full, best_aSb


def compute_pose_errors_3d(
    aTi_list_gt: List[Pose3], aligned_bTi_list_est: List[Optional[Pose3]], verbose: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute average pose errors over all cameras (separately in rotation and translation).

    Note: pose graphs must already be aligned.

    Args:
        aTi_list_gt: ground truth 3d pose graph.
        aligned_bTi_list_est: estimated pose graph aligned to the ground truth's "a" frame.
        verbose: whether to print out information about pose inputs/errors.

    Returns:
        mean_rot_err: average rotation error per camera, measured in degrees.
        mean_trans_err: average translation error per camera.
        rot_errors: array of (K,) rotation errors, measured in degrees.
        trans_errors: array of (K,) translation errors.
    """
    if verbose:
        print("aTi_list_gt: ", aTi_list_gt)
        print("aligned_bTi_list_est", aligned_bTi_list_est)

    rotation_errors = []
    translation_errors = []
    for (aTi, aTi_) in zip(aTi_list_gt, aligned_bTi_list_est):
        if aTi is None or aTi_ is None:
            continue
        rot_err = gtsfm_geometry_comparisons.compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation())
        trans_err = np.linalg.norm(aTi.translation() - aTi_.translation())

        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)

    if verbose:
        print("Rotation Errors: ", np.round(rotation_errors, 1))
        print("Translation Errors: ", np.round(translation_errors, 1))
    mean_rot_err = np.mean(rotation_errors)
    mean_trans_err = np.mean(translation_errors)
    return mean_rot_err, mean_trans_err, rotation_errors, translation_errors

