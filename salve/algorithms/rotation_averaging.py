"""Wrapper around GTSAM's Shonan rotation averaging."""

import logging
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose2,
    LevenbergMarquardtParams,
    Pose2,
    Rot2,
    ShonanAveraging2,
    ShonanAveragingParameters2,
)

from salve.utils.rotation_utils import rotmat2theta_deg

logger = logging.getLogger(__name__)


def ShonanAveraging2_BetweenFactorPose2s_wrapper(
    i2Ri1_dict: Dict[Tuple[int, int], np.ndarray], use_huber: bool = False
) -> List[np.ndarray]:
    """Requires consecutive ordering [0,...,N-1]

    Note: Shonan will only converge for certain amounts of noise. 63 degrees is the limit to converge?

    Args:
        i2Ri1_dict: 2d relative rotations
        use_huber: whether to use a robust cost in averaging, e.g. Huber norm.

    Returns:
        wRi_list: global rotations
    """
    lm_params = LevenbergMarquardtParams.CeresDefaults()
    shonan_params = ShonanAveragingParameters2(lm_params)
    shonan_params.setUseHuber(use_huber)
    shonan_params.setCertifyOptimality(not use_huber)

    noise_model = gtsam.noiseModel.Unit.Create(3)
    between_factors = gtsam.BetweenFactorPose2s()

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():

        theta_deg = rotmat2theta_deg(i2Ri1)
        # print("Bad edges:")
        # print(f"({i1},{i2}): {theta_deg:.6f}")
        i2Ri1 = Rot2.fromDegrees(theta_deg)
        i2Ti1 = Pose2(i2Ri1, np.zeros(2))
        between_factors.append(BetweenFactorPose2(i2, i1, i2Ti1, noise_model))

    obj = ShonanAveraging2(between_factors, shonan_params)
    initial = obj.initializeRandomly()

    if use_huber:
        pmin = 2
        pmax = 2
    else:
        pmin = 2
        pmax = 100
    result_values, _ = obj.run(initial, min_p=pmin, max_p=pmax)

    wRi_list = [result_values.atRot2(i).matrix() for i in range(result_values.size())]
    return wRi_list


def globalaveraging2d(i2Ri1_dict: Dict[Tuple[int, int], Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.
    Note: run() functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
    All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
    camera in a single, global coordinate frame.

    Args:
        num_images: number of images. Since we have one pose per image, it is also the number of poses.
        i2Ri1_dict: relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
    Returns:
        Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
           `num_images`. The list may contain `None` where the global rotation could not be computed (either
           underconstrained system or ill-constrained system), or where the camera pose had no valid observation
           in the input to run().
    """
    edges = i2Ri1_dict.keys()

    if len(edges) == 0:
        return None

    num_images = max([max(i1, i2) for i1, i2 in edges]) + 1

    connected_nodes = set()
    for (i1, i2) in i2Ri1_dict.keys():
        connected_nodes.add(i1)
        connected_nodes.add(i2)

    connected_nodes = sorted(list(connected_nodes))

    # given original index, this map gives back a new temporary index, starting at 0
    reordered_idx_map = {}
    for (new_idx, i) in enumerate(connected_nodes):
        reordered_idx_map[i] = new_idx

    # now, map the original indices to reordered indices
    i2Ri1_dict_reordered = {}
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        i1_ = reordered_idx_map[i1]
        i2_ = reordered_idx_map[i2]
        i2Ri1_dict_reordered[(i1_, i2_)] = i2Ri1

    wRi_list_subset = ShonanAveraging2_BetweenFactorPose2s_wrapper(i2Ri1_dict=i2Ri1_dict_reordered)

    wRi_list = [None] * num_images
    for remapped_i, original_i in enumerate(connected_nodes):
        wRi_list[original_i] = wRi_list_subset[remapped_i]

    return wRi_list
