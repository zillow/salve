
"""Translation direction filtering via Minimum Feedback Arc Set (MFAS).

Reference: https://github.com/borglab/gtsfm/blob/master/gtsfm/averaging/translation/averaging_1dsfm.py
"""

from typing import Dict, List, Tuple

import gtsam
import numpy as np
from gtsam import BinaryMeasurementUnit3, BinaryMeasurementsUnit3, MFAS, Unit3

import gtsfm.utils.coordinate_conversions as conversion_utils

import afp.utils.rotation_utils as rotation_utils

# Hyperparameters for 1D-SFM
# maximum number of times 1dsfm will project the Unit3's to a 1d subspace for outlier rejection
MAX_PROJECTION_DIRECTIONS = 200
OUTLIER_WEIGHT_THRESHOLD = 0.1

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01
HUBER_LOSS_K = 1.345  # default value from GTSAM

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0



def run_mfas(wRi_list_2d: List[np.ndarray], i2Ui1_dict_2d: Dict[Tuple[int,int], np.ndarray],
    robust_measurement_noise: bool = True,
    max_1dsfm_projection_directions: int = MAX_PROJECTION_DIRECTIONS,
    outlier_weight_threshold: float = OUTLIER_WEIGHT_THRESHOLD
):
    """

    Note: we don't use a dict, since we have a multigraph.
    """
    noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
    if robust_measurement_noise:
        huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
        noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

    import pdb; pdb.set_trace()
    wRi_list = [None] * len(wRi_list_2d)

    for i, wRi_2d in enumerate(wRi_list_2d):
        if wRi_2d is None:
            continue
        wRi_list[i] = rotation_utils.rot2x2_to_Rot3(wRi_2d)

    i2Ui1_dict = {}
    for (i1,i2), i2Ui1_2d in i2Ui1_dict_2d.items():
        i2Ui1_dict[(i1,i2)] = Unit3(np.array([i2Ui1_2d[0], i2Ui1_2d[1], 0]))

    import pdb; pdb.set_trace()

    # Note: all measurements are relative translation directions in the
    # world frame.

    # convert translation direction in global frame using rotations.
    w_i2Ui1_measurements = BinaryMeasurementsUnit3()
    for (i1, i2), i2Ui1 in i2Ui1_dict.items():
        if wRi_list[i2] is not None:
            w_i2Ui1_measurements.append(
                BinaryMeasurementUnit3(i2, i1, Unit3(wRi_list[i2].rotate(i2Ui1.point3())), noise_model)
            )

    # sample projection directions
    projection_directions = _sample_random_directions(max_1dsfm_projection_directions)

    # compute outlier weights using MFAS
    outlier_weights = []

    # TODO: parallelize this step.
    for direction in projection_directions:
        algorithm = MFAS(w_i2Ui1_measurements, direction)
        outlier_weights.append(algorithm.computeOutlierWeights())

    # compute average outlier weight
    avg_outlier_weights = {}
    for outlier_weight_dict in outlier_weights:
        for index_pair, weight in outlier_weight_dict.items():
            if index_pair in avg_outlier_weights:
                avg_outlier_weights[index_pair] += weight / len(outlier_weights)
            else:
                avg_outlier_weights[index_pair] = weight / len(outlier_weights)

    # filter out outlier measurements
    w_i2Ui1_inlier_measurements = BinaryMeasurementsUnit3()
    inliers = []
    outliers = []
    for w_i2Ui1 in w_i2Ui1_measurements:
        # key1 is i2 and key2 is i1 above.
        i1 = w_i2Ui1.key2()
        i2 = w_i2Ui1.key1()
        if avg_outlier_weights[(i2, i1)] < outlier_weight_threshold:
            w_i2Ui1_inlier_measurements.append(w_i2Ui1)
            inliers.append((i1, i2))
        else:
            outliers.append((i1, i2))

    import pdb; pdb.set_trace()

def _sample_random_directions(num_samples: int) -> List[Unit3]:
    """Samples num_samples Unit3 3D directions.
    The sampling is done in 2D spherical coordinates (azimuth, elevation), and then converted to Cartesian coordinates.
    Sampling in spherical coordinates was found to have precision-recall for 1dsfm outlier rejection.
    Args:
        num_samples: Number of samples required.
    Returns:
        List of sampled Unit3 directions.
    """
    sampled_azimuth = np.random.uniform(low=-np.pi, high=np.pi, size=(num_samples, 1))
    sampled_elevation = np.random.uniform(low=0.0, high=np.pi, size=(num_samples, 1))
    sampled_azimuth_elevation = np.concatenate((sampled_azimuth, sampled_elevation), axis=1)

    return conversion_utils.spherical_to_cartesian_directions(sampled_azimuth_elevation)

