"""Data structure to hold an SfM reconstruction, as produced by OpenMVG or OpenSfM."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict

import numpy as np
from gtsam import Pose3


@dataclass
class SfmReconstruction:
    """Container for SfM results, including camera parameters, camera poses, 3d points, and their colors.

    Modeled after types.Reconstruction from OpenSfM.
    (see https://github.com/mapillary/OpenSfM/blob/main/opensfm/types.py#L25)
    """
    camera: SimpleNamespace
    pose_dict: Dict[int, Pose3]
    points: np.ndarray
    rgb: np.ndarray

    @property
    def wTi_list(self) -> np.ndarray:
        """Convert dictionary of poses to an ordered list of poses (some of which may be None)."""
        N = max(self.pose_dict.values()) + 1
        wTi_list = [self.pose_dict.get(i, None) for i in range(N)]
        return wTi_list
