
"""
Data structure to hold an SfM reconstruction, produced by OpenMVG or OpenSfM.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict

import numpy as np
from gtsam import Pose3


@dataclass
class SfmReconstruction:
    """
    Modeleed after types.Reconstruction from OpenSfM.
    """
    camera: SimpleNamespace
    pose_dict: Dict[int, Pose3]
    points: np.ndarray
    rgb: np.ndarray

    @property
    def wTi_list(self) -> np.ndarray:
        """ """
        N = max(self.pose_dict.values()) + 1
        wTi_list = [reconstruction.pose_dict.get(i, None) for i in range(N)]
        return wTi_list
