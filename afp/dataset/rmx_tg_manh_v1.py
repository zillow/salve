
"""
Corresponds to model output for the "rmx-tg-manh-v1_predictions"

Represents total (visible) geometry with Manhattanization shape post processing
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PanoStructurePredictionRmxTgManhV1:
    """
    from `rmx-tg-manh-v1_predictions` results.

    Note: Total geometry with Manhattan world assumptions is not very useful, as Manhattanization is
    too strong of an assumption for real homes.
    """

    ceiling_height: float
    floor_height: float
    corners_in_uv: np.ndarray  # (N,2)
    wall_wall_probabilities: np.ndarray  # (N,1)

    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """Render the predicted wall corners onto the equirectangular projection, for visualization.

        Args:
            img_h: image height (in pixels).
            img_w: image width (in pixels).
        """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:, 0] *= img_w
        uv[:, 1] *= img_h

        floor_uv = uv[::2]
        ceiling_uv = uv[1::2]

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker="o")
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker="o")

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxTgManhV1":
        """Generate an object from JSON data loaded as a dictionary.

        Dictionary with keys:
            'ceiling_height', 'floor_height', 'corners_in_uv', 'wall_wall_probabilities'
        """
        return cls(
            ceiling_height=json_data["room_shape"]["ceiling_height"],
            floor_height=json_data["room_shape"]["floor_height"],
            corners_in_uv=np.array(json_data["room_shape"]["corners_in_uv"]),
            wall_wall_probabilities=np.array(json_data["room_shape"]["wall_wall_probabilities"]),
        )
