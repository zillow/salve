"""
Utilities and classes for working with RCNN predictions
These come from "rmx-dwo-rcnn_predictions", representing RCNN DWO predictions.

Reference for RCNN: https://www.zillow.com/tech/training-models-to-detect-windows-doors-in-panos/
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class RcnnDwoPred:
    """(x,y) are normalized to [0,1]

    Args:
        category:
        prob:
        xmin:
        ymin:
        xmax:
        ymax:
    """

    category: int
    prob: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_json(cls, json_data: Any) -> "RcnnDwoPred":
        """given 6-tuple"""

        if len(json_data) != 6:
            raise RuntimeError("Data schema violated for RCNN DWO prediction.")

        category, prob, xmin, ymin, xmax, ymax = json_data
        return cls(category, prob, xmin, ymin, xmax, ymax)


@dataclass
class PanoStructurePredictionRmxDwoRCNN:
    """
    should be [2.0, 0.999766529, 0.27673912, 0.343023, 0.31810075, 0.747359931
            class prob x y x y

    make sure the the prob is not below 0.5 (may need above 0.1)

    3 clases -- { 1, 2, }
    """

    dwo_preds: List[RcnnDwoPred]

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxDwoRCNN":
        """ """
        dwo_preds = []
        for dwo_data in json_data[0]:
            dwo_preds += [RcnnDwoPred.from_json(dwo_data)]

        return cls(dwo_preds=dwo_preds)
