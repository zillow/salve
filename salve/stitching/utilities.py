""" TODO: ADD DOCSTRING """

from copy import deepcopy
from typing import Any, Dict, List

from salve.stitching.constants import WDO_CODE
from salve.stitching.models.feature2d import Feature2dU


def get_dwo_edge_feature2ds_from_prediction(preds: Dict[str, Any], height: float) -> List[List[Feature2dU]]:
    """TODO

    Args:
        preds: TODO
        height: TODO

    Returns:
        features: TODO
    """
    features = []
    for wdo in preds["wdo"][0]:
        type = WDO_CODE[int(wdo[0]) - 1]
        confidence = wdo[1]
        if confidence > 0.5:
            features.append([Feature2dU(u=wdo[2], feature_type=type), Feature2dU(u=wdo[4], feature_type=type)])
    return features
