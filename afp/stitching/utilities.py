from copy import deepcopy
from typing import Dict, List

from stitching.constants import WDO_CODE
from stitching.models.feature2d import Feature2dU


def get_dwo_edge_feature2ds_from_prediction(preds: dict, height: float) -> List[List[Feature2dU]]:
    features = []
    for wdo in preds["wdo"][0]:
        type = WDO_CODE[int(wdo[0]) - 1]
        confidence = wdo[1]
        if confidence > 0.5:
            features.append([Feature2dU(u=wdo[2], feature_type=type), Feature2dU(u=wdo[4], feature_type=type)])
    return features
