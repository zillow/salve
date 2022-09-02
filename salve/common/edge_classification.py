"""Data structure representing a CNN-classification result on a relative pose hypothesis (from a W/D/O alignment)."""

import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gtsfm.utils.io as io_utils
import numpy as np

from salve.common.sim2 import Sim2
import salve.common.posegraph2d as posegraph2d


@dataclass(frozen=False)
class EdgeClassification:
    """Represents a model prediction for a particular alignment hypothesis between Panorama i1 and Panorama i2.

    Note: i1 and i2 are panorama id's.
    """

    i1: int  # ID of panorama 1
    i2: int  # ID of panorama 2
    prob: float
    y_hat: int  # predicted category
    y_true: int  # true category
    pair_idx: int
    wdo_pair_uuid: str
    configuration: str  # identity or rotated


def get_edge_classifications_from_serialized_preds(
    serialized_preds_json_dir: str, allowed_wdo_types: List[str] = ["door", "window", "opening"]
) -> Dict[Tuple[str, str], List[EdgeClassification]]:
    """Convert serialized predictions into EdgeClassification objects.

    Given a directory of JSON files containing model predictions into predictions per ZinD building and per floor.

    Args:
        serialized_preds_json_dir: path to directory where model predictions (per edge) have been serialized as JSON.
            The serializations are stored per batch, and thus are mixed across ZInD buildings and floors.
        allowed_wdo_types: allowed types of semantic objects (W/D/O) to use for reconstruction. Others will be ignored.

    Returns:
        floor_edgeclassifications_dict: a mapping from (building_id, floor_id) to corresponding EdgeClassification
            measurements.
    """
    floor_edgeclassifications_dict = defaultdict(list)

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    for json_fpath in json_fpaths:

        json_data = io_utils.read_json_file(json_fpath)
        y_hat_list = json_data["y_hat"]
        y_true_list = json_data["y_true"]
        y_hat_prob_list = json_data["y_hat_probs"]
        fp0_list = json_data["fp0"]
        fp1_list = json_data["fp1"]

        for y_hat, y_true, y_hat_prob, fp0, fp1 in zip(y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list):
            # Note: not guaranteed that i1 < i2
            i1 = int(Path(fp0).stem.split("_")[-1])
            i2 = int(Path(fp1).stem.split("_")[-1])
            building_id = Path(fp0).parent.stem

            s = Path(fp0).stem.find("floor_0")
            e = Path(fp0).stem.find("_partial")
            floor_id = Path(fp0).stem[s:e]

            pair_idx = Path(fp0).stem.split("_")[1]

            is_identity = "identity" in Path(fp0).stem
            configuration = "identity" if is_identity else "rotated"

            # Rip out the WDO indices (`wdo_pair_uuid`), given a filename such as
            # `pair_3905___door_3_0_identity_floor_rgb_floor_01_partial_room_02_pano_38.jpg`
            k = Path(fp0).stem.split("___")[1].find(f"_{configuration}")
            assert k != -1
            wdo_pair_uuid = Path(fp0).stem.split("___")[1][:k]
            # split `door_3_0` to `door`
            wdo_type = wdo_pair_uuid.split("_")[0]
            if wdo_type not in allowed_wdo_types:
                continue

            floor_edgeclassifications_dict[(building_id, floor_id)] += [
                EdgeClassification(
                    i1=i1,
                    i2=i2,
                    prob=y_hat_prob,
                    y_hat=y_hat,
                    y_true=y_true,
                    pair_idx=pair_idx,
                    wdo_pair_uuid=wdo_pair_uuid,
                    configuration=configuration,
                )
            ]
    return floor_edgeclassifications_dict


def get_alignment_hypothesis_for_measurement(
    m: EdgeClassification, hypotheses_save_root: str, building_id: str, floor_id: str
) -> Sim2:
    """Given a classification result (measurement), find corresponding Sim(2) alignment JSON file on disk.

    Args:
        m:  measurement.
        hypotheses_save_root:
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.

    Returns:
        i2Si1: Similarity(2) transformation.
    """
    # label_dirname = "gt_alignment_exact"
    # fpaths = glob.glob(f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_dirname}/{m.i1}_{m.i2}.json")

    # look up the associated Sim(2) file for this prediction, by looping through the pair idxs again
    label_dirname = "gt_alignment_approx" if m.y_true else "incorrect_alignment"
    fpaths = glob.glob(
        f"{hypotheses_save_root}/{building_id}/{floor_id}"
        f"/{label_dirname}/{m.i1}_{m.i2}__{m.wdo_pair_uuid}_{m.configuration}.json"
    )
    if not len(fpaths) == 1:
        import pdb; pdb.set_trace()
    i2Si1 = Sim2.from_json(fpaths[0])
    return i2Si1
