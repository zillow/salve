"""
Converts a HorizonNet inference result to PanoData and PoseGraph2d objects. Also supports rendering the inference
result with oracle pose.
"""

import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import gtsfm.utils.io as io_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
import salve.dataset.zind_data as zind_data
import salve.utils.csv_utils as csv_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.rmx_madori_v1 import PanoStructurePredictionRmxMadoriV1
from salve.dataset.rmx_tg_manh_v1 import PanoStructurePredictionRmxTgManhV1
from salve.dataset.rmx_dwo_rcnn import PanoStructurePredictionRmxDwoRCNN
from salve.dataset.mhnet_prediction import MHNetPanoStructurePrediction

import salve.dataset.zind_partition as zind_partition

from salve.dataset.zind_partition import DATASET_SPLITS

# Path to batch of unzipped prediction files, from Yuguang
PANO_MAPPING_TSV_FPATH = "/Users/johnlambert/Downloads/salve/Yuguang_ZinD_prod_mapping_exported_panos.csv"


REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
]

IMAGE_HEIGHT_PX = 512
IMAGE_WIDTH_PX = 1024


def export_horizonnet_zind_predictions(
    raw_dataset_dir: str, predictions_data_root: str, export_dir: str
) -> bool:
    """Load W/D/O's predicted for each pano of each floor by HorizonNet.

    TODO: rename this function, since no pose graph is loaded here.

    Note: we read in mapping from spreadsheet, mapping from their ZInD index to these guid
        https://drive.google.com/drive/folders/1A7N3TESuwG8JOpx_TtkKCy3AtuTYIowk?usp=sharing

        For example:
            "b912c68c-47da-40e5-a43a-4e1469009f7f":
            ZinD Image: /Users/johnlam/Downloads/complete_07_10_new/1012/panos/floor_01_partial_room_15_pano_19.jpg
            Prod: Image URL https://d2ayvmm1jte7yn.cloudfront.net/vrmodels/e9c3eb49-6cbc-425f-b301-7da0aff161d2/floor_map/b912c68c-47da-40e5-a43a-4e1469009f7f/pano/cf94fcb5a5/straightened.jpg # noqa
            See this corresponds to 1012 (not 109).

    Args:
        query_building_id: string representing ZinD building ID to fetch the per-floor inferred pose graphs for.
            Should be a zfilled-4 digit string, e.g. "0001"
        raw_dataset_dir:
        predictions_data_root:
        export_dir: 

    Returns:
        Boolean indicating export success for specified building.
    """
    pano_mapping_rows = csv_utils.read_csv(PANO_MAPPING_TSV_FPATH, delimiter=",")

    json_fpaths = glob.glob(f"{predictions_data_root}/*.json")
    import pdb; pdb.set_trace()
    for json_fpath in json_fpaths:

        zind_building_id, pano_fname = Path(json_fpath).stem.split("___")

        i = pano_fname.split("_")[-1]

        # Export vanishing angles.
        #vanishing_angles_dict[ int(i) ] = floor_map_json["panos"][pano_guid]["vanishing_angle"]

        img_fpaths = glob.glob(f"{raw_dataset_dir}/{zind_building_id}/panos/{pano_fname}.jpg")
        if not len(img_fpaths) == 1:
            print("\tShould only be one image for this (building id, pano id) tuple.")
            print(f"\tPano {i} was missing")
            plt.close("all")
            continue

        img_fpath = img_fpaths[0]
        img = imageio.imread(img_fpath)

        img_resized = cv2.resize(img, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
        img_h, img_w, _ = img_resized.shape

        floor_id = get_floor_id_from_img_fpath(img_fpath)

        gt_pose_graph = posegraph2d.get_gt_pose_graph(
            building_id=zind_building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
        )

        model_name = "rmx-madori-v1_predictions"

        model_prediction_fpath = json_fpath

        if not Path(model_prediction_fpath).exists():
            print(
                "Home too old, no Madori predictions currently available for this building id (Yuguang will re-compute later).",
                building_guid,
                zind_building_id,
            )
            # skip this building.
            return None

        prediction_data = io_utils.read_json_file(model_prediction_fpath)
        assert len(prediction_data) == 1

        # have to clip like zind_building_id == "0302" and i == 31
        for wdo_type in ["window", "door", "opening"]:
            prediction_data[0]["predictions"]["wall_features"][wdo_type] = np.mod(prediction_data[0]["predictions"]["wall_features"][wdo_type], 1.0).tolist()

        save_fpath = f"{export_dir}/horizon_net/{zind_building_id}/{i}.json"
        Path(save_fpath).parent.mkdir(exist_ok=True, parents=True)
        io_utils.save_json_file(save_fpath, prediction_data[0])

        if model_name == "rmx-madori-v1_predictions":
            pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
            if pred_obj is None:  # malformatted pred for some reason
                continue

        render_on_pano = True # False #  True True # np.random.rand() < 0.02 # 
        if render_on_pano:
            plt.imshow(img_resized)
            pred_obj.render_layout_on_pano(img_h, img_w)
            plt.title(f"Pano {i} from Building {zind_building_id}")
            plt.tight_layout()
            os.makedirs(f"prod_pred_model_visualizations_2022_08_16_bridge/{model_name}_bev", exist_ok=True)
            plt.savefig(
                f"prod_pred_model_visualizations_2022_08_16_bridge/{model_name}_bev/{zind_building_id}_{i}.jpg",
                dpi=400,
            )
            # plt.show()
            plt.close("all")
            plt.figure(figsize=(20, 10))

        # vanishing_angles_building_fpath = f"{export_dir}/vanishing_angle/{query_building_id}.json"
        # io_utils.save_json_file(vanishing_angles_building_fpath, vanishing_angles_dict)
        # # Success
        # return True

    return False


def get_floor_id_from_img_fpath(img_fpath: str) -> str:
    """Fetch the corresponding embedded floor ID from a panorama file path.

    For example,
    "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg" -> "floor_01"
    """
    fname = Path(img_fpath).name
    k = fname.find("_partial")
    floor_id = fname[:k]

    return floor_id


if __name__ == "__main__":
    """ """
    predictions_data_root = "/Users/johnlambert/Downloads/inference_0815"

    #raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    export_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_08_16"
    success = export_horizonnet_zind_predictions(
        raw_dataset_dir=raw_dataset_dir,
        predictions_data_root=predictions_data_root,
        export_dir=export_dir
    )

