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

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
import salve.dataset.zind_data as zind_data
import salve.utils.csv_utils as csv_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.rmx_madori_v1 import PanoStructurePredictionRmxMadoriV1
from salve.dataset.rmx_tg_manh_v1 import PanoStructurePredictionRmxTgManhV1
from salve.dataset.rmx_dwo_rcnn import PanoStructurePredictionRmxDwoRCNN

# Path to batch of unzipped prediction files, from Yuguang
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Prediction_Prod_Model/ZInD_pred"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/mnt/data/johnlam/zind2_john"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/home/johnlam/zind2_john"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/Users/johnlam/Downloads/zind2_john"
RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/srv/scratch/jlambert30/salve/zind2_john"

# Path to CSV w/ info about prod-->ZInD remapping.
# PANO_MAPPING_TSV_FPATH = "/home/ZILLOW.LOCAL/johnlam/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/home/johnlam/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/Users/johnlam/Downloads/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/srv/scratch/jlambert30/salve/Yuguang_ZinD_prod_mapping_exported_panos.csv"
PANO_MAPPING_TSV_FPATH = "/Users/johnlambert/Downloads/salve/Yuguang_ZinD_prod_mapping_exported_panos.csv"


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
]

IMAGE_HEIGHT_PX = 512
IMAGE_WIDTH_PX = 1024


def export_horizonnet_zind_predictions(
    query_building_id: str, raw_dataset_dir: str, predictions_data_root: str = RMX_MADORI_V1_PREDICTIONS_DIRPATH
) -> Optional[Dict[str, PoseGraph2d]]:
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

    Returns:
        floor_pose_graphs: mapping from floor_id to predicted pose graph
            (without poses, but just W/D/O predictions and layout prediction)
    """
    pano_mapping_rows = csv_utils.read_csv(PANO_MAPPING_TSV_FPATH, delimiter=",")

    # Note: pano_guid is unique across the entire dataset.
    panoguid_to_panoid = {}
    for pano_metadata in pano_mapping_rows:
        pano_guid = pano_metadata["pano_guid"]
        dgx_fpath = pano_metadata["file"]
        pano_id = zind_data.pano_id_from_fpath(dgx_fpath)
        panoguid_to_panoid[pano_guid] = pano_id

    # TSV contains mapping between Prod building IDs and ZinD building IDs
    tsv_fpath = REPO_ROOT / "ZInD_Re-processing.tsv"
    tsv_rows = csv_utils.read_csv(tsv_fpath, delimiter="\t")
    for row in tsv_rows:
        # building_guid = row["floor_map_guid_new"] # use for Batch 1 from Yuguang
        building_guid = row["floormap_guid_prod"]  # use for Batch 2 from Yuguang
        # e.g. building_guid resembles "0a7a6c6c-77ce-4aa9-9b8c-96e2588ac7e8"

        zind_building_id = row["new_home_id"].zfill(4)

        # print("on ", zind_building_id)
        if zind_building_id != query_building_id:
            continue

        import pdb; pdb.set_trace()
        if building_guid == "":
            print(f"Invalid building_guid, skipping ZinD Building {zind_building_id}...")
            return None

        print(f"On ZinD Building {zind_building_id}")
        # if int(zind_building_id) not in [7, 16, 14, 17, 24]:# != 1:
        #     continue

        pano_guids = [
            Path(dirpath).stem
            for dirpath in glob.glob(f"{predictions_data_root}/{building_guid}/floor_map/{building_guid}/pano/*")
        ]
        if len(pano_guids) == 0:
            # e.g. building '0258' is missing predictions.
            print(f"No Pano GUIDs provided for {building_guid} (ZinD Building {zind_building_id}).")
            return None

        floor_map_json_fpath = f"{predictions_data_root}/{building_guid}/floor_map.json"
        if not Path(floor_map_json_fpath).exists():
            print(f"JSON file missing for {zind_building_id}")
            return None

        floor_map_json = io_utils.read_json_file(floor_map_json_fpath)
        floor_pose_graphs = {}

        plt.figure(figsize=(20, 10))
        for pano_guid in pano_guids:

            if pano_guid not in panoguid_to_panoid:
                print(f"Missing the panorama for Building {zind_building_id} -> {pano_guid}")
                continue
            i = panoguid_to_panoid[pano_guid]

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{zind_building_id}/panos/floor*_pano_{i}.jpg")
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

            if floor_id not in floor_pose_graphs:
                # start populating the pose graph for each floor pano-by-pano
                floor_pose_graphs[floor_id] = PoseGraph2d(
                    building_id=zind_building_id,
                    floor_id=floor_id,
                    nodes={},
                    scale_meters_per_coordinate=gt_pose_graph.scale_meters_per_coordinate,
                )

            model_name = "rmx-madori-v1_predictions"
            # plot the image in question
            # for model_name in model_names:
            # print(f"\tLoaded {model_name} prediction for Pano {i}")
            model_prediction_fpath = (
                f"{predictions_data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
            )
            if not Path(model_prediction_fpath).exists():
                print(
                    "Home too old, no Madori predictions currently available for this building id (Yuguang will re-compute later).",
                    building_guid,
                    zind_building_id,
                )
                # skip this building.
                return None

            prediction_data = io_utils.read_json_file(model_prediction_fpath)

            if model_name == "rmx-madori-v1_predictions":
                pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                if pred_obj is None:  # malformatted pred for some reason
                    continue
                #
                pano_data = pred_obj.convert_to_pano_data(
                    img_h,
                    img_w,
                    pano_id=i,
                    gt_pose_graph=gt_pose_graph,
                    img_fpath=img_fpath,
                    vanishing_angle_deg=floor_map_json["panos"][pano_guid]["vanishing_angle"],
                )
                floor_pose_graphs[floor_id].nodes[i] = pano_data


            render_on_pano = False
            if render_on_pano:
                plt.imshow(img_resized)
                pred_obj.render_layout_on_pano(img_h, img_w)
                plt.title(f"Pano {i} from Building {zind_building_id}")
                plt.tight_layout()
                os.makedirs(f"prod_pred_model_visualizations_2021_10_16_bridge/{model_name}_bev", exist_ok=True)
                plt.savefig(
                    f"prod_pred_model_visualizations_2021_10_16_bridge/{model_name}_bev/{zind_building_id}_{i}.jpg",
                    dpi=400,
                )
                # plt.show()
                plt.close("all")
                plt.figure(figsize=(20, 10))

        return floor_pose_graphs

    raise RuntimeError("Unknown error loading inferred pose graphs")
    return None



if __name__ == "__main__":
    """ """
    predictions_data_root = "/srv/scratch/jlambert30/salve/zind2_john"
    raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"
    zind_building_ids = ["0715"]

    #for each home in ZInD:
    for query_building_id in zind_building_ids:
        export_horizonnet_zind_predictions(
            query_building_id=query_building_id,
            raw_dataset_dir=raw_dataset_dir,
            predictions_data_root=predictions_data_root
        )


