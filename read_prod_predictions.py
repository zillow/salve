"""
Converts an inference result to PanoData and PoseGraph2d objects. Also supports rendering the inference
result with oracle pose.
"""

import copy
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import argoverse.utils.json_utils as json_utils
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

import afp.common.posegraph2d as posegraph2d
import afp.dataset.zind_data as zind_data
from afp.common.pano_data import PanoData, WDO
from afp.common.posegraph2d import PoseGraph2d
from afp.dataset.rmx_madori_v1 import PanoStructurePredictionRmxMadoriV1
from afp.dataset.rmx_tg_manh_v1 import PanoStructurePredictionRmxTgManhV1
from afp.dataset.rmx_dwo_rcnn import PanoStructurePredictionRmxDwoRCNN


MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
    "rmx-dwo-rcnn_predictions",  #  RCNN DWO predictions
    "rmx-joint-v1_predictions",  # Older version joint model prediction
    "rmx-manh-joint-v2_predictions",  # Older version joint model prediction + Manhattanization shape post processing
    "rmx-rse-v1_predictions",  # Basic HNet trained with production shapes
    "rmx-tg-manh-v1_predictions",  # Total (visible) geometry with Manhattanization shape post processing
]
# could also try partial manhattanization (separate model) -- get link from Yuguang

RED = (1.0, 0, 0)
GREEN = (0, 1.0, 0)
BLUE = (0, 0, 1.0)

# in accordance with color scheme in afp/common/pano_data.py
WINDOW_COLOR = RED
DOOR_COLOR = GREEN
OPENING_COLOR = BLUE


def read_csv(fpath: str, delimiter: str = ",") -> List[Dict[str, Any]]:
    """Read in a .csv or .tsv file as a list of dictionaries."""
    rows = []

    with open(fpath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        for row in reader:
            rows.append(row)

    return rows


def load_inferred_floor_pose_graphs(query_building_id: str) -> None:
    """

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
    """
    # raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    # data_root = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Prediction_Prod_Model/ZInD_pred"
    
    # path to batch of unzipped prediction files, from Yuguang
    data_root = "/Users/johnlam/Downloads/zind2_john"

    # TSV contains mapping between Prod building IDs and ZinD building IDs
    tsv_fpath = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Re-processing.tsv"
    pano_mapping_tsv_fpath = "/Users/johnlam/Downloads/Yuguang_ZinD_prod_mapping_exported_panos.csv"

    pano_mapping_rows = read_csv(pano_mapping_tsv_fpath, delimiter=",")

    # Note: pano_guid is unique across the entire dataset.
    panoguid_to_panoid = {}
    for pano_metadata in pano_mapping_rows:
        pano_guid = pano_metadata["pano_guid"]
        dgx_fpath = pano_metadata["file"]
        pano_id = zind_data.pano_id_from_fpath(dgx_fpath)
        panoguid_to_panoid[pano_guid] = pano_id

    tsv_rows = read_csv(tsv_fpath, delimiter="\t")
    for row in tsv_rows:
        # building_guid = row["floor_map_guid_new"] # use for Batch 1 from Yuguang
        building_guid = row["floormap_guid_prod"] # use for Batch 2 from Yuguang
        # e.g. building_guid resembles "0a7a6c6c-77ce-4aa9-9b8c-96e2588ac7e8"

        zind_building_id = row["new_home_id"].zfill(4)

        # print("on ", zind_building_id)
        if zind_building_id != query_building_id:
            continue

        if building_guid == "":
            print(f"Invalid building_guid, skipping ZinD Building {zind_building_id}...")
            return None

        print(f"On ZinD Building {zind_building_id}")
        # if int(zind_building_id) not in [7, 16, 14, 17, 24]:# != 1:
        #     continue

        pano_guids = [
            Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/*")
        ]

        floor_map_json_fpath = f"{data_root}/{building_guid}/floor_map.json"
        if not Path(floor_map_json_fpath).exists():
            print(f"JSON file missing for {zind_building_id}")
            return None

        floor_map_json = json_utils.read_json_file(floor_map_json_fpath)
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

            img_resized = cv2.resize(img, (1024, 512))
            img_h, img_w, _ = img_resized.shape
            # plt.imshow(img_resized)

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

            model_names = ["rmx-madori-v1_predictions"]
            # plot the image in question
            for model_name in model_names:
                print(f"\tLoaded {model_name} prediction for Pano {i}")
                model_prediction_fpath = (
                    f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
                )
                if not Path(model_prediction_fpath).exists():
                    print(
                        "Home too old, no Madori predictions currently available for this building id (Yuguang will re-compute later).",
                        building_guid,
                        zind_building_id,
                    )
                    # skip this building.
                    return None

                prediction_data = json_utils.read_json_file(model_prediction_fpath)

                if model_name == "rmx-madori-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                    if pred_obj is None:  # malformatted pred for some reason
                        continue
                    # pred_obj.render_layout_on_pano(img_h, img_w)
                    pano_data = pred_obj.convert_to_pano_data(
                        img_h, img_w, pano_id=i, gt_pose_graph=gt_pose_graph, img_fpath=img_fpath
                    )
                    floor_pose_graphs[floor_id].nodes[i] = pano_data

                elif model_name == "rmx-dwo-rcnn_predictions":
                    pred_obj = PanoStructurePredictionRmxDwoRCNN.from_json(prediction_data["predictions"])
                    # if not prediction_data["predictions"] == prediction_data["raw_predictions"]:
                    #     import pdb; pdb.set_trace()
                    # print("\tDWO RCNN: ", pred_obj)
                elif model_name == "rmx-tg-manh-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxTgManhV1.from_json(prediction_data[0]["predictions"])
                    pred_obj.render_layout_on_pano(img_h, img_w)
                else:
                    continue

            # plt.title(f"Pano {i} from Building {zind_building_id}")
            # plt.tight_layout()
            # os.makedirs(f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}_bev", exist_ok=True)
            # plt.savefig(
            #     f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}_bev/{zind_building_id}_{i}.jpg", dpi=400
            # )
            # # plt.show()
            # plt.close("all")
            # plt.figure(figsize=(20, 10))

        return floor_pose_graphs

    raise RuntimeError("Unknown error loading inferred pose graphs")
    return None


def get_floor_id_from_img_fpath(img_fpath: str) -> str:
    """Fetch the corresponding embedded floor ID from a panorama file path.

    For example,
    "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg" -> "floor_01"
    """
    fname = Path(img_fpath).name
    k = fname.find("_partial")
    floor_id = fname[:k]

    return floor_id


def test_get_floor_id_from_img_fpath() -> None:
    """Verify we can fetch the floor ID from a panorama file path."""
    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg"
    floor_id = get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_01"

    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/1386/panos/floor_02_partial_room_18_pano_53.jpg"
    floor_id = get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_02"



def main() -> None:
    """
    For each of the 1575 ZinD homes, check to see if we have Rmx-Madori-V1 predictions. If we do, load
    the predictions as a PoseGraph2d object, and render overlaid with oracle pose estimation.
    """
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    rendering_save_dir = "ZinD_Inferred_GT_bridgeapi_2021_10_05_rendered_2021_10_14"
    model_name = "rmx-madori-v1_predictions"

    # Generate all possible building IDs for ZinD.
    building_ids = [str(v).zfill(4) for v in range(1575)]

    for building_id in building_ids:
        floor_pose_graphs = load_inferred_floor_pose_graphs(query_building_id=building_id)
        if floor_pose_graphs is None:
            # prediction files must have been missing, so we skip.
            continue

        for floor_id, floor_pose_graph in floor_pose_graphs.items():

            # load the GT pose graph to rip out the GT pose for each pano.
            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            floor_pose_graph.render_estimated_layout(
                show_plot=False,
                save_plot=True,
                plot_save_dir=f"{model_name}__oracle_pose",
                gt_floor_pg=gt_pose_graph,
            )

            floor_pose_graph.save_as_zind_data_json(
                save_fpath=f"{rendering_save_dir}/{building_id}/{floor_id}.json"
            )


if __name__ == "__main__":
    main()
