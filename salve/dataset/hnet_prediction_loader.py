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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
    "rmx-dwo-rcnn_predictions",  #  RCNN DWO predictions
    "rmx-joint-v1_predictions",  # Older version joint model prediction
    "rmx-manh-joint-v2_predictions",  # Older version joint model prediction + Manhattanization shape post processing
    "rmx-rse-v1_predictions",  # Basic HNet trained with production shapes
    "rmx-tg-manh-v1_predictions",  # Total (visible) geometry with Manhattanization shape post processing
]
# could also try partial manhattanization (separate model)


IMAGE_HEIGHT_PX = 512
IMAGE_WIDTH_PX = 1024


def load_hnet_predictions(
    building_id: str, raw_dataset_dir: str, predictions_data_root: str
) -> Optional[Dict[str, Dict[int, PanoStructurePredictionRmxMadoriV1]]]:
    """Load raw pixelwise HorizonNet predictions...

    Args:
        building_id:
        raw_dataset_dir
        predictions_data_root: path to HorizonNet predictions.

    Returns:
        Mapping from floor_id to a dictionary of per-pano predictions.
    """
    floor_hnet_predictions = defaultdict(dict)

    # find available floors
    floor_ids = posegraph2d.compute_available_floors_for_building(
        building_id=building_id, raw_dataset_dir=raw_dataset_dir
    )

    for floor_id in floor_ids:
        floor_gt_pose_graph = posegraph2d.get_gt_pose_graph(
            building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
        )
        for i in floor_gt_pose_graph.pano_ids():

            # if the file doesn't exist, return None for now (TODO: throw an error)
            model_prediction_fpath = f"{predictions_data_root}/{building_id}/{i}.json"
            if not Path(model_prediction_fpath).exists():
                print(
                    f"Missing predictions for building {building_id}.",
                    building_id,
                )
                # skip this building.
                return None

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/floor*_pano_{i}.jpg")
            if not len(img_fpaths) == 1:
                print("\tShould only be one image for this (building id, pano id) tuple.")
                print(f"\tPano {i} was missing")
                plt.close("all")
                continue

            img_fpath = img_fpaths[0]

            discovered_floor_id = get_floor_id_from_img_fpath(img_fpath)
            assert floor_id == discovered_floor_id
            model_name = "rmx-madori-v1_predictions"
            prediction_data = io_utils.read_json_file(model_prediction_fpath)

            if model_name == "rmx-madori-v1_predictions":
                pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data["predictions"])
                if pred_obj is None:  # malformatted pred for some reason
                    continue

                floor_hnet_predictions[floor_id][i] = pred_obj

            render_on_pano = False
            if render_on_pano:
                plt.figure(figsize=(20, 10))
                img = imageio.imread(img_fpath)
                img_resized = cv2.resize(img, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
                img_h, img_w, _ = img_resized.shape

                plt.imshow(img_resized)
                pred_obj.render_layout_on_pano(img_h, img_w)
                plt.title(f"Pano {i} from Building {building_id}")
                plt.tight_layout()
                os.makedirs(f"HorizonNet_pred_model_visualizations_2022_07_18_bridge/{model_name}_bev", exist_ok=True)
                plt.savefig(
                    f"HorizonNet_pred_model_visualizations_2022_07_18_bridge/{model_name}_bev/{building_id}_{i}.jpg",
                    dpi=400,
                )
                # plt.show()
                plt.close("all")
                plt.figure(figsize=(20, 10))

    return floor_hnet_predictions

    raise RuntimeError("Unknown error loading inferred pose graphs")
    return None


def load_vanishing_angles(predictions_data_root: str, building_id: str) -> Dict[int, float]:
    """Load pre-computed vanishing angles for each panorama ID."""
    json_fpath = Path(predictions_data_root) / "vanishing_angle" / f"{building_id}.json"
    vanishing_angles_map = io_utils.read_json_file(json_fpath)
    return {int(k): v for k, v in vanishing_angles_map.items()}


def load_inferred_floor_pose_graphs(
    building_id: str, raw_dataset_dir: str, predictions_data_root: str
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
        building_id: string representing ZInD building ID to fetch the per-floor inferred pose graphs for.
            Should be a zfilled-4 digit string, e.g. "0001"
        raw_dataset_dir: path to ZInD dataset.
        predictions_data_root:

    Returns:
        floor_pose_graphs: mapping from floor_id to predicted pose graph
            (without poses, but just W/D/O predictions and layout prediction)
    """
    floor_pose_graphs = {}

    hnet_predictions_dict = load_hnet_predictions(
        building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=predictions_data_root
    )
    if hnet_predictions_dict is None:
        print(f"HorizonNet predictions could not be loaded for {building_id}")
        return None
    building_vanishing_angles_dict = load_vanishing_angles(
        predictions_data_root=predictions_data_root, building_id=building_id
    )

    # Populate the pose graph for each floor, pano-by-pano.
    for floor_id, floor_predictions in hnet_predictions_dict.items():

        # Load GT just to get the `scale_meters_per_coordinate` scaling factor.
        floor_gt_pose_graph = posegraph2d.get_gt_pose_graph(
            building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
        )

        if floor_id not in floor_pose_graphs:
            # Initialize a new PoseGraph2d for this new floor.
            floor_pose_graphs[floor_id] = PoseGraph2d(
                building_id=building_id,
                floor_id=floor_id,
                nodes={},
                scale_meters_per_coordinate=floor_gt_pose_graph.scale_meters_per_coordinate,
            )

        # `i` represents the panorama's ID.
        for i, pred_obj in floor_predictions.items():

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/floor*_pano_{i}.jpg")
            if not len(img_fpaths) == 1:
                raise ValueError("There should be a unique image for this panorama ID.")
            img_fpath = img_fpaths[0]

            IMG_H = 512
            IMG_W = 1024
            pano_data = pred_obj.convert_to_pano_data(
                img_h=IMG_H,
                img_w=IMG_W,
                pano_id=i,
                gt_pose_graph=floor_gt_pose_graph,
                img_fpath=img_fpath,
                vanishing_angle_deg=building_vanishing_angles_dict[i],
            )
            floor_pose_graphs[floor_id].nodes[i] = pano_data

    return floor_pose_graphs


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

    # raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    rendering_save_dir = "ZinD_Inferred_GT_bridgeapi_2021_10_05_rendered_2021_10_14"
    model_name = "rmx-madori-v1_predictions"

    NUM_ZIND_BUILDINGS = 1575

    # Generate all possible building IDs for ZinD.
    building_ids = [str(v).zfill(4) for v in range(NUM_ZIND_BUILDINGS)]

    for building_id in building_ids:

        # if building_id != "0767":
        # if building_id != "0879":
        #     continue

        floor_pose_graphs = load_inferred_floor_pose_graphs(building_id=building_id, raw_dataset_dir=raw_dataset_dir)
        continue
        if floor_pose_graphs is None:
            # prediction files must have been missing, so we skip.
            continue

        for floor_id, floor_pose_graph in floor_pose_graphs.items():

            # load the GT pose graph to rip out the GT pose for each pano.
            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            floor_reconstruction_report.render_floorplans_side_by_side(
                est_floor_pose_graph=floor_pose_graph,
                show_plot=False,
                save_plot=True,
                plot_save_dir=f"{model_name}__oracle_pose_2021_11_12",
                gt_floor_pg=gt_pose_graph,
            )

            floor_pose_graph.save_as_zind_data_json(save_fpath=f"{rendering_save_dir}/{building_id}/{floor_id}.json")


if __name__ == "__main__":
    main()
