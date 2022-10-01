"""Utility to load ModifiedHorizonNet (MHNet) predictions.

Converts a ModifiedHorizonNet inference result to PanoData and PoseGraph2d objects. Also supports rendering the
inference result with oracle pose.
"""

import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import imageio
import matplotlib.pyplot as plt

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.utils.io as io_utils
import salve.common.posegraph2d as posegraph2d
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.mhnet_prediction import MHNetPanoStructurePrediction


def load_hnet_predictions(
    building_id: str, raw_dataset_dir: str, predictions_data_root: str
) -> Dict[str, Dict[int, MHNetPanoStructurePrediction]]:
    """Load raw pixelwise HorizonNet predictions for each pano in a specified ZInD building.

    # TODO: remove dependency on getting pano paths from ZInD in this function (put them inside the predictions).

    Args:
        building_id: unique ID of ZInD building.
        raw_dataset_dir: path to ZInD dataset.
        predictions_data_root: path to HorizonNet predictions.

    Returns:
        Mapping from floor_id to a dictionary of per-pano predictions.
    """
    floor_hnet_predictions = defaultdict(dict)

    # Find available floors for this ZInD building.
    floor_ids = posegraph2d.compute_available_floors_for_building(
        building_id=building_id, raw_dataset_dir=raw_dataset_dir
    )

    for floor_id in floor_ids:
        floor_gt_pose_graph = posegraph2d.get_gt_pose_graph(
            building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
        )
        for i in floor_gt_pose_graph.pano_ids():
            # TODO: get corresponding prediction file based on the filename, not on the integer ID

            # If the file doesn't exist, return None for now (TODO: throw an error)
            model_prediction_fpaths = glob.glob(f"{predictions_data_root}/horizon_net/{building_id}/*_{i}.json")
            if len(model_prediction_fpaths) == 0:
                print("\tShould only be one prediction for this (building id, pano id) tuple.")
                print(f"\tPrediction {i} was missing")
                plt.close("all")
                continue

            elif len(model_prediction_fpaths) > 1:
                # Note: Building 1348 has two panos with ID `5` each, which are very similar. Choose the one below for `5`.
                if building_id == "1348" and i == 5:
                    model_prediction_fpath = Path(
                        f"{predictions_data_root}/horizon_net/1348/floor_01_partial_room_12_pano_5.json"
                    )
                # Note: Building 0363 has two panos with ID `34` each.
                if building_id == "0363" and i == 34:
                    model_prediction_fpath = Path(
                        f"{predictions_data_root}/horizon_net/0363/floor_02_partial_room_05_pano_34.json"
                    )
            else:
                model_prediction_fpath = Path(model_prediction_fpaths[0])
            fname_stem = Path(model_prediction_fpath).stem
            img_fpath = Path(f"{raw_dataset_dir}/{building_id}/panos/{fname_stem}.jpg")
            pred_obj = MHNetPanoStructurePrediction.from_json_fpath(
                json_fpath=model_prediction_fpath, image_fpath=img_fpath
            )
            floor_hnet_predictions[floor_id][i] = pred_obj

            render_on_pano = False
            if render_on_pano:
                _visualize_overlaid_layout_on_pano(img_fpath, pred_obj, building_id, i)

    return floor_hnet_predictions


def _visualize_overlaid_layout_on_pano(
    img_fpath: str,
    pred_obj: MHNetPanoStructurePrediction,
    building_id: str,
    i: int
) -> None:
    """Visualize overlaid layout on panorama."""
    plt.figure(figsize=(20, 10))
    img = imageio.imread(img_fpath)
    img_resized = cv2.resize(img, (pred_obj.image_width, pred_obj.image_height))
    plt.imshow(img_resized)
    pred_obj.render_layout_on_pano()
    plt.title(f"Pano {i} from Building {building_id}")
    plt.tight_layout()
    save_dir = "HorizonNet_pred_model_visualizations_2022_01_01_bridge/mhnet_overlaid"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{building_id}_{i}.jpg", dpi=400)
    # plt.show()
    plt.close("all")
    plt.figure(figsize=(20, 10))


def load_vanishing_angles(predictions_data_root: str, building_id: str) -> Dict[int, float]:
    """Load pre-computed vanishing angles for each panorama ID."""
    json_fpath = Path(predictions_data_root) / "vanishing_angle" / f"{building_id}.json"
    vanishing_angles_map = io_utils.read_json_file(json_fpath)
    return {int(k): v for k, v in vanishing_angles_map.items()}


def load_inferred_floor_pose_graphs(
    building_id: str, raw_dataset_dir: str, predictions_data_root: str
) -> Optional[Dict[str, PoseGraph2d]]:
    """Load W/D/O's & layout predicted for each pano of each floor by ModifiedHorizonNet (MHNet).

    TODO: rename this function, since no pose graph is loaded here.
    TODO: remove dependency on getting pano paths from ZInD in this function (put them inside the predictions).

    Args:
        building_id: string representing ZInD building ID to fetch the per-floor inferred pose graphs for.
            Should be a zfilled-4 digit string, e.g. "0001"
        raw_dataset_dir: path to ZInD dataset.
        predictions_data_root: path to HorizonNet predictions.

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

    # TODO: add back vanishing angle utilization (and check where they are being used downstream here).
    building_vanishing_angles_dict = defaultdict(int)
    # building_vanishing_angles_dict = load_vanishing_angles(
    #     predictions_data_root=predictions_data_root, building_id=building_id
    # )

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

        # Index `i` represents the panorama's ID.
        for i, pred_obj in floor_predictions.items():

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/floor*_pano_{i}.jpg")
            if not len(img_fpaths) == 1:
                # Note: Building 1348 has two panos with ID `5` each.
                known_duplicate1 = (building_id == "1348" and i == 5)
                # Note: Building 0363 has two panos with ID `34` each.
                known_duplicate2 = (building_id == "0363" and i == 34)
                if not (known_duplicate1 or known_duplicate2):
                    raise ValueError(f"There should be a unique image for panorama ID {i} from Bldg. {building_id}.")
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


def load_inferred_floor_pose_graph(
    building_id: str, floor_id: str, raw_dataset_dir: str, predictions_data_root: str
) -> PoseGraph2d:
    """

    Args:
        building_id: string representing ZInD building ID to fetch the inferred floor pose graphs for.
            Should be a zfilled-4 digit string, e.g. "0001"
        floor_id: 
        raw_dataset_dir: path to ZInD dataset.
        predictions_data_root: path to ModifiedHorizonNet (MHNet) predictions.

    Returns:
        floor_pose_graph: predicted pose graph (without poses, but just W/D/O predictions and layout prediction).
    """
    floor_pose_graphs = load_inferred_floor_pose_graphs(
        building_id=building_id,
        raw_dataset_dir=raw_dataset_dir,
        predictions_data_root=predictions_data_root,
    )
    if floor_pose_graphs is None:
        raise ValueError(
            f"ModifiedHorizonNet (MHNet) predictions missing for all floors of ZInD Building {building_id}."
        )
    if floor_id not in floor_pose_graphs:
        raise ValueError(
            f"ModifiedHorizonNet (MHNet) predictions missing for {floor_id} of ZInD Building {building_id}."
        )
    return floor_pose_graphs[floor_id]


def get_floor_id_from_img_fpath(img_fpath: str) -> str:
    """Fetch the corresponding embedded floor ID from a panorama file path.

    For example,
    "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg" -> "floor_01"
    """
    fname = Path(img_fpath).name
    k = fname.find("_partial")
    floor_id = fname[:k]

    return floor_id

