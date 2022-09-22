"""Verify if predictions are missing for any ZInD tours."""

import os
import shutil
from pathlib import Path

import cv2
import gtsfm.utils.io as io_utils
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

from salve.dataset.mhnet_prediction import MHNetPanoStructurePrediction
from salve.dataset.zind_partition import DATASET_SPLITS


def main() -> None:
    """ """
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    #preds_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_07_11"
    #preds_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_07_18_"
    #preds_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_aggregated_2022_08_05"
    preds_dir = "/Users/johnlambert/Downloads/zind_final_schema_1768_majority_2022_09_05_aggregated"

    missing_building_ids = []

    #save_fpath = "/Users/johnlambert/Downloads/salve/zind_missing_predictions_2022_08_15.json"
    save_fpath = "/Users/johnlambert/Downloads/salve/zind_missing_predictions_2022_09_05.json"
    missing_preds_list = []

    for split in ["val", "train", "test"]:

        zind_building_ids = DATASET_SPLITS[split]
        for building_id in zind_building_ids:

            # if building_id not in [
            #     "0969","0245","0297","0299","0308","0336","0353","0354","0406","0420","0429","0431","0453","0490","0496","0528","0534","0564","0575","0579","0583","0588","0605","0629","0668","0681","0684","0691","0792","0800","0809","0819","0854","0870","0905","0957","0963","0964","0966","1001","1027","1028","1041","1050","1068","1069","1075","1130","1160","1175","1184","1185","1203","1207","1210","1218","1239","1248","1326","1328","1330","1368","1383","1388","1398","1401","1404","1409","1479","1490","1494","1500","1538","1544","1551","1566",
            # ]:
            #     continue

            # check number of panos predicted for
            pano_fpaths = Path(raw_dataset_dir).glob(f"{building_id}/panos/*.jpg")

            json_preds_fpaths = list(Path(f"{preds_dir}/horizon_net/{building_id}").glob("*.json"))
            if len(json_preds_fpaths) == 0:
                print(f"No preds for {building_id}")
                missing_building_ids.append(building_id)
                for pano_fpath in pano_fpaths:
                    pano_id = int(Path(pano_fpath).stem.split("_")[-1])
                    missing_preds_list.append((building_id, pano_id))
                continue

            #import pdb; pdb.set_trace()
            # discovered_pred_ids = set([fp.stem for fp in json_preds_fpaths])
            # discovered_pano_ids = set([fp.stem.split("_")[-1] for fp in pano_fpaths])

            discovered_pred_ids = [fp.stem for fp in json_preds_fpaths]
            discovered_pano_ids = [fp.stem for fp in pano_fpaths]

            if sorted(discovered_pred_ids) != sorted(discovered_pano_ids):
                print(f"\tFor building {building_id}, should exist for {len(discovered_pano_ids)}, but preds only for {len(discovered_pred_ids)}")
                missing_pred_ids = set(discovered_pano_ids) - set(discovered_pred_ids)
                print("\t\tMissing", [Path(fp).name for fp in missing_pred_ids])
                for pano_id in missing_pred_ids:
                    missing_preds_list.append((building_id, pano_id))

            model_name = "rmx-madori-v1_predictions"
            # Validate each file.
            for model_prediction_fpath in json_preds_fpaths:
                #pano_id = int(Path(model_prediction_fpath).stem)
                pano_id = Path(model_prediction_fpath).stem

                img_fpath =f"{raw_dataset_dir}/{building_id}/panos/{pano_id}.jpg"
                assert Path(img_fpath).exists()

                pred_obj = MHNetPanoStructurePrediction.from_json_fpath(
                    json_fpath=Path(model_prediction_fpath),
                    image_fpath=Path(img_fpath)
                )
                
                if pred_obj is None:
                    missing_preds_list.append((building_id, pano_id))
                    print(f"Invalid pred for {building_id} ->, {model_prediction_fpath}")
                    continue
                if pred_obj.floor_boundary.shape != (1024,):
                    print(f"Invalid pred for {building_id} ->, {model_prediction_fpath}")
                    missing_preds_list.append((building_id, pano_id))

                render_on_pano = np.random.rand() < 0.001
                if render_on_pano:
                    plt.figure(figsize=(20, 10))
                    img = imageio.imread(img_fpath)
                    img_resized = cv2.resize(img, (pred_obj.image_width, pred_obj.image_height))

                    plt.imshow(img_resized)
                    pred_obj.render_layout_on_pano()
                    plt.title(f"Pano {pano_id} from Building {building_id}")
                    plt.tight_layout()
                    os.makedirs(f"HorizonNet_pred_model_visualizations_2022_09_05_bridge/{model_name}_bev", exist_ok=True)
                    plt.savefig(
                        f"HorizonNet_pred_model_visualizations_2022_09_05_bridge/{model_name}_bev/{building_id}_{pano_id}.jpg",
                        dpi=400,
                    )
                    # plt.show()
                    plt.close("all")
                    plt.figure(figsize=(20, 10))


    print("Missing list: ", len(missing_preds_list))
    print("Missing building ids: ", sorted(missing_building_ids))

    io_utils.save_json_file(save_fpath, missing_preds_list)



def copy_missing_files() -> None:
    """ """

    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    read_fpath = "/Users/johnlambert/Downloads/salve/zind_missing_predictions_2022_08_15.json"
    missing_data = io_utils.read_json_file(read_fpath)

    missing_filepaths = []

    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"
    missing_img_dir = "/Users/johnlambert/Downloads/missing_zind_predictions_2022_08_15"

    for (building_id, pano_id) in missing_data:

        pano_fpaths = list(Path(raw_dataset_dir).glob(f"{building_id}/panos/*_{pano_id}.jpg"))
        for pano_fpath in pano_fpaths:
            # erroneously, sometimes two panos have the same pano ID
            # if len(pano_fpaths) != 1:
            #     import pdb; pdb.set_trace()
            shutil.copyfile(pano_fpath, f"{missing_img_dir}/{building_id}___{Path(pano_fpath).name}")
            missing_filepaths.append(str(pano_fpath))

    save_fpath = "/Users/johnlambert/Downloads/salve/zind_missing_predictions_filepaths_2022_08_15.json"
    io_utils.save_json_file(save_fpath, missing_filepaths)


def determine_duplicate_integer_image_ids():
    """
    0363: Duplicate among  [6, 8, 10, 14, 16, 18, 25, 27, 28, 30, 32, 34, 34, 35, 37, 38, 40, 41, 45, 48, 49, 50]
    1348: Duplicate among  [3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    """
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    for split, building_ids in DATASET_SPLITS.items():
        for building_id in building_ids:

            pano_fpaths = Path(raw_dataset_dir).glob(f"{building_id}/panos/*.jpg")
            image_ids = [int(pano_fpath.stem.split("_")[-1]) for pano_fpath in pano_fpaths]

            if len(image_ids) != len(set(image_ids)):
                print(f"{building_id}: Duplicate among ", sorted(image_ids))




if __name__ == "__main__":
    main()
    #copy_missing_files()
    #determine_duplicate_integer_image_ids()

