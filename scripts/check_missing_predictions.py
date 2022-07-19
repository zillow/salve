"""Verify if predictions are missing for any ZInD tours."""

from pathlib import Path

import gtsfm.utils.io as io_utils

from salve.dataset.rmx_madori_v1 import PanoStructurePredictionRmxMadoriV1
from salve.dataset.zind_partition import DATASET_SPLITS


def main() -> None:
    """ """
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    #preds_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_07_11"
    preds_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_07_18_"

    missing_building_ids = []

    for split in ["train", "val", "test"]:

        zind_building_ids = DATASET_SPLITS[split]
        for building_id in zind_building_ids:

            if building_id not in [
                "0969","0245","0297","0299","0308","0336","0353","0354","0406","0420","0429","0431","0453","0490","0496","0528","0534","0564","0575","0579","0583","0588","0605","0629","0668","0681","0684","0691","0792","0800","0809","0819","0854","0870","0905","0957","0963","0964","0966","1001","1027","1028","1041","1050","1068","1069","1075","1130","1160","1175","1184","1185","1203","1207","1210","1218","1239","1248","1326","1328","1330","1368","1383","1388","1398","1401","1404","1409","1479","1490","1494","1500","1538","1544","1551","1566",
            ]:
                continue

            json_preds_fpaths = list(Path(f"{preds_dir}/horizon_net/{building_id}").glob("*.json"))
            if len(json_preds_fpaths) == 0:
                print(f"No preds for {building_id}")
                missing_building_ids.append(building_id)
                continue

            # check number of panos predicted for
            pano_fpaths = Path(raw_dataset_dir).glob(f"{building_id}/panos/*.jpg")
            #import pdb; pdb.set_trace()
            discovered_pred_ids = set([fp.stem for fp in json_preds_fpaths])
            discovered_pano_ids = set([fp.stem.split("_")[-1] for fp in pano_fpaths])

            if discovered_pred_ids != discovered_pano_ids:
                print(f"\tFor building {building_id}, should exist for {len(discovered_pano_ids)}, but preds only for {len(discovered_pred_ids)}")

            # Validate each file.
            for model_prediction_fpath in json_preds_fpaths:
                prediction_data = io_utils.read_json_file(model_prediction_fpath)
                pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data["predictions"])
                
                if pred_obj is None:
                    print(f"Invalid pred for {building_id} ->, {model_prediction_fpath}")
                    continue
                if pred_obj.floor_boundary.shape != (1024,):
                    print(f"Invalid pred for {building_id} ->, {model_prediction_fpath}")

    print("Missing building ids: ", sorted(missing_building_ids))


if __name__ == "__main__":
    main()