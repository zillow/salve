
"""Compute the average number of floors in each home of ZinD"""

import glob
from collections import defaultdict
from pathlib import Path

import argoverse.utils.json_utils as json_utils
import numpy as np

from salve.dataset.zind_partition import DATASET_SPLITS


def count_floor_stats_per_home():
    """
    Train: 2168 floors, 1.72 floors per building. #Panos per floor: 24.92
    Val: 278 floors, 1.77 floors per building. #Panos per floor: 23.97
    Test: 291 floors, 1.84 floors per building. #Panos per floor: 23.20
    """
    #split = "train"
    #split = "val"
    split = "test"

    #raw_dataset_dir = "/home/johnlam/zind_bridgeapi_2021_10_05"
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    floor_counter = defaultdict(int)

    building_floor_panocounter = {}

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:
        print(f"On Building {building_id}")
        # for rendering test data only
        if building_id not in DATASET_SPLITS[split]:
            continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        if not Path(json_annot_fpath).exists():
            print(f"zind_data.json file missing for {building_id}")

        floor_map_json = json_utils.read_json_file(json_annot_fpath)

        if "merger" not in floor_map_json:
            print(f"No merger data in {building_id}: {json_annot_fpath}")
            continue

        merger_data = floor_map_json["merger"]
        for floor_id in merger_data.keys():

            print(f"\tFound Building {building_id} {floor_id}")
            floor_counter[building_id] += 1

            # count the number of panos on this floor
            floor_panos_wildcard = f"{raw_dataset_dir}/{building_id}/panos/{floor_id}_partial_room_*_pano_*.jpg"
            num_panos_on_floor = len(glob.glob(floor_panos_wildcard))
            building_floor_panocounter[(building_id, floor_id)] = num_panos_on_floor

    avg_panos_per_floor = np.mean(list(building_floor_panocounter.values()))
    print(f"Average number of panos per floor: {avg_panos_per_floor:.2f}")

    mean_num_floors = np.mean(list(floor_counter.values()))
    total_num_floors = np.sum(list(floor_counter.values()))
    print(f"Number of floors in {split}: {total_num_floors}")
    print(f"mean_num_floors: {mean_num_floors:.2f}")


def count_avg_number_wdo_per_pano() -> None:
    """ """

    dirpath = "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65/1529/floor_02/incorrect_alignment"



if __name__ == "__main__":
    # count_floor_stats_per_home()
    count_avg_number_wdo_per_pano()
