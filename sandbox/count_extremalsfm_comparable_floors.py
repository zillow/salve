
"""Compute which home floors in ZinD have 5 or less rooms (suitable for "Extremal SfM" (See Shabani et al, ICCV 21)"""

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import gtsfm.utils.io as io_utils
import numpy as np

from salve.dataset.zind_partition import DATASET_SPLITS


def count_floor_stats_per_home(raw_dataset_dir: str):
    """
    Train: 2168 floors, 1.72 floors per building. #Panos per floor: 24.92
    Val: 278 floors, 1.77 floors per building. #Panos per floor: 23.97
    Test: 291 floors, 1.84 floors per building. #Panos per floor: 23.20
    """
    #split = "train"
    #split = "val"
    split = "test"

    floor_counter = defaultdict(int)

    building_floor_panocounter = {}

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    num_missing = set(DATASET_SPLITS[split]) - set(building_ids)
    print(f"Found {num_missing} missing homes.")

    for building_id in building_ids:
        print(f"\tOn Building {building_id}")
        # for rendering test data only
        if building_id not in DATASET_SPLITS[split]:
            continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        if not Path(json_annot_fpath).exists():
            print(f"zind_data.json file missing for {building_id}")

        floor_map_json = io_utils.read_json_file(json_annot_fpath)

        if "merger" not in floor_map_json:
            print(f"\tNo merger data in {building_id}: {json_annot_fpath}")
            continue

        merger_data = floor_map_json["merger"]
        for floor_id in merger_data.keys():

            print(f"\t\tFound Building {building_id} {floor_id}")
            floor_counter[building_id] += 1

            # count the number of panos on this floor
            floor_panos_wildcard = f"{raw_dataset_dir}/{building_id}/panos/{floor_id}_partial_room_*_pano_*.jpg"
            num_panos_on_floor = len(glob.glob(floor_panos_wildcard))
            building_floor_panocounter[(building_id, floor_id)] = num_panos_on_floor

            if num_panos_on_floor <= 4:
                print(f"Building {building_id}, {floor_id} has only {num_panos_on_floor}")

    num_floors_5panos_or_less = (np.array(list(building_floor_panocounter.values())) <= 4).sum()
    print(f"Found {num_floors_5panos_or_less} floors with <=5 panos.")

    num_floors = len(building_floor_panocounter.keys())
    print(f"This is {num_floors_5panos_or_less / num_floors * 100: .2f} percent of all floors", num_floors)

    plt.hist(list(building_floor_panocounter.values()), bins=20)
    plt.savefig("panos_per_floor_hist.png", dpi=500)
    avg_panos_per_floor = np.mean(list(building_floor_panocounter.values()))
    print(f"Average number of panos per floor: {avg_panos_per_floor:.2f}")



if __name__ == "__main__":

    #raw_dataset_dir = "/home/johnlam/zind_bridgeapi_2021_10_05"
    # raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"

    count_floor_stats_per_home(raw_dataset_dir)
