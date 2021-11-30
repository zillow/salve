import glob
import os
from pathlib import Path
from typing import Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

from afp.common.pano_data import PanoData, FloorData


def render_building(building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'
    """
    FLOORPLANS_OUTPUT_DIR = "/Users/johnlam/Downloads/ZinD_Vis_2021_07_13"

    floor_map_json = read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        # if floor_id != 'floor_02':
        #     continue
        # import pdb; pdb.set_trace()

        fd = FloorData.from_json(floor_data, floor_id)

        wdo_objs_seen_on_floor = set()

        for pano_obj in fd.panos:
            wdo_objs_seen_on_floor = pano_obj.plot_room_layout(
                coord_frame="global", wdo_objs_seen_on_floor=wdo_objs_seen_on_floor, show_plot=False
            )

        plt.legend(loc="upper right")
        plt.title(f"Building {building_id}: {floor_id}")
        plt.axis("equal")
        building_dir = f"{FLOORPLANS_OUTPUT_DIR}/{building_id}"
        os.makedirs(building_dir, exist_ok=True)
        plt.savefig(f"{building_dir}/{floor_id}.jpg", dpi=500)
        # plt.show()

        plt.close("all")

        # dominant_rotation, _ = determine_rotation_angle(room_vertices_global)
        # if dominant_rotation is None:
        #     continue
        # floor_dominant_rotation[floor_id] = dominant_rotation

        # geometry_to_visualize = ["raw", "complete", "visible", "redraw"]
        # for geometry_type in geometry_to_visualize:

        #         zfm_dict = zfm._zfm[geometry_type]
        #         floor_maps_files_list = zfm.render_top_down_zfm(zfm_dict, output_folder_floor_plan)
        #     for floor_id, zfm_poly_list in zfm_dict.items():


def export_pano_visualizations(raw_dataset_dir: str) -> None:
    """ """

    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        if building_id not in [
            "1530",
            "1442",
            "1482",
            "1490",
            "1441",
            "1427",
            "1634",
            "1635",
            "1626",
            "1584",
            "1578",
            "1583",
            "1394",
        ]:  # != '000': # '1442':
            continue

        print(f"Render floor maps for {building_id}")
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        render_building(building_id=building_id, pano_dir=pano_dir, json_annot_fpath=json_annot_fpath)


if __name__ == "__main__":
    """ """
    # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    export_pano_visualizations(raw_dataset_dir)
