"""Generate visualizations of annotated floorplans and camera poses in a bird's eye view."""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Set

import argoverse.utils.json_utils as json_utils
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2

import afp.common.posegraph2d as posegraph2d
from afp.common.pano_data import PanoData, FloorData


def render_building(
    building_id: str, pano_dir: str, json_annot_fpath: str, viz_save_dir: str, raw_dataset_dir: str
) -> None:
    """For a single building, render each floor.

    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'

    Args:
        building_id: unique ID of ZInD building.
        pano_dir:
        json_annot_fpath: path to GT annotation file (zind_data.json) for this building.
        viz_save_dir: directory where visualizations will be saved to.
        raw_dataset_dir: dirpath to downloaded ZInD dataset.
    """
    floor_map_json = json_utils.read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        # need the PoseGraph2d object to get scaling factor.
        gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        fd = FloorData.from_json(floor_data, floor_id)

        wdo_objs_seen_on_floor = set()

        for pano_obj in fd.panos:
            wdo_objs_seen_on_floor = pano_obj.plot_room_layout(
                coord_frame="worldmetric",
                wdo_objs_seen_on_floor=wdo_objs_seen_on_floor,
                show_plot=False,
                scale_meters_per_coordinate=gt_floor_pose_graph.scale_meters_per_coordinate,
            )

        plt.legend(loc="upper right")
        plt.title(f"Building {building_id}: {floor_id}")
        plt.axis("equal")
        os.makedirs(viz_save_dir, exist_ok=True)
        plt.savefig(f"{viz_save_dir}/{building_id}__{floor_id}.jpg", dpi=500)
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


def export_floorplan_visualizations(raw_dataset_dir: str, viz_save_dir: str) -> None:
    """For all buildings, render each floor."""

    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        print(f"Render floor maps for {building_id}")
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        render_building(
            building_id=building_id,
            pano_dir=pano_dir,
            json_annot_fpath=json_annot_fpath,
            viz_save_dir=viz_save_dir,
            raw_dataset_dir=raw_dataset_dir,
        )


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05",
        help="dirpath to downloaded ZInD dataset.",
    )
    parser.add_argument(
        "--viz_save_dir",
        type=str,
        default="/Users/johnlam/Downloads/ZinD_Vis_2021_11_30",
        help="directory where visualizations will be saved to.",
    )

    args = parser.parse_args()
    export_floorplan_visualizations(args.raw_dataset_dir, args.viz_save_dir)
