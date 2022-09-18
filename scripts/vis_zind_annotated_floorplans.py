"""Generate visualizations of annotated floorplans and camera poses in a bird's eye view."""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Set

import matplotlib.pyplot as plt
import numpy as np

import salve.common.posegraph2d as posegraph2d
import salve.utils.logger_utils as logger_utils
import salve.utils.matplotlib_utils as matplotlib_utils
import salve.utils.io as io_utils
from salve.common.pano_data import PanoData, FloorData
from salve.common.sim2 import Sim2

logger = logger_utils.get_logger()

def render_building(
    raw_dataset_dir: str, building_id: str, viz_save_dir: str, 
) -> None:
    """For a single building, render each floor and save plot.

    We use the "merger" (pre-redraw) ZInD annotations, instead of "redraw".
    See https://github.com/zillow/zind#what-are-the-important-issues-associated-with-the-dataset-that-we-need-to-be-aware-of
    for more information.

    Args:
        raw_dataset_dir: dirpath to downloaded ZInD dataset.
        building_id: unique ID of ZInD building.
        viz_save_dir: directory where visualizations will be saved to.
    """
    # Get path to GT annotation file (zind_data.json) for this building.
    json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
    pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
    floor_map_json = io_utils.read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        raise ValueError(f"Building {building_id} missing `merger` data")

    merger_data = floor_map_json["merger"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        # Need the PoseGraph2d object to get scaling factor.
        gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
        fd = FloorData.from_json(floor_data, floor_id)

        for pano_obj in fd.panos:
            pano_obj.plot_room_layout(
                coord_frame="worldmetric",
                show_plot=False,
                scale_meters_per_coordinate=gt_floor_pose_graph.scale_meters_per_coordinate,
            )
        ax = None
        # Note: many of the Matplotlib "labels" are duplicated, but can be de-duplicated here during legend creation.
        #matplotlib_utils.legend_without_duplicate_labels(ax)
        plt.legend(loc="upper right")
        plt.title(f"Building {building_id}: {floor_id}")
        plt.axis("equal")
        os.makedirs(viz_save_dir, exist_ok=True)
        plt.savefig(f"{viz_save_dir}/{building_id}__{floor_id}.jpg", dpi=500)
        # plt.show()
        plt.close("all")


def export_floorplan_visualizations(raw_dataset_dir: str, viz_save_dir: str) -> None:
    """For all buildings, render each floor."""

    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        logger.info(f"Render floor maps for {building_id}")
        render_building(
            raw_dataset_dir=raw_dataset_dir,
            building_id=building_id,
            viz_save_dir=viz_save_dir,
        )


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05",
        help="Path to directory containing downloaded ZInD dataset (from Bridge API).",
    )
    parser.add_argument(
        "--viz_save_dir",
        type=str,
        default="/Users/johnlambert/Downloads/ZinD_Vis_2022_02_30",
        help="directory where visualizations will be saved to.",
    )

    args = parser.parse_args()
    export_floorplan_visualizations(raw_dataset_dir=args.raw_dataset_dir, viz_save_dir=args.viz_save_dir)
