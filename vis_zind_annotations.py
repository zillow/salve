import glob
import os
from pathlib import Path
from typing import Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

from pano_data import PanoData, FloorData

from sim3_align_dw import rotmat2d


RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
wdo_color_dict = {"windows": RED, "doors": GREEN, "openings": BLUE}


def plot_room_layout(
    pano_obj: PanoData, coord_frame: str, wdo_objs_seen_on_floor: Optional[Set] = None, show_plot: bool = True
) -> None:
    """
    coord_frame: either 'local' or 'global'
    """
    R_refl = np.array([[-1.0, 0], [0, 1]])
    world_Sim2_reflworld = Sim2(R_refl, t=np.zeros(2), s=1.0)

    hohopano_Sim2_zindpano = Sim2(R=rotmat2d(-90), t=np.zeros(2), s=1.0)

    assert coord_frame in ["global", "local"]

    if coord_frame == "global":
        room_vertices = pano_obj.room_vertices_global_2d
    else:
        room_vertices = pano_obj.room_vertices_local_2d

    room_vertices = hohopano_Sim2_zindpano.transform_from(room_vertices)
    room_vertices = world_Sim2_reflworld.transform_from(room_vertices)

    color = np.random.rand(3)
    plt.scatter(room_vertices[:, 0], room_vertices[:, 1], 10, marker=".", color=color)
    plt.plot(room_vertices[:, 0], room_vertices[:, 1], color=color, alpha=0.5)
    # draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot([last_vert[0], first_vert[0]], [last_vert[1], first_vert[1]], color=color, alpha=0.5)

    pano_position = np.zeros((1, 2))
    if coord_frame == "global":
        pano_position_local = pano_position
        pano_position_reflworld = pano_obj.global_SIM2_local.transform_from(pano_position_local)
        pano_position_world = world_Sim2_reflworld.transform_from(pano_position_reflworld)
        pano_position = pano_position_world
    else:
        pano_position_refllocal = pano_position
        pano_position_local = world_Sim2_reflworld.transform_from(pano_position_refllocal)
        pano_position = pano_position_local

    plt.scatter(pano_position[0, 0], pano_position[0, 1], 30, marker="+", color=color)

    point_ahead = np.array([1, 0]).reshape(1, 2)
    if coord_frame == "global":
        point_ahead = pano_obj.global_SIM2_local.transform_from(point_ahead)

    point_ahead = world_Sim2_reflworld.transform_from(point_ahead)

    # TODO: add patch to Argoverse, enforcing ndim on the input to `transform_from()`
    plt.plot([pano_position[0, 0], point_ahead[0, 0]], [pano_position[0, 1], point_ahead[0, 1]], color=color)
    pano_id = pano_obj.id

    pano_text = pano_obj.label + f" {pano_id}"
    TEXT_LEFT_OFFSET = 0.15
    # mean_loc = np.mean(room_vertices_global, axis=0)
    # mean position
    noise = np.clip(np.random.randn(2), a_min=-0.05, a_max=0.05)
    text_loc = pano_position[0] + noise
    plt.text(
        (text_loc[0] - TEXT_LEFT_OFFSET), text_loc[1], pano_text, color=color, fontsize="xx-small"
    )  # , 'x-small', 'small', 'medium',)

    for wdo_idx, wdo in enumerate(pano_obj.doors + pano_obj.windows + pano_obj.openings):

        if coord_frame == "global":
            wdo_points = wdo.vertices_global_2d
        else:
            wdo_points = wdo.vertices_local_2d

        wdo_points = hohopano_Sim2_zindpano.transform_from(wdo_points)
        wdo_points = world_Sim2_reflworld.transform_from(wdo_points)

        # zfm_points_list = Polygon.list_to_points(wdo_points)
        # zfm_poly_type = PolygonTypeMapping[wdo_type]
        # wdo_poly = Polygon(type=zfm_poly_type, name=json_file_name, points=zfm_points_list)

        # # Add the WDO element to the list of polygons/lines
        # wdo_poly_list.append(wdo_poly)

        wdo_type = wdo.type
        wdo_color = wdo_color_dict[wdo_type]

        if (wdo_objs_seen_on_floor is not None) and (wdo_type not in wdo_objs_seen_on_floor):
            label = wdo_type
        else:
            label = None
        plt.scatter(wdo_points[:, 0], wdo_points[:, 1], 10, color=wdo_color, marker="o", label=label)
        plt.plot(wdo_points[:, 0], wdo_points[:, 1], color=wdo_color, linestyle="dotted")
        plt.text(wdo_points[:, 0].mean(), wdo_points[:, 1].mean(), f"{wdo.type}_{wdo_idx}")

        if wdo_objs_seen_on_floor is not None:
            wdo_objs_seen_on_floor.add(wdo_type)

    if show_plot:
        plt.axis("equal")
        plt.show()
        plt.close("all")

    return wdo_objs_seen_on_floor


def render_building(building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'
    """
    FLOORPLANS_OUTPUT_DIR = "/Users/johnlam/Downloads/ZinD_Vis_2021_06_25"

    floor_map_json = read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    merger_data = floor_map_json["merger"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        fd = FloorData.from_json(floor_data, floor_id)

        wdo_objs_seen_on_floor = set()
        # Calculate the dominant rotation of one room on this floor.
        # This dominant rotation will be used to align the floor map.

        for pano_obj in fd.panos:
            wdo_objs_seen_on_floor = plot_room_layout(
                pano_obj, coord_frame="global", wdo_objs_seen_on_floor=wdo_objs_seen_on_floor, show_plot=False
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
    import pdb

    pdb.set_trace()
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    args = []

    for building_id in building_ids:
        print(f"Render floor maps for {building_id}")
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        render_building(building_id=building_id, pano_dir=pano_dir, json_annot_fpath=json_annot_fpath)


if __name__ == "__main__":
    """ """
    raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    export_pano_visualizations(raw_dataset_dir)
