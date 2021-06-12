
"""
Based on code found at
https://gitlab.zgtools.net/zillow/rmx/research/scripts-insights/open_platform_utils/-/raw/zind_cleanup/zfm360/visualize_zfm360_data.py
"""


import collections
import glob
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np

from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

# The type of supported polygon/wall/point objects.
class PolygonType(Enum):
    ROOM = "room"
    WINDOW = "window"
    DOOR = "door"
    OPENING = "opening"
    PRIMARY_CAMERA = "primary_camera"
    SECONDARY_CAMERA = "secondary_camera"
    PIN_LABEL = "pin_label"


PolygonTypeMapping = {"windows": PolygonType.WINDOW, "doors": PolygonType.DOOR, "openings": PolygonType.OPENING}


# multiply all x-coordinates or y-coordinates by -1, to transfer origin from upper-left, to bottom-left
# (Reflection about either axis, would need additional rotation if reflect over x-axis)


class WDO(NamedTuple)
    """ """
    x1: float
    y1: float
    x2: float
    y2: float
    bottom_z: float
    top_z: float

    def vertices_local(self):
        return (x1,y1), (x2,y2)


class PanoData(NamedTuple):
    """ """
    global_SIM2_local: Sim2
    room_vertices_local_2d: np.ndarray
    wdo_vertices_local: List[WDO]

class FloorData(NamedTuple):
    """ """
    floor_id
    panos: List[PanoData]


def main():
    """
    Questions: what is tour_data_mapping.json? -> for internal people, GUIDs to production people
    Last edge of polygon (to close it) is not provided -- right??
    are all polygons closed? or just polylines?
    What is 'scale_meters_per_coordinate'?

    What is merger vs. redraw?

    Sim(2)

    s(Rp + t)  -> Sim(2)
    sRp + t -> ICP (Zillow)

    sRp + st

    Maybe compose with other Sim(2)
    """
    teaser_dirpath = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{teaser_dirpath}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:
        json_annot_fpath = f"{teaser_dirpath}/{building_id}/zfm_data.json"
        pano_dir = f"{teaser_dirpath}/{building_id}/panos"
        render_building(building_id, pano_dir, json_annot_fpath)


OUTPUT_DIR = "/Users/johnlam/Downloads/ZinD_Vis_2021_06_10"



def generate_Sim2_from_floorplan_transform(transform_data: Dict[str,Any]) -> Sim2:
    """Generate a Similarity(2) object from a dictionary storing transformation parameters.

    Note: ZinD stores (sRp + t), instead of s(Rp + t), so we have to divide by s to create Sim2.

    Args:
        transform_data: dictionary of the form
            {'translation': [0.015, -0.0022], 'rotation': -352.53, 'scale': 0.40}
    """
    scale = transform_data["scale"]
    t = np.array(transform_data["translation"]) / scale
    theta_deg = transform_data["rotation"]

    theta_rad = np.deg2rad(theta_deg)

    s = np.sin(theta_rad)
    c = np.cos(theta_rad)

    # fmt: off
    R = np.array(
        [
            [c, -s],
            [s, c]
        ]
    )
    # fmt: on

    global_SIM2_local = Sim2(R=R, t=t, s=scale)
    return global_SIM2_local


def render_building(building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'
    """
    floor_map_json = read_json_file(json_annot_fpath)

    merger_data = floor_map_json["merger"]
    redraw_data = floor_map_json["redraw"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():
        wdo_objs_seen_on_floor = set()
        # Calculate the dominant rotation of one room on this floor.
        # This dominant rotation will be used to align the floor map.
        for complete_room_data in floor_data.values():
            for partial_room_data in complete_room_data.values():
                for pano_data in partial_room_data.values():
                    #if pano_data["is_primary"]:

                    print('Camera height: ', pano_data['camera_height'])

                    global_SIM2_local = generate_Sim2_from_floorplan_transform(pano_data["floor_plan_transformation"])

                    room_vertices_local = np.asarray(pano_data["layout_raw"]["vertices"])
                    room_vertices_global = global_SIM2_local.transform_from(room_vertices_local)

                    color = np.random.rand(3)
                    plt.scatter(-room_vertices_global[:,0], room_vertices_global[:,1], 10, marker='.', color=color)
                    plt.plot(-room_vertices_global[:,0], room_vertices_global[:,1], color=color, alpha=0.5)
                    # draw edge to close each polygon
                    last_vert = room_vertices_global[-1]
                    first_vert = room_vertices_global[0]
                    plt.plot([-last_vert[0],-first_vert[0]], [last_vert[1],first_vert[1]], color=color, alpha=0.5)

                    pano_position = global_SIM2_local.transform_from(np.zeros((1,2)))
                    plt.scatter(-pano_position[0,0], pano_position[0,1], 30, marker='+', color=color)

                    point_ahead = global_SIM2_local.transform_from(np.array([1,0]).reshape(1,2))
                    
                    # TODO: add patch to Argoverse, enforcing ndim on the input to `transform_from()`
                    plt.plot(
                        [-pano_position[0,0], -point_ahead[0,0]],
                        [pano_position[0,1], point_ahead[0,1]],
                        color=color
                    )

                    image_path = pano_data["image_path"]
                    print(image_path)
                    pano_id = Path(image_path).stem.split('_')[-1]

                    pano_text = pano_data['label'] + f' {pano_id}'
                    TEXT_LEFT_OFFSET = 0.15
                    #mean_loc = np.mean(room_vertices_global, axis=0)
                    # mean position
                    noise = np.clip(np.random.randn(2), a_min=-0.05, a_max=0.05)
                    text_loc = pano_position[0] + noise
                    plt.text(-1 * (text_loc[0] - TEXT_LEFT_OFFSET), text_loc[1], pano_text, color=color, fontsize='xx-small') #, 'x-small', 'small', 'medium',)


                    # DWO objects
                    geometry_type = "layout_raw" #, "layout_complete", "layout_visible"]
                    for wdo_type in ["windows", "doors", "openings"]:
                        wdo_vertices_local = np.asarray(pano_data[geometry_type][wdo_type])

                        # Skip if there are no elements of this type
                        if len(wdo_vertices_local) == 0:
                            continue

                        # as (x1,y1), (x2,y2), bottom_z, top_z
                        # Transform the local W/D/O vertices to the global frame of reference
                        if len(wdo_vertices_local) > 2 and isinstance(wdo_vertices_local[2], float):  # with cvat bbox annotation
                            num_wdo = len(wdo_vertices_local) // 4
                            wdo_left_right_bound = []
                            for wdo_idx in range(num_wdo):
                                wdo_left_right_bound.extend(wdo_vertices_local[wdo_idx * 4 : wdo_idx * 4 + 2])
                            wdo_vertices_global = global_SIM2_local.transform_from(np.array(wdo_left_right_bound))
                        else:  # without cvat bbox annotation
                            raise RuntimeError("Expected CVAT, did not get CVAT")
                            wdo_vertices_global = global_SIM2_local.transform_from(wdo_vertices_local)

                        # Loop through the global coordinates
                        # Every two points in the list define windows/doors/openings by left and right boundaries, so
                        # for N elements we will have 2 * N pair of points, thus we iterate on every successive pair
                        for wdo_points in zip(wdo_vertices_global[::2], wdo_vertices_global[1::2]):

                            #import pdb; pdb.set_trace()
                            wdo_points = np.array(wdo_points)
                            
                            # zfm_points_list = Polygon.list_to_points(wdo_points)
                            # zfm_poly_type = PolygonTypeMapping[wdo_type]
                            # wdo_poly = Polygon(type=zfm_poly_type, name=json_file_name, points=zfm_points_list)

                            # # Add the WDO element to the list of polygons/lines
                            # wdo_poly_list.append(wdo_poly)

                            RED = [1,0,0]
                            GREEN = [0,1,0]
                            BLUE = [0,0,1]
                            wdo_color_dict = {"windows": RED, "doors": GREEN, "openings": BLUE}

                            wdo_color = wdo_color_dict[wdo_type]
                            label = wdo_type if wdo_type not in wdo_objs_seen_on_floor else None
                            plt.scatter(-wdo_points[:,0], wdo_points[:,1], 10, color=wdo_color, marker='o', label=label)
                            plt.plot(-wdo_points[:,0], wdo_points[:,1], color=wdo_color, linestyle='dotted')
                            
                            wdo_objs_seen_on_floor.add(wdo_type)


        plt.legend(loc="upper right")
        plt.title(f"Building {building_id}: {floor_id}")
        plt.axis('equal')
        building_dir = f"{OUTPUT_DIR}/{building_id}"
        os.makedirs(building_dir, exist_ok=True)
        plt.savefig(f"{building_dir}/{floor_id}.jpg", dpi=500)
        #plt.show()
        
        plt.close('all')

                    # dominant_rotation, _ = determine_rotation_angle(room_vertices_global)
                    # if dominant_rotation is None:
                    #     continue
                    # floor_dominant_rotation[floor_id] = dominant_rotation



    # geometry_to_visualize = ["raw", "complete", "visible", "redraw"]
    # for geometry_type in geometry_to_visualize:





    #         zfm_dict = zfm._zfm[geometry_type]
    #         floor_maps_files_list = zfm.render_top_down_zfm(zfm_dict, output_folder_floor_plan)
    #     for floor_id, zfm_poly_list in zfm_dict.items():







if __name__ == '__main__':
    main()