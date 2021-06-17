
"""
Based on code found at
https://gitlab.zgtools.net/zillow/rmx/research/scripts-insights/open_platform_utils/-/raw/zind_cleanup/zfm360/visualize_zfm360_data.py

Data can be found at:
/mnt/data/zhiqiangw/ZInD_release/complete_zind_paper_final_localized_json_6_3_21
"""


import collections
import glob
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Similarity3

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


class WDO(NamedTuple):
    """define windows/doors/openings by left and right boundaries"""
    global_SIM2_local: Sim2
    pt1: Tuple[float,float] # (x1,y1)
    pt2: Tuple[float,float] # (x2,y2)
    bottom_z: float
    top_z: float
    type: str

    @property
    def vertices_local_2d(self) -> np.ndarray:
        """ """
        return np.array([self.pt1, self.pt2])

    # TODO: come up with better name for vertices in BEV, vs. all 4 vertices
    @property
    def vertices_global_2d(self) -> np.ndarray:
        """ """
        return self.global_SIM2_local.transform_from(self.vertices_local_2d)

    @property
    def vertices_local_3d(self) -> np.ndarray:
        """ """
        x1,y1 = self.pt1
        x2,y2 = self.pt2
        return np.array([ [x1,y1,bottom_z], [x2,y2,top_z] ])

    @property
    def vertices_global_3d(self) -> np.ndarray:
        """ """
        return self.global_SIM2_local.transform_from(self.vertices_local_3d)

    @property
    def polygon_vertices_local_3d(self) -> np.ndarray:
        """Note: first vertex is repeated as last vertex"""
        x1,y1 = self.pt1
        x2,y2 = self.pt2
        return np.array([ [x1,y1,self.bottom_z], [x1,y1,self.top_z], [x2,y2,self.top_z], [x2,y2,self.bottom_z], [x1,y1,self.bottom_z] ])


    @classmethod
    def from_object_array(cls, wdo_data: Any, global_SIM2_local: Sim2, type: str) -> "WDO":
        """
        """
        return cls(
            global_SIM2_local=global_SIM2_local,
            pt1=wdo_data[0],
            pt2=wdo_data[1],
            bottom_z=wdo_data[2],
            top_z=wdo_data[3],
            type=type
        )


class PanoData(NamedTuple):
    """ """
    id: int
    global_SIM2_local: Sim2
    room_vertices_local_2d: np.ndarray
    image_path: str
    label: str
    doors: Optional[List[WDO]] = None
    windows: Optional[List[WDO]] = None
    openings: Optional[List[WDO]] = None

    @property
    def room_vertices_global_2d(self):
        """ """
        return self.global_SIM2_local.transform_from(self.room_vertices_local_2d)

    @classmethod
    def from_json(cls, pano_data: Any) -> "PanoData":
        """
        From JSON ("pano_data")
        """
        print('Camera height: ', pano_data['camera_height'])

        global_SIM2_local = generate_Sim2_from_floorplan_transform(pano_data["floor_plan_transformation"])
        room_vertices_local_2d = np.asarray(pano_data["layout_raw"]["vertices"])

        doors = []
        windows = []
        openings = []
        image_path = pano_data["image_path"]
        label = pano_data['label']

        pano_id = int(Path(image_path).stem.split('_')[-1])

        # DWO objects
        geometry_type = "layout_raw" #, "layout_complete", "layout_visible"]
        for wdo_type in ["windows", "doors", "openings"]:
            wdos = []
            wdo_data = np.asarray(pano_data[geometry_type][wdo_type], dtype=object)

            # Skip if there are no elements of this type
            if len(wdo_data) == 0:
                continue

            # as (x1,y1), (x2,y2), bottom_z, top_z
            # Transform the local W/D/O vertices to the global frame of reference
            assert len(wdo_data) > 2 and isinstance(wdo_data[2], float)  # with cvat bbox annotation

            num_wdo = len(wdo_data) // 4
            for wdo_idx in range(num_wdo):
                wdo = WDO.from_object_array(wdo_data[ wdo_idx * 4 : (wdo_idx+1) * 4], global_SIM2_local, wdo_type)
                wdos.append(wdo)

            if wdo_type == "windows":
                windows = wdos
            elif wdo_type == "doors":
                doors = wdos
            elif wdo_type == "openings":
                openings = wdos

        return cls(pano_id, global_SIM2_local, room_vertices_local_2d, image_path, label, doors, windows, openings)



class FloorData(NamedTuple):
    """ """
    floor_id: str
    panos: List[PanoData]

    @classmethod
    def from_json(cls, floor_data: Any, floor_id: str) -> "FloorData":
        """
        From JSON ("floor_data")
        """
        pano_objs = []

        for complete_room_data in floor_data.values():
            for partial_room_data in complete_room_data.values():
                for pano_data in partial_room_data.values():
                    #if pano_data["is_primary"]:

                    pano_obj = PanoData.from_json(pano_data)
                    pano_objs.append(pano_obj)

        return cls(floor_id, pano_objs)


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

        align_by_wdo(building_id, pano_dir, json_annot_fpath)


OUTPUT_DIR = "/Users/johnlam/Downloads/ZinD_Vis_2021_06_17"


def rotmat2d(theta_deg: float) -> np.ndarray:
    """Generate 2x2 rotation matrix, given a rotation angle in degrees."""
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
    return R


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

    R = rotmat2d(theta_deg)

    global_SIM2_local = Sim2(R=R, t=t, s=scale)
    return global_SIM2_local



def align_by_wdo(building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """ """
    floor_map_json = read_json_file(json_annot_fpath)

    merger_data = floor_map_json["merger"]
    redraw_data = floor_map_json["redraw"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        fd = FloorData.from_json(floor_data, floor_id)
        if floor_id != 'floor01' and building_id != '000':
            continue

        pano_dict = {pano_obj.id: pano_obj for pano_obj in fd.panos}

        pano_ids = list(pano_dict.keys())
        for i1 in pano_ids:

            for i2 in pano_ids:

                if i1 == i2:
                    continue

                print(f"On {i1}-{i2}")
                _ = plot_room_layout(pano_dict[i1], coord_frame="local")
                _ = plot_room_layout(pano_dict[i2], coord_frame="local")

                i2Ti1, error = align_rooms_by_wd(pano_dict[i1], pano_dict[i2])

                # given wTi1, wTi2, then i2Ti1 = i2Tw * wTi1 = i2Ti1
                i2Ti1_gt = pano_dict[i2].global_SIM2_local.inverse().compose(pano_dict[i1].global_SIM2_local)

                #import pdb; pdb.set_trace()

                print("\t", i2Ti1_gt.scale)
                print("\t", i2Ti1.scale)

                print("\t", np.round(i2Ti1_gt.translation,1))
                print("\t", np.round(i2Ti1.translation,1))

                print()
                print()

    # for every room, try to align it to another.
    # while there are any unmatched rooms?
    # try all possible matches at every step. compute costs, and choose the best greedily.




def align_rooms_by_wd(pano1_obj: PanoData, pano2_obj: PanoData) -> Similarity3:
    """
    Window-Window correspondences must be established. May have to find all possible pairwise choices, or ICP?

    Cohen16: A single match between an indoor and outdoor window determines an alignment hypothesis
    Computing the alignment boils down to finding a similarity transformation between the models, which
    can be computed from three point correspondences in the general case and from two point matches if
    the gravity direction is known.

    TODO: maybe perform alignment in 2d, instead of 3d? so we have less unknowns?
    
    Args:
        pano1_obj
        pano2_obj
        alignment_object: "door" or "window"

    What heuristic tells us if they should be identity or mirrored in configuration?
    Are there any new WDs that should not be visible? walls should not cross on top of each other? know same-room connections, first
    
    Cannot expect a match for each door or window. Find nearest neighbor -- but then choose min dist on rows or cols?
    may not be visible?
    - Rooms cannot be joined at windows.
    - Rooms may be joined at a door.
    - Predicted wall cannot lie behind another wall, if line of sight is clear.
    """
    import scipy.spatial
    from sim3_align_dw import align_points_sim3

    best_alignment_error = np.nan

    for alignment_object in ["door"]:#, "window"]:

        pano1_wds = pano1_obj.windows if alignment_object=="window" else pano1_obj.doors
        pano2_wds = pano2_obj.windows if alignment_object=="window" else pano2_obj.doors

        # try every possible pairwise combination, for this object type
        for i, pano1_wd in enumerate(pano1_wds):
            pano1_wd_pts = pano1_wd.polygon_vertices_local_3d
            #sample_points_along_bbox_boundary(wd), # TODO: add in the 3d linear interpolation

            for j, pano2_wd in enumerate(pano2_wds):

                if i > j:
                    # only compute the upper diagonal, since symmetric
                    continue

                pano2_wd_pts = pano2_wd.polygon_vertices_local_3d
                #sample_points_along_bbox_boundary(wd)

                for configuration in ["identity", "rotated"]:

                    if configuration == "rotated":
                        pano2_wd_pts = pano2_wd_pts[ np.array([3,2,1,0,3]) ] # instead of 0,1,2,3,0

                    i2Ti1, aligned_pts1 = align_points_sim3(pano2_wd_pts, pano1_wd_pts)

                    # TODO: score hypotheses by reprojection error, or registration nearest neighbor-distances
                    #evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
                    
                    visualize = True
                    if visualize:
                        plt.scatter(pano2_wd_pts[:,0], pano2_wd_pts[:,1], 200, color='r', marker='.')
                        plt.scatter(aligned_pts1[:,0], aligned_pts1[:,1], 50, color='b', marker='.')

                    all_pano1_pts = get_all_pano_wd_vertices(pano1_obj)
                    all_pano2_pts = get_all_pano_wd_vertices(pano2_obj)

                    all_pano1_pts = i2Ti1.transform_from(all_pano1_pts[:,:2])

                    if visualize:
                        plt.scatter(all_pano1_pts[:,0], all_pano1_pts[:,1], 200, color='g', marker='+')
                        plt.scatter(all_pano2_pts[:,0], all_pano2_pts[:,1], 50, color='m', marker='+')

                    # should be in the same coordinate frame, now
                    affinity_matrix = scipy.spatial.distance.cdist(all_pano1_pts, all_pano2_pts[:,:2])
                    print(np.round(affinity_matrix,1))
                    closest_dists = np.min(affinity_matrix, axis=1)
                    avg_error = closest_dists.mean()
                    print(f"\t{alignment_object} {i}/{j}: Cost {avg_error:.2f}")

                    best_alignment_error = min(best_alignment_error, avg_error)

                    if visualize:
                        plot_room_walls(pano1_obj, i2Ti1)
                        plot_room_walls(pano2_obj)

                        pano1_id = pano1_obj.id
                        pano2_id = pano2_obj.id
                        plt.title(f"Match: ({pano1_id},{pano2_id})")
                        plt.axis('equal')
                        plt.show()
                        plt.close('all')

    return i2Ti1, best_alignment_error



def plot_room_walls(pano_obj: PanoData, i2Ti1: Optional[Sim2] = None) -> None:
    """ """

    room_vertices = pano_obj.room_vertices_local_2d
    if i2Ti1:
        room_vertices = i2Ti1.transform_from(room_vertices)

    color = np.random.rand(3)
    plt.scatter(room_vertices[:,0], room_vertices[:,1], 10, marker='.', color=color)
    plt.plot(room_vertices[:,0], room_vertices[:,1], color=color, alpha=0.5)
    # draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot([last_vert[0],first_vert[0]], [last_vert[1],first_vert[1]], color=color, alpha=0.5)



def get_all_pano_wd_vertices(pano_obj: PanoData) -> np.ndarray:
    """

    Returns:
        pts: array of shape (N,3)
    """
    pts = np.zeros((0,3))

    for wd in pano_obj.windows + pano_obj.doors:
        wd_pts = wd.polygon_vertices_local_3d

        pts = np.vstack([pts, wd_pts])

    return pts


# def sample_points_along_bbox_boundary(wdo: WDO) -> np.ndarray:
#     """ """

#     from argoverse.utils.interpolate import interp_arc
#     from argoverse.utils.polyline_density import get_polyline_length

#     x1,y1 = wdo.local_vertices_3d

#     num_interp_pts = int(query_l2 * NUM_PTS_PER_TRAJ / ref_l2)
#     dense_interp_polyline = interp_arc(num_interp_pts, polyline_to_interp[:, 0], polyline_to_interp[:, 1])

#     return pts


# def test_sample_points_along_bbox_boundary() -> None:
#     """ """
#     global_SIM2_local = Sim2(R=np.eye(2), t=np.zeros(2), s=1.0)

#     wdo = WDO(
#         global_SIM2_local= global_SIM2_local,
#         pt1= (x1,y1),
#         pt2= (x2,y2),
#         bottom_z=0,
#         top_z=10,
#         type="door"
#     )

#     pts = sample_points_along_bbox_boundary(wdo)




def plot_room_layout(pano_obj: PanoData, coord_frame: str, wdo_objs_seen_on_floor: Optional[Set] = None, show_plot: bool = True) -> None:
    """
    coord_frame: either 'local' or 'global'
    """
    R_refl = np.array([[-1.,0],[0,1]])
    world_Sim2_reflworld = Sim2(R_refl, t=np.zeros(2), s=1.0)

    assert coord_frame in ["global","local"]

    if coord_frame == "global":
        room_vertices = pano_obj.room_vertices_global_2d
    else:
        room_vertices = pano_obj.room_vertices_local_2d

    room_vertices = world_Sim2_reflworld.transform_from(room_vertices)

    color = np.random.rand(3)
    plt.scatter(room_vertices[:,0], room_vertices[:,1], 10, marker='.', color=color)
    plt.plot(room_vertices[:,0], room_vertices[:,1], color=color, alpha=0.5)
    # draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot([last_vert[0],first_vert[0]], [last_vert[1],first_vert[1]], color=color, alpha=0.5)

    pano_position = np.zeros((1,2))
    if coord_frame == "global":
        pano_position_local = pano_position
        pano_position_reflworld = pano_obj.global_SIM2_local.transform_from(pano_position_local)
        pano_position_world = world_Sim2_reflworld.transform_from(pano_position_reflworld)
        pano_position = pano_position_world
    else:
        pano_position_refllocal = pano_position
        pano_position_local = world_Sim2_reflworld.transform_from(pano_position_refllocal)
        pano_position = pano_position_local

    plt.scatter(pano_position[0,0], pano_position[0,1], 30, marker='+', color=color)

    point_ahead = np.array([1,0]).reshape(1,2)
    if coord_frame == "global":
        point_ahead = pano_obj.global_SIM2_local.transform_from(point_ahead)
    
    point_ahead = world_Sim2_reflworld.transform_from(point_ahead)
    
    # TODO: add patch to Argoverse, enforcing ndim on the input to `transform_from()`
    plt.plot(
        [pano_position[0,0], point_ahead[0,0]],
        [pano_position[0,1], point_ahead[0,1]],
        color=color
    )
    pano_id = pano_obj.id

    pano_text = pano_obj.label + f' {pano_id}'
    TEXT_LEFT_OFFSET = 0.15
    #mean_loc = np.mean(room_vertices_global, axis=0)
    # mean position
    noise = np.clip(np.random.randn(2), a_min=-0.05, a_max=0.05)
    text_loc = pano_position[0] + noise
    plt.text((text_loc[0] - TEXT_LEFT_OFFSET), text_loc[1], pano_text, color=color, fontsize='xx-small') #, 'x-small', 'small', 'medium',)

    for wdo_idx, wdo in enumerate(pano_obj.doors + pano_obj.windows + pano_obj.openings):

        if coord_frame == "global":
            wdo_points = wdo.vertices_global_2d
        else:
            wdo_points = wdo.vertices_local_2d

        wdo_points = world_Sim2_reflworld.transform_from(wdo_points)
            
        # zfm_points_list = Polygon.list_to_points(wdo_points)
        # zfm_poly_type = PolygonTypeMapping[wdo_type]
        # wdo_poly = Polygon(type=zfm_poly_type, name=json_file_name, points=zfm_points_list)

        # # Add the WDO element to the list of polygons/lines
        # wdo_poly_list.append(wdo_poly)

        RED = [1,0,0]
        GREEN = [0,1,0]
        BLUE = [0,0,1]
        wdo_color_dict = {"windows": RED, "doors": GREEN, "openings": BLUE}

        wdo_type = wdo.type
        wdo_color = wdo_color_dict[wdo_type]
        label = wdo_type if wdo_type not in wdo_objs_seen_on_floor else None
        plt.scatter(wdo_points[:,0], wdo_points[:,1], 10, color=wdo_color, marker='o', label=label)
        plt.plot(wdo_points[:,0], wdo_points[:,1], color=wdo_color, linestyle='dotted')
        plt.text(wdo_points[:,0].mean(), wdo_points[:,1].mean(), f"{wdo.type}_{wdo_idx}")

        if wdo_objs_seen_on_floor:
            wdo_objs_seen_on_floor.add(wdo_type)
        
    if show_plot:
        plt.axis('equal')
        plt.show()
        plt.close('all')

    return wdo_objs_seen_on_floor


def render_building(building_id: str, pano_dir: str, json_annot_fpath: str) -> None:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'
    """
    floor_map_json = read_json_file(json_annot_fpath)

    merger_data = floor_map_json["merger"]
    redraw_data = floor_map_json["redraw"]

    floor_dominant_rotation = {}
    for floor_id, floor_data in merger_data.items():

        fd = FloorData.from_json(floor_data, floor_id)

        wdo_objs_seen_on_floor = set()
        # Calculate the dominant rotation of one room on this floor.
        # This dominant rotation will be used to align the floor map.

        for pano_obj in fd.panos:
            wdo_objs_seen_on_floor = plot_room_layout(pano_obj, coord_frame="global", wdo_objs_seen_on_floor=wdo_objs_seen_on_floor, show_plot=False)


        plt.legend(loc="upper right")
        plt.title(f"Building {building_id}: {floor_id}")
        plt.axis('equal')
        building_dir = f"{OUTPUT_DIR}/{building_id}"
        os.makedirs(building_dir, exist_ok=True)
        #plt.savefig(f"{building_dir}/{floor_id}.jpg", dpi=500)
        plt.show()
        
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


def test_reflections_1() -> None:
    """
    Compose does not work properly for chained reflection and rotation.
    """

    pts_local = np.array(
        [
            [2,1],
            [1,1],
            [1,2],
            [1,3],
            [2,3],
            [2,2],
            [1,2]
        ])

    plt.scatter(pts_local[:,0], pts_local[:,1], 10, color='r', marker='.')
    plt.plot(pts_local[:,0], pts_local[:,1], color='r')

    R = rotmat2d(45) # 45 degree rotation
    t = np.array([1,1])
    s = 2.0
    world_Sim2_local = Sim2(R, t, s)

    pts_world = world_Sim2_local.transform_from(pts_local)

    plt.scatter(pts_world[:,0], pts_world[:,1], 10, color='b', marker='.')
    plt.plot(pts_world[:,0], pts_world[:,1], color='b')

    plt.scatter(pts_world[:,0], pts_world[:,1], 10, color='b', marker='.')
    plt.plot(pts_world[:,0], pts_world[:,1], color='b')

    R_refl = np.array([[-1.,0],[0,1]])
    reflectedworld_Sim2_world = Sim2(R_refl, t=np.zeros(2), s=1.0)

    #import pdb; pdb.set_trace()
    pts_reflworld = reflectedworld_Sim2_world.transform_from(pts_world)

    plt.scatter(pts_reflworld[:,0], pts_reflworld[:,1], 100, color='g', marker='.')
    plt.plot(pts_reflworld[:,0], pts_reflworld[:,1], color='g', alpha=0.3)

    plt.scatter(-pts_world[:,0], pts_world[:,1], 10, color='m', marker='.')
    plt.plot(-pts_world[:,0], pts_world[:,1], color='m', alpha=0.3)

    plt.axis('equal')
    plt.show()


def test_reflections_2() -> None:
    """
    Try reflection -> rotation -> compare relative poses

    rotation -> reflection -> compare relative poses

    Relative pose is identical, but configuration will be different in the absolute world frame
    """
    pts_local = np.array(
        [
            [2,1],
            [1,1],
            [1,2],
            [1,3],
            [2,3],
            [2,2],
            [1,2]
        ])

    R_refl = np.array([[-1.,0],[0,1]])
    identity_Sim2_reflected = Sim2(R_refl, t=np.zeros(2), s=1.0)

    #pts_refl = identity_Sim2_reflected.transform_from(pts_local)
    pts_refl = pts_local

    R = rotmat2d(45) # 45 degree rotation
    t = np.array([1,1])
    s = 1.0
    world_Sim2_i1 = Sim2(R, t, s)

    R = rotmat2d(45) # 45 degree rotation
    t = np.array([1,2])
    s = 1.0
    world_Sim2_i2 = Sim2(R, t, s)

    pts_i1 = world_Sim2_i1.transform_from(pts_refl)
    pts_i2 = world_Sim2_i2.transform_from(pts_refl)

    pts_i1 = identity_Sim2_reflected.transform_from(pts_i1)
    pts_i2 = identity_Sim2_reflected.transform_from(pts_i2)

    plt.scatter(pts_i1[:,0], pts_i1[:,1], 10, color='b', marker='.')
    plt.plot(pts_i1[:,0], pts_i1[:,1], color='b')

    plt.scatter(pts_i2[:,0], pts_i2[:,1], 10, color='g', marker='.')
    plt.plot(pts_i2[:,0], pts_i2[:,1], color='g')

    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
    #test_reflections_2()




