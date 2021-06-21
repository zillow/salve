
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
import scipy.spatial
from argoverse.utils.polyline_density import get_polyline_length
from argoverse.utils.interpolate import interp_arc
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Similarity3
from shapely.geometry import Point, Polygon

from sim3_align_dw import align_points_sim3

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

    def get_wd_normal_2d(self) -> np.ndarray:
        """return 2-vector describing normal to line segment (rotate CCW from vector linking pt1->pt2)"""
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        vx = x2 - x1
        vy = y2 - y1
        n = np.array([-vy, vx])
        # normalize to unit length
        return n / np.linalg.norm(n)

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

    def get_rotated_version(self) -> "WDO":
        """Rotate WDO by 180 degrees"""
        self_rotated = WDO(
            global_SIM2_local=self.global_SIM2_local,
            pt1=self.pt2,
            pt2=self.pt1,
            bottom_z=self.bottom_z,
            top_z=self.top_z,
            type=self.type
        )

        return self_rotated


def test_get_wd_normal_2d() -> None:
    """ """

    # flat horizontal line for window
    wd1 = WDO(
        global_SIM2_local=None,
        pt1=(-2,0),
        pt2=(2,0),
        bottom_z=-1,
        top_z=1,
        type="window"
    )
    n1 = wd1.get_wd_normal_2d()
    gt_n1 = np.array([0,1])
    assert np.allclose(n1, gt_n1)
    

    # upwards diagonal for window, y=x
    wd2 = WDO(
        global_SIM2_local=None,
        pt1=(0,0),
        pt2=(3,3),
        bottom_z=-1,
        top_z=1,
        type="window"
    )
    import pdb; pdb.set_trace()
    n2 = wd2.get_wd_normal_2d()
    gt_n2 = np.array([-1,1]) / np.sqrt(2)
    
    assert np.allclose(n2, gt_n2)
    



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
        #print('Camera height: ', pano_data['camera_height'])
        assert pano_data['camera_height'] == 1.0

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
        #render_building(building_id, pano_dir, json_annot_fpath)

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
        if not (floor_id == 'floor_01' and building_id == '000'):
            continue

        floor_n_valid_configurations = 0
        floor_n_invalid_configurations = 0

        pano_dict = {pano_obj.id: pano_obj for pano_obj in fd.panos}

        pano_ids = list(pano_dict.keys())
        for i1 in pano_ids:

            for i2 in pano_ids:

                # compute only upper diagonal, since symmetric
                if i1 >= i2:
                    continue

                print(f"On {i1}-{i2}")
                # _ = plot_room_layout(pano_dict[i1], coord_frame="local")
                # _ = plot_room_layout(pano_dict[i2], coord_frame="local")

                possible_alignment_info, num_invalid_configurations = align_rooms_by_wd(pano_dict[i1], pano_dict[i2])

                TODO: prune to the unique ones

                floor_n_valid_configurations += len(possible_alignment_info)
                floor_n_invalid_configurations += num_invalid_configurations

                # given wTi1, wTi2, then i2Ti1 = i2Tw * wTi1 = i2Ti1
                i2Ti1_gt = pano_dict[i2].global_SIM2_local.inverse().compose(pano_dict[i1].global_SIM2_local)
                gt_fname = f"verifier_dataset/{building_id}/{floor_id}/gt_alignment/{i1}_{i2}.json"
                save_Sim2(gt_fname, i2Ti1_gt)

                for k, (i2Ti1, alignment_object) in enumerate(possible_alignment_info):

                    proposed_fname = f"verifier_dataset/{building_id}/{floor_id}/proposed_alignment/{i1}_{i2}_{alignment_object}___variant_{k}.json"
                    save_Sim2(proposed_fname, i2Ti1)

                    print(f"\t {i2Ti1_gt.scale:.2f}")
                    print(f"\t {i2Ti1.scale:.2f}")

                    print("\t", np.round(i2Ti1_gt.translation,1))
                    print("\t", np.round(i2Ti1.translation,1))

                    print()
                    print()

        print(f"floor_n_valid_configurations: {floor_n_valid_configurations}")
        print(f"floor_n_invalid_configurations: {floor_n_invalid_configurations}")

    # for every room, try to align it to another.
    # while there are any unmatched rooms?
    # try all possible matches at every step. compute costs, and choose the best greedily.




def save_Sim2(save_fpath: str, i2Ti1: Sim2) -> None:
    """
    Args:
        save_fpath
        i2Ti1: transformation that takes points in frame1, and brings them into frame2
    """
    if not Path(save_fpath).exists():
        os.makedirs( Path(save_fpath).parent, exist_ok=True)

    from argoverse.utils.json_utils import save_json_dict
    dict_for_serialization = {
        "R": i2Ti1.rotation.flatten().tolist(),
        "t": i2Ti1.translation.flatten().tolist(),
        "s": i2Ti1.scale,
    }
    save_json_dict(save_fpath, dict_for_serialization)


def test_get_relative_angle() -> None:
    """ """
    vec1 = np.array([5,0])
    vec2 = np.array([0,9])

    angle_deg = get_relative_angle(vec1, vec2)
    assert angle_deg == 90.0


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Convert vector to equivalent unit-length version."""
    return v / np.linalg.norm(v)


def get_relative_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get angle in degrees of two vectors. We normalize them first to unit length."""
    
    # expect vector to be 2d or 3d
    assert vec1.size in [2,3]
    assert vec2.size in [2,3]

    vec1 = normalize_vec(vec1)
    vec2 = normalize_vec(vec2)

    dot_product = np.dot(vec1, vec2)
    assert dot_product >= -1.0 and dot_product <= 1.0
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg




def test_align_rooms_by_wd() -> None:
    """ """
    wTi5 = Sim2(
        R=np.array([[ 0.999897  , -0.01435102],[ 0.01435102,  0.999897  ]], dtype=np.float32),
        t=np.array([ 0.7860708, -1.57248  ], dtype=np.float32),
        s=0.4042260417272217
    )
    wTi8 = Sim2(
        R=np.array([[ 0.02998102, -0.99955046],[ 0.99955046,  0.02998102]], dtype=np.float32),
        t=np.array([ 0.91035557, -3.2141    ], dtype=np.float32),
        s=0.4042260417272217
    )

    # fmt: off
    pano1_obj = PanoData(
        id=5,
        global_SIM2_local=wTi5,
        room_vertices_local_2d=np.array(
            [
                [ 1.46363621, -2.43808616],
                [ 1.3643741 ,  0.5424695 ],
                [ 0.73380685,  0.52146958],
                [ 0.7149462 ,  1.08780075],
                [ 0.4670652 ,  1.07954551],
                [ 0.46914653,  1.01704912],
                [-1.2252865 ,  0.96061904],
                [-1.10924507, -2.5237714 ]
            ]),
        image_path='panos/floor_01_partial_room_05_pano_5.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_SIM2_local=wTi5,
                pt1=[-1.0367953294361147, -2.5213585867749635],
                pt2=[-0.4661345615720372, -2.5023537435761822],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_SIM2_local=wTi5,
                pt1=[0.823799786466513, -2.45939477144822],
                pt2=[1.404932996095547, -2.4400411621788427],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )


    pano2_obj = PanoData(
        id=8,
        global_SIM2_local=wTi8,
        room_vertices_local_2d=np.array(
            [
                [-0.7336625 , -1.3968136 ],
                [ 2.23956454, -1.16554334],
                [ 2.19063694, -0.53652654],
                [ 2.75557561, -0.4925832 ],
                [ 2.73634178, -0.2453117 ],
                [ 2.67399906, -0.25016098],
                [ 2.54252291,  1.44010577],
                [-0.93330008,  1.16974146]
            ]), 
        image_path='panos/floor_01_partial_room_05_pano_8.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_SIM2_local=wTi8,
                pt1=[-0.9276784906829552, 1.0974698581331057],
                pt2=[-0.8833992085857922, 0.5282122352406332],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_SIM2_local=wTi8,
                pt1=[-0.7833093301499523, -0.758550412558342],
                pt2=[-0.7382174598580689, -1.338254727497497],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )

    # fmt: on
    possible_alignment_info, _ = align_rooms_by_wd(pano1_obj, pano2_obj)
    assert len(possible_alignment_info) == 3


def align_rooms_by_wd(pano1_obj: PanoData, pano2_obj: PanoData, visualize: bool = True) -> Similarity3:
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
    - Rooms can be joined at windows.
    - Rooms may be joined at a door.
    - Predicted wall cannot lie behind another wall, if line of sight is clear.

    Returns a list of tuples (i2Ti1, alignment_object) where i2Ti1 is an alignment transformation
    """
    pano1_id = pano1_obj.id
    pano2_id = pano2_obj.id

    num_invalid_configurations = 0
    possible_alignment_info = []

    #import pdb; pdb.set_trace()
    for alignment_object in ["door","window"]: #[,"window"]:

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

                if alignment_object == "door":
                    plausible_configurations = ["identity", "rotated"]
                elif alignment_object == "window":
                    plausible_configurations = ["identity"]

                for configuration in plausible_configurations:

                    print(f"\t{alignment_object} {i}/{j} {configuration}")

                    if configuration == "rotated":
                        pano2_wd_ = pano2_wd.get_rotated_version()
                    else:
                        pano2_wd_ = pano2_wd

                    pano2_wd_pts = pano2_wd_.polygon_vertices_local_3d
                    #sample_points_along_bbox_boundary(wd)

                    # if visualize:
                    #     plt.close("all")
                    #     plot_room_walls(pano1_obj)
                    #     plot_room_walls(pano2_obj)

                    #     plt.plot(pano1_wd.polygon_vertices_local_3d[:,0], pano1_wd.polygon_vertices_local_3d[:,1], color="r", linewidth=5, alpha=0.1)
                    #     plt.plot(pano2_wd_.polygon_vertices_local_3d[:,0], pano2_wd_.polygon_vertices_local_3d[:,1], color="b", linewidth=5, alpha=0.1)

                    #     plt.axis("equal")
                    #     plt.title("Step 1: Before alignment")
                    #     os.makedirs(f"debug_plots/{pano1_id}_{pano2_id}", exist_ok=True)
                    #     plt.savefig(f"debug_plots/{pano1_id}_{pano2_id}/step1_{i}_{j}.jpg")
                    #     #plt.show()
                    #     plt.close("all")

                    i2Ti1, aligned_pts1 = align_points_sim3(pano2_wd_pts, pano1_wd_pts)

                    # TODO: score hypotheses by reprojection error, or registration nearest neighbor-distances
                    #evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
                    
                    if visualize:
                        plt.scatter(pano2_wd_pts[:,0], pano2_wd_pts[:,1], 200, color='r', marker='.')
                        plt.scatter(aligned_pts1[:,0], aligned_pts1[:,1], 50, color='b', marker='.')

                    all_pano1_pts = get_all_pano_wd_vertices(pano1_obj)
                    all_pano2_pts = get_all_pano_wd_vertices(pano2_obj)

                    all_pano1_pts = i2Ti1.transform_from(all_pano1_pts[:,:2])

                    if visualize:
                        plt.scatter(all_pano1_pts[:,0], all_pano1_pts[:,1], 200, color='g', marker='+')
                        plt.scatter(all_pano2_pts[:,0], all_pano2_pts[:,1], 50, color='m', marker='+')

                    pano1_room_vertices = pano1_obj.room_vertices_local_2d
                    pano1_room_vertices = i2Ti1.transform_from(pano1_room_vertices)
                    pano2_room_vertices = pano2_obj.room_vertices_local_2d

                    is_valid = determine_invalid_wall_overlap(
                        pano1_id,
                        pano2_id,
                        i,
                        j,
                        pano1_room_vertices,
                        pano2_room_vertices,
                        wall_buffer_m=0.3
                    )

                    # window_normals_compatible = True
                    # WINDOW_NORMAL_ALIGNMENT_THRESH = 10
                    # # if touch at windows, normals must be in the same direction
                    # if alignment_object == "window":
                    #     import pdb; pdb.set_trace()
                    #     #import pdb; pdb.set_trace()
                    #     window_normal1 = pano1_wd.get_wd_normal_2d()
                    #     window_normal1 = i2Ti1.rotation @ window_normal1 # this must be a rotation, and no translation may be added here
                    #     window_normal2 = pano2_wd_.get_wd_normal_2d()
                    #     print("\tN1: ", np.round(window_normal1, 1))
                    #     print("\tN2: ", np.round(window_normal2, 1))

                    #     plt.plot([0, window_normal1[0]], [0, window_normal1[1]], color="r", linewidth=10, alpha=0.1)
                    #     plt.plot([0, window_normal2[0]], [0, window_normal2[1]], color="b", linewidth=1, alpha=0.1)

                    #     rel_angle_deg = get_relative_angle(vec1=window_normal1, vec2=window_normal2)
                    #     print("\tRelative angle (deg.) = ", rel_angle_deg)
                        
                    #     if rel_angle_deg > WINDOW_NORMAL_ALIGNMENT_THRESH:
                    #         is_valid = False
                    #         window_normals_compatible = False
                    #         #print("Alignment of window normals is invalid/implausible")

                    if is_valid:
                        possible_alignment_info += [ (i2Ti1, alignment_object) ]
                        classification = "valid"
                    else:
                        num_invalid_configurations += 1
                        classification = "invalid"
                    
                    if visualize:
                        plot_room_walls(pano1_obj, i2Ti1, linewidth=10)
                        plot_room_walls(pano2_obj, linewidth=1)

                        plt.scatter(aligned_pts1[:,0], aligned_pts1[:,1], 10, color="r", marker="+")
                        plt.plot(aligned_pts1[:,0], aligned_pts1[:,1], color="r", linewidth=10, alpha=0.2)

                        plt.scatter(pano2_wd_pts[:,0], pano2_wd_pts[:,1], 10, color="g", marker="+")
                        plt.plot(pano2_wd_pts[:,0], pano2_wd_pts[:,1], color="g", linewidth=5, alpha=0.1)

                        # plt.plot(inter_poly_verts[:,0],inter_poly_verts[:,1], color='m')

                        plt.title(f"Step 3: Match: ({pano1_id},{pano2_id}): valid={is_valid}, aligned via {alignment_object}, \n  config={configuration}")
                        # window_normals_compatible={window_normals_compatible},
                        plt.axis('equal')
                        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
                        plt.savefig(f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step3_{i}_{j}.jpg")

                        #plt.show()
                        plt.close('all')


    #assert best_alignment_object in ["door","window"]
    return possible_alignment_info, num_invalid_configurations



def shrink_polygon(polygon: Polygon, shrink_factor: float = 0.10) -> Polygon:
    """
    Args:
        shrink_factor: shrink by 10%
    """
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    # find the minimum volume enclosing bounding box, and treat its center as polygon centroid
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * shrink_factor

    polygon_shrunk = polygon.buffer(-shrink_distance) #shrink
    return polygon_shrunk


def interp_evenly_spaced_points(polyline: np.ndarray, interval_m) -> np.ndarray:
    """Nx2 polyline to Mx2 polyline, for waypoint every `interval_m` meters"""

    length_m = get_polyline_length(polyline)
    n_waypoints = int(np.ceil(length_m / interval_m))
    interp_polyline = interp_arc(t=n_waypoints, px=polyline[:,0], py=polyline[:,1])

    return interp_polyline


def determine_invalid_wall_overlap(
    pano1_id: int,
    pano2_id: int,
    i: int,
    j: int,
    pano1_room_vertices: np.ndarray, pano2_room_vertices: np.ndarray, wall_buffer_m: float = 0.3, visualize: bool = True
) -> bool:
    """
    TODO: consider adding allowed_overlap_pct: float = 0.01
        Args:
            pano1_id: panorama id
            pano2_id: panorama id
            i: id of WDO to match from pano1
            j: id of WDO to match from pano2

        TODO: use `wall_buffer_m`

        Returns:
    """
    polygon1 = Polygon(pano1_room_vertices)
    polygon2 = Polygon(pano2_room_vertices)

    pano1_room_vertices_interp = interp_evenly_spaced_points(pano1_room_vertices, interval_m=0.1) # meters
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1) # meters

    # # should be in the same coordinate frame, now
    # inter_poly = polygon1.intersection(polygon2)
    # inter_poly_area = inter_poly.area

    # inter_poly_verts = np.array(list(inter_poly.exterior.coords))

    shrunk_poly1 = shrink_polygon(polygon1)
    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    shrunk_poly2 = shrink_polygon(polygon2)
    shrunk_poly2_verts = np.array(list(shrunk_poly2.exterior.coords))

    def count_verts_inside_poly(polygon: Polygon, query_verts: np.ndarray) -> int:
        """
        Args:
        query vertices
        """
        num_violations = 0
        for vert in query_verts:
            v_pt = Point(vert)
            if polygon.contains(v_pt) or polygon.contains(v_pt):
                num_violations += 1
        return num_violations

    # TODO: interpolate evenly spaced points along edges, if this gives any benefit?
    # Cannot be the case that poly1 vertices fall within a shrinken version of polygon2
    # also, cannot be the case that poly2 vertices fall within a shrunk version of polygon1
    pano1_violations = count_verts_inside_poly(shrunk_poly1, query_verts=pano2_room_vertices_interp)
    pano2_violations = count_verts_inside_poly(shrunk_poly2, query_verts=pano1_room_vertices_interp)
    num_violations = pano1_violations + pano2_violations

    is_valid = num_violations == 0

    if visualize:
        # plot the overlap region
        plt.close("all")

        # plot the interpolated points via scatter, but keep the original lines
        plt.scatter(pano1_room_vertices_interp[:,0], pano1_room_vertices_interp[:,1], 10, color="m")
        plt.plot(pano1_room_vertices[:,0], pano1_room_vertices[:,1], color="m", linewidth=20, alpha=0.1)

        plt.scatter(pano2_room_vertices_interp[:,0], pano2_room_vertices_interp[:,1], 10, color="g")
        plt.plot(pano2_room_vertices[:,0], pano2_room_vertices[:,1], color="g", linewidth=10, alpha=0.1)

        plt.scatter(shrunk_poly1_verts[:,0], shrunk_poly1_verts[:,1], 10, color="r")
        plt.plot(shrunk_poly1_verts[:,0], shrunk_poly1_verts[:,1], color="r", linewidth=1, alpha=0.1)

        plt.scatter(shrunk_poly2_verts[:,0], shrunk_poly2_verts[:,1], 10, color="b")
        plt.plot(shrunk_poly2_verts[:,0], shrunk_poly2_verts[:,1], color="b", linewidth=1, alpha=0.1)


        plt.title(f"Step 2: Number of violations: {num_violations}")
        plt.axis("equal")
        #plt.show()

        classification = "invalid" if num_violations > 0 else "valid"

        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
        plt.savefig(f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step2_{i}_{j}.jpg")

    return is_valid


def test_determine_invalid_wall_overlap1() -> None:
    """large horseshoe, and small horseshoe

    .---.---.
    |       |
    .   .xxx.
    |       x|
    .   .xxx.
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [2,4],
            [3,4],
            [3,3],
            [2,3]
        ])

    # fmt: on
    wall_buffer_m = 0.2 # 20 centimeter noise buffer
    allowed_overlap_pct = 0.01 # TODO: allow 1% of violations ???

    is_valid = determine_invalid_wall_overlap(
        pano1_room_vertices,
        pano2_room_vertices,
        wall_buffer_m
    )
    assert not is_valid




def test_determine_invalid_wall_overlap2() -> None:
    """identical shape

    .---.---.
    |       |
    .       .
    |       |
    .       .
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])

    # fmt: on
    wall_buffer_m = 0.2 # 20 centimeter noise buffer
    allowed_overlap_pct = 0.01 # TODO: allow 1% of violations ???

    is_valid = determine_invalid_wall_overlap(
        pano1_room_vertices,
        pano2_room_vertices,
        wall_buffer_m
    )
    assert is_valid



def plot_room_walls(pano_obj: PanoData, i2Ti1: Optional[Sim2] = None, linewidth: float = 2.0, alpha: float = 0.5) -> None:
    """ """

    room_vertices = pano_obj.room_vertices_local_2d
    if i2Ti1:
        room_vertices = i2Ti1.transform_from(room_vertices)

    color = np.random.rand(3)
    plt.scatter(room_vertices[:,0], room_vertices[:,1], 10, marker='.', color=color, alpha=alpha)
    plt.plot(room_vertices[:,0], room_vertices[:,1], color=color, alpha=alpha, linewidth=linewidth)
    # draw edge to close each polygon
    last_vert = room_vertices[-1]
    first_vert = room_vertices[0]
    plt.plot([last_vert[0],first_vert[0]], [last_vert[1],first_vert[1]], color=color, alpha=alpha, linewidth=linewidth)


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

    hohopano_Sim2_zindpano = Sim2(R=rotmat2d(-90),t=np.zeros(2),s=1.0)

    assert coord_frame in ["global","local"]

    if coord_frame == "global":
        room_vertices = pano_obj.room_vertices_global_2d
    else:
        room_vertices = pano_obj.room_vertices_local_2d

    room_vertices = hohopano_Sim2_zindpano.transform_from(room_vertices)
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

        wdo_points = hohopano_Sim2_zindpano.transform_from(wdo_points)
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

        if (wdo_objs_seen_on_floor is not None) and (wdo_type not in wdo_objs_seen_on_floor):
            label = wdo_type
        else:
            label = None
        plt.scatter(wdo_points[:,0], wdo_points[:,1], 10, color=wdo_color, marker='o', label=label)
        plt.plot(wdo_points[:,0], wdo_points[:,1], color=wdo_color, linestyle='dotted')
        plt.text(wdo_points[:,0].mean(), wdo_points[:,1].mean(), f"{wdo.type}_{wdo_idx}")

        if wdo_objs_seen_on_floor is not None:
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
    #test_determine_invalid_wall_overlap1()
    #test_determine_invalid_wall_overlap2()
    #test_get_wd_normal_2d()
    #test_get_relative_angle()
    #test_align_rooms_by_wd()

