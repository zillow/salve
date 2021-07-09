
"""
Transformation (R,t) followed by a reflection over the x-axis is equivalent to 
Transformation by (R^T,-t) followed by no reflection.
"""


from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2

from sim3_align_dw import rotmat2d

class WDO(NamedTuple):
    """define windows/doors/openings by left and right boundaries"""
    global_Sim2_local: Sim2
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
        return self.global_Sim2_local.transform_from(self.vertices_local_2d)

    @property
    def vertices_local_3d(self) -> np.ndarray:
        """ """
        x1,y1 = self.pt1
        x2,y2 = self.pt2
        return np.array([ [x1,y1,bottom_z], [x2,y2,top_z] ])

    @property
    def vertices_global_3d(self) -> np.ndarray:
        """ """
        return self.global_Sim2_local.transform_from(self.vertices_local_3d)

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
    def from_object_array(cls, wdo_data: Any, global_Sim2_local: Sim2, type: str) -> "WDO":
        """
        """
        pt1 = wdo_data[0]
        pt2 = wdo_data[1]
        pt1[0] *= -1
        pt2[0] *= -1
        return cls(
            global_Sim2_local=global_Sim2_local,
            pt1=pt1,
            pt2=pt2,
            bottom_z=wdo_data[2],
            top_z=wdo_data[3],
            type=type
        )

    def get_rotated_version(self) -> "WDO":
        """Rotate WDO by 180 degrees"""
        self_rotated = WDO(
            global_Sim2_local=self.global_Sim2_local,
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
        global_Sim2_local=None,
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
        global_Sim2_local=None,
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
    global_Sim2_local: Sim2
    room_vertices_local_2d: np.ndarray
    image_path: str
    label: str
    doors: Optional[List[WDO]] = None
    windows: Optional[List[WDO]] = None
    openings: Optional[List[WDO]] = None

    @property
    def room_vertices_global_2d(self):
        """ """
        return self.global_Sim2_local.transform_from(self.room_vertices_local_2d)

    @classmethod
    def from_json(cls, pano_data: Any) -> "PanoData":
        """
        From JSON ("pano_data")
        """
        #print('Camera height: ', pano_data['camera_height'])
        assert pano_data['camera_height'] == 1.0

        doors = []
        windows = []
        openings = []
        image_path = pano_data["image_path"]
        label = pano_data['label']

        pano_id = int(Path(image_path).stem.split('_')[-1])

        global_Sim2_local = generate_Sim2_from_floorplan_transform(pano_data["floor_plan_transformation"])
        room_vertices_local_2d = np.asarray(pano_data["layout_raw"]["vertices"])
        room_vertices_local_2d[:,0] *= -1

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
                wdo = WDO.from_object_array(wdo_data[ wdo_idx * 4 : (wdo_idx+1) * 4], global_Sim2_local, wdo_type)
                wdos.append(wdo)

            if wdo_type == "windows":
                windows = wdos
            elif wdo_type == "doors":
                doors = wdos
            elif wdo_type == "openings":
                openings = wdos

        return cls(pano_id, global_Sim2_local, room_vertices_local_2d, image_path, label, doors, windows, openings)



    def plot_room_layout(
        self, coord_frame: str, wdo_objs_seen_on_floor: Optional[Set] = None, show_plot: bool = True
    ) -> None:
        """Plot the room shape for this panorama, either in a global or local frame

        Args:
            coord_frame: either 'local' or 'global'
            wdo_objs_seen_on_floor
            show_plot
        
        Returns:

        """
        #hohopano_Sim2_zindpano = Sim2(R=rotmat2d(-90), t=np.zeros(2), s=1.0)

        assert coord_frame in ["global", "local"]

        if coord_frame == "global":
            room_vertices = self.room_vertices_global_2d
        else:
            room_vertices = self.room_vertices_local_2d

        #room_vertices = hohopano_Sim2_zindpano.transform_from(room_vertices)

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
            pano_position_world = self.global_SIM2_local.transform_from(pano_position_local)
            pano_position = pano_position_world

        plt.scatter(pano_position[0, 0], pano_position[0, 1], 30, marker="+", color=color)

        point_ahead = np.array([1, 0]).reshape(1, 2)
        if coord_frame == "global":
            point_ahead = self.global_SIM2_local.transform_from(point_ahead)

        plt.plot([pano_position[0, 0], point_ahead[0, 0]], [pano_position[0, 1], point_ahead[0, 1]], color=color)
        pano_id = self.id

        pano_text = self.label + f" {pano_id}"
        TEXT_LEFT_OFFSET = 0.15
        # mean_loc = np.mean(room_vertices_global, axis=0)
        # mean position
        noise = np.clip(np.random.randn(2), a_min=-0.05, a_max=0.05)
        text_loc = pano_position[0] + noise
        plt.text(
            (text_loc[0] - TEXT_LEFT_OFFSET), text_loc[1], pano_text, color=color, fontsize="xx-small"
        )  # , 'x-small', 'small', 'medium',)

        for wdo_idx, wdo in enumerate(self.doors + self.windows + self.openings):

            if coord_frame == "global":
                wdo_points = wdo.vertices_global_2d
            else:
                wdo_points = wdo.vertices_local_2d

            #wdo_points = hohopano_Sim2_zindpano.transform_from(wdo_points)
            wdo_type = wdo.type
            wdo_color = wdo_color_dict[wdo_type]

            if (wdo_objs_seen_on_floor is not None) and (wdo_type not in wdo_objs_seen_on_floor):
                label = wdo_type
            else:
                label = None
            plt.scatter(wdo_points[:, 0], wdo_points[:, 1], 10, color=wdo_color, marker="o", label=label)
            plt.plot(wdo_points[:, 0], wdo_points[:, 1], color=wdo_color, linestyle="dotted")
            #plt.text(wdo_points[:, 0].mean(), wdo_points[:, 1].mean(), f"{wdo.type}_{wdo_idx}")

            if wdo_objs_seen_on_floor is not None:
                wdo_objs_seen_on_floor.add(wdo_type)

        if show_plot:
            plt.axis("equal")
            plt.show()
            plt.close("all")

        return wdo_objs_seen_on_floor



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



def generate_Sim2_from_floorplan_transform(transform_data: Dict[str,Any]) -> Sim2:
    """Generate a Similarity(2) object from a dictionary storing transformation parameters.

    Note: ZinD stores (sRp + t), instead of s(Rp + t), so we have to divide by s to create Sim2.

    Args:
        transform_data: dictionary of the form
            {'translation': [0.015, -0.0022], 'rotation': -352.53, 'scale': 0.40}
    """
    scale = transform_data["scale"]
    t = np.array(transform_data["translation"]) / scale
    t *= np.array([-1., 1.])
    theta_deg = transform_data["rotation"]

    R = rotmat2d(-theta_deg)

    assert np.allclose(R.T @ R, np.eye(2))

    global_Sim2_local = Sim2(R=R, t=t, s=scale)
    return global_Sim2_local

