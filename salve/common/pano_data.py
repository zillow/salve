"""
Transformation (R,t) followed by a reflection over the x-axis is equivalent to 
Transformation by (R^T,-t) followed by no reflection.

Sim(2)

s(Rp + t)  -> Sim(2)
sRp + t -> ICP (Zillow)

sRp + st
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

import salve.utils.rotation_utils as rotation_utils
from salve.common.sim2 import Sim2
from salve.common.wdo import WDO


RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
WDO_COLOR_DICT = {"windows": RED, "doors": GREEN, "openings": BLUE}


COLORMAP = np.random.rand(100, 3)


class CoordinateFrame(str, Enum):
    """Defines scale of coordinates, their axes, and their frame of reference.

    LOCAL corresponds to Cartesian ego-frame of panorama.
    WORLD_NORMALIZED corresponds to TODO
    WORLD_METRIC corresponds to TODO
    """
    LOCAL: str = 'local'
    WORLD_NORMALIZED: str = 'worldnormalized'
    WORLD_METRIC: str = 'worldmetric'


@dataclass(frozen=False)
class PanoData:
    """Container for ground truth relevant to a single panorama.

    Attributes:
        id: TODO
        global_Sim2_local: TODO
        room_vertices_local_2d: stored in TODO coordinate frame...
        image_path: TODO
        label: TODO
        doors: TODO
        windows: TODO
        openings: TODO
        vanishing_angle_deg: defined as ...
    """

    id: int
    global_Sim2_local: Sim2
    room_vertices_local_2d: np.ndarray
    image_path: str
    label: str
    doors: Optional[List[WDO]] = None
    windows: Optional[List[WDO]] = None
    openings: Optional[List[WDO]] = None
    vanishing_angle_deg: Optional[float] = None
    
    @property
    def room_vertices_global_2d(self):
        """ """
        return self.global_Sim2_local.transform_from(self.room_vertices_local_2d)

    @classmethod
    def from_json(cls, pano_data: Any) -> "PanoData":
        """
        From JSON ("pano_data")
        """
        # print('Camera height: ', pano_data['camera_height'])
        assert pano_data["camera_height"] == 1.0

        doors = []
        windows = []
        openings = []
        image_path = pano_data["image_path"]
        label = pano_data["label"]

        pano_id = int(Path(image_path).stem.split("_")[-1])

        global_Sim2_local = generate_Sim2_from_floorplan_transform(pano_data["floor_plan_transformation"])
        room_vertices_local_2d = np.asarray(pano_data["layout_raw"]["vertices"])
        room_vertices_local_2d[:, 0] *= -1

        # DWO objects
        geometry_type = "layout_raw"  # , "layout_complete", "layout_visible"]
        for wdo_type in ["windows", "doors", "openings"]:
            wdos = []
            wdo_data = np.asarray(pano_data[geometry_type][wdo_type], dtype=object)

            # Skip if there are no elements of this type
            if len(wdo_data) == 0:
                continue

            # as (x1,y1), (x2,y2), (bottom_z, top_z)
            # Transform the local W/D/O vertices to the global frame of reference
            assert len(wdo_data) % 3 == 0  # should be a multiple of 3

            num_wdo = len(wdo_data) // 3
            for wdo_idx in range(num_wdo):
                wdo = WDO.from_object_array(wdo_data[wdo_idx * 3 : (wdo_idx + 1) * 3], global_Sim2_local, wdo_type)
                wdos.append(wdo)

            if wdo_type == "windows":
                windows = wdos
            elif wdo_type == "doors":
                doors = wdos
            elif wdo_type == "openings":
                openings = wdos

        return cls(pano_id, global_Sim2_local, room_vertices_local_2d, image_path, label, doors, windows, openings)

    def plot_room_layout(
        self,
        coord_frame: str,
        wdo_objs_seen_on_floor: Optional[Set] = None,
        show_plot: bool = True,
        scale_meters_per_coordinate: Optional[float] = None,
    ) -> None:
        """Plot the room shape for this panorama, either in a global or local frame

        Args:
            coord_frame: either 'local' (cartesian ego-frame) or  'worldnormalized' or 'worldmetric'
            wdo_objs_seen_on_floor: TODO
            show_plot: TODO
            scale_meters_per_coordinate: TODO

        Returns:
            wdo_objs_seen_on_floor
        """
        # hohopano_Sim2_zindpano = Sim2(R=rotation_utils.rotmat2d(-90), t=np.zeros(2), s=1.0)

        # TODO: replace with their Enum equivalent.
        if coord_frame not in ["worldmetric", "worldnormalized" "local"]:
            raise ValueError(f"Unknown coordinate frame provided: {coord_frame}.")

        if coord_frame in ["worldmetric", "worldnormalized"]:
            room_vertices = self.room_vertices_global_2d
        else:
            room_vertices = self.room_vertices_local_2d

        if coord_frame == "worldmetric":
            if scale_meters_per_coordinate is None:
                print("Failure: need scale as arg to convert to meters. Skipping rendering...")
                return
            else:
                room_vertices *= scale_meters_per_coordinate

        elif coord_frame == "worldnormalized":
            scale_meters_per_coordinate = 1.0

        # room_vertices = hohopano_Sim2_zindpano.transform_from(room_vertices)

        num_colormap_colors = COLORMAP.shape[0]
        color = COLORMAP[self.id % num_colormap_colors]

        plt.scatter(room_vertices[:, 0], room_vertices[:, 1], 10, marker=".", color=color)
        plt.plot(room_vertices[:, 0], room_vertices[:, 1], color=color, alpha=0.5)
        # draw edge to close each polygon
        last_vert = room_vertices[-1]
        first_vert = room_vertices[0]
        plt.plot([last_vert[0], first_vert[0]], [last_vert[1], first_vert[1]], color=color, alpha=0.5)

        pano_position = np.zeros((1, 2))
        if coord_frame in ["worldnormalized", "worldmetric"]:
            pano_position_local = pano_position
            pano_position_world = self.global_Sim2_local.transform_from(pano_position_local)
            pano_position = pano_position_world * scale_meters_per_coordinate

        plt.scatter(pano_position[0, 0], pano_position[0, 1], 30, marker="+", color=color)

        point_ahead = np.array([1, 0]).reshape(1, 2)
        if coord_frame in ["worldnormalized", "worldmetric"]:
            point_ahead = self.global_Sim2_local.transform_from(point_ahead) * scale_meters_per_coordinate

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

            if coord_frame in ["worldnormalized", "worldmetric"]:
                wdo_points = wdo.vertices_global_2d * scale_meters_per_coordinate
            else:
                wdo_points = wdo.vertices_local_2d

            # wdo_points = hohopano_Sim2_zindpano.transform_from(wdo_points)
            wdo_type = wdo.type
            wdo_color = WDO_COLOR_DICT[wdo_type]

            #if (wdo_objs_seen_on_floor is not None) and (wdo_type not in wdo_objs_seen_on_floor):
            label = wdo_type
            #else:
            #    label = None
            plt.scatter(wdo_points[:, 0], wdo_points[:, 1], 10, color=wdo_color, marker="o", label=label)
            plt.plot(wdo_points[:, 0], wdo_points[:, 1], color=wdo_color, linestyle="dotted")
            # plt.text(wdo_points[:, 0].mean(), wdo_points[:, 1].mean(), f"{wdo.type}_{wdo_idx}")

            if wdo_objs_seen_on_floor is not None:
                wdo_objs_seen_on_floor.add(wdo_type)

        if show_plot:
            plt.axis("equal")
            plt.show()
            plt.close("all")

        return wdo_objs_seen_on_floor


class FloorData(NamedTuple):
    """Container for panorama information on a single floor of a building."""

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
                    # if pano_data["is_primary"]:

                    pano_obj = PanoData.from_json(pano_data)
                    pano_objs.append(pano_obj)

        return cls(floor_id, pano_objs)


def generate_Sim2_from_floorplan_transform(transform_data: Dict[str, Any]) -> Sim2:
    """Generate a Similarity(2) object from a dictionary storing transformation parameters.

    Note: ZinD stores (sRp + t), instead of s(Rp + t), so we have to divide by s to create Sim2.

    Args:
        transform_data: dictionary of the form
            {'translation': [0.015, -0.0022], 'rotation': -352.53, 'scale': 0.40}

    Returns:
        TODO...
    """
    scale = transform_data["scale"]
    t = np.array(transform_data["translation"]) / scale
    # TODO: explain...
    t *= np.array([-1.0, 1.0])
    theta_deg = transform_data["rotation"]

    # Note: to account for reflection, we swap the sign here to use R^T
    R = rotation_utils.rotmat2d(-theta_deg)

    assert np.allclose(R.T @ R, np.eye(2))

    global_Sim2_local = Sim2(R=R, t=t, s=scale)
    return global_Sim2_local
