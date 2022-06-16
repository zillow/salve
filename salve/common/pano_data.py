"""
Transformation (R,t) followed by a reflection over the x-axis is equivalent to 
Transformation by (R^T,-t) followed by no reflection.

Sim(2)

s(Rp + t)  -> Sim(2)
sRp + t -> ICP (Zillow)

sRp + st
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

import salve.utils.rotation_utils as rotation_utils
from salve.common.sim2 import Sim2


RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
wdo_color_dict = {"windows": RED, "doors": GREEN, "openings": BLUE}


COLORMAP = np.random.rand(100, 3)


@dataclass(frozen=False)
class WDO:
    """define windows/doors/openings by left and right boundaries"""

    global_Sim2_local: Sim2
    pt1: Tuple[float, float]  # (x1,y1)
    pt2: Tuple[float, float]  # (x2,y2)
    bottom_z: float
    top_z: float
    type: str

    @property
    def centroid(self) -> np.ndarray:
        """Compute centroid of WDO 2d line segment"""
        return np.array([self.pt1, self.pt2]).mean(axis=0)

    @property
    def width(self) -> float:
        """Determine the width of the WDO.

        We define this as length of the line segment from start to end vertices.
        """
        return np.linalg.norm(np.array(self.pt1) - np.array(self.pt2))

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
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return np.array([[x1, y1, self.bottom_z], [x2, y2, self.top_z]])

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
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return np.array(
            [
                [x1, y1, self.bottom_z],
                [x1, y1, self.top_z],
                [x2, y2, self.top_z],
                [x2, y2, self.bottom_z],
                [x1, y1, self.bottom_z],
            ]
        )

    @classmethod
    def from_object_array(cls, wdo_data: Any, global_Sim2_local: Sim2, type: str) -> "WDO":
        """

        Args:
            wdo_data: array of shape (3,2)
            global_Sim2_local
            type: type of WDO, e.g.
        """
        pt1 = wdo_data[0].tolist()
        pt2 = wdo_data[1].tolist()
        bottom_z, top_z = wdo_data[2]
        pt1[0] *= -1
        pt2[0] *= -1
        return cls(global_Sim2_local=global_Sim2_local, pt1=pt1, pt2=pt2, bottom_z=bottom_z, top_z=top_z, type=type)

    def get_rotated_version(self) -> "WDO":
        """Rotate WDO by 180 degrees, as if seen from other side of doorway."""
        self_rotated = WDO(
            global_Sim2_local=self.global_Sim2_local,
            pt1=self.pt2,
            pt2=self.pt1,
            bottom_z=self.bottom_z,
            top_z=self.top_z,
            type=self.type,
        )

        return self_rotated

    def transform_from(self, i2Ti1: Sim2) -> "WDO":
        """If this WDO is in i1's frame, this will transfer the WDO into i2's frame."""
        pt1_ = tuple(i2Ti1.transform_from(np.array(self.pt1).reshape(1, 2)).squeeze().tolist())
        pt2_ = tuple(i2Ti1.transform_from(np.array(self.pt2).reshape(1, 2)).squeeze().tolist())

        # global_Sim2_local represented wTi1, so wTi1 * i1Ti2 = wTi2
        i1Ti2 = i2Ti1.inverse()
        self_transformed = WDO(
            global_Sim2_local=self.global_Sim2_local.compose(i1Ti2),  # TODO: update this as well by multiply with i1Ti2
            pt1=pt1_,
            pt2=pt2_,
            bottom_z=self.bottom_z,
            top_z=self.top_z,
            type=self.type,
        )
        return self_transformed

    def apply_Sim2(self, a_Sim2_b: Sim2, gt_scale: float) -> "WDO":
        """Convert the WDO's pose to a new global reference frame `a` for Sim(3) alignment.
        Previous was in global frame `b`.

        Consider this WDO to be the j'th WDO in some list/set.
        """
        aligned_self = copy.deepcopy(self)

        b_Sim2_j = self.global_Sim2_local
        a_Sim2_j = a_Sim2_b.compose(b_Sim2_j)
        # equivalent of `transformFrom()` on Pose2 object.
        aligned_self.global_Sim2_local = Sim2(R=a_Sim2_j.rotation, t=a_Sim2_j.translation * a_Sim2_j.scale, s=gt_scale)
        return aligned_self


@dataclass(frozen=False)
class PanoData:
    """Container for ground truth relevant to a single panorama.

    Args:
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
            wdo_objs_seen_on_floor
            show_plot

        Returns:
            wdo_objs_seen_on_floor
        """
        # hohopano_Sim2_zindpano = Sim2(R=rotation_utils.rotmat2d(-90), t=np.zeros(2), s=1.0)

        assert coord_frame in ["worldmetric", "worldnormalized" "local"]

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
            wdo_color = wdo_color_dict[wdo_type]

            if (wdo_objs_seen_on_floor is not None) and (wdo_type not in wdo_objs_seen_on_floor):
                label = wdo_type
            else:
                label = None
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
    """
    scale = transform_data["scale"]
    t = np.array(transform_data["translation"]) / scale
    t *= np.array([-1.0, 1.0])
    theta_deg = transform_data["rotation"]

    # note: to account for reflection, we swap the sign here to use R^T
    R = rotation_utils.rotmat2d(-theta_deg)

    assert np.allclose(R.T @ R, np.eye(2))

    global_Sim2_local = Sim2(R=R, t=t, s=scale)
    return global_Sim2_local
