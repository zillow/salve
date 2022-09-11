"""Utilities and data structures for HorizonNet model predictions (using a joint shape + W/D/O model)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import gtsfm.utils.io as io_utils
import matplotlib.pyplot as plt
import numpy as np
import rdp

from salve.common.pano_data import PanoData, WDO
from salve.common.posegraph2d import PoseGraph2d
import salve.utils.zind_pano_utils as zind_pano_utils


RAMER_DOUGLAS_PEUCKER_EPSILON = 0.02

RED = (1.0, 0, 0)
GREEN = (0, 1.0, 0)
BLUE = (0, 0, 1.0)

# Colormap in accordance with color scheme in salve/common/pano_data.py
WINDOW_COLOR = RED
DOOR_COLOR = GREEN
OPENING_COLOR = BLUE

YELLOW = (1.0, 1.0, 0.0)
LAYOUT_COLOR = YELLOW


@dataclass
class RmxMadoriV1DWO:
    """Stores the start and end horizontal coordinates of a window, door, or opening.

    Args:
        s: horizontal starting coordinate, in the range [0,1]
        e: horizontal ending coordinate, in the range [0,1]
    """

    s: float
    e: float

    @classmethod
    def from_json(cls, json_data: Any) -> "RmxMadoriV1DWO":
        """ """
        if len(json_data) != 2:
            raise RuntimeError("Schema error...")

        s, e = json_data
        return cls(s=s, e=e)


@dataclass
class PanoStructurePredictionRmxMadoriV1:
    """Attributes predicted for a single panorama by HorizonNet.

    Attributes:
        image_height: image height (in pixels).
        image_width: image width (in pixels).
        corners_in_uv: array of shape (C,2) in range [0,1] indicating normalized (u,v) coordinates of room corner
            locations, interleaved as (floor corner 1, ceiling corner 1), (floor corner 2, ceiling corner 2), ...,
            (floor corner C//2, ceiling corner C//2).
        floor_boundary: array of shape (1024,) in range [0, image_height] indicating of floor boundary.
        floor_boundary_uncertainty: array of shape (1024,) in range [0, image_height] indicating uncertainty of floor boundary.
        doors: in range [0,1]
        openings: in range [0,1]
        windows: in range [0,1]
        image_fpath: path to corresponding image (360 deg. panorama).
    """

    corners_in_uv: np.ndarray  # (K,2)
    image_height: int
    image_width: int

    floor_boundary: np.ndarray
    floor_boundary_uncertainty: np.ndarray
    doors: List[RmxMadoriV1DWO]
    openings: List[RmxMadoriV1DWO]
    windows: List[RmxMadoriV1DWO]
    image_fpath: Path

    @classmethod
    def from_json_fpath(cls, json_fpath: Path, image_fpath: Path) -> PanoStructurePredictionRmxMadoriV1:
        """Generate an object from dictionary containing data loaded from JSON.

        Args:
            json_fpath: path to JSON file. JSON data conforms to the schema in `salve/horizon_net_schema.json`.
            image_fpath: path to corresponding image (360 deg. panorama).
        """
        if not isinstance(image_fpath, Path):
            raise ValueError("Image file path provided to `PanoStructurePredictionRmxMadoriV1` must"
                " be a `pathlib.Path` object.")

        if not isinstance(json_fpath, Path):
            raise ValueError("JSON file path provided to `PanoStructurePredictionRmxMadoriV1` must"
                " be a `pathlib.Path` object.")

        if not Path(json_fpath).exists():
            raise ValueError(f"No JSON file found at {json_fpath} while loading `PanoStructurePredictionRmxMadoriV1`.")

        json_data = io_utils.read_json_file(json_fpath)
        json_data = json_data["predictions"]

        doors = [RmxMadoriV1DWO.from_json(d) for d in json_data["wall_features"]["door"]]
        windows = [RmxMadoriV1DWO.from_json(w) for w in json_data["wall_features"]["window"]]
        openings = [RmxMadoriV1DWO.from_json(o) for o in json_data["wall_features"]["opening"]]

        doors = merge_wdos_straddling_img_border(doors)
        windows = merge_wdos_straddling_img_border(windows)
        openings = merge_wdos_straddling_img_border(openings)

        return cls(
            image_height=json_data["image_height"],
            image_width=json_data["image_width"],
            corners_in_uv=np.array(json_data["room_shape"]["corners_in_uv"]),
            floor_boundary=np.array(json_data["room_shape"]["raw_predictions"]["floor_boundary"]),
            floor_boundary_uncertainty=np.array(
                json_data["room_shape"]["raw_predictions"]["floor_boundary_uncertainty"]
            ),
            doors=doors,
            openings=openings,
            windows=windows,
            image_fpath=image_fpath
        )

    def get_floor_corners_image(self) -> np.ndarray:
        """Get predicted floor corners, in pixel coordinates.

        Returns:
            floor_uv: array of shape (C//2, 2), where C is the total number of floor & ceiling corners
                in the room.
        """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:, 0] *= self.image_width
        uv[:, 1] *= self.image_height

        floor_uv = uv[::2]
        return floor_uv

    def get_ceiling_corners_image(self) -> np.ndarray:
        """Get predicted ceiling corners, in pixel coordinates.

        Returns:
            ceiling_uv: array of shape (C//2, 2), where C is the total number of floor & ceiling corners
                in the room.
        """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:, 0] *= self.image_width
        uv[:, 1] *= self.image_height

        ceiling_uv = uv[1::2]
        return ceiling_uv

    def render_layout_on_pano(self) -> None:
        """Render the predicted wall-floor boundary and wall corners onto the equirectangular projection.

        Note: assumes plt.imshow(panorama_image) has been called previously.
        """
        img_h = self.image_height
        img_w = self.image_width

        linewidth = 5 #20 # use 20 for paper figures, but 5 for debug visualizations.

        floor_uv = self.get_floor_corners_image()
        ceiling_uv = self.get_ceiling_corners_image()

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker="o")
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker="o")

        for wdo_instances, color in zip(
            [self.windows, self.doors, self.openings], [WINDOW_COLOR, DOOR_COLOR, OPENING_COLOR]
        ):
            for wdo in wdo_instances:
                plt.plot([wdo.s * img_w, wdo.s * img_w], [0, img_h - 1], color=color, linewidth=linewidth)
                plt.plot([wdo.e * img_w, wdo.e * img_w], [0, img_h - 1], color=color, linewidth=linewidth)

        if len(self.floor_boundary) != 1024:
            print(f"\tFloor boundary shape was {len(self.floor_boundary)}")
            return
        # plt.scatter(np.arange(1024), self.floor_boundary, 10, color="y", marker=".")
        plt.plot(np.arange(1024), self.floor_boundary, color=LAYOUT_COLOR, linewidth=linewidth)

    def convert_to_pano_data(
        self, img_h: int, img_w: int, pano_id: int, gt_pose_graph: PoseGraph2d, img_fpath: str, vanishing_angle_deg: float
    ) -> PanoData:
        """Convert HorizonNet W/D/O and layout predictions into a PanoData object.

        We run the Ramer-Douglas-Peucker simplification algorithm on the 1024 vertex contour
        with an epsilon of about 0.02 in room coordinate space.

        Args:
            img_h: height of panorama image, in pixels.
            img_w: width of panorama image, in pixels.
            pano_id: integer ID of panorama
            gt_pose_graph: ground-truth 2d pose graph, with GT shapes and GT global poses.
            img_fpath: file path to panorama image.
            vanishing_angle_deg: 

        Returns:
            PanoData containing predicted W/D/O's and predicted layout for this panorama.
        """
        camera_height_m = gt_pose_graph.get_camera_height_m(pano_id)
        camera_height_m = 1.0 # TODO: REMOVE THIS

        u, v = np.arange(1024), np.round(self.floor_boundary)  # .astype(np.int32)
        pred_floor_wall_boundary_pixel = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
        image_width = 1024

        # layout pts in `worldmetric` system
        room_vertices_local_2d = zind_pano_utils.convert_points_px_to_worldmetric(
            points_px=pred_floor_wall_boundary_pixel, image_width=img_w, camera_height_m=camera_height_m
        )

        # See https://cartography-playground.gitlab.io/playgrounds/douglas-peucker-algorithm/
        # https://rdp.readthedocs.io/en/latest/
        room_vertices_local_2d = rdp.rdp(room_vertices_local_2d[:, :2], epsilon=RAMER_DOUGLAS_PEUCKER_EPSILON)

        windows = []
        doors = []
        openings = []

        for wdo_type, wdo_instances_single_type in zip(
            ["windows", "doors", "openings"], [self.windows, self.doors, self.openings]
        ):
            for wdo in wdo_instances_single_type:
                wdo_s_u = wdo.s * img_w
                wdo_e_u = wdo.e * img_w
                wdo_s_u = np.clip(wdo_s_u, a_min=0, a_max=img_w - 1)
                wdo_e_u = np.clip(wdo_e_u, a_min=0, a_max=img_w - 1)
                # self.floor_boundary contains the `v` coordinates at each `u`.
                wdo_s_v = self.floor_boundary[round(wdo_s_u)]
                wdo_e_v = self.floor_boundary[round(wdo_e_u)]
                # fmt: off
                wdo_endpoints_px = np.array([[wdo_s_u, wdo_s_v], [wdo_e_u, wdo_e_v]])
                # fmt: on
                wdo_endpoints_worldmetric = zind_pano_utils.convert_points_px_to_worldmetric(
                    points_px=wdo_endpoints_px, image_width=img_w, camera_height_m=camera_height_m
                )
                x1, x2 = wdo_endpoints_worldmetric[:, 0]
                y1, y2 = wdo_endpoints_worldmetric[:, 1]

                inferred_wdo = WDO(
                    global_Sim2_local=gt_pose_graph.nodes[pano_id].global_Sim2_local,  # using GT pose for now
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    bottom_z=-np.nan,
                    top_z=np.nan,
                    type=wdo_type,
                )
                if wdo_type == "windows":
                    windows.append(inferred_wdo)

                elif wdo_type == "doors":
                    doors.append(inferred_wdo)

                elif wdo_type == "openings":
                    openings.append(inferred_wdo)

        pano_data = PanoData(
            id=pano_id,
            global_Sim2_local=gt_pose_graph.nodes[pano_id].global_Sim2_local,  # using GT pose for now
            room_vertices_local_2d=room_vertices_local_2d,
            image_path=img_fpath,
            label=gt_pose_graph.nodes[pano_id].label,
            doors=doors,
            windows=windows,
            openings=openings,
            vanishing_angle_deg=vanishing_angle_deg
        )
        return pano_data

    def render_bev(pano_data: PanoData) -> None:
        """
        Render the estimated layout for a single panorama.
        """
        import pdb

        pdb.set_trace()

        # plt.close("All")
        #         plt.scatter(, 10, color="m", marker=".")
        # plt.axis("equal")
        # gt_pose_graph.nodes[pano_id].plot_room_layout(coord_frame="local", show_plot=False)

        # for wdo_instances_single_type, color in zip(
        #     [self.windows, self.doors, self.openings], [WINDOW_COLOR, DOOR_COLOR, OPENING_COLOR]
        # ):
        #     for wdo in wdo_instances_single_type:

        #         plt.plot(, color=color, linewidth=6)

        # n = ray_dirs.shape[0]
        # rgb = np.zeros((n, 3)).astype(np.uint8)
        # rgb[:, 0] = 255
        # import visualization.open3d_vis_utils as open3d_vis_utils
        # # pcd = open3d_vis_utils.create_colored_spheres_open3d(
        # #     point_cloud=ray_dirs, rgb=rgb, sphere_radius=0.1
        # # )
        # pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=ray_dirs, rgb=rgb)
        # import open3d
        # open3d.visualization.draw_geometries([pcd])

        # plt.subplot(1, 2, 2)
        # gt_pose_graph.nodes[pano_id].plot_room_layout(coord_frame="local", show_plot=False)

        plt.axis("equal")

        # plt.show()
        # plt.close("all")


def merge_wdos_straddling_img_border(wdo_instances: List[RmxMadoriV1DWO]) -> List[RmxMadoriV1DWO]:
    """Merge an object that has been split by the panorama seam (merge two pieces into one).

    Args:
        wdo_instances: W/D/O instances of a single type (all doors, all windows, or all openings).

    Returns:
        wdo_instances_merged: W/D/O instances of a single type, after merging has been performed.
    """
    if len(wdo_instances) <= 1:
        # We require at least two objects of the same type to attempt a merge
        return wdo_instances

    wdo_instances_merged = []

    # First, ensure that the W/D/Os are provided left to right, sorted.

    # For each set. If one end is located within 50 px of the border (or 1% of image width),
    # then it may have been a byproduct of a seam.
    straddles_left = [wdo.s < 0.01 for wdo in wdo_instances]
    straddles_right = [wdo.e > 0.99 for wdo in wdo_instances]

    any_straddles_left_border = any(straddles_left)
    any_straddles_right_border = any(straddles_right)
    merge_is_allowed = any_straddles_left_border and any_straddles_right_border

    if not merge_is_allowed:
        return wdo_instances

    straddles_left = np.array(straddles_left)
    straddles_right = np.array(straddles_right)

    left_idx = np.argmax(straddles_left)
    right_idx = np.argmax(straddles_right)

    for i, wdo in enumerate(wdo_instances):
        if i in [left_idx, right_idx]:
            continue
        wdo_instances_merged.append(wdo)

    # Merge with last (far-right) if exists, if it also straddles far-right edge.
    # Merge with first (far-left) if exists, and it straddles far-left edge.
    left_wdo = wdo_instances[left_idx]
    right_wdo = wdo_instances[right_idx]
    merged_wdo = RmxMadoriV1DWO(s=right_wdo.s, e=left_wdo.e)
    wdo_instances_merged.append(merged_wdo)

    return wdo_instances_merged

