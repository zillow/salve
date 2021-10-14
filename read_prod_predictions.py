"""
Converts an inference result to PanoData and PoseGraph2d objects.

Reference for RCNN: https://www.zillow.com/tech/training-models-to-detect-windows-doors-in-panos/
"""

import copy
import csv
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import argoverse.utils.json_utils as json_utils
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

import afp.common.posegraph2d as posegraph2d
import afp.dataset.zind_data as zind_data
import afp.utils.zind_pano_utils as zind_pano_utils
from afp.common.pano_data import PanoData, WDO
from afp.common.posegraph2d import PoseGraph2d


MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
    "rmx-dwo-rcnn_predictions",  #  RCNN DWO predictions
    "rmx-joint-v1_predictions",  # Older version joint model prediction
    "rmx-manh-joint-v2_predictions",  # Older version joint model prediction + Manhattanization shape post processing
    "rmx-rse-v1_predictions",  # Basic HNet trained with production shapes
    "rmx-tg-manh-v1_predictions",  # Total (visible) geometry with Manhattanization shape post processing
]
# could also try partial manhattanization (separate model) -- get link from Yuguang


RED = (1.0, 0, 0)
GREEN = (0, 1.0, 0)
BLUE = (0, 0, 1.0)

# in accordance with color scheme in afp/common/pano_data.py
WINDOW_COLOR = RED
DOOR_COLOR = GREEN
OPENING_COLOR = BLUE


def read_csv(fpath: str, delimiter: str = ",") -> List[Dict[str, Any]]:
    """Read in a .csv or .tsv file as a list of dictionaries."""
    rows = []

    with open(fpath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        for row in reader:
            rows.append(row)

    return rows


def main() -> None:
    """
    Read in mapping from Excel, mapping from their ZInD index to these guid
        https://drive.google.com/drive/folders/1A7N3TESuwG8JOpx_TtkKCy3AtuTYIowk?usp=sharing

    "b912c68c-47da-40e5-a43a-4e1469009f7f":
    # /Users/johnlam/Downloads/complete_07_10_new/1012/panos/floor_01_partial_room_15_pano_19.jpg
    # https://d2ayvmm1jte7yn.cloudfront.net/vrmodels/e9c3eb49-6cbc-425f-b301-7da0aff161d2/floor_map/b912c68c-47da-40e5-a43a-4e1469009f7f/pano/cf94fcb5a5/straightened.jpg
    # 1012 (not 109) and it is by order
    """
    # raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    data_root = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Prediction_Prod_Model/ZInD_pred"

    building_guids = [Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/*")]

    # e.g. building_guid resembles "0a7a6c6c-77ce-4aa9-9b8c-96e2588ac7e8"

    pano_mapping_tsv_fpath = "/Users/johnlam/Downloads/Yuguang_ZinD_prod_mapping_exported_panos.csv"
    pano_mapping_rows = read_csv(pano_mapping_tsv_fpath, delimiter=",")

    # Note: pano_guid is unique across the entire dataset.
    panoguid_to_panoid = {}
    panoguid_to_vrmodelurl = {}
    for pano_metadata in pano_mapping_rows:
        pano_guid = pano_metadata["pano_guid"]
        dgx_fpath = pano_metadata["file"]
        pano_id = zind_data.pano_id_from_fpath(dgx_fpath)
        panoguid_to_panoid[pano_guid] = pano_id
        panoguid_to_vrmodelurl[pano_guid] = pano_metadata["url"]  # .replace('https://www.zillowstatic.com/')

    # floor_map
    tsv_fpath = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Re-processing.tsv"
    tsv_rows = read_csv(tsv_fpath, delimiter="\t")
    for row in tsv_rows:
        building_guid = row["floor_map_guid_new"]
        zind_building_id = row["new_home_id"].zfill(4)

        if building_guid == "":
            print("Invalid building_guid, skipping...")
            continue

        print(f"On ZinD Building {zind_building_id}")
        # if int(zind_building_id) not in [7, 16, 14, 17, 24]:# != 1:
        #     continue

        pano_guids = [
            Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/*")
        ]

        floor_map_json_fpath = f"{data_root}/{building_guid}/floor_map.json"
        if not Path(floor_map_json_fpath).exists():
            import pdb

            pdb.set_trace()
        floor_map_json = json_utils.read_json_file(floor_map_json_fpath)

        for pano_guid, pano_metadata in floor_map_json["panos"].items():
            # import pdb; pdb.set_trace()
            # vrmodelurl = panoguid_to_vrmodelurl[pano_guid]
            pass
            # these differ, for some reason
            # assert vrmodelurl == pano_metadata["url"]


        floor_pose_graphs = {}

        plt.figure(figsize=(20, 10))
        for pano_guid in pano_guids:

            if pano_guid not in panoguid_to_panoid:
                print(f"Missing the panorama for Building {zind_building_id} -> {pano_guid}")
                continue
            i = panoguid_to_panoid[pano_guid]

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{zind_building_id}/panos/floor*_pano_{i}.jpg")
            if not len(img_fpaths) == 1:
                print("\tShould only be one image for this (building id, pano id) tuple.")
                print(f"\tPano {i} was missing")
                plt.close("all")
                continue

            img_fpath = img_fpaths[0]
            img = imageio.imread(img_fpath)

            img_resized = cv2.resize(img, (1024, 512))
            img_h, img_w, _ = img_resized.shape
            # plt.imshow(img_resized)

            floor_id = get_floor_id_from_img_fpath(img_fpath)
            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=zind_building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            if floor_id not in floor_pose_graphs:
                # start populating the pose graph for each floor pano-by-pano
                floor_pose_graphs[floor_id] = PoseGraph2d(
                    building_id=zind_building_id,
                    floor_id=floor_id,
                    nodes= {},
                    scale_meters_per_coordinate = gt_pose_graph.scale_meters_per_coordinate
                )

            model_names = ["rmx-madori-v1_predictions"]  # MODEL_NAMES, "rmx-tg-manh-v1_predictions"]
            # plot the image in question
            for model_name in model_names:
                print(f"\tLoaded {model_name} prediction for Pano {i}")
                model_prediction_fpath = (
                    f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
                )
                if not Path(model_prediction_fpath).exists():
                    import pdb

                    pdb.set_trace()
                prediction_data = json_utils.read_json_file(model_prediction_fpath)

                if model_name == "rmx-madori-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                    if pred_obj is None:  # malformatted pred for some reason
                        continue
                    # pred_obj.render_layout_on_pano(img_h, img_w)
                    pano_data = pred_obj.convert_to_pano_data(img_h, img_w, pano_id=i, gt_pose_graph=gt_pose_graph, img_fpath=img_fpath)
                    floor_pose_graphs[floor_id].nodes[i] = pano_data

                elif model_name == "rmx-dwo-rcnn_predictions":
                    pred_obj = PanoStructurePredictionRmxDwoRCNN.from_json(prediction_data["predictions"])
                    # if not prediction_data["predictions"] == prediction_data["raw_predictions"]:
                    #     import pdb; pdb.set_trace()
                    # print("\tDWO RCNN: ", pred_obj)
                elif model_name == "rmx-tg-manh-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxTgManhV1.from_json(prediction_data[0]["predictions"])
                    pred_obj.render_layout_on_pano(img_h, img_w)
                else:
                    continue

            # plt.title(f"Pano {i} from Building {zind_building_id}")
            # plt.tight_layout()
            # os.makedirs(f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}_bev", exist_ok=True)
            # plt.savefig(
            #     f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}_bev/{zind_building_id}_{i}.jpg", dpi=400
            # )
            # # plt.show()
            # plt.close("all")
            # plt.figure(figsize=(20, 10))

        for floor_id, floor_pose_graph in floor_pose_graphs.items():

            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=zind_building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            floor_pose_graph.render_estimated_layout(
                show_plot=False,
                save_plot=True,
                plot_save_dir=f"{model_name}__oracle_pose",
                gt_floor_pg=gt_pose_graph,
                # plot_save_fpath: Optional[str] = None,
            )

            floor_pose_graph.save_as_zind_data_json(save_fpath=f"ZinD_Inferred_GT_bridgeapi_2021_10_05/{zind_building_id}/{floor_id}.json")


def get_floor_id_from_img_fpath(img_fpath: str) -> str:
    """Fetch the corresponding embedded floor ID from a panorama file path.

    For example, 
    "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg" -> "floor_01"
    """
    fname = Path(img_fpath).name
    k = fname.find("_partial")
    floor_id = fname[:k]

    return floor_id


def test_get_floor_id_from_img_fpath() -> None:
    """Verify we can fetch the floor ID from a panorama file path."""
    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg"
    floor_id = get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_01"

    img_fpath = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/1386/panos/floor_02_partial_room_18_pano_53.jpg"
    floor_id = get_floor_id_from_img_fpath(img_fpath)
    assert floor_id == "floor_02"


@dataclass
class RcnnDwoPred:
    """(x,y) are normalized to [0,1]"""

    category: int
    prob: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_json(cls, json_data: Any) -> "RcnnDwoPred":
        """given 6-tuple"""

        if len(json_data) != 6:
            raise RuntimeError("Data schema violated for RCNN DWO prediction.")

        category, prob, xmin, ymin, xmax, ymax = json_data
        return cls(category, prob, xmin, ymin, xmax, ymax)


@dataclass
class PanoStructurePredictionRmxDwoRCNN:
    """
    should be [2.0, 0.999766529, 0.27673912, 0.343023, 0.31810075, 0.747359931
            class prob x y x y

    make sure the the prob is not below 0.5 (may need above 0.1)

    3 clases -- { 1, 2, }
    """

    dwo_preds: List[RcnnDwoPred]

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxDwoRCNN":
        """ """
        dwo_preds = []
        for dwo_data in json_data[0]:
            dwo_preds += [RcnnDwoPred.from_json(dwo_data)]

        return cls(dwo_preds=dwo_preds)


@dataclass
class PanoStructurePredictionRmxTgManhV1:
    """
    from `rmx-tg-manh-v1_predictions` results.

    Note: Total geometry with Manhattan world assumptions is not very useful, as Manhattanization is
    too strong of an assumption for real homes.
    """

    ceiling_height: float
    floor_height: float
    corners_in_uv: np.ndarray  # (N,2)
    wall_wall_probabilities: np.ndarray  # (N,1)

    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """ """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:, 0] *= img_w
        uv[:, 1] *= img_h

        floor_uv = uv[::2]
        ceiling_uv = uv[1::2]

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker="o")
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker="o")

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxTgManhV1":
        """
        Dictionary with keys:
            'ceiling_height', 'floor_height', 'corners_in_uv', 'wall_wall_probabilities'
        """
        return cls(
            ceiling_height=json_data["room_shape"]["ceiling_height"],
            floor_height=json_data["room_shape"]["floor_height"],
            corners_in_uv=np.array(json_data["room_shape"]["corners_in_uv"]),
            wall_wall_probabilities=np.array(json_data["room_shape"]["wall_wall_probabilities"]),
        )


@dataclass
class RmxMadoriV1DWO:
    """start and end"""

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
    """ """

    ceiling_height: float
    floor_height: float
    corners_in_uv: np.ndarray  # (N,2)
    wall_wall_probabilities: np.ndarray  # (M,)
    wall_uncertainty_score: np.ndarray  # (N,)

    floor_boundary: np.ndarray
    wall_wall_boundary: np.ndarray
    floor_boundary_uncertainty: np.ndarray
    doors: List[RmxMadoriV1DWO]
    openings: List[RmxMadoriV1DWO]
    windows: List[RmxMadoriV1DWO]

    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """ """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:, 0] *= img_w
        uv[:, 1] *= img_h

        floor_uv = uv[::2]
        ceiling_uv = uv[1::2]

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker="o")
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker="o")

        for wdo_instances, color in zip(
            [self.windows, self.doors, self.openings], [WINDOW_COLOR, DOOR_COLOR, OPENING_COLOR]
        ):
            for wdo in wdo_instances:
                plt.plot([wdo.s * img_w, wdo.s * img_w], [0, img_h - 1], color=color, linewidth=5)
                plt.plot([wdo.e * img_w, wdo.e * img_w], [0, img_h - 1], color=color, linewidth=5)

        if len(self.floor_boundary) != 1024:
            print(f"\tFloor boundary shape was {len(self.floor_boundary)}")
            return
        plt.scatter(np.arange(1024), self.floor_boundary, 10, color="y", marker=".")

    def convert_to_pano_data(self, img_h: int, img_w: int, pano_id: int, gt_pose_graph: PoseGraph2d, img_fpath: str) -> PanoData:
        """Render the wall-floor boundary in a bird's eye view.

        Args:
            img_h: height of panorama image, in pixels.
            img_w: width of panorama image, in pixels.
            pano_id: integer ID of panorama
            gt_pose_graph: ground-truth 2d pose graph, with GT shapes and GT global poses.
            img_fpath: file path to panorama image.
        """
        camera_height_m = gt_pose_graph.get_camera_height_m(pano_id)
        camera_height_m = 1.0

        u, v = np.arange(1024), np.round(self.floor_boundary)  # .astype(np.int32)
        pred_floor_wall_boundary_pixel = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
        image_width = 1024

        layout_pts_worldmetric = convert_points_px_to_worldmetric(
            points_px=pred_floor_wall_boundary_pixel, image_width=img_w, camera_height_m=camera_height_m
        )

        # ignore y values, which are along the vertical axis
        room_vertices_local_2d = layout_pts_worldmetric[:, np.array([0,2]) ]

        # TODO: remove this when saving (only for plotting a ready-to-go PanoData instance)
        room_vertices_local_2d[:,0] *= -1

        windows = []
        doors = []
        openings = []

        for wdo_type, wdo_instances_single_type in zip(["windows", "doors", "openings"], [self.windows, self.doors, self.openings]):
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
                wdo_endpoints_worldmetric = convert_points_px_to_worldmetric(
                    points_px=wdo_endpoints_px, image_width=img_w, camera_height_m=camera_height_m
                )
                
                x1, x2 = wdo_endpoints_worldmetric[:, 0]
                y1, y2 = wdo_endpoints_worldmetric[:, 2]

                # TODO: remove this when saving (only for plotting a ready-to-go PanoData instance)
                x1 = -x1
                x2 = -x2

                inferred_wdo = WDO(
                    global_Sim2_local=gt_pose_graph.nodes[pano_id].global_Sim2_local, # using GT pose for now
                    pt1=(x1,y1),
                    pt2=(x2,y2),
                    bottom_z=None,
                    top_z=None,
                    type=wdo_type
                )
                if wdo_type == "windows":
                    windows.append(inferred_wdo)

                elif wdo_type == "doors":
                    doors.append(inferred_wdo)

                elif wdo_type == "openings":
                    openings.append(inferred_wdo)

        pano_data = PanoData(
            id=pano_id,
            global_Sim2_local=gt_pose_graph.nodes[pano_id].global_Sim2_local, # using GT pose for now
            room_vertices_local_2d=room_vertices_local_2d,
            image_path=img_fpath,
            label=gt_pose_graph.nodes[pano_id].label,
            doors=doors,
            windows=windows,
            openings=openings,
        )
        return pano_data


    def render_bev(pano_data: PanoData) -> None:
        """
        Render the estimated layout for a single panorama.
        """
        import pdb; pdb.set_trace()

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

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxMadoriV1":
        """
        Args:
            json_data: nested dictionaries with strucure
                "room_shape":
                  keys: 'ceiling_height', 'floor_height', 'corners_in_uv', 'wall_wall_probabilities', 'wall_uncertainty_score', 'raw_predictions'
                "wall_features":
                  keys: 'window', 'door', 'opening'
        """
        doors = [RmxMadoriV1DWO.from_json(d) for d in json_data["wall_features"]["door"]]
        windows = [RmxMadoriV1DWO.from_json(w) for w in json_data["wall_features"]["window"]]
        openings = [RmxMadoriV1DWO.from_json(o) for o in json_data["wall_features"]["opening"]]

        doors = merge_wdos_straddling_img_border(doors)
        windows = merge_wdos_straddling_img_border(windows)
        openings = merge_wdos_straddling_img_border(openings)

        if len(json_data["room_shape"]["raw_predictions"]["floor_boundary"]) == 0:
            return None

        return cls(
            ceiling_height=json_data["room_shape"]["ceiling_height"],
            floor_height=json_data["room_shape"]["floor_height"],
            corners_in_uv=np.array(json_data["room_shape"]["corners_in_uv"]),
            wall_wall_probabilities=np.array(json_data["room_shape"]["wall_wall_probabilities"]),
            wall_uncertainty_score=np.array(json_data["room_shape"]["wall_uncertainty_score"]),
            floor_boundary=np.array(json_data["room_shape"]["raw_predictions"]["floor_boundary"]),
            wall_wall_boundary=np.array(json_data["room_shape"]["raw_predictions"]["wall_wall_boundary"]),
            floor_boundary_uncertainty=np.array(
                json_data["room_shape"]["raw_predictions"]["floor_boundary_uncertainty"]
            ),
            doors=doors,
            openings=openings,
            windows=windows,
        )


def convert_points_px_to_worldmetric(points_px: np.ndarray, image_width: int, camera_height_m: int) -> np.ndarray:
    """Convert pixel coordinates to Cartesian coordinates with a known scale (i.e. the units are meters).

    Args:
        points_px: 2d points in pixel coordaintes

    Returns:
        points_worldmetric: 
    """

    points_sph = zind_pano_utils.zind_pixel_to_sphere(points_px, width=image_width)
    points_cartesian = zind_pano_utils.zind_sphere_to_cartesian(points_sph)
    points_worldmetric = zind_pano_utils.zind_intersect_cartesian_with_floor_plane(points_cartesian, camera_height_m)
    return points_worldmetric


def merge_wdos_straddling_img_border(wdo_instances: List[RmxMadoriV1DWO]) -> List[RmxMadoriV1DWO]:
    """Merge an object that has been split by the panorama seam (merge two pieces into one).

    Args:
        wdo_instances: of a single type (all doors, all windows, or all openings)

    Returns:
        wdo_instances_merged
    """
    if len(wdo_instances) <= 1:
        # we require at least two objects of the same type to attempt a merge
        return wdo_instances

    wdo_instances_merged = []

    # first ensure that the WDOs are provided left to right, sorted

    # for each set. if one end is located within 50 px of the border (or 1% of image width), than it may have been a byproduct of a seam
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

    # merge with last (far-right) if exists, if it also straddles far-right edge
    # merge with first (far-left) if exists, and it straddles far-left edge
    left_wdo = wdo_instances[left_idx]
    right_wdo = wdo_instances[right_idx]
    merged_wdo = RmxMadoriV1DWO(s=right_wdo.s, e=left_wdo.e)
    wdo_instances_merged.append(merged_wdo)

    return wdo_instances_merged


def test_merge_wdos_straddling_img_border_windows() -> None:
    """
    On ZinD Building 0000, Pano 17
    """
    windows = []
    windows_merged = merge_wdos_straddling_img_border(wdo_instances=windows)
    assert len(windows_merged) == 0
    assert isinstance(windows_merged, list)


def test_merge_wdos_straddling_img_border_doors() -> None:
    """
    On ZinD Building 0000, Pano 17
    """
    doors = [
        RmxMadoriV1DWO(s=0.14467253176930597, e=0.3704789833822092),
        RmxMadoriV1DWO(s=0.45356793743890517, e=0.46920821114369504),
        RmxMadoriV1DWO(s=0.47702834799608995, e=0.5278592375366569),
        RmxMadoriV1DWO(s=0.5376344086021505, e=0.5865102639296188),
        RmxMadoriV1DWO(s=0.6217008797653959, e=0.8084066471163245),
    ]
    doors_merged = merge_wdos_straddling_img_border(wdo_instances=doors)

    assert doors == doors_merged
    assert len(doors_merged) == 5 # should be same as input


def test_merge_wdos_straddling_img_border_openings() -> None:
    """
    On ZinD Building 0000, Pano 17

    Other good examples are:
    Panos 16, 22, 33 for building 0000. 
    Pano 21 for building 0001, 
    """
    openings = [
        RmxMadoriV1DWO(s=0.0009775171065493646, e=0.10361681329423265),
        RmxMadoriV1DWO(s=0.9354838709677419, e=1.0),
    ]
    openings_merged = merge_wdos_straddling_img_border(wdo_instances=openings)

    assert len(openings_merged) == 1
    assert openings_merged[0] == RmxMadoriV1DWO(s=0.9354838709677419, e=0.10361681329423265)


# batch_transform_input_manifest_rmx-tg-manh-v1.json
# batch_transform_input_manifest_rmx-dwo-rcnn.json
# batch_transform_input_manifest_rmx-joint-v1.json
# batch_transform_input_manifest_rmx-madori-v1.json
# batch_transform_input_manifest_rmx-manh-joint-v2.json
# batch_transform_input_manifest_rmx-rse-v1.json


if __name__ == "__main__":
    main()
