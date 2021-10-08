
"""

TODO: wrap doors, windows, oepnings around image border (merge them if starts or ends within 50 px of edge)

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

MODEL_NAMES = [
    "rmx-madori-v1_predictions", # Ethan’s new shape DWO joint model
    "rmx-dwo-rcnn_predictions", #  RCNN DWO predictions
    "rmx-joint-v1_predictions", # Older version joint model prediction
    "rmx-manh-joint-v2_predictions", # Older version joint model prediction + Manhattanization shape post processing
    "rmx-rse-v1_predictions", # Basic HNet trained with production shapes
    "rmx-tg-manh-v1_predictions" # Total (visible) geometry with Manhattanization shape post processing
]
# could also try partial manhattanization (separate model) -- get link from Yuguang


WINDOW_COLOR = "y" # yellow
DOOR_COLOR = "k" # black
OPENING_COLOR =  "m" # magenta


# Yuguang: what is "Old home ID" vs. "New home ID"
# zind building 002 --totally off, building 007, 016, 14, 17, 24

# 013 looks good, 23 looks good.


def read_csv(fpath: str, delimiter: str = ",") -> List[Dict[str,Any]]:
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
    #raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
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
        panoguid_to_vrmodelurl[pano_guid] = pano_metadata["url"] # .replace('https://www.zillowstatic.com/')

    # floor_map
    tsv_fpath = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Re-processing.tsv"
    tsv_rows = read_csv(tsv_fpath, delimiter="\t")
    for row in tsv_rows:
        building_guid = row["floor_map_guid_new"]
        # if building_guid != "b912c68c-47da-40e5-a43a-4e1469009f7f":
        #     continue

        zind_building_id = row["new_home_id"].zfill(4)

        if building_guid == "":
            print("Invalid building_guid, skipping...")
            continue

        # if zind_building_id in ["000", "001", "002"]:
        #     continue

        print(f"On ZinD Building {zind_building_id}")
        # if int(zind_building_id) not in [7, 16, 14, 17, 24]:# != 1:
        #     continue

        #import pdb; pdb.set_trace()

        pano_guids = [Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/*")]

        floor_map_json_fpath = f"{data_root}/{building_guid}/floor_map.json"
        if not Path(floor_map_json_fpath).exists():
            import pdb; pdb.set_trace()
        floor_map_json = json_utils.read_json_file(floor_map_json_fpath)

        for pano_guid, pano_metadata in floor_map_json['panos'].items():
            #import pdb; pdb.set_trace()
            #vrmodelurl = panoguid_to_vrmodelurl[pano_guid]
            pass
            # these differ, for some reason
            #assert vrmodelurl == pano_metadata["url"]

        # get floor height.
        gt_pose_graph = posegraph2d.get_gt_pose_graph(building_id=zind_building_id, floor_id="floor_01", raw_dataset_dir=raw_dataset_dir)

        # zillow_floor_map["scale_meters_per_coordinate"][floor_id]
        scale_meters_per_coordinate = gt_pose_graph.scale_meters_per_coordinate

        # from pano_data["floor_plan_transformation"]["scale"]
        floor_plan_transformation_scale = gt_pose_graph.nodes[9].global_Sim2_local.scale

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

            # if i > 3:
            #     continue

            img_fpath = img_fpaths[0]
            img = imageio.imread(img_fpath)
            
            img_resized = cv2.resize(img, (1024,512))
            img_h, img_w, _ = img_resized.shape
            plt.imshow(img_resized)

            model_names = ["rmx-madori-v1_predictions"] # MODEL_NAMES, "rmx-tg-manh-v1_predictions"]
            # plot the image in question
            for model_name in model_names:
                print(f"\tLoaded {model_name} prediction for Pano {i}")
                model_prediction_fpath = f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
                if not Path(model_prediction_fpath).exists():
                    import pdb; pdb.set_trace()
                prediction_data = json_utils.read_json_file(model_prediction_fpath)

                if model_name == "rmx-madori-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                    pred_obj.render_layout_on_pano(img_h, img_w)

                elif model_name == "rmx-dwo-rcnn_predictions":
                    pred_obj = PanoStructurePredictionRmxDwoRCNN.from_json(prediction_data["predictions"])
                    # if not prediction_data["predictions"] == prediction_data["raw_predictions"]:
                    #     import pdb; pdb.set_trace()
                    #print("\tDWO RCNN: ", pred_obj)
                elif model_name == "rmx-tg-manh-v1_predictions":
                    pred_obj = PanoStructurePredictionRmxTgManhV1.from_json(prediction_data[0]["predictions"])
                    pred_obj.render_layout_on_pano(img_h, img_w)
                else:
                    continue

            plt.title(f"Pano {i} from Building {zind_building_id}")
            os.makedirs(f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}", exist_ok=True)
            plt.savefig(f"prod_pred_model_visualizations_2021_10_07_bridge/{model_name}/{zind_building_id}_{i}.jpg", dpi=400)
            #plt.show()
            plt.close("all")
            plt.figure(figsize=(20,10))

@dataclass
class RcnnDwoPred:
    """ (x,y) are normalized to [0,1] """
    category: int
    prob: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_json(cls, json_data: Any) -> "RcnnDwoPred":
        """ given 6-tuple """

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
            dwo_preds += [ RcnnDwoPred.from_json(dwo_data) ]

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
    corners_in_uv: np.ndarray # (N,2)
    wall_wall_probabilities: np.ndarray # (N,1)


    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """ """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:,0] *= img_w
        uv[:,1] *= img_h

        floor_uv = uv[::2]
        ceiling_uv = uv[1::2]

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker='o')
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker='o')


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
            wall_wall_probabilities=np.array(json_data["room_shape"]["wall_wall_probabilities"])
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
    corners_in_uv: np.ndarray # (N,2)
    wall_wall_probabilities: np.ndarray # (M,)
    wall_uncertainty_score: np.ndarray # (N,)

    floor_boundary: np.ndarray
    wall_wall_boundary: np.ndarray
    floor_boundary_uncertainty: np.ndarray
    doors: List[RmxMadoriV1DWO]
    openings: List[RmxMadoriV1DWO]
    windows: List[RmxMadoriV1DWO]

    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """ """
        render_egoview = False
        if render_egoview:
            uv = copy.deepcopy(self.corners_in_uv)
            uv[:,0] *= img_w
            uv[:,1] *= img_h

            floor_uv = uv[::2]
            ceiling_uv = uv[1::2]

            plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker='o')
            plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker='o')

            # import pdb; pdb.set_trace()

            # yellow -> window
            # black -> door
            # magenta -> opening

            for wdo_instances, color in zip([self.windows, self.doors, self.openings], [WINDOW_COLOR, DOOR_COLOR, OPENING_COLOR]):
                for wdo in wdo_instances:
                    plt.plot([wdo.s * img_w, wdo.s * img_w], [0,img_h-1], color)
                    plt.plot([wdo.e * img_w, wdo.e * img_w], [0,img_h-1], color)

            if len(self.floor_boundary) != 1024:
                print(f"\tFloor boundary shape was {len(self.floor_boundary)}")
                return
            plt.scatter(np.arange(1024), self.floor_boundary, 10, color='y', marker='.')
        else:
            self.render_bev()

    def render_bev(self) -> None:
        """Render the wall-floor boundary in a bird's eye view."""
        floor_height = 0.5

        plt.close("All")
        import afp.utils.pano_utils as pano_utils

        u, v = np.arange(1024), np.round(self.floor_boundary).astype(np.int32)
        pred_floor_wall_pixel_corners = np.hstack([u.reshape(-1,1), v.reshape(-1,1)])
        floor_height = 999
        image_width = 1024

        import pdb; pdb.set_trace()

        floor_real_scale = zillow_floor_map["scale_meters_per_coordinate"][floor_id]
        room_shape_real_scale = pano_data["floor_plan_transformation"]["scale"]
        real_scale = floor_real_scale * room_shape_real_scale
        camera_height = pano_data["camera_height"]
        floor_height = camera_height * real_scale

        pred_floor_wall_sphere_corners = pixel_to_sphere(pred_floor_wall_pixel_corners, width=image_width)
        pred_floor_wall_cartesian_corners = sphere_to_cartesian(pred_floor_wall_sphere_corners)
        ray_dirs = intersect_cartesian_with_floor_plane(pred_floor_wall_cartesian_corners, floor_height)


        # # get unit-norm rays
        # ray_dirs_all = pano_utils.get_uni_sphere_xyz(H=512, W=1024)
        # ray_dirs = ray_dirs_all[v, u]

        import pdb; pdb.set_trace()
        # ray_dirs /= ray_dirs[:, 1].reshape(-1, 1) # scale so that y has unit norm
        # ray_dirs *= floor_height

        n = ray_dirs.shape[0]
        rgb = np.zeros((n,3)).astype(np.uint8)
        rgb[:, 0] = 255
        import visualization.open3d_vis_utils as open3d_vis_utils
        import pdb; pdb.set_trace()
        # pcd = open3d_vis_utils.create_colored_spheres_open3d(
        #     point_cloud=ray_dirs, rgb=rgb, sphere_radius=0.1
        # )
        pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=ray_dirs, rgb=rgb)
        import open3d
        
        open3d.visualization.draw_geometries([pcd])


        import pdb; pdb.set_trace()
        plt.scatter(ray_dirs[:,0], ray_dirs[:,2], 10, color='m', marker='.')
        plt.show()
        plt.close('all')

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
        doors = [ RmxMadoriV1DWO.from_json(d) for d in json_data["wall_features"]["door"]]
        windows = [RmxMadoriV1DWO.from_json(w) for w in json_data["wall_features"]["window"]]
        openings = [RmxMadoriV1DWO.from_json(o) for o in json_data["wall_features"]["opening"]]

        return cls(
            ceiling_height=json_data["room_shape"]['ceiling_height'],
            floor_height=json_data["room_shape"]['floor_height'],
            corners_in_uv=np.array(json_data["room_shape"]['corners_in_uv']),
            wall_wall_probabilities=np.array(json_data["room_shape"]['wall_wall_probabilities']),
            wall_uncertainty_score=np.array(json_data["room_shape"]['wall_uncertainty_score']),
            floor_boundary=np.array(json_data["room_shape"]['raw_predictions']["floor_boundary"]),
            wall_wall_boundary=np.array(json_data["room_shape"]['raw_predictions']["wall_wall_boundary"]),
            floor_boundary_uncertainty=np.array(json_data["room_shape"]['raw_predictions']["floor_boundary_uncertainty"]),
            doors=doors,
            openings=openings,
            windows=windows
        )


EPS_RAD = 1e-10

# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def intersect_cartesian_with_floor_plane(cartesian_coordinates: np.ndarray, floor_height: float) -> np.ndarray:
    """
    In order to get the floor coordinates, intersect with the floor plane
    """
    return (
        cartesian_coordinates
        * floor_height
        / cartesian_coordinates[:, 1].reshape(-1, 1)
    )

# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def sphere_to_cartesian(points_sph: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to cartesian.

    Args:
        points_sph: List of points given in spherical coordinates. We support
            two formats, both in a row major-order: [theta, phi, rho] or
            [theta, phi], where in the second form we assume all points lie
            on the unit sphere, i.e. rho = 1.0 for all points.

        theta is the azimuthal angle in [-pi, pi],
        phi is the elevation angle in [-pi/2, pi/2]
        rho is the radial distance in (0, inf)

    Note:
        If rho is omitted, we will assume the radial distances is 1 for all.
        thus points_sph.shape can be (num_points, 2) or (num_points, 3).

    Return:
        List of points in cartesian coordinates [x, y, z], where the shape
        is (num_points, 3)
    """
    if not isinstance(points_sph, np.ndarray) or points_sph.ndim == 1:
        points_sph = np.reshape(points_sph, (1, -1))
        output_shape = (3,)
    else:
        output_shape = (points_sph.shape[0], 3)  # type: ignore

    num_points = points_sph.shape[0]
    assert num_points > 0

    num_coords = points_sph.shape[1]
    assert num_coords == 2 or num_coords == 3

    theta = points_sph[:, 0]

    # Validate the azimuthal angles.
    assert np.all(np.greater_equal(theta, -math.pi - EPS_RAD))
    assert np.all(np.less_equal(theta, math.pi + EPS_RAD))

    phi = points_sph[:, 1]

    # Validate the elevation angles.
    assert np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS_RAD))
    assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS_RAD))

    if num_coords == 2:
        rho = np.ones_like(theta)
    else:
        rho = points_sph[:, 2]

    # Validate the radial distances.
    assert np.all(np.greater(rho, 0.0))

    rho_cos_phi = rho * np.cos(phi)

    x_arr = rho_cos_phi * np.sin(theta)
    y_arr = rho * np.sin(phi)
    z_arr = -rho_cos_phi * np.cos(theta)

    return np.column_stack((x_arr, y_arr, z_arr)).reshape(output_shape)


# from
# https://gitlab.zgtools.net/zillow/rmx/libs/egg.panolib/-/blob/main/panolib/sphereutil.py#L96
def pixel_to_sphere(points_pix: np.ndarray, width: int) -> np.ndarray:
    """Convert pixel coordinates into spherical coordinates from a 360 pano
    with a given width.

    Note:
        We assume the width covers the full 360 degrees horizontally, and the
        height is derived as width/2 and covers the full 180 degrees
        vertical, i.e. we support mapping only on full FoV panos.

    Args:
        points_pix: List of points given in pano image coordinates [x, y],
            thus points_sph.shape is (num_points, 2)

        width: The width of the pano image (defines the azimuth scale).

    Return:
        List of points in spherical coordinates [theta, phi], where the
        spherical point [theta=0, phi=0] maps to the image center.
        Shape of the result is (num_points, 2).
    """
    if not isinstance(points_pix, np.ndarray) or points_pix.ndim == 1:
        points_pix = np.reshape(points_pix, (1, -1))
        output_shape = (2,)
    else:
        output_shape = (points_pix.shape[0], 2)  # type: ignore

    num_points = points_pix.shape[0]
    assert num_points > 0

    num_coords = points_pix.shape[1]
    assert num_coords == 2

    height = width / 2
    assert width > 1 and height > 1

    # We only consider the azimuth and elevation angles.
    x_arr = points_pix[:, 0]
    assert np.all(np.greater_equal(x_arr, 0.0))
    assert np.all(np.less(x_arr, width))

    y_arr = points_pix[:, 1]
    assert np.all(np.greater_equal(y_arr, 0.0))
    assert np.all(np.less(y_arr, height))

    # Convert the x-coordinates to azimuth spherical coordinates, where
    # theta=0 maps to the horizontal center.
    theta = x_arr / (width - 1)  # Map to [0, 1]
    theta *= 2.0 * math.pi  # Map to [0, 2*pi]
    theta -= math.pi  # Map to [-pi, pi]

    # Convert the y-coordinates to elevation spherical coordinates, where
    # phi=0 maps to the vertical center.
    phi = y_arr / (height - 1)  # Map to [0, 1]
    phi = 1.0 - phi  # Flip so that y=0 corresponds to pi/2
    phi *= math.pi  # Map to [0, pi]
    phi -= math.pi / 2.0  # Map to [-pi/2, pi/2]

    return np.column_stack((theta, phi)).reshape(output_shape)



	# batch_transform_input_manifest_rmx-tg-manh-v1.json
	# batch_transform_input_manifest_rmx-dwo-rcnn.json
	# batch_transform_input_manifest_rmx-joint-v1.json
	# batch_transform_input_manifest_rmx-madori-v1.json
	# batch_transform_input_manifest_rmx-manh-joint-v2.json
	# batch_transform_input_manifest_rmx-rse-v1.json


if __name__ == "__main__":
	main()
