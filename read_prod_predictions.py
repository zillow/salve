
"""

"""
import copy
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import argoverse.utils.json_utils as json_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np


model_names = [
    "rmx-madori-v1_predictions", # Ethan’s new shape DWO joint model
    "rmx-dwo-rcnn_predictions", #  RCNN DWO predictions
    "rmx-joint-v1_predictions", # Older version joint model prediction
    "rmx-manh-joint-v2_predictions", # Older version joint model prediction + Manhattanization shape post processing
    "rmx-rse-v1_predictions", # Basic HNet trained with production shapes
    "rmx-tg-manh-v1_predictions" # Total geometry with Manhattanization shape post processing
]


# Yuguang: what is "Old home ID" vs. "New home ID"
def main() -> None:
    """
    Read in mapping from Excel, mapping from their ZInD index to these guid
        https://drive.google.com/drive/folders/1A7N3TESuwG8JOpx_TtkKCy3AtuTYIowk?usp=sharing

    "b912c68c-47da-40e5-a43a-4e1469009f7f":
    # /Users/johnlam/Downloads/complete_07_10_new/1012/panos/floor_01_partial_room_15_pano_19.jpg
    # https://d2ayvmm1jte7yn.cloudfront.net/vrmodels/e9c3eb49-6cbc-425f-b301-7da0aff161d2/floor_map/b912c68c-47da-40e5-a43a-4e1469009f7f/pano/cf94fcb5a5/straightened.jpg
    # 1012 (not 109) and it is by order
    """
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    data_root = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Prediction_Prod_Model/ZInD_pred"

    building_guids = [Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/*")]

    # e.g. building_guid resembles "0a7a6c6c-77ce-4aa9-9b8c-96e2588ac7e8"

    # floor_map
    tsv_fpath = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Re-processing.tsv"
    with open(tsv_fpath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")

        for row in reader:
            building_guid = row["floor_map_guid_new"]
            if building_guid != "b912c68c-47da-40e5-a43a-4e1469009f7f":
                continue

            zind_building_id = row["old_home_id"]
            print(f"On ZinD Building {zind_building_id}")
            pano_guids = [Path(dirpath).stem for dirpath in glob.glob(f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/*")]

            floor_map_json_fpath = f"{data_root}/{building_guid}/floor_map.json"
            floor_map_json = json_utils.read_json_file(floor_map_json_fpath)

            panoguid_to_panoid = {}
            for pano_guid, pano_metadata in floor_map_json['panos'].items():
                i = pano_metadata["order"]
                panoguid_to_panoid[pano_guid] = i

            panoid_to_panoguid = {i: pano_guid for pano_guid, i in panoguid_to_panoid.items()}


            for i in sorted(panoid_to_panoguid.keys()):
                pano_guid = panoid_to_panoguid[i]

                img_fpaths = glob.glob(f"{raw_dataset_dir}/{zind_building_id}/panos/floor*_pano_{i}.jpg")
                if not len(img_fpaths) == 1:
                    print("\tShould only be one image for this (building id, pano id) tuple.")
                    print(f"\tPano {i} was missing")
                    continue

                img_fpath = img_fpaths[0]
                img = imageio.imread(img_fpath)

                img_h, img_w, _ = img.shape
                plt.imshow(img)

                # plot the image in question
                for model_name in model_names:
                    print(f"\tLoaded {model_name} prediction for Pano {i}")
                    model_prediction_fpath = f"{data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
                    prediction_data = json_utils.read_json_file(model_prediction_fpath)

                    if model_name == "rmx-madori-v1_predictions":
                        pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                        pred_obj.render_layout_on_pano(img_h, img_w)

                    elif model_name == "rmx-dwo-rcnn_predictions":
                        pred_obj = PanoStructurePredictionRmxDwoRCNN.from_json(prediction_data["predictions"])
                        # if not prediction_data["predictions"] == prediction_data["raw_predictions"]:
                        #     import pdb; pdb.set_trace()
                        print("\tDWO RCNN: ", pred_obj)
                    else:
                        continue

                plt.title(f"Pano {i} from Building {zind_building_id}")
                plt.show()




@dataclass
class PanoStructurePredictionRmxDwoRCNN:
    """ """
    vals: List[float]

    @classmethod
    def from_json(cls, json_data: Any) -> "PanoStructurePredictionRmxDwoRCNN":
        """ """
        pass

        return cls(json_data)



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

    def render_layout_on_pano(self, img_h: int, img_w: int) -> None:
        """ """
        uv = copy.deepcopy(self.corners_in_uv)
        uv[:,0] *= img_w
        uv[:,1] *= img_h

        floor_uv = uv[::2]
        ceiling_uv = uv[1::2]

        plt.scatter(floor_uv[:, 0], floor_uv[:, 1], 100, color="r", marker='+')
        plt.scatter(ceiling_uv[:, 0], ceiling_uv[:, 1], 100, color="g", marker='+')


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
        return cls(
            ceiling_height=json_data["room_shape"]['ceiling_height'],
            floor_height=json_data["room_shape"]['floor_height'],
            corners_in_uv=np.array(json_data["room_shape"]['corners_in_uv']),
            wall_wall_probabilities=np.array(json_data["room_shape"]['wall_wall_probabilities']),
            wall_uncertainty_score=np.array(json_data["room_shape"]['wall_uncertainty_score']),
            floor_boundary=np.array(json_data["room_shape"]['raw_predictions']["floor_boundary"]),
            wall_wall_boundary=np.array(json_data["room_shape"]['raw_predictions']["wall_wall_boundary"]),
            floor_boundary_uncertainty=np.array(json_data["room_shape"]['raw_predictions']["floor_boundary_uncertainty"])
        )


	# batch_transform_input_manifest_rmx-tg-manh-v1.json
	# batch_transform_input_manifest_rmx-dwo-rcnn.json
	# batch_transform_input_manifest_rmx-joint-v1.json
	# batch_transform_input_manifest_rmx-madori-v1.json
	# batch_transform_input_manifest_rmx-manh-joint-v2.json
	# batch_transform_input_manifest_rmx-rse-v1.json


if __name__ == "__main__":
	main()