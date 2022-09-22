"""

See JSON schema info here: http://json-schema.org/understanding-json-schema/reference/numeric.html

Validated using https://github.com/python-jsonschema/jsonschema
"""

import glob
import json
from pathlib import Path

import gtsfm.utils.io as io_utils
import numpy as np


def main():
    """ """
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    postprocessed_dirpath = "/Users/johnlambert/Downloads/zind_final_schema_1768_majority_2022_09_05"

    #preds_dirpath = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_09_05/inference_0815"
    preds_dirpath = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_aggregated_2022_08_05/horizon_net"

    #import pdb; pdb.set_trace()

    json_fpaths = glob.glob(f"{preds_dirpath}/**/*.json", recursive=True)
    for json_fpath in json_fpaths:

        #building_id, fname_stem = Path(json_fpath).stem.split("___")
        building_id = Path(json_fpath).parent.stem
        i = Path(json_fpath).stem
        img_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/floor*_pano_{i}.jpg")
        if len(img_fpaths) != 1:
            import pdb; pdb.set_trace()
        fname_stem = Path(img_fpaths[0]).stem

        pano_json_data = io_utils.read_json_file(json_fpath) #[0]
        #import pdb; pdb.set_trace()

        assert "predictions" in pano_json_data

        predictions = pano_json_data["predictions"]
        assert "room_shape" in predictions
        assert "wall_features" in predictions
        room_shape_data = predictions["room_shape"]

        assert "corners_in_uv" in room_shape_data
        assert "raw_predictions" in room_shape_data

        raw_predictions = room_shape_data["raw_predictions"]
        assert "floor_boundary" in raw_predictions
        assert "floor_boundary_uncertainty" in raw_predictions

        floor_boundary = raw_predictions["floor_boundary"]
        floor_boundary_uncertainty = raw_predictions["floor_boundary_uncertainty"]
        
        if len(floor_boundary) != 1024:
            print(f"Skip {building_id}, {json_fpath}")
            assert (Path("/Users/johnlambert/Downloads/zind_final_schema_1768_missing_2022_09_05") / building_id / f"{fname_stem}.json").exists()
            continue

        if len(floor_boundary_uncertainty) != 1024:
            print(f"Skip {building_id}, {json_fpath}")
            assert (Path("/Users/johnlambert/Downloads/zind_final_schema_1768_missing_2022_09_05") / building_id / f"{fname_stem}.json").exists()
            continue

        assert len(floor_boundary) == 1024
        assert len(floor_boundary_uncertainty) == 1024
        wall_features_data = predictions["wall_features"]
        assert "window" in wall_features_data
        assert "door" in wall_features_data
        assert "opening" in wall_features_data

        #if "ceiling_height" in pano_json_data:
        #if "floor_height" in pano_json_data:

        keys_to_delete = [
            "ceiling_height",
            "floor_height",
            "wall_wall_probabilities",
            "wall_uncertainty_score",
            "wall_uncertainty_score_in_room_cs",
        ]
        for key_to_delete in keys_to_delete:

            if key_to_delete not in pano_json_data["predictions"]["room_shape"]:
                print(f"{building_id}: No {key_to_delete} to delete.")
                continue
            del pano_json_data["predictions"]["room_shape"][key_to_delete]
        
        del pano_json_data["predictions"]["room_shape"]["raw_predictions"]["wall_wall_boundary"]

        pano_json_data["predictions"]["room_shape"]["raw_predictions"]["floor_boundary"] = np.round(floor_boundary, 6).tolist()
        pano_json_data["predictions"]["room_shape"]["raw_predictions"]["floor_boundary_uncertainty"] = np.round(floor_boundary_uncertainty, 6).tolist()

        new_pano_json_data = {
            "predictions": {
                "image_width": 1024,
                "image_height": 512,
                "room_shape": pano_json_data["predictions"]["room_shape"],
                "wall_features": pano_json_data["predictions"]["wall_features"]
            }
        }
        # Add resolution information.
        # new_pano_json_data["predictions"]["image_width"] = 1024
        # new_pano_json_data["predictions"]["image_height"] = 512
        # new_pano_json_data["predictions"] = pano_json_data["predictions"]
        pano_json_data = new_pano_json_data

        #import pdb; pdb.set_trace()

        (Path(postprocessed_dirpath) / building_id).mkdir(parents=True, exist_ok=True)

        new_json_path = Path(postprocessed_dirpath) / building_id / f"{fname_stem}.json"
        json_save(new_json_path, pano_json_data)

        #quit()


def json_save(path, data):    
    file = open(path,'w',encoding='utf-8')
    file.write(json.dumps(data, indent=4))
    file.close()


from jsonschema import validate

from tqdm import tqdm

def validate_files():

    schema = io_utils.read_json_file("/Users/johnlambert/Downloads/salve/horizon_net_schema.json")
    postprocessed_dirpath = "/Users/johnlambert/Downloads/zind_final_schema_1768_majority_2022_09_05"
    json_fpaths = glob.glob(f"{postprocessed_dirpath}/**/*.json", recursive=True)
    print("Validating ", len(json_fpaths), " json files against schema.")
    for json_fpath in tqdm(json_fpaths):
        instance_json_data = io_utils.read_json_file(json_fpath)
        validate(instance=instance_json_data, schema=schema)


if __name__ == "__main__":
    #main()
    validate_files()

