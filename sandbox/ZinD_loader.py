""" """

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Pose3

from gtsfm.loader.colmap_loader import ColmapLoader


class ZindLoader:
    """
    SfM loader for the ZinD dataset.
    """

    def __init__(self, dataset_dirpath: str, building_id: str, floor_id: str) -> None:
        """ """
        building_dirpath = f"{dataset_dirpath}/{building_id}"

        self._json_annot_fpath = f"{building_dirpath}/zfm_data.json"
        self._pano_dir = f"{building_dirpath}/panos"
        self._floor_id = floor_id

        floor_pano_tuples = self.get_pano_pose_imgfpath_tuples()

        import matplotlib.pyplot as plt

        for (pano_id, image_fpath, pano_position) in floor_pano_tuples:

            color = np.random.rand(3)
            plt.scatter(pano_position[0], pano_position[1], 30, marker="+", color=color)

            pano_text = f"{pano_id}"
            TEXT_LEFT_OFFSET = 0.15
            noise = np.clip(np.random.randn(2), a_min=-0.05, a_max=0.05)
            text_loc = pano_position[:2] + noise
            plt.text(
                (text_loc[0] - TEXT_LEFT_OFFSET), text_loc[1], pano_text, color=color, fontsize="xx-small"
            )  # , 'x-small', 'small', 'medium',)

        plt.axis("equal")
        plt.show()

    def get_pano_pose_imgfpath_tuples(self) -> List[Tuple[int, str, np.ndarray]]:
        """Generate 3-tuples consistenting of (panorama id, image file path, camera position)

        floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'

        We use the 'merger' annotations.
        """
        floor_pano_tuples = []

        floor_map_json = read_json_file(self._json_annot_fpath)
        merger_data = floor_map_json["merger"]

        floor_data = merger_data[self._floor_id]
        for complete_room_data in floor_data.values():
            for partial_room_data in complete_room_data.values():
                for pano_data in partial_room_data.values():

                    z = pano_data["camera_height"]
                    print("Camera height: ", z)

                    global_SIM2_local = generate_Sim2_from_floorplan_transform(pano_data["floor_plan_transformation"])
                    pano_position = global_SIM2_local.transform_from(np.zeros((1, 2)))
                    x, y = pano_position.squeeze()
                    # need reflection
                    x *= -1

                    pano_position = np.array([x, y, z])
                    image_fpath = pano_data["image_path"]
                    print(image_fpath)
                    pano_id = int(Path(image_fpath).stem.split("_")[-1])
                    floor_pano_tuples.append((pano_id, image_fpath, pano_position))

        return floor_pano_tuples


def get_camera_pose(self, index: int) -> Optional[Pose3]:
    """ """

    wTi_list


def generate_Sim2_from_floorplan_transform(transform_data: Dict[str, Any]) -> Sim2:
    """Generate a Similarity(2) object from a dictionary storing transformation parameters.

    Note: ZinD stores (sRp + t), instead of s(Rp + t), so we have to divide by s to create Sim2.

    Args:
    transform_data: dictionary of the form
    {'translation': [0.015, -0.0022], 'rotation': -352.53, 'scale': 0.40}
    """
    scale = transform_data["scale"]
    t = np.array(transform_data["translation"]) / scale
    theta_deg = transform_data["rotation"]

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

    global_SIM2_local = Sim2(R=R, t=t, s=scale)
    return global_SIM2_local


if __name__ == "__main__":

    teaser_dirpath = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    building_id = "000"
    floor_id = "floor_01"
    ZindLoader(teaser_dirpath, building_id, floor_id)
