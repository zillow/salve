"""Utilities for loading OpenMVG SfM results.

https://github.com/openMVG/openMVG/blob/develop/docs/sphinx/rst/software/SfM/SfM.rst#notes-about-spherical-sfm
"""

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gtsam import Pose3, Rot3

import salve.utils.io as io_utils
from salve.baselines.sfm_reconstruction import SfmReconstruction


def panoid_from_key(key: str) -> int:
    """Extract panorama id from panorama image file name.

    Given 'floor_01_partial_room_01_pano_11.jpg', return 11 as an integer.
    """
    return int(Path(key).stem.split("_")[-1])


def load_openmvg_reconstructions_from_json(json_fpath: str, building_id: str, floor_id: str) -> List[SfmReconstruction]:
    """Read OpenMVG-specific format ("sfm_data.json") to AutoFloorPlan generic types.

    Args:
        building_id: unique ID for ZinD building.
        floor_id: unique ID for floor of a ZinD building.
        reconstruction_fpath: path to reconstruction/sfm_data.json file.

    Returns:
        reconstructions: list of length 1, representing single scene reconstruction. We return a list for API parity
           with OpenSfM's wrapper (OpenSfM returns multiple connected components).
    """
    data = io_utils.read_json_file(json_fpath)

    assert data["sfm_data_version"] == "0.3"

    intrinsics = data["intrinsics"] # noqa
    # print("OpenMVG Estimated Intrinsics: ", intrinsics)
    view_metadata = data["views"] # noqa
    # print("OpenMVG Estimated View Metadata: ", view_metadata)
    extrinsics = data["extrinsics"]

    key_to_fname_dict = {}
    for view in data["views"]:
        openmvg_key = view["key"]
        filename = view["value"]["ptr_wrapper"]["data"]["filename"]
        key_to_fname_dict[openmvg_key] = filename

    pose_dict = {}

    for ext_info in extrinsics:
        openmvg_key = ext_info["key"]
        R = np.array(ext_info["value"]["rotation"])
        # See https://github.com/openMVG/openMVG/issues/671
        # and http://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/#pinhole-camera-model
        t = -R @ np.array(ext_info["value"]["center"])

        cTw = Pose3(Rot3(R), t)
        wTc = cTw.inverse()

        filename = key_to_fname_dict[openmvg_key]

        pano_id = panoid_from_key(filename)
        pose_dict[pano_id] = wTc

    reconstruction = SfmReconstruction(
        camera=None,  # could provide spherical properties
        pose_dict=pose_dict,
        points=np.zeros((0, 3)),
        rgb=np.zeros((0, 3)).astype(np.uint8),
    )

    # OpenSfM only returns the largest connected component for incremental
    # (See Pierre Moulon's comment here: https://github.com/openMVG/openMVG/issues/1938)
    return [reconstruction]


def find_seed_pair(image_dirpath: str) -> Tuple[str, str]:
    """Choose a seed pair for incremental SfM.

    We use the heuristic criterion that any two images that are next to each other in trajectory/capture order
    may be suitable as a seed pair.

    Args:
        image_dirpath: path to directory containing images from a ZInD floor.

    Returns:
        seed_fname1: file name of panorama 1.
        seed_fname2: file name of panorama 2.
    """
    image_fpaths = glob.glob(f"{image_dirpath}/*.jpg")

    if len(image_fpaths) < 2:
        raise ValueError("Less than two images were not found in the image directory, so no seed can be assigned.")

    # Choose a seed pair. Sort by temporal order (last key).
    image_fpaths.sort(key=lambda x: panoid_from_key(x))

    # Find an adjacent pair.
    frame_idxs = np.array([int(Path(x).stem.split("_")[-1]) for x in image_fpaths])
    temporal_dist = np.diff(frame_idxs)

    # This image, and the next frame in the pair, are useful.
    valid_seed_idxs = np.where(np.absolute(temporal_dist) == 1)[0]

    seed_idx_1 = valid_seed_idxs[0]
    seed_idx_2 = seed_idx_1 + 1

    seed_fname1 = Path(image_fpaths[seed_idx_1]).name
    seed_fname2 = Path(image_fpaths[seed_idx_2]).name
    return seed_fname1, seed_fname2

