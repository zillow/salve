"""Utilities for loading OpenSfM results from disk.

Note: I/O is based off of the original OpenSfM code here:
https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214
"""

import glob
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Similarity3

import salve.utils.io as io_utils
import salve.utils.subprocess_utils as subprocess_utils
from salve.baselines.sfm_reconstruction import SfmReconstruction
from salve.dataset.zind_partition import DATASET_SPLITS

logger = logging.getLogger(__name__)


def panoid_from_key(key: str) -> int:
    """Extract panorama id from panorama image file name.

    Given 'floor_01_partial_room_01_pano_11.jpg', return 11
    """
    return int(Path(key).stem.split("_")[-1])


def point_from_json(
    key: str, obj: Dict[str, Any]  # types.Reconstruction
) -> Tuple[Tuple[float, float, float], Tuple[int, int, int]]:
    """Read a point from a json object.

    Note: OpenSfm's version returns a `pymap.Landmark` object.
    """
    # point = reconstruction.create_point(key, obj["coordinates"])
    point = obj["coordinates"]
    color = obj["color"]
    return point, color


def shot_in_reconstruction_from_json(
    # reconstruction,#: types.Reconstruction,
    key: str,
    obj: Dict[str, Any],
    is_pano_shot: bool = False,
) -> Pose3:
    """Read `shot` from a json object and append it to a reconstruction.

    Note: OpenSfm's version returns a `pymap.Shot` object.
    """
    pose = pose_from_json(obj)

    # if is_pano_shot:
    #     shot = reconstruction.create_pano_shot(key, obj["camera"], pose)
    # else:
    #     # equivalent to self.map.create_shot(shot_id, camera_id, pose)
    #     # https://github.com/mapillary/OpenSfM/blob/master/opensfm/types.py#L162
    #     shot = reconstruction.create_shot(key, obj["camera"], pose)
    return pose  # shot


def pose_from_json(obj: Dict[str, Any]) -> Pose3:
    """Convert the stored extrinsics on disk to a camera pose.

    Note: The OpenSfM Pose class contains a rotation field, representing the local coordinate system as an axis-angle vector.
    The direction of this 3D vector represents the axis around which to rotate.
    The length of this vector is the angle to rotate around said axis. It is in radians.

    See: https://github.com/mapillary/OpenSfM/blob/master/doc/source/cam_coord_system.rst

    Args:
        dictionary with keys
            "rotation": [X, Y, Z],      # Estimated rotation as an angle-axis vector
            "translation": [X, Y, Z],   # Estimated translation

    Returns:
        wTc: pose of camera in the world frame.
    """
    R = VectorToRotationMatrix(np.array(obj["rotation"]))
    if "translation" in obj:
        t = obj["translation"]

    # Equivalent to Pybind functions below
    # .def_property("rotation", &geometry::Pose::RotationWorldToCameraMin,
    #               &geometry::Pose::SetWorldToCamRotation)
    # .def_property("translation", &geometry::Pose::TranslationWorldToCamera,
    #               &geometry::Pose::SetWorldToCamTranslation)

    # OpenSfM stores extrinsics cTw, not poses wTc in a world frame.
    # https://github.com/mapillary/OpenSfM/issues/793
    cTw = Pose3(R, t)
    wTc = cTw.inverse()
    return wTc


def VectorToRotationMatrix(r: np.ndarray) -> Rot3:
    """Convert vector to rotation matrix.

    Args:
        array of shape (3,) rpresenting 3d rotation in axis-angle format.

    Returns:
        array of shape (3,3) representing 3d rotation matrix.
    """
    n = np.linalg.norm(r)  # get encoded angle (in radians)

    r = r.reshape(3, 1)

    # AxisAngle accepts (unitAxis, angle) vs. Eigen::AngleAxisd which accepts (angle, unitAxis)
    if n == 0:  # avoid division by 0
        R = gtsam.Rot3.AxisAngle(r, 0)
        # return Eigen::AngleAxisd(0, r).toRotationMatrix()
    else:
        # R = gtsam.Rot3.AxisAngle(unitAxis=r/n, angle=n)
        R = gtsam.Rot3.AxisAngle(r / n, n)
        # return Eigen::AngleAxisd(n, r / n).toRotationMatrix()
    return R


def camera_from_json(key: str, obj: Dict[str, Any]):  # -> pygeometry.Camera:
    """Read camera from a json object

    TODO
    """
    camera = None
    pt = obj.get("projection_type", "perspective")
    # if pt == "perspective":
    #     camera = pygeometry.Camera.create_perspective(
    #         obj["focal"], obj.get("k1", 0.0), obj.get("k2", 0.0)
    #     )
    # elif pt == "brown":
    #     camera = pygeometry.Camera.create_brown(
    #         obj["focal_x"],
    #         obj["focal_y"] / obj["focal_x"],
    #         [obj.get("c_x", 0.0), obj.get("c_y", 0.0)],
    #         [
    #             obj.get("k1", 0.0),
    #             obj.get("k2", 0.0),
    #             obj.get("k3", 0.0),
    #             obj.get("p1", 0.0),
    #             obj.get("p2", 0.0),
    #         ],
    #     )
    # elif pt == "radial":
    #     camera = pygeometry.Camera.create_radial(
    #         obj["focal_x"],
    #         obj["focal_y"] / obj["focal_x"],
    #         [obj.get("c_x", 0.0), obj.get("c_y", 0.0)],
    #         [
    #             obj.get("k1", 0.0),
    #             obj.get("k2", 0.0),
    #         ],
    #     )
    # elif pt == "simple_radial":
    #     camera = pygeometry.Camera.create_simple_radial(
    #         obj["focal_x"],
    #         obj["focal_y"] / obj["focal_x"],
    #         [obj.get("c_x", 0.0), obj.get("c_y", 0.0)],
    #         obj.get("k1", 0.0),
    #     )
    # elif pt == "dual":
    #     camera = pygeometry.Camera.create_dual(
    #         obj.get("transition", 0.5),
    #         obj["focal"],
    #         obj.get("k1", 0.0),
    #         obj.get("k2", 0.0),
    # )
    if pt == "spherical" or pt == "equirectangular":
        # see https://github.com/mapillary/OpenSfM/blob/master/opensfm/src/geometry/python/pybind.cc#L169
        # camera = pygeometry.Camera.create_spherical()
        # https://github.com/mapillary/OpenSfM/blob/master/opensfm/src/geometry/src/camera.cc#L107
        camera = SimpleNamespace(**{"projection_type": "SPHERICAL", "id": None, "width": None, "height": None})

    elif pt == "perspective":

        # atio between the focal length and the sensor size
        f = obj["focal"] * max(obj["width"], obj["height"])
        camera = SimpleNamespace(**{"projection_type": pt, "width": obj["width"], "height": obj["height"], "focal": f})
    else:
        raise NotImplementedError

    camera.id = key
    camera.width = int(obj.get("width", 0))
    camera.height = int(obj.get("height", 0))
    return camera


def load_opensfm_reconstruction_from_json(obj: Dict[str, Any]) -> SfmReconstruction:
    """TODO

    We ignore "reference_lla", since set to dummy values, e.g. {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}

    See https://github.com/mapillary/OpenSfM/blob/master/doc/source/dataset.rst#reconstruction-file-format
    """
    # Extract cameras
    for key, value in obj["cameras"].items():
        camera = camera_from_json(key, value)
        # reconstruction.add_camera(camera)

    # We do not extract camera `biases` from `obj`, since they are all filled with dummy values like
    # {'rotation': [-0.0, -0.0, -0.0], 'translation': [0.0, 0.0, 0.0], 'scale': 1.0}

    # # Extract rig models
    # if "rig_cameras" in obj:
    #     for key, value in obj["rig_cameras"].items():
    #         reconstruction.add_rig_camera(rig_camera_from_json(key, value))

    pose_dict = {}
    # Extract "shots".
    for key, value in obj["shots"].items():
        pose = shot_in_reconstruction_from_json(key, value)
        pano_id = panoid_from_key(key)

        # for perspective camera
        # pano_id = int(Path(key).stem[1:])
        # pano_id = key
        pose_dict[pano_id] = pose

    # # Extract rig instances from shots
    # if "rig_instances" in obj:
    #     for key, value in obj["rig_instances"].items():
    #         rig_instance_from_json(reconstruction, key, value)

    # Extract "points".
    if "points" in obj:
        points = []
        rgb = []
        for key, value in obj["points"].items():
            point, color = point_from_json(key, value)
            points.append(point)
            rgb.append(color)
        points = np.array(points)
        rgb = np.array(rgb).astype(np.uint8)

    # # Extract pano_shots
    # if "pano_shots" in obj:
    #     for key, value in obj["pano_shots"].items():
    #         is_pano_shot = True
    #         shot_in_reconstruction_from_json(reconstruction, key, value, is_pano_shot)

    # # Extract reference topocentric frame
    # if  in obj:
    #     lla = obj["reference_lla"]
    #     reconstruction.reference = geo.TopocentricConverter(
    #         lla["latitude"], lla["longitude"], lla["altitude"]
    #     )

    reconstruction = SfmReconstruction(camera, pose_dict, points, rgb)
    logger.info("Reconstruction found with %d cameras and %d points", len(pose_dict), points.shape[0])
    return reconstruction


def load_opensfm_reconstructions_from_json(reconstruction_json_fpath: str) -> List[SfmReconstruction]:
    """Read a reconstruction from a json object (TODO MAKE PLURAL)

    Based on https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214
    """
    if not Path(reconstruction_json_fpath).exists():
        reconstructions = []
        return reconstructions

    objs = io_utils.read_json_file(reconstruction_json_fpath)
    reconstructions = [load_opensfm_reconstruction_from_json(obj) for obj in objs]
    return reconstructions

