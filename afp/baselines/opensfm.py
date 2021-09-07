"""

https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214

"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gtsam
import numpy as np
from argoverse.utils.json_utils import read_json_file
from gtsam import Pose3, Rot3, Similarity3

from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
from afp.utils.logger_utils import get_logger


logger = get_logger()


@dataclass
class OpenSfmReconstruction:
    camera: SimpleNamespace
    pose_dict: Dict[int, Pose3]
    points: np.ndarray
    rgb: np.ndarray


def panoid_from_key(key: str) -> int:
    """Extract panorama id from panorama image file name.

    Given 'floor_01_partial_room_01_pano_11.jpg', return 11
    """
    return int(Path(key).stem.split("_")[-1])


def assign_shot_attributes(obj: Dict[str, Any], shot) -> None:  # pymap.Shot
    shot.metadata = json_to_pymap_metadata(obj)
    if "scale" in obj:
        shot.scale = obj["scale"]
    if "covariance" in obj:
        shot.covariance = np.array(obj["covariance"])
    if "merge_cc" in obj:
        shot.merge_cc = obj["merge_cc"]
    if "vertices" in obj and "faces" in obj:
        shot.mesh.vertices = obj["vertices"]
        shot.mesh.faces = obj["faces"]


def json_to_pymap_metadata(obj: Dict[str, Any]):  # -> pymap.ShotMeasurements:
    metadata = pymap.ShotMeasurements()
    if obj.get("orientation") is not None:
        metadata.orientation.value = obj.get("orientation")
    if obj.get("capture_time") is not None:
        metadata.capture_time.value = obj.get("capture_time")
    if obj.get("gps_dop") is not None:
        metadata.gps_accuracy.value = obj.get("gps_dop")
    if obj.get("gps_position") is not None:
        metadata.gps_position.value = obj.get("gps_position")
    if obj.get("skey") is not None:
        metadata.sequence_key.value = obj.get("skey")
    if obj.get("accelerometer") is not None:
        metadata.accelerometer.value = obj.get("accelerometer")
    if obj.get("compass") is not None:
        compass = obj.get("compass")
        if "angle" in compass:
            metadata.compass_angle.value = compass["angle"]
        if "accuracy" in compass:
            metadata.compass_accuracy.value = compass["accuracy"]
    return metadata


def point_from_json(
    key: str, obj: Dict[str, Any]  # types.Reconstruction
) -> Tuple[Tuple[float, float, float], Tuple[int, int, int]]:  # -> pymap.Landmark:
    """
    Read a point from a json object
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
):  # -> pymap.Shot:
    """
    Read shot from a json object and append it to a reconstruction
    """
    pose = pose_from_json(obj)
    # import pdb; pdb.set_trace()

    # if is_pano_shot:
    #     shot = reconstruction.create_pano_shot(key, obj["camera"], pose)
    # else:
    #     # equivalent to self.map.create_shot(shot_id, camera_id, pose)
    #     # https://github.com/mapillary/OpenSfM/blob/master/opensfm/types.py#L162
    #     shot = reconstruction.create_shot(key, obj["camera"], pose)
    # assign_shot_attributes(obj, shot)
    return pose  # shot


def pose_from_json(obj: Dict[str, Any]) -> Pose3:  # -> pygeometry.Pose:
    """
    The OpenSfM Pose class contains a rotation field, representing the local coordinate system as an axis-angle vector.

    The direction of this 3D vector represents the axis around which to rotate.
    The length of this vector is the angle to rotate around said axis. It is in radians.

    See: https://github.com/mapillary/OpenSfM/blob/master/doc/source/cam_coord_system.rst

    Args:
        dictionary with keys
            "rotation": [X, Y, Z],      # Estimated rotation as an angle-axis vector
            "translation": [X, Y, Z],   # Estimated translation

    Returns:
        TODO: confirm if cTw or wTc
    """
    R = VectorToRotationMatrix(np.array(obj["rotation"]))
    if "translation" in obj:
        t = obj["translation"]

    # Equivalent to Pybind
    # .def_property("rotation", &geometry::Pose::RotationWorldToCameraMin,
    #               &geometry::Pose::SetWorldToCamRotation)
    # .def_property("translation", &geometry::Pose::TranslationWorldToCamera,
    #               &geometry::Pose::SetWorldToCamTranslation)

    return Pose3(R, t)


def VectorToRotationMatrix(r: np.ndarray) -> Rot3:
    """
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


def bias_from_json(obj: Dict[str, Any]):  # -> pygeometry.Similarity:
    """
    Args:

    Returns:
        Similarity(3)
    """
    return Similarity3(R=obj["rotation"], t=obj["translation"], s=obj["scale"])


def camera_from_json(key: str, obj: Dict[str, Any]):  # -> pygeometry.Camera:
    """
    Read camera from a json object
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

    else:
        raise NotImplementedError
    camera.id = key
    camera.width = int(obj.get("width", 0))
    camera.height = int(obj.get("height", 0))
    return camera


def load_opensfm_reconstruction_from_json(obj: Dict[str, Any]):
    """

    We ignore "reference_lla", since set to dummy values, e.g. {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}

    See https://github.com/mapillary/OpenSfM/blob/master/doc/source/dataset.rst#reconstruction-file-format
    """
    # Extract cameras
    for key, value in obj["cameras"].items():
        camera = camera_from_json(key, value)
        # reconstruction.add_camera(camera)

    # Extract camera biases
    # Ignore these, since all filled with dummy values like
    # {'rotation': [-0.0, -0.0, -0.0], 'translation': [0.0, 0.0, 0.0], 'scale': 1.0}
    if "biases" in obj:
        for key, value in obj["biases"].items():
            transform = bias_from_json(value)
            # reconstruction.set_bias(key, transform)

    # # Extract rig models
    # if "rig_cameras" in obj:
    #     for key, value in obj["rig_cameras"].items():
    #         reconstruction.add_rig_camera(rig_camera_from_json(key, value))

    pose_dict = {}
    # Extract shots
    for key, value in obj["shots"].items():
        pose = shot_in_reconstruction_from_json(key, value)
        pano_id = panoid_from_key(key)
        pose_dict[pano_id] = pose

    # # Extract rig instances from shots
    # if "rig_instances" in obj:
    #     for key, value in obj["rig_instances"].items():
    #         rig_instance_from_json(reconstruction, key, value)

    # Extract points
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

    reconstruction = OpenSfmReconstruction(camera, pose_dict, points, rgb)
    logger.info("Reconstruction found with %d cameras and %d points", len(pose_dict), points.shape[0])
    return reconstruction


def load_opensfm_reconstructions_from_json(
    reconstruction_json_fpath: str,
) -> List[OpenSfmReconstruction]:  # : # -> types.Reconstruction:
    """
    Read a reconstruction from a json object

    Based on https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214
    """
    # reconstruction = types.Reconstruction()

    objs = read_json_file(reconstruction_json_fpath)
    reconstructions = [load_opensfm_reconstruction_from_json(obj) for obj in objs]
    return reconstructions


def measure_opensfm_localization_accuracy(reconstruction_json_fpath: str):
    """ """
    building_id = "1442"
    floor_id = "floor_01"
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    reconstructions = load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    import pdb

    pdb.set_trace()

    for reconstruction in reconstructions:
        est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(
            wRi_list, wti_list, gt_floor_pose_graph, building_id, floor_id
        )
        mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )
        est_floor_pose_graph.render_estimated_layout(
            show_plot=False, save_plot=True, plot_save_dir=plot_save_dir, gt_floor_pg=gt_floor_pose_graph
        )


if __name__ == "__main__":
    """ """
    reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/ZinD_1442_floor_01/reconstruction.json"

    # load_opensfm_reconstructions_from_json(reconstruction_json_fpath)
    measure_opensfm_localization_accuracy(reconstruction_json_fpath)
