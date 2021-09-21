"""

https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214

"""

import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gtsam
import gtsfm.utils.geometry_comparisons as geometry_comparisons
import matplotlib.pyplot as plt
import numpy as np
import open3d
import argoverse.utils.json_utils as json_utils
from argoverse.utils.subprocess_utils import run_command
from colour import Color
from gtsam import Pose3, Rot3, Similarity3

# import below is from gtsfm
import visualization.open3d_vis_utils as open3d_vis_utils

from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
from afp.common.posegraph3d import PoseGraph3d
from afp.utils.logger_utils import get_logger


logger = get_logger()


@dataclass
class OpenSfmReconstruction:
    camera: SimpleNamespace
    pose_dict: Dict[int, Pose3]
    points: np.ndarray
    rgb: np.ndarray

    @property
    def wTi_list(self) -> np.ndarray:
        """ """
        N = max(self.pose_dict.values()) + 1
        wTi_list = [reconstruction.pose_dict.get(i, None) for i in range(N)]
        return wTi_list


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


def pose_from_json(obj: Dict[str, Any]) -> Pose3:
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

    # OpenSfM stores extrinsics, not poses in a world frame.
    return Pose3(R, t).inverse()


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

        # for perspective camera
        # pano_id = int(Path(key).stem[1:])
        # pano_id = key
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

    objs = json_utils.read_json_file(reconstruction_json_fpath)
    reconstructions = [load_opensfm_reconstruction_from_json(obj) for obj in objs]
    return reconstructions


def get_colormap(N: int) -> np.ndarray:
    """

    Args:
        N: number of unique colors to generate.

    Returns:
        colormap: uint8 array of shape (N,3)
    """
    colormap = np.array([[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), N)]).squeeze()
    colormap = (colormap * 255).astype(np.uint8)
    return colormap


def draw_coordinate_frame(wTc: Pose3, axis_length: float = 1.0) -> List[open3d.geometry.LineSet]:
    """Draw 3 orthogonal axes representing a camera coordinate frame.

    Args:
        wTc: Pose of any camera in the world frame.
        axis_length:

    Returns:
        line_sets: list of Open3D LineSet objects
    """
    RED = np.array([1, 0, 0])
    GREEN = np.array([0, 1, 0])
    BLUE = np.array([0, 0, 1])
    colors = (RED, GREEN, BLUE)

    line_sets = []
    for axis, color in zip([0, 1, 2], colors):

        lines = [[0, 1]]
        verts_worldfr = np.zeros((2, 3))

        verts_camfr = np.zeros((2, 3))
        verts_camfr[0, axis] = axis_length

        for i in range(2):
            verts_worldfr[i] = wTc.transformFrom(verts_camfr[i])

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(verts_worldfr),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(color.reshape(1, 3))
        line_sets.append(line_set)

    return line_sets


def plot_3d_poses(aTi_list_gt: List[Optional[Pose3]], bTi_list_est: List[Optional[Pose3]]) -> None:
    """ """

    def get_colormapped_spheres(wTi_list: List[Optional[Pose3]]) -> np.ndarray:
        """ """
        num_valid_poses = sum([1 if wTi is not None else 0 for wTi in wTi_list])
        colormap = get_colormap(N=num_valid_poses)

        curr_color_idx = 0
        point_cloud = []
        rgb = []
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            point_cloud += [wTi.translation()]
            rgb += [colormap[curr_color_idx]]
            curr_color_idx += 1

        point_cloud = np.array(point_cloud)
        rgb = np.array(rgb)
        return point_cloud, rgb

    point_cloud_est, rgb_est = get_colormapped_spheres(bTi_list_est)
    point_cloud_gt, rgb_gt = get_colormapped_spheres(aTi_list_gt)
    geo1 = open3d_vis_utils.create_colored_spheres_open3d(point_cloud_est, rgb_est, sphere_radius=0.2)
    geo2 = open3d_vis_utils.create_colored_spheres_open3d(point_cloud_gt, rgb_gt, sphere_radius=0.5)

    def get_coordinate_frames(wTi_list: List[Optional[Pose3]]) -> List[open3d.geometry.LineSet]:
        frames = []
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            frames.extend(draw_coordinate_frame(wTi))
        return frames

    frames1 = get_coordinate_frames(aTi_list_gt)
    frames2 = get_coordinate_frames(bTi_list_est)

    open3d.visualization.draw_geometries(geo1 + geo2 + frames1 + frames2)


def get_zillow_T_opensfm() -> Pose3:
    """
    Transform OpenSfM camera to ZinD camera.
    """
    Rx = np.pi / 2
    Ry = 0.0
    Rz = 0.0
    zillow_R_opensfm = Rot3.RzRyRx(Rx, Ry, Rz)
    zillow_T_opensfm = Pose3(zillow_R_opensfm, np.zeros(3))
    return zillow_T_opensfm


def measure_opensfm_localization_accuracy(
    reconstruction_json_fpath: str, building_id: str, floor_id: str, raw_dataset_dir: str
) -> None:
    """

    Args:
        reconstruction_json_fpath:
        building_id:
        floor_id:
        raw_dataset_dir:
    """
    gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    reconstructions = load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    floor_results_dicts = []
    for r, reconstruction in enumerate(reconstructions):

        # create a 3d pose graph
        aTi_list_gt = gt_floor_pose_graph.as_3d_pose_graph()
        bTi_list_est = [reconstruction.pose_dict.get(i, None) for i in range(len(aTi_list_gt))]

        aTi_list_gt = [aTi if bTi_list_est[i] is not None else None for i, aTi in enumerate(aTi_list_gt)]

        zillow_T_opensfm = get_zillow_T_opensfm()

        # world frame <-> opensfm camera <-> zillow camera
        bTi_list_est = [bTi.compose(zillow_T_opensfm) if bTi is not None else None for bTi in bTi_list_est]
        # plot_3d_poses(aTi_list_gt, bTi_list_est)

        # align it to the 2d pose graph using Sim(3)
        aligned_bTi_list_est, _ = geometry_comparisons.align_poses_sim3_ignore_missing(aTi_list_gt, bTi_list_est)

        # plot_3d_poses(aTi_list_gt, aligned_bTi_list_est) # visualize after alignment

        # project to 2d
        est_floor_pose_graph = PoseGraph3d.from_wTi_list(aligned_bTi_list_est, building_id, floor_id)
        est_floor_pose_graph = est_floor_pose_graph.project_to_2d(gt_floor_pose_graph)

        logger.info(
            "Reconstruction found with %d cameras and %d points",
            len(reconstruction.pose_dict),
            reconstruction.points.shape[0],
        )

        # then measure the aligned error
        mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_aligned_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )

        os.makedirs(f"/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_viz/{building_id}_{floor_id}", exist_ok=True)
        plot_save_fpath = f"/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_viz/{building_id}_{floor_id}/opensfm_reconstruction_{r}.jpg"
        # render estimated layout
        est_floor_pose_graph.render_estimated_layout(
            show_plot=False, save_plot=True, plot_save_dir=None, gt_floor_pg=gt_floor_pose_graph, plot_save_fpath=plot_save_fpath
        )

        floor_results_dict = {
            "id": f"Reconstruction {r}",
            "num_cameras": len(reconstruction.pose_dict),
            "num_points": reconstruction.points.shape[0],
            "mean_abs_rot_err": mean_abs_rot_err,
            "mean_abs_trans_err": mean_abs_trans_err
        }
        floor_results_dicts.append(floor_results_dict)
    
    json_save_fpath = f"/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_results/{building_id}_{floor_id}.json"
    json_utils.save_json_dict(json_save_fpath, floor_results_dicts)



def test_measure_opensfm_localization_accuracy():
    pass

    # TODO: write unit test

    reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/skydio-32/reconstruction.json"

    import pdb

    pdb.set_trace()
    reconstructions = load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    for r, reconstruction in enumerate(reconstructions):

        from visualization.open3d_vis_utils import draw_scene_open3d

        fnames = [
            "S1014644.JPG",
            "S1014645.JPG",
            "S1014646.JPG",
            "S1014647.JPG",
            "S1014648.JPG",
            "S1014649.JPG",
            "S1014650.JPG",
            "S1014651.JPG",
            "S1014652.JPG",
            "S1014653.JPG",
            "S1014654.JPG",
            "S1014655.JPG",
            "S1014656.JPG",
            "S1014684.JPG",
            "S1014685.JPG",
            "S1014686.JPG",
            "S1014687.JPG",
            "S1014688.JPG",
            "S1014689.JPG",
            "S1014690.JPG",
            "S1014691.JPG",
            "S1014692.JPG",
            "S1014693.JPG",
            "S1014694.JPG",
            "S1014695.JPG",
            "S1014696.JPG",
            "S1014724.JPG",
            "S1014725.JPG",
            "S1014726.JPG",
            "S1014734.JPG",
            "S1014735.JPG",
            "S1014736.JPG",
        ]

        # point_cloud = np.zeros((0,3))
        # rgb = np.zeros((0,3))

        point_cloud = reconstruction.points
        rgb = reconstruction.rgb

        wTi_list = [reconstruction.pose_dict[fname] if fname in reconstruction.pose_dict else None for fname in fnames]
        N = len(wTi_list)
        # import pdb; pdb.set_trace()
        fx = reconstruction.camera.focal * 1000
        px = reconstruction.camera.width / 2
        py = reconstruction.camera.height / 2
        from gtsam import Cal3Bundler

        calibrations = [Cal3Bundler(fx=fx, k1=0, k2=0, u0=px, v0=py)] * N
        args = SimpleNamespace(**{"point_rendering_mode": "point"})
        draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)



def run_opensfm_over_all_zind() -> None:
    """ """
    OVERRIDES_FPATH = "/Users/johnlam/Downloads/OpenSfM/data/camera_models_overrides.json"

    OPENSFM_REPO_ROOT = "/Users/johnlam/Downloads/OpenSfM"
    # reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/ZinD_1442_floor_01/reconstruction.json"

    # building_id = "1442"
    # floor_id = "floor_01"
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    building_ids = [ Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*") ]
    building_ids.sort()
    
    for building_id in building_ids[90:]:
        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        for floor_id in floor_ids:
            try:
                src_pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
                pano_fpaths = glob.glob(f"{src_pano_dir}/{floor_id}_*.jpg")

                if len(pano_fpaths) == 0:
                    continue

                FLOOR_OPENSFM_DATADIR = f"{OPENSFM_REPO_ROOT}/data/ZinD_{building_id}_{floor_id}__2021_09_13"
                os.makedirs(f"{FLOOR_OPENSFM_DATADIR}/images", exist_ok=True)
                reconstruction_json_fpath = f"{FLOOR_OPENSFM_DATADIR}/reconstruction.json"
                
                dst_dir = f"{FLOOR_OPENSFM_DATADIR}/images"
                for pano_fpath in pano_fpaths:
                    fname = Path(pano_fpath).name
                    dst_fpath = f"{dst_dir}/{fname}"
                    shutil.copyfile(src=pano_fpath, dst=dst_fpath)

                # See https://opensfm.readthedocs.io/en/latest/using.html#providing-your-own-camera-parameters
                shutil.copyfile(OVERRIDES_FPATH, f"{FLOOR_OPENSFM_DATADIR}/camera_models_overrides.json")

                #import pdb; pdb.set_trace()
                cmd = f"bin/opensfm_run_all data/ZinD_{building_id}_{floor_id}__2021_09_13 2>&1 | tee {FLOOR_OPENSFM_DATADIR}/opensfm.log"
                run_command(cmd)

                # shutil.rmtree()

                # load_opensfm_reconstructions_from_json(reconstruction_json_fpath)
                measure_opensfm_localization_accuracy(reconstruction_json_fpath, building_id, floor_id, raw_dataset_dir)

            except Exception as e:
                logger.exception(f"OpenSfM failed for {building_id} {floor_id}")
                print(f"failed on Building {building_id} {floor_id}")
                continue


def get_buildingid_floorid_from_json_fpath(fpath: str) -> Tuple[str,str]:
    """
    From a JSON results fpath, get tour metadata.
    """
    json_fname_stem = Path(fpath).stem
    k = json_fname_stem.find('_f')
    building_id = json_fname_stem[:k]
    floor_id = json_fname_stem[k+1:]
    return building_id, floor_id



def analyze_opensfm_results():
    """ """
    json_results_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_results"

    num_ccs_per_floor = []
    cc_idx_arr = []
    num_cameras_in_cc = []
    avg_rot_err_per_cc = []
    avg_trans_err_per_cc = []

    num_dropped_cameras_per_floor = []
    percent_reconstructed_cameras_per_floor = []
    percent_in_largest_cc_per_floor = []

    json_fpaths = glob.glob(f"{json_results_dir}/*.json")
    for json_fpath in json_fpaths:
        data = json_utils.read_json_file(json_fpath)

        num_cameras = 0
        num_points = 0
        mean_abs_rot_err = 0
        mean_abs_trans_err = 0
        num_reconst_cameras_on_floor = 0
        num_reconst_cameras_largest_cc = 0

        num_ccs_per_floor.append(len(data))

        # loop through the connected components
        # CC's are sorted by cardinality
        for cc_idx, cc_info in enumerate(data):

            if cc_idx >= 10:
                continue

            num_cameras = cc_info["num_cameras"]
            num_points = cc_info["num_points"]
            mean_abs_rot_err = cc_info["mean_abs_rot_err"]
            mean_abs_trans_err = cc_info["mean_abs_trans_err"]

            num_reconst_cameras_on_floor += num_cameras
            if cc_idx == 0:
                num_reconst_cameras_largest_cc = num_cameras

            print(f"CC {cc_idx} has {num_cameras} cameras.")

            cc_idx_arr.append(cc_idx)
            num_cameras_in_cc.append(num_cameras)
            avg_rot_err_per_cc.append(mean_abs_rot_err)
            avg_trans_err_per_cc.append(mean_abs_trans_err)

        # how many cameras get left out?
        #import pdb; pdb.set_trace()
        building_id, floor_id = get_buildingid_floorid_from_json_fpath(json_fpath)
        panos_dirpath = f"/Users/johnlam/Downloads/OpenSfM/data/ZinD_{building_id}_{floor_id}__2021_09_13/images"
        num_panos = len(glob.glob(f"{panos_dirpath}/*.jpg"))
        num_dropped_cameras = num_panos - num_reconst_cameras_on_floor
        if num_panos == 0:
            import pdb; pdb.set_trace()
        percent_reconstructed = num_reconst_cameras_on_floor / num_panos * 100
        num_dropped_cameras_per_floor.append(num_dropped_cameras)
        percent_reconstructed_cameras_per_floor.append(percent_reconstructed)

        percent_in_largest_cc = num_reconst_cameras_largest_cc / num_panos * 100
        percent_in_largest_cc_per_floor.append(percent_in_largest_cc)

        if percent_in_largest_cc > 100:
            import pdb; pdb.set_trace()

    print("Mean percent_in_largest_cc_per_floor: ", np.mean(percent_in_largest_cc_per_floor))
    print("Median percent_in_largest_cc_per_floor: ", np.median(percent_in_largest_cc_per_floor))

    plt.hist(percent_in_largest_cc_per_floor, bins=20)
    plt.title("Histogram % of All Panos Localized in Largest CC")
    plt.ylabel("Counts")
    plt.xlabel("% of All Panos Localized in Largest CC")
    plt.show()

    plt.hist(num_dropped_cameras_per_floor)
    plt.title("num_dropped_cameras_per_floor)")
    plt.ylabel("Counts")
    plt.show()

    plt.hist(percent_reconstructed_cameras_per_floor, bins=20)
    plt.ylabel("Counts")
    plt.xlabel("Camera Localization Percent as Union of all CCs")
    plt.title("percent_reconstructed_cameras_per_floor")
    plt.show()

    print("Mean num_dropped_cameras_per_floor: ", np.mean(num_dropped_cameras_per_floor))
    print("Median num_dropped_cameras_per_floor: ", np.median(num_dropped_cameras_per_floor))

    print("Mean percent_reconstructed_cameras_per_floor: ", np.mean(percent_reconstructed_cameras_per_floor))
    print("Median percent_reconstructed_cameras_per_floor: ", np.median(percent_reconstructed_cameras_per_floor))

    print("Mean of Avg. Rot. Error within CC: ", np.mean(avg_rot_err_per_cc))
    print("Median of Avg. Rot. Error within CC: ", np.median(avg_rot_err_per_cc))
    
    print("Mean of Avg. Trans. Error within CC: ", np.mean(avg_trans_err_per_cc))
    print("Median of Avg. Trans. Error within CC: ", np.median(avg_trans_err_per_cc))

    plt.hist(avg_trans_err_per_cc, bins=np.linspace(0,5,20))
    plt.ylabel("Counts")
    plt.xlabel("Avg. Trans. Error per CC")
    plt.show()

    plt.hist(avg_rot_err_per_cc, bins=np.linspace(0,5,20))
    plt.ylabel("Counts")
    plt.xlabel("Avg. Rot. Error per CC")
    plt.show()

    # average number of cameras in first 10 components
    camera_counts_per_cc_idx = np.zeros(10)
    for (cc_idx, num_cameras) in zip(cc_idx_arr, num_cameras_in_cc):
        camera_counts_per_cc_idx[cc_idx] += num_cameras
    camera_counts_per_cc_idx /= len(json_fpaths)
    
    plt.bar(x=range(10), height=camera_counts_per_cc_idx)
    plt.xticks(range(10))
    plt.title("Avg. # Cameras in i'th CC")
    plt.xlabel("i'th CC")
    plt.ylabel("Avg. # Cameras")
    plt.show()
    #import pdb; pdb.set_trace()
    quit()

    # Histogram of number of CCs per floor
    plt.hist(num_ccs_per_floor, bins=np.arange(0,20)-0.5) # center the bins
    plt.xticks(range(20))
    plt.xlabel("Number of CCs per Floor")
    plt.ylabel("Counts")
    plt.title("Histogram of Number of CCs per Floor")
    plt.show()

    # average rot error vs. number of cameras in component
    plt.scatter(num_cameras_in_cc, avg_rot_err_per_cc, 10, color='r', marker='.')
    plt.title("Avg Rot Error per CC vs. Num Cameras in CC")
    plt.xlabel("Num Cameras in CC")
    plt.ylabel("Avg Rot Error per CC (degrees)")
    plt.show()

    # average trans error vs. number of cameras in component
    plt.scatter(num_cameras_in_cc, avg_trans_err_per_cc, 10, color='r', marker='.')
    plt.title("Avg Translation Error per CC vs. Num Cameras in CC")
    plt.xlabel("Num Cameras in CC")
    plt.ylabel("Avg Translation Error per CC")
    plt.show()

    # histogram of CC size
    plt.hist(num_cameras_in_cc, bins=np.arange(0,20)-0.5) # center the bins
    plt.xticks(range(20))
    plt.xlabel("Number of Cameras in CC")
    plt.ylabel("Counts")
    plt.title("Histogram of Number of Cameras per CC")
    plt.show()


   

if __name__ == "__main__":
    """
    cd ~/Downloads/OpenSfM
    python ~/Downloads/jlambert-auto-floorplan/afp/baselines/opensfm.py

    """
    #run_opensfm_over_all_zind()
    # test_measure_opensfm_localization_accuracy()

    analyze_opensfm_results()



