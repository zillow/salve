"""

https://github.com/mapillary/OpenSfM/blob/master/opensfm/io.py#L214

"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gtsam
import gtsfm.utils.geometry_comparisons as geometry_comparisons
import numpy as np
from argoverse.utils.json_utils import read_json_file
from colour import Color
from gtsam import Pose3, Rot3, Similarity3

import open3d
# import below is from gtsfm
import visualization.open3d_vis_utils as open3d_vis_utils


from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
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
        camera = SimpleNamespace(**{"projection_type": pt, "width": obj["width"], "height": obj["height"], "focal": f })
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
        #pano_id = key
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


def get_colormap(N: int) -> np.ndarray:
    """

    Args:
        N: number of unique colors to generate.

    Returns:
        colormap: uint8 array of shape (N,3)
    """
    colormap = np.array(
        [[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), N)]
    ).squeeze()
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
    RED = np.array([1,0,0])
    GREEN = np.array([0,1,0])
    BLUE = np.array([0,0,1])
    colors = (RED, GREEN, BLUE)

    line_sets = []
    for axis, color in zip([0,1,2], colors):

        lines = [[0, 1]]
        verts_worldfr = np.zeros((2,3))
        
        verts_camfr = np.zeros((2,3))
        verts_camfr[0,axis] = axis_length
       
        for i in range(2):
            verts_worldfr[i] = wTc.transformFrom(verts_camfr[i])

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(verts_worldfr),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(color.reshape(1,3))
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



def measure_opensfm_localization_accuracy(reconstruction_json_fpath: str, building_id: str, floor_id: str, raw_dataset_dir: str) -> None:
    """

    Args:
        reconstruction_json_fpath:
        building_id:
        floor_id:
        raw_dataset_dir:
    """

    cmd = "bin/opensfm_run_all data/ZinD_1442_floor_01"

    gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    reconstructions = load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    for r, reconstruction in enumerate(reconstructions):

        # create a 3d pose graph
        aTi_list_gt = gt_floor_pose_graph.as_3d_pose_graph()
        bTi_list_est = [reconstruction.pose_dict.get(i, None) for i in range(len(aTi_list_gt))]

        aTi_list_gt = [aTi if bTi_list_est[i] is not None else None for i, aTi in enumerate(aTi_list_gt)]

        Rx = np.pi / 2
        Ry = 0.0
        Rz = 0.0
        zillow_R_opensfm = Rot3.RzRyRx(Rx, Ry, Rz)
        zillow_T_opensfm = Pose3(zillow_R_opensfm, np.zeros(3))
        
        # world frame <-> opensfm camera <-> zillow camera
        bTi_list_est = [ bTi.compose(zillow_T_opensfm) if bTi is not None else None for bTi in bTi_list_est]
        #plot_3d_poses(aTi_list_gt, bTi_list_est)

        # align it to the 2d pose graph using Sim(3)
        aligned_bTi_list_est, _ = geometry_comparisons.align_poses_sim3_ignore_missing(aTi_list_gt, bTi_list_est)

        #plot_3d_poses(aTi_list_gt, aligned_bTi_list_est) # visualize after alignment

        # project to 2d
        est_floor_pose_graph = PoseGraph3d.from_wTi_list(aligned_bTi_list_est, building_id, floor_id)
        est_floor_pose_graph = est_floor_pose_graph.project_to_2d(gt_floor_pose_graph)

        logger.info("Reconstruction found with %d cameras and %d points", len(reconstruction.pose_dict), reconstruction.points.shape[0])

        # then measure the aligned error
        mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_aligned_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )

        plot_save_dir = f"opensfm_1442_reconstruction_{r}"
        # render estimated layout
        est_floor_pose_graph.render_estimated_layout(
            show_plot=False, save_plot=True, plot_save_dir=plot_save_dir, gt_floor_pg=gt_floor_pose_graph
        )




@dataclass
class PoseGraph3d:
    building_id: str
    floor_id: str
    pose_dict: Dict[int, Pose3]

    def project_to_2d(self, gt_floor_pose_graph: PoseGraph2d) -> PoseGraph2d:
        """
        Args:
            gt_floor_pose_graph

        Returns:
            est_floor_pose_graph
        """
        #import pdb; pdb.set_trace()

        n = len(gt_floor_pose_graph.as_3d_pose_graph())

        wRi_list = []
        wti_list = []
        for i in range(n):
            wTi = self.pose_dict.get(i, None)
            if wTi is not None:
                wRi = wTi.rotation().matrix()[:2,:2]
                wti = wTi.translation()[:2]
            else:
                wRi, wti = None, None

            wRi_list.append(wRi)
            wti_list.append(wti)

        est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(
            wRi_list, wti_list, gt_floor_pose_graph, self.building_id, self.floor_id
        )
        return est_floor_pose_graph

    @classmethod
    def from_wTi_list(cls, wTi_list: List[Optional[Pose3]], building_id: str, floor_id: str) -> "PoseGraph3d":
        """
        Args:
            wTi_list
            building_id
            floor_id
        """
        #import pdb; pdb.set_trace()

        pose_dict = {}
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            pose_dict[i] = wTi

        return cls(building_id, floor_id, pose_dict)


def test_measure_opensfm_localization_accuracy():
    pass

    # TODO: write unit test

    reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/skydio-32/reconstruction.json"

    import pdb; pdb.set_trace()
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
            "S1014736.JPG"
        ]

        # point_cloud = np.zeros((0,3))
        # rgb = np.zeros((0,3))

        point_cloud = reconstruction.points
        rgb = reconstruction.rgb

        wTi_list = [ reconstruction.pose_dict[fname] if fname in reconstruction.pose_dict else None for fname in fnames]
        N = len(wTi_list)
        #import pdb; pdb.set_trace()
        fx = reconstruction.camera.focal * 1000
        px = reconstruction.camera.width / 2
        py = reconstruction.camera.height / 2
        from gtsam import Cal3Bundler
        calibrations = [Cal3Bundler(fx=fx, k1=0, k2=0, u0=px, v0=py)] * N
        args = SimpleNamespace(**{"point_rendering_mode": "point"})
        draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)



if __name__ == "__main__":
    """ """
    reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/ZinD_1442_floor_01/reconstruction.json"


    building_id = "1442"
    floor_id = "floor_01"
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    # load_opensfm_reconstructions_from_json(reconstruction_json_fpath)
    measure_opensfm_localization_accuracy(reconstruction_json_fpath, building_id, floor_id, raw_dataset_dir)

    #test_measure_opensfm_localization_accuracy()


