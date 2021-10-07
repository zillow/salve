"""
Utility to evaluate an SfM algorithm baseline, such as OpenMVG or OpenSfM.

# TODO: only measure localization precision for connected components with 3+ cameras. (or measure separately)
# 2-camera connected components should alawys achieve perfect translation alignment under Sim(3)? (TODO: write unit tests).
(not true -- close -- but rotation also plays a role in it.)
"""

import glob
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gtsam
import gtsfm.utils.geometry_comparisons as geometry_comparisons
import matplotlib.pyplot as plt
import numpy as np
import open3d
import argoverse.utils.json_utils as json_utils
from colour import Color
from gtsam import Pose3, Rot3

# import below is from gtsfm
import visualization.open3d_vis_utils as open3d_vis_utils

import afp.baselines.opensfm as opensfm_utils
import afp.baselines.openmvg as openmvg_utils
from afp.baselines.openmvg import OPENMVG_DEMO_ROOT
from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
from afp.common.posegraph3d import PoseGraph3d
from afp.utils.logger_utils import get_logger


logger = get_logger()


def get_opensfm_T_zillow() -> Pose3:
    """Transform OpenSfM spherical camera to ZinD spherical camera.

    See https://github.com/mapillary/OpenSfM/issues/794
    """
    # in radians
    Rx = np.pi / 2
    Ry = 0.0
    Rz = 0.0
    opensfm_R_zillow = Rot3.RzRyRx(Rx, Ry, Rz)
    opensfm_T_zillow = Pose3(opensfm_R_zillow, np.zeros(3))
    return opensfm_T_zillow


def get_openmvg_T_zillow() -> Pose3:
    """Transform OpenMVG spherical camera to ZinD spherical camera.

    See: https://github.com/openMVG/openMVG/issues/1938

    Note: x,y,z axes correspond to red, green, blue colors.

    """
    # in radians
    # Rx = 0.0  # -np.pi/2 # -90 deg.
    # Ry = -np.pi / 2  # 0.0
    # Rz = 0.0  # - np.pi/2 # 90 deg.

    Rx = np.pi / 2
    Ry = 0.0
    Rz = 0.0

    openmvg_R_zillow = Rot3.RzRyRx(Rx, Ry, Rz)
    openmvg_T_zillow = Pose3(openmvg_R_zillow, np.zeros(3))
    return openmvg_T_zillow


def save_empty_json_results_file(building_id: str, floor_id: str, algorithm_name: str) -> None:
    """ """
    floor_results_dicts = [
        {
            "id": f"Reconstruction 0",
            "num_cameras": 0,
            "num_points": 0,
            "mean_abs_rot_err": np.nan,
            "mean_abs_trans_err": np.nan
        }
    ]
    os.makedirs(f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_results", exist_ok=True)
    json_save_fpath = (
        f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_results/{building_id}_{floor_id}.json"
    )
    json_utils.save_json_dict(json_save_fpath, floor_results_dicts)


def measure_algorithm_localization_accuracy(
    building_id: str, floor_id: str, raw_dataset_dir: str, algorithm_name: str, reconstruction_json_fpath: Optional[str] = None
) -> None:
    """Evaluate reconstruction from a single floor against GT poses, via Sim(3) alignment.

    Note: we do not refer to 3rd part SfM implementations as "baselines", but rather as "algorithms", as "baseline" means
    something quite different in the context of SfM.

    Args:
        building_id:
        floor_id:
        raw_dataset_dir:
        algorithm_name
        reconstruction_json_fpath:
    """
    if algorithm_name == "opensfm":
        reconstructions = opensfm_utils.load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    elif algorithm_name == "openmvg":
        reconstructions = openmvg_utils.load_openmvg_reconstructions_from_json(building_id, floor_id)
        if len(reconstructions[0].pose_dict) == 0:
            save_empty_json_results_file(building_id, floor_id, algorithm_name=algorithm_name)
            return

    gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    floor_results_dicts = []
    for r, reconstruction in enumerate(reconstructions):

        # create a 3d pose graph
        aTi_list_gt = gt_floor_pose_graph.as_3d_pose_graph()
        bTi_list_est = [reconstruction.pose_dict.get(i, None) for i in range(len(aTi_list_gt))]

        aTi_list_gt = [aTi if bTi_list_est[i] is not None else None for i, aTi in enumerate(aTi_list_gt)]

        if algorithm_name == "opensfm":
            opensfm_T_zillow = get_opensfm_T_zillow()
            algocam_T_zillowcam = opensfm_T_zillow
            # world frame <-> opensfm camera <-> zillow camera
        elif algorithm_name == "openmvg":

            openmvg_T_zillowcam = get_openmvg_T_zillow()
            algocam_T_zillowcam = openmvg_T_zillowcam
            #algocam_T_zillowcam = Pose3()

        bTi_list_est = [bTi.compose(algocam_T_zillowcam) if bTi is not None else None for bTi in bTi_list_est]

        # plot_3d_poses(aTi_list_gt, bTi_list_est)

        # align it to the 2d pose graph using Sim(3)
        aligned_bTi_list_est, _ = geometry_comparisons.align_poses_sim3_ignore_missing(aTi_list_gt, bTi_list_est)

        # plot_3d_poses(aTi_list_gt, aligned_bTi_list_est)  # visualize after alignment

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

        os.makedirs(
            f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_viz/{building_id}_{floor_id}", exist_ok=True
        )
        plot_save_fpath = f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_viz/{building_id}_{floor_id}/{algorithm_name}_reconstruction_{r}.jpg"
        # render estimated layout
        est_floor_pose_graph.render_estimated_layout(
            show_plot=False,
            save_plot=True,
            plot_save_dir=None,
            gt_floor_pg=gt_floor_pose_graph,
            plot_save_fpath=plot_save_fpath,
        )

        floor_results_dict = {
            "id": f"Reconstruction {r}",
            "num_cameras": len(reconstruction.pose_dict),
            "num_points": reconstruction.points.shape[0],
            "mean_abs_rot_err": mean_abs_rot_err,
            "mean_abs_trans_err": mean_abs_trans_err,
        }
        floor_results_dicts.append(floor_results_dict)

    os.makedirs(f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_results", exist_ok=True)
    json_save_fpath = (
        f"/Users/johnlam/Downloads/jlambert-auto-floorplan/{algorithm_name}_zind_results/{building_id}_{floor_id}.json"
    )
    json_utils.save_json_dict(json_save_fpath, floor_results_dicts)


def plot_3d_poses(aTi_list_gt: List[Optional[Pose3]], bTi_list_est: List[Optional[Pose3]]) -> None:
    """

    Ground truth poses are rendered large (sphere of radius 0.5)
    Estimated poses are rendered small (spehere of radius 0.2)

    Args:
        aTi_list_gt: list of ground truth camera poses.
        bTi_list_est: list of estimated camera poses.
    """

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


def get_colormap(N: int) -> np.ndarray:
    """Obtain an RGB colormap from red to green, with N unique colors.

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

    Note: x,y,z axes correspond to red, green, blue colors.

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


def count_panos_on_floor(raw_dataset_dir: str, building_id: str, floor_id: str) -> int:
    """Count the number of panoramas on a specified floor of a specified building."""
    src_pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
    pano_fpaths = glob.glob(f"{src_pano_dir}/{floor_id}_*.jpg")
    return len(pano_fpaths)


def test_count_panos_on_floor() -> None:
    """Ensure that the number of panoramas on a floor is counted correctly."""

    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    building_id = "379"

    num_floor0_panos = count_panos_on_floor(raw_dataset_dir, building_id, floor_id="floor_00")
    assert num_floor0_panos == 8

    num_floor1_panos = count_panos_on_floor(raw_dataset_dir, building_id, floor_id="floor_01")
    assert num_floor1_panos == 13


def analyze_algorithm_results(json_results_dir: str, raw_dataset_dir: str) -> None:
    """Analyze the accuracy of global pose estimation (camera localization) from a third-party SfM algorithm.

    Analzes the average completeness of the recovered poses, and global pose estimation precision.

    Args:
        json_results_dir:
        raw_dataset_dir
    """
    num_ccs_per_floor = []
    # stats below are aggregated over all CCs, independent of which floor or building they came from.
    cc_idx_arr = []
    num_cameras_in_cc = []
    avg_rot_err_per_cc = []
    avg_trans_err_per_cc = []

    # stats over all floors, independent of which building they came from.
    num_dropped_cameras_per_floor = []
    percent_reconstructed_cameras_per_floor = []
    percent_in_largest_cc_per_floor = []

    json_fpaths = glob.glob(f"{json_results_dir}/*.json")
    num_reconstructed_floors = len(json_fpaths)

    for json_fpath in json_fpaths:
        # each entry in the list contains info about a single CC
        all_cc_data = json_utils.read_json_file(json_fpath)

        # set to zero, in case the floor has no CCs, to avoid defaulting to previous variable value
        num_cameras = 0
        mean_abs_rot_err = 0
        mean_abs_trans_err = 0
        num_reconst_cameras_on_floor = 0 # represents the cardinality of union of all CCs
        num_reconst_cameras_largest_cc = 0

        num_ccs_per_floor.append(len(all_cc_data))

        if num_ccs_per_floor == 0:
            import pdb; pdb.set_trace()

        # loop through the connected components
        # CC's are sorted by cardinality
        for cc_idx, cc_info in enumerate(all_cc_data):

            if cc_idx >= 10:
                continue

            # Note: we ignore the number of 3d reconstructed points, accessible via cc_info["num_points"]
            num_cameras = cc_info["num_cameras"]
            mean_abs_rot_err = cc_info["mean_abs_rot_err"]
            mean_abs_trans_err = cc_info["mean_abs_trans_err"]

            num_reconst_cameras_on_floor += num_cameras
            if cc_idx == 0:
                num_reconst_cameras_largest_cc = num_cameras

            #print(f"CC {cc_idx} has {num_cameras} cameras.")

            cc_idx_arr.append(cc_idx)
            num_cameras_in_cc.append(num_cameras)
            avg_rot_err_per_cc.append(mean_abs_rot_err)
            avg_trans_err_per_cc.append(mean_abs_trans_err)

        # how many cameras get left out of any CC, on this floor?
        building_id, floor_id = get_buildingid_floorid_from_json_fpath(json_fpath)
        print(f"\tAnalyzing Building {building_id}, {floor_id}")

        num_panos = count_panos_on_floor(raw_dataset_dir, building_id, floor_id)
        # find # of cameras that didn't appear anywhere in the union of all CCs
        num_dropped_cameras = num_panos - num_reconst_cameras_on_floor
        num_dropped_cameras_per_floor.append(num_dropped_cameras)
        if num_panos == 0:
            import pdb
            pdb.set_trace()

        # what % is found in the union?
        percent_reconstructed = num_reconst_cameras_on_floor / num_panos * 100
        percent_reconstructed_cameras_per_floor.append(percent_reconstructed)

        percent_in_largest_cc = num_reconst_cameras_largest_cc / num_panos * 100
        percent_in_largest_cc_per_floor.append(percent_in_largest_cc)

        if percent_in_largest_cc > 100:
            raise RuntimeError("% in largest CC must be <100%.")

    print(f"Computed numbers over {num_reconstructed_floors} reconstructed floors.")

    print("Mean percent_in_largest_cc_per_floor: ", np.mean(percent_in_largest_cc_per_floor))
    print("Median percent_in_largest_cc_per_floor: ", np.median(percent_in_largest_cc_per_floor))

    plt.hist(percent_in_largest_cc_per_floor, bins=20)
    plt.title("Histogram % of All Panos Localized in Largest CC")
    plt.ylabel("Counts")
    plt.xlabel("% of All Panos Localized in Largest CC")
    plt.show()

    plt.hist(num_dropped_cameras_per_floor, bins=20)
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

    print("Mean of Avg. Rot. Error within CC: ", np.nanmean(avg_rot_err_per_cc))
    print("Median of Avg. Rot. Error within CC: ", np.nanmedian(avg_rot_err_per_cc))

    print("Mean of Avg. Trans. Error within CC: ", np.nanmean(avg_trans_err_per_cc))
    print("Median of Avg. Trans. Error within CC: ", np.nanmedian(avg_trans_err_per_cc))

    plt.hist(avg_trans_err_per_cc, bins=np.linspace(0, 5, 20))
    plt.ylabel("Counts")
    plt.xlabel("Avg. Trans. Error per CC")
    plt.show()

    plt.hist(avg_rot_err_per_cc, bins=np.linspace(0, 5, 20))
    plt.ylabel("Counts")
    plt.xlabel("Avg. Rot. Error per CC")
    plt.show()

    # average number of cameras in first 10 components
    camera_counts_per_cc_idx = np.zeros(10)
    for (cc_idx, num_cameras) in zip(cc_idx_arr, num_cameras_in_cc):
        camera_counts_per_cc_idx[cc_idx] += num_cameras
    camera_counts_per_cc_idx /= num_reconstructed_floors

    plt.bar(x=range(10), height=camera_counts_per_cc_idx)
    plt.xticks(range(10))
    plt.title("Avg. # Cameras in i'th CC")
    plt.xlabel("i'th CC")
    plt.ylabel("Avg. # Cameras")
    plt.show()

    # Histogram of number of CCs per floor
    plt.hist(num_ccs_per_floor, bins=np.arange(0, 20) - 0.5)  # center the bins
    plt.xticks(range(20))
    plt.xlabel("Number of CCs per Floor")
    plt.ylabel("Counts")
    plt.title("Histogram of Number of CCs per Floor")
    plt.show()

    # average rot error vs. number of cameras in component
    plt.scatter(num_cameras_in_cc, avg_rot_err_per_cc, 10, color="r", marker=".")
    plt.title("Avg Rot Error per CC vs. Num Cameras in CC")
    plt.xlabel("Num Cameras in CC")
    plt.ylabel("Avg Rot Error per CC (degrees)")
    plt.show()

    # average trans error vs. number of cameras in component
    plt.scatter(num_cameras_in_cc, avg_trans_err_per_cc, 10, color="r", marker=".")
    plt.title("Avg Translation Error per CC vs. Num Cameras in CC")
    plt.xlabel("Num Cameras in CC")
    plt.ylabel("Avg Translation Error per CC")
    plt.show()

    # histogram of CC size
    plt.hist(num_cameras_in_cc, bins=np.arange(0, 20) - 0.5)  # center the bins
    plt.xticks(range(20))
    plt.xlabel("Number of Cameras in CC")
    plt.ylabel("Counts")
    plt.title("Histogram of Number of Cameras per CC")
    plt.show()


def get_buildingid_floorid_from_json_fpath(fpath: str) -> Tuple[str, str]:
    """From a JSON results fpath, get tour metadata.

    Args:
        fpath: file path to JSON file, of the form `1167_floor_02.json`
    """
    json_fname_stem = Path(fpath).stem
    k = json_fname_stem.find("_f")
    building_id = json_fname_stem[:k]
    floor_id = json_fname_stem[k + 1 :]
    return building_id, floor_id



def eval_openmvg_errors_all_tours():
    """
    Evaluate dumped sfm_data.json from every tour against ground truth.
    """
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    building_ids = [Path(dirpath).stem for dirpath in glob.glob(f"{raw_dataset_dir}/*")]
    building_ids.sort()

    for building_id in building_ids:
        floor_ids = ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04", "floor_05"]

        for floor_id in floor_ids:

            matches_dirpath = f"{OPENMVG_DEMO_ROOT}/ZinD_{building_id}_{floor_id}__2021_09_21/matches"
            if not Path(matches_dirpath).exists():
                continue

            reconstruction_json_fpath = f"{OPENMVG_DEMO_ROOT}/ZinD_{building_id}_{floor_id}__2021_09_21/reconstruction/sfm_data.json"
            
            # whether we want consider failed reconstructions
            if Path(matches_dirpath).exists() and not Path(reconstruction_json_fpath).exists():
                save_empty_json_results_file(building_id, floor_id, algorithm_name="openmvg")

            if not Path(reconstruction_json_fpath).exists():
                continue

            print(f"Running OpenMVG on {building_id}, {floor_id}")

            measure_algorithm_localization_accuracy(
                building_id=building_id,
                floor_id=floor_id,
                raw_dataset_dir=raw_dataset_dir,
                algorithm_name="openmvg",
                reconstruction_json_fpath=None #reconstruction_json_fpath,
            )


def main():
    """ """
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"

    """
    # for OpenSFM -- analyze error results dumped to JSON files.
    json_results_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_results"
    analyze_algorithm_results(json_results_dir, raw_dataset_dir)
    """


    # For OpenMVG
    # run_openmvg_all_tours()

    # reconstruction_json_fpath = "/Users/johnlam/Downloads/openmvg_demo/ZinD_1183_floor_01__2021_09_21/reconstruction/sfm_data.json"
    # building_id = "1183"
    # floor_id = "floor_01"

    # reconstruction_json_fpath

    eval_openmvg_errors_all_tours()
    # then analyze the mean statistics
    json_results_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/openmvg_zind_results"
    analyze_algorithm_results(json_results_dir, raw_dataset_dir)


if __name__ == "__main__":
    main()
