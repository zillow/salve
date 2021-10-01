
"""
Utility to evaluate an SfM algorithm baseline, such as OpenMVG or OpenSfM.
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
from afp.common.posegraph2d import PoseGraph2d, get_gt_pose_graph
from afp.common.posegraph3d import PoseGraph3d
from afp.utils.logger_utils import get_logger


logger = get_logger()


def get_zillow_T_opensfm() -> Pose3:
    """
    Transform OpenSfM camera to ZinD camera.
    """
    # in radians
    Rx = np.pi / 2
    Ry = 0.0
    Rz = 0.0
    zillow_R_opensfm = Rot3.RzRyRx(Rx, Ry, Rz)
    zillow_T_opensfm = Pose3(zillow_R_opensfm, np.zeros(3))
    return zillow_T_opensfm


def get_zillow_T_openmvg() -> Pose3:
    """ 
    Transform OpenMVG camera to ZinD camera.

    Note: x,y,z axes correspond to red, green, blue colors.

    """
    # in radians
    Rx = 0.0 #-np.pi/2 # -90 deg.
    Ry = -np.pi/2 # 0.0
    Rz = 0.0 #- np.pi/2 # 90 deg.
    zillow_R_openmvg = Rot3.RzRyRx(Rx, Ry, Rz)
    zillow_T_openmvg = Pose3(zillow_R_openmvg, np.zeros(3))
    return zillow_T_openmvg


def measure_algorithm_localization_accuracy(
    reconstruction_json_fpath: str, building_id: str, floor_id: str, raw_dataset_dir: str, algorithm_name: str
) -> None:
    """

    Note: we do not refer to these as "baselines", as "baseline" means something quite different for SfM.

    Args:
        reconstruction_json_fpath:
        building_id:
        floor_id:
        raw_dataset_dir:
    """
    if algorithm_name == "opensfm":
    	reconstructions = opensfm_utils.load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    elif algorithm_name == "openmvg":
    	reconstructions = openmvg_utils.load_openmvg_reconstructions_from_json(building_id, floor_id)

    gt_floor_pose_graph = get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

    floor_results_dicts = []
    for r, reconstruction in enumerate(reconstructions):

        # create a 3d pose graph
        aTi_list_gt = gt_floor_pose_graph.as_3d_pose_graph()
        bTi_list_est = [reconstruction.pose_dict.get(i, None) for i in range(len(aTi_list_gt))]

        aTi_list_gt = [aTi if bTi_list_est[i] is not None else None for i, aTi in enumerate(aTi_list_gt)]

        if algorithm_name == "opensfm":
            zillow_T_opensfm = get_zillow_T_opensfm()
            zillowcam_T_algocam = zillow_T_opensfm
            # world frame <-> opensfm camera <-> zillow camera
        elif algorithm_name == "openmvg":

            zillowcam_T_openmvg = get_zillow_T_openmvg()
            zillowcam_T_algocam = zillowcam_T_openmvg
            #zillowcam_T_algocam = Pose3()

        bTi_list_est = [bTi.compose(zillowcam_T_algocam) if bTi is not None else None for bTi in bTi_list_est]
        
        import pdb; pdb.set_trace()
        plot_3d_poses(aTi_list_gt, bTi_list_est)

        # align it to the 2d pose graph using Sim(3)
        aligned_bTi_list_est, _ = geometry_comparisons.align_poses_sim3_ignore_missing(aTi_list_gt, bTi_list_est)

        plot_3d_poses(aTi_list_gt, aligned_bTi_list_est) # visualize after alignment

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



def plot_3d_poses(aTi_list_gt: List[Optional[Pose3]], bTi_list_est: List[Optional[Pose3]]) -> None:
    """

    Ground truth poses are rendered large (sphere of radius 0.5)
    Estimated poses are rendered small (spehere of radius 0.2)
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


def analyze_algorithm_results(json_results_dict: str) -> None:
    """ """

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


def main():
    """ """

    # for OpenSFM
    # json_results_dir = "/Users/johnlam/Downloads/jlambert-auto-floorplan/opensfm_zind_results"
    # analyze_algorithm_results(json_results_dir)


    # For OpenMVG
    #run_openmvg_all_tours()

    reconstruction_json_fpath = "/Users/johnlam/Downloads/openmvg_demo/ZinD_1183_floor_01__2021_09_21/reconstruction/sfm_data.json"
    building_id = "1183"
    floor_id = "floor_01"
    raw_dataset_dir = "/Users/johnlam/Downloads/complete_07_10_new"
    measure_algorithm_localization_accuracy(reconstruction_json_fpath, building_id, floor_id, raw_dataset_dir, algorithm_name="openmvg")

    # then analyze the mean statistics




if __name__ == "__main__":
	main()




