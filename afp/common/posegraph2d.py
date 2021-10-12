"""
Class to represent 2d pose graphs, render them, and compute error between two of them.
"""

import copy
import glob
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2
from gtsam import Point3, Rot3, Pose3
from gtsfm.utils.geometry_comparisons import align_poses_sim3_ignore_missing

from afp.common.pano_data import FloorData, PanoData, generate_Sim2_from_floorplan_transform
from afp.utils.rotation_utils import rotmat2d, wrap_angle_deg

REDTEXT = "\033[91m"
ENDCOLOR = "\033[0m"


class PoseGraph2d(NamedTuple):
    """Pose graph for a single floor.

    Note: edges are not included here, since there are different types of adjacency (spatial vs. visible)

    Args:
        building_id
        floor_id
        nodes:
    """

    building_id: int
    floor_id: str
    nodes: Dict[int, PanoData]
    scale_meters_per_coordinate: float

    def __repr__(self) -> str:
        """ """
        return f"Graph has {len(self.nodes.keys())} nodes in Building {self.building_id}, {self.floor_id}: {self.nodes.keys()}"


    def get_camera_height_m(self, pano_id: int) -> float:
        """Obtain the actual height of a RICOH Theta camera, from the saved ZinD dataset information.

        Args:
            pano_id: 
            TODO: this is same for every pano on a floor, argument is unnecessary (can be removed)
               but programmatically verify this first.

        Returns:
            camera_height_m: camera height above the floor, in meters.
        """
        if pano_id not in self.nodes:
            print(f"Pano id {pano_id} not found among {self.nodes.keys()}")
            import pdb; pdb.set_trace()

        # from zillow_floor_map["scale_meters_per_coordinate"][floor_id]
        worldmetric_s_worldnormalized = self.scale_meters_per_coordinate

        # from pano_data["floor_plan_transformation"]["scale"]
        worldnormalized_s_egonormalized = self.nodes[pano_id].global_Sim2_local.scale

        worldmetric_s_egonormalized = worldmetric_s_worldnormalized * worldnormalized_s_egonormalized

        cam_height_egonormalized = 1.0
        camera_height_m = worldmetric_s_egonormalized * cam_height_egonormalized

        print(f"Camera height (meters): {camera_height_m:.2f}")
        return camera_height_m


    @classmethod
    def from_floor_data(cls, building_id: str, fd: FloorData, scale_meters_per_coordinate: float) -> "PoseGraph2d":
        """ """
        print("scale_meters_per_coordinate: ", scale_meters_per_coordinate)
        return cls(
            building_id=building_id,
            floor_id=fd.floor_id,
            nodes={p.id: p for p in fd.panos},
            scale_meters_per_coordinate=scale_meters_per_coordinate,
        )

    @classmethod
    def from_json(cls, json_fpath: str) -> "PoseGraph2d":
        """ """
        pass

    @classmethod
    def from_wRi_list(cls, wRi_list: List[np.ndarray], building_id: str, floor_id: str) -> "PoseGraph2d":
        """
        from 2x2 rotations

        Fill other pano metadata with dummy values. Alternatively, could populate them from the GT pose graph.
        """
        nodes = {}
        for i, wRi in enumerate(wRi_list):
            if wRi is None:
                continue

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local=Sim2(R=wRi, t=np.zeros(2), s=1.0),
                room_vertices_local_2d=np.zeros((0, 2)),
                image_path="",
                label="",
                doors=None,
                windows=None,
                openings=None,
            )

        return cls(building_id=building_id, floor_id=floor_id, nodes=nodes)

    def as_3d_pose_graph(self) -> List[Optional[Pose3]]:
        """

        Returns:
            wTi_list: list of length N, where N is the one greater than the largest index
                in the dictionary.
        """
        num_images = max(self.nodes.keys()) + 1
        wTi_list = [None] * num_images
        for i, pano_obj in self.nodes.items():

            wRi = pano_obj.global_Sim2_local.rotation
            wti = pano_obj.global_Sim2_local.translation
            wti = np.array([wti[0], wti[1], 0.0])
            wRi = rot2x2_to_Rot3(wRi)

            wTi = Pose3(wRi, Point3(wti))

            wTi_list[i] = wTi

        return wTi_list

    @classmethod
    def from_wRi_wti_lists(
        cls,
        wRi_list: List[np.ndarray],
        wti_list: List[np.ndarray],
        gt_floor_pg: "PoseGraph2d",
        building_id: str,
        floor_id: str,
    ) -> "PoseGraph2d":
        """
        2x2
        and 2,

        Fill other pano metadata with values from the ground truth pose graph.
        """
        nodes = {}
        for i, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
            if wRi is None or wti is None:
                continue

            # update the global pose associated with each WDO object
            global_Sim2_local = Sim2(R=wRi, t=wti, s=1.0)
            doors = copy.deepcopy(gt_floor_pg.nodes[i].doors)
            windows = copy.deepcopy(gt_floor_pg.nodes[i].windows)
            openings = copy.deepcopy(gt_floor_pg.nodes[i].openings)

            for door in doors:
                door.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            for window in windows:
                window.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            for opening in openings:
                opening.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local=global_Sim2_local,
                room_vertices_local_2d=gt_floor_pg.nodes[i].room_vertices_local_2d,
                image_path=gt_floor_pg.nodes[i].image_path,
                label=gt_floor_pg.nodes[i].label,
                doors=doors,
                windows=windows,
                openings=openings,
            )

        return cls(building_id=building_id, floor_id=floor_id, nodes=nodes)

    def as_json(self, json_fpath: str) -> None:
        """ """
        pass

    def measure_aligned_abs_pose_error(self, gt_floor_pg: "PoseGraph2d") -> Tuple[float, float]:
        """ """
        aTi_list_gt = gt_floor_pg.as_3d_pose_graph()  # reference
        bTi_list_est = self.as_3d_pose_graph()

        mean_rot_err, mean_trans_err = compute_pose_errors(aTi_list_gt, bTi_list_est)
        return mean_rot_err, mean_trans_err

    def measure_unaligned_abs_pose_error(self, gt_floor_pg: "PoseGraph2d") -> Tuple[float, float]:
        """Measure the absolute pose errors (in both rotations and translations) for each localized pano.

        Args:
            gt_floor_pg

        Returns:
            mean_rot_err
            mean_trans_err
        """
        aTi_list_gt = gt_floor_pg.as_3d_pose_graph()  # reference
        bTi_list_est = self.as_3d_pose_graph()

        # if the estimate pose graph is missing a few nodes, pad it up to the GT list length
        pad_len = len(aTi_list_gt) - len(bTi_list_est)
        bTi_list_est.extend([None] * pad_len)

        # align the pose graphs
        aligned_bTi_list_est = align_poses_sim3_ignore_missing(aTi_list_gt, bTi_list_est)

        mean_rot_err, mean_trans_err = compute_pose_errors(aTi_list_gt, aligned_bTi_list_est)
        return mean_rot_err, mean_trans_err

    def measure_avg_abs_rotation_err(self, gt_floor_pg: "PoseGraph2d") -> float:
        """Measure how the absolute poses satisfy the individual binary measurement constraints.

        If `self` is the estimate, then we measure our error w.r.t. GT argument.

        Args:
            gt_floor_pg: ground truth pose graph for a single floor

        Returns:
            mean_relative_rot_err: average error on each relative rotation.
        """
        raise NotImplementedError("Add alignment")
        # TODO: have to do a pose-graph alignment first, or Karcher-mean alignment first.

        errs = []
        for pano_id, est_pano_obj in self.nodes.items():

            theta_deg_est = est_pano_obj.global_Sim2_local.theta_deg
            theta_deg_gt = gt_floor_pg.nodes[pano_id].global_Sim2_local.theta_deg

            print(f"\tPano {pano_id}: GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}")

            # need to wrap around at 360
            err = wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print(
            f"Mean absolute rot. error: {mean_err:.1f}. Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos."
        )
        return mean_err

    def measure_avg_rel_rotation_err(
        self, gt_floor_pg: "PoseGraph2d", gt_edges: List[Tuple[int, int]], verbose: bool
    ) -> float:
        """

        Args:
            gt_edges: list of (i1,i2) pairs representing panorama pairs where a WDO is found closeby between the two
        """
        errs = []
        for (i1, i2) in gt_edges:

            if not (i1 in self.nodes and i2 in self.nodes):
                continue

            wTi1_gt = gt_floor_pg.nodes[i1].global_Sim2_local
            wTi2_gt = gt_floor_pg.nodes[i2].global_Sim2_local
            i2Ti1_gt = wTi2_gt.inverse().compose(wTi1_gt)

            wTi1 = self.nodes[i1].global_Sim2_local
            wTi2 = self.nodes[i2].global_Sim2_local
            i2Ti1 = wTi2.inverse().compose(wTi1)

            theta_deg_est = i2Ti1.theta_deg
            theta_deg_gt = i2Ti1_gt.theta_deg

            if verbose:
                print(f"\tPano pair ({i1},{i2}): GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}")

            # need to wrap around at 360
            err = wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print_str = f"Mean relative rot. error: {mean_err:.1f}. Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos"
        print_str += f", estimated {len(errs)} / {len(gt_edges)} GT edges"
        print(REDTEXT + print_str + ENDCOLOR)

        return mean_err

    def render_estimated_layout(
        self,
        show_plot: bool = True,
        save_plot: bool = False,
        plot_save_dir: str = "floorplan_renderings",
        gt_floor_pg: "PoseGraph2d" = None,
        plot_save_fpath: Optional[str] = None,
    ) -> None:
        """
        Either render (show plot) or save plot to disk.
        """
        if gt_floor_pg is not None:
            plt.suptitle("left: GT floorplan. Right: estimated floorplan.")
            plt.subplot(1, 2, 1)
            gt_floor_pg.render_estimated_layout(show_plot=False, save_plot=False, plot_save_dir=None, gt_floor_pg=None)
            plt.axis("equal")
            plt.subplot(1, 2, 2)

        for i, pano_obj in self.nodes.items():
            pano_obj.plot_room_layout(coord_frame="global", show_plot=False)

        plt.title(f"Building {self.building_id}, {self.floor_id}")
        plt.axis("equal")
        if save_plot:
            if plot_save_dir is not None and plot_save_fpath is None:
                os.makedirs(plot_save_dir, exist_ok=True)
                save_fpath = f"{plot_save_dir}/{self.building_id}_{self.floor_id}.jpg"
            elif plot_save_dir is None and plot_save_fpath is not None:
                save_fpath = plot_save_fpath
            plt.savefig(save_fpath, dpi=500)
            plt.close("all")

        if show_plot:
            plt.axis("equal")
            plt.show()

    def draw_edge(self, i1: int, i2: int, color: str) -> None:
        """ """
        t1 = self.nodes[i1].global_Sim2_local.transform_from(np.zeros((1, 2)))
        t2 = self.nodes[i2].global_Sim2_local.transform_from(np.zeros((1, 2)))

        t1 = t1.squeeze()
        t2 = t2.squeeze()

        plt.plot([t1[0], t2[0]], [t1[1], t2[1]], c=color, linestyle="dotted", alpha=0.6)


def test_measure_abs_pose_error_shifted() -> None:
    """Pose graph is shifted to the left by 1 meter, but Sim(3) alignment should fix this. Should have zero error.

    TODO: fix rotations to be +90

    GT pose graph:

       | pano 1 = (0,4)
     --o
       | .
       .   .
       .     .
       |       |
       o-- ... o--
    pano 0          pano 2 = (4,0)
      (0,0)

    Estimated PG:
       | pano 1 = (-1,4)
     --o
       | .
       .   .
       .     .
       |       |
       o-- ... o--
    pano 0          pano 2 = (3,0)
      (-1,0)
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotmat2d(0), rotmat2d(-90), rotmat2d(0)]
    wti_list = [np.array([-1, 0]), np.array([-1, 4]), np.array([3, 0])]

    est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, building_id, floor_id)

    wRi_list_gt = [rotmat2d(0), rotmat2d(-90), rotmat2d(0)]
    wti_list_gt = [np.array([0, 0]), np.array([0, 4]), np.array([4, 0])]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list_gt, wti_list, building_id, floor_id)

    avg_rot_error, avg_trans_error = est_floor_pose_graph.measure_abs_pose_error(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(avg_rot_error, 0.0, atol=1e-3)
    assert np.isclose(avg_trans_error, 0.0, atol=1e-3)


def test_measure_avg_abs_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure absolute rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotmat2d(-5), rotmat2d(-95), rotmat2d(0)]

    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotmat2d(0), rotmat2d(-90), rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(mean_abs_rot_err, 10 / 3, atol=1e-3)


def test_measure_avg_rel_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    wRi_list = [rotmat2d(-5), rotmat2d(-95), rotmat2d(0)]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotmat2d(0), rotmat2d(-90), rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0, 1)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    # both are incorrect by the same amount, cancelling out to zero error
    assert mean_rel_rot_err == 0

    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    assert np.isclose(mean_rel_rot_err, 10 / 3, atol=1e-3)


def test_measure_avg_rel_rotation_err_unestimated() -> None:
    """Estimate average relative pose (rotation) error when some nodes are unestimated.

    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

    TODO: fix rotations to be +90

    GT rotation graph:

      | 1
    --o
      | .
      .   .
      .     .
      |       |
      o-- ... o--
    0          2
    """
    building_id = "000"
    floor_id = "floor_01"

    # only 1 edge can be measured for correctness
    wRi_list = [rotmat2d(-5), rotmat2d(-90), None]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [rotmat2d(0), rotmat2d(-90), rotmat2d(0)]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0, 1), (1, 2), (0, 2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(
        gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges
    )
    assert mean_rel_rot_err == 5.0


def get_single_building_pose_graphs(building_id: str, pano_dir: str, json_annot_fpath: str) -> Dict[str, PoseGraph2d]:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'

    Returns:
        floor_pg_dict: mapping from floor_id to pose graph
    """
    floor_map_json = read_json_file(json_annot_fpath)

    scale_meters_per_coordinate_dict = floor_map_json["scale_meters_per_coordinate"]

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    floor_pg_dict = {}

    merger_data = floor_map_json["merger"]
    for floor_id, floor_data in merger_data.items():

        fd = FloorData.from_json(floor_data, floor_id)
        pg = PoseGraph2d.from_floor_data(
            building_id=building_id, fd=fd, scale_meters_per_coordinate=scale_meters_per_coordinate_dict[floor_id]
        )

        floor_pg_dict[floor_id] = pg

    return floor_pg_dict


def export_dataset_pose_graphs(raw_dataset_dir: str) -> None:
    """ """

    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        if building_id != "000":  # '1442':
            continue

        print(f"Render floor maps for {building_id}")
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        floor_pg_dict = get_single_building_pose_graphs(
            building_id=building_id, pano_dir=pano_dir, json_annot_fpath=json_annot_fpath
        )


def get_gt_pose_graph(building_id: int, floor_id: str, raw_dataset_dir: str) -> PoseGraph2d:
    """ """
    pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
    json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
    floor_pg_dict = get_single_building_pose_graphs(building_id, pano_dir, json_annot_fpath)
    return floor_pg_dict[floor_id]


def rot2x2_to_Rot3(R: np.ndarray) -> Rot3:
    """
    2x2 rotation matrix to Rot3 object
    """
    R_Rot3 = np.eye(3)
    R_Rot3[:2, :2] = R
    return Rot3(R_Rot3)


def compute_pose_errors(aTi_list_gt: List[Pose3], aligned_bTi_list_est: List[Optional[Pose3]]) -> Tuple[float, float]:
    """

    Args:
        aTi_list_gt
        aligned_bTi_list_est:

    Returns:
        mean_rot_err
        mean_trans_err
    """
    rotation_errors = []
    translation_errors = []
    for (aTi, aTi_) in zip(aTi_list_gt, aligned_bTi_list_est):
        if aTi is None or aTi_ is None:
            continue
        rot_err = geometry_comparisons.compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation())
        trans_err = np.linalg.norm(aTi.translation() - aTi_.translation())

        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

    mean_rot_err = np.mean(rotation_errors)
    mean_trans_err = np.mean(translation_errors)
    print(f"Mean translation error: {mean_trans_err:.1f}, Mean rotation error: {mean_rot_err:.1f}")
    return mean_rot_err, mean_trans_err


if __name__ == "__main__":
    """ """
    raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # export_dataset_pose_graphs(raw_dataset_dir)

    # test_measure_avg_rel_rotation_err()
    # test_measure_avg_abs_rotation_err()
    # test_measure_avg_rel_rotation_err_unestimated()
    # test_measure_abs_pose_error()

    test_measure_abs_pose_error_shifted()
