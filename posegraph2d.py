
import glob
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

from gtsam import Rot3

from pano_data import FloorData, PanoData, generate_Sim2_from_floorplan_transform
from vis_depth import rotmat2d


REDTEXT = '\033[91m'
ENDCOLOR = '\033[0m'

class PoseGraph2d(NamedTuple):
    """Pose graph for a single floor.

    Note: edges are not included here, since there are different types of adjacency (spatial vs. vsible)

    Args:
        building_id
        floor_id
        nodes:
    """
    building_id: int
    floor_id: str
    nodes: Dict[int, PanoData]

    def __repr__(self) -> str:
        """ """
        return f"Graph has {len(self.nodes.keys())} nodes in Building {self.building_id}, {self.floor_id}: {self.nodes.keys()}"

    @classmethod
    def from_floor_data(cls, building_id: str, fd: FloorData) -> "PoseGraph2d":
        """ """
        return cls(building_id=building_id, floor_id=fd.floor_id, nodes={p.id: p for p in fd.panos})

    @classmethod
    def from_json(cls, json_fpath: str) -> "PoseGraph2d":
        """ """
        pass

    @classmethod
    def from_wRi_list(cls, wRi_list: List[Rot3], building_id: str, floor_id: str) -> "PoseGraph2d":
        """

        Fill other pano metadata with dummy values. Alternatively, could populate them from the GT pose graph.
        """
        nodes = {}
        for i, wRi in enumerate(wRi_list):
            if wRi is None:
                continue

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local = Sim2(R=wRi, t=np.zeros(2), s=1.0),
                room_vertices_local_2d = np.zeros((0,2)),
                image_path="",
                label="",
                doors = None,
                windows = None,
                openings = None
            )

        return cls(building_id=building_id, floor_id=floor_id, nodes=nodes)


    def as_json(self, json_fpath: str) -> None:
        """ """
        pass


    def measure_avg_abs_rotation_err(self, gt_floor_pg: "PoseGraph2d") -> float:
        """Measure how the absolute poses satisfy the individual binary measurement constraints.

        If `self` is the estimate, then we measure our error w.r.t. GT argument.

        Args:
            gt_floor_pg: ground truth pose graph for a single floor

        Returns:
            mean_relative_rot_err: average error on each relative rotation.
        """
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
        print(f"Mean absolute rot. error: {mean_err:.1f}. Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos.")
        return mean_err


    def measure_avg_rel_rotation_err(self, gt_floor_pg: "PoseGraph2d", gt_edges=List[Tuple[int,int]]) -> float:
        """

        Args:
            gt_edges: list of (i1,i2) pairs representing panorama pairs where a WDO is found closeby between the two
        """
        errs = []
        for (i1,i2) in gt_edges:

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

            print(f"\tPano pair ({i1},{i2}): GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}")

            # need to wrap around at 360
            err = wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print_str = f"Mean relative rot. error: {mean_err:.1f}. Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos"
        print_str += f", estimated {len(errs)} / {len(gt_edges)} GT edges"
        print(REDTEXT + print_str + ENDCOLOR)
        
        return mean_err


def test_measure_avg_abs_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure absolute rotation errors are evaluated properly.

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

    wRi_list = [
        rotmat2d(-5),
        rotmat2d(-95),
        rotmat2d(0)
    ]
    
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [
        rotmat2d(0),
        rotmat2d(-90),
        rotmat2d(0)
    ]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    mean_abs_rot_err = est_floor_pose_graph.measure_avg_abs_rotation_err(gt_floor_pg=gt_floor_pose_graph)

    assert np.isclose(mean_abs_rot_err, 10/3, atol=1e-3)

    # TODO: compute if some pano poses are unestimated (i.e. None)


def test_measure_avg_rel_rotation_err() -> None:
    """
    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

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

    wRi_list = [
        rotmat2d(-5),
        rotmat2d(-95),
        rotmat2d(0)
    ]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [
        rotmat2d(0),
        rotmat2d(-90),
        rotmat2d(0)
    ]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0,1)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges)
    # both are incorrect by the same amount, cancelling out to zero error
    assert mean_rel_rot_err == 0

    gt_edges = [(0,1),(1,2),(0,2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges)
    assert np.isclose(mean_rel_rot_err, 10/3, atol=1e-3)
    


def test_measure_avg_rel_rotation_err_unestimated() -> None:
    """Estimate average relative pose (rotation) error when some nodes are unestimated.

    Create a dummy scenario, to make sure relative rotation errors are evaluated properly.

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
    wRi_list = [
        rotmat2d(-5),
        rotmat2d(-90),
        None
    ]
    est_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list, building_id, floor_id)

    wRi_list_gt = [
        rotmat2d(0),
        rotmat2d(-90),
        rotmat2d(0)
    ]
    gt_floor_pose_graph = PoseGraph2d.from_wRi_list(wRi_list_gt, building_id, floor_id)

    gt_edges = [(0,1),(1,2),(0,2)]
    mean_rel_rot_err = est_floor_pose_graph.measure_avg_rel_rotation_err(gt_floor_pg=gt_floor_pose_graph, gt_edges=gt_edges)
    assert mean_rel_rot_err == 5.0


def wrap_angle_deg(angle1: float, angle2: float):
    """
    https://stackoverflow.com/questions/28036652/finding-the-shortest-distance-between-two-angles/28037434
    """
    # mod n will wrap x to [0,n)
    diff = (angle2 - angle1 + 180) % 360 - 180
    if diff < -180:
        return np.absolute(diff + 360)
    else:
        return np.absolute(diff)


def test_wrap_angle_deg() -> None:
    """ """
    angle1 = 180
    angle2 = -180
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -180
    angle2 = 180
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 0

    angle1 = -45
    angle2 = -47
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 1
    angle2 = -1
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 2

    angle1 = 10
    angle2 = 11.5
    orientation_err = wrap_angle_deg(angle1, angle2)
    assert orientation_err == 1.5


def get_single_building_pose_graphs(building_id: str, pano_dir: str, json_annot_fpath: str) -> Dict[str, PoseGraph2d]:
    """
    floor_map_json has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'

    Returns:
        floor_pg_dict: mapping from floor_id to pose graph
    """
    floor_map_json = read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    floor_pg_dict = {}

    merger_data = floor_map_json["merger"]
    for floor_id, floor_data in merger_data.items():

        fd = FloorData.from_json(floor_data, floor_id)
        pg = PoseGraph2d.from_floor_data(building_id=building_id, fd=fd)

        floor_pg_dict[floor_id] = pg

    return floor_pg_dict


def export_dataset_pose_graphs(raw_dataset_dir: str) -> None:
    """ """

    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    for building_id in building_ids:

        if building_id != '000': # '1442':
            continue

        print(f"Render floor maps for {building_id}")
        pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
        floor_pg_dict = get_single_building_pose_graphs(building_id=building_id, pano_dir=pano_dir, json_annot_fpath=json_annot_fpath)


def get_gt_pose_graph(building_id: int, floor_id: str, raw_dataset_dir: str) -> PoseGraph2d:
    """ """
    pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
    json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zfm_data.json"
    floor_pg_dict = get_single_building_pose_graphs(building_id, pano_dir, json_annot_fpath)
    return floor_pg_dict[floor_id]


if __name__ == "__main__":
    """ """
    raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    #export_dataset_pose_graphs(raw_dataset_dir)

    #test_measure_avg_rel_rotation_err()
    #test_measure_avg_abs_rotation_err()
    test_measure_avg_rel_rotation_err_unestimated()





