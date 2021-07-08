
import glob
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

from gtsam import Rot3

from pano_data import FloorData, PanoData, generate_Sim2_from_floorplan_transform


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
    def from_floor_data(cls, building_id: int, fd: FloorData) -> "PoseGraph2d":
        """ """
        return cls(building_id=building_id, floor_id=fd.floor_id, nodes={p.id: p for p in fd.panos})

    @classmethod
    def from_json(cls, json_fpath: str) -> "PoseGraph2d":
        """ """
        pass

    @classmethod
    def from_wRi_list(cls, wRi_list: List[Rot3], building_id: int, floor_id: str) -> "PoseGraph2d":
        """ """
        import pdb; pdb.set_trace()

        for i, wRi in enumerate(wRi_list):
            if wRi is None:
                continue

            nodes[i] = PanoData(
                id=i,
                global_SIM2_local = Sim2(R=wRi, t=np.zeros(3), s=1.0),
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


    def measure_avg_abs_rotation_err(gt_floor_pg: "PoseGraph2d") -> float:
        """Measure how the absolute poses satisfy the individual binary measurement constraints.

        If `self` is the estimate, then we measure our error w.r.t. GT argument.

        Args:
            gt_floor_pg: ground truth pose graph for a single floor

        Returns:
            mean_relative_rot_err: average error on each relative rotation.
        """
        errs = []
        for pano_id, est_pano_obj in self.nodes.items():

            theta_deg_est = est_pano_obj.global_Sim2_local.theta_deg
            theta_deg_gt = gt_floor_pg.nodes[pano_id].global_Sim2_local.theta_deg

            print(f"Pano {pano_id}: GT {theta_deg_gt:.2f} vs. {theta_deg_est:.2f}")

            # need to wrap around at 360
            err = wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print(f"Mean abs rot. error: {mean_err:.2f}. Estimated rotation for {len(self.nodes)} of {len(self.gt_floor_pg.nodes)} GT panos.")
        return mean_err


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
    export_dataset_pose_graphs(raw_dataset_dir)





