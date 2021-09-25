
"""
Utilityfor creating 3d pose graphs.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from gtsam import Pose3

from afp.common.posegraph2d import PoseGraph2d


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
        # import pdb; pdb.set_trace()

        n = len(gt_floor_pose_graph.as_3d_pose_graph())

        wRi_list = []
        wti_list = []
        for i in range(n):
            wTi = self.pose_dict.get(i, None)
            if wTi is not None:
                wRi = wTi.rotation().matrix()[:2, :2]
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
        # import pdb; pdb.set_trace()

        pose_dict = {}
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            pose_dict[i] = wTi

        return cls(building_id, floor_id, pose_dict)
