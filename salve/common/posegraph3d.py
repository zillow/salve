"""Utility for creating 3d pose graphs."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from gtsam import Pose3

from salve.common.posegraph2d import PoseGraph2d


@dataclass
class PoseGraph3d:
    """Represents a 3d pose graph for a specific floor of a ZInD building.

    Attributes:
        building_id: unique ID of ZInD building containing panos to which the poses correspond to.
        floor_id: unique ID of floor of aforementioned ZInD building.
        pose_dict: mapping from panoram ID to Pose(3) object.
    """

    building_id: str
    floor_id: str
    pose_dict: Dict[int, Pose3]

    def project_to_2d(self, gt_floor_pose_graph: PoseGraph2d) -> PoseGraph2d:
        """Project 3d pose graph object to 2d.

        Args:
            gt_floor_pose_graph: ground truth 2d pose graph for a specific ZInD floor.
                Its poses are ignored, and we use it only to scrape other pano metadata
                (image paths, building id, floor id, room vertices).

        Returns:
            floor_pose_graph_2d: 2d representation of 3d pose graph object instance.
        """
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

        floor_pose_graph_2d = PoseGraph2d.from_wRi_wti_lists(
            wRi_list=wRi_list, wti_list=wti_list, gt_floor_pg=gt_floor_pose_graph
        )
        return floor_pose_graph_2d

    @classmethod
    def from_wTi_list(cls, wTi_list: List[Optional[Pose3]], building_id: str, floor_id: str) -> "PoseGraph3d":
        """Construct a 3d pose graph object from a list of Pose(3) objects.

        Args:
            wTi_list: zero-indexed list of Pose(3) objects.
            building_id: unique ID of ZInD building containing panos to which the poses correspond to.
            floor_id: unique ID of floor of aforementioned ZInD building.
        """
        pose_dict = {}
        for i, wTi in enumerate(wTi_list):
            if wTi is None:
                continue
            pose_dict[i] = wTi

        return cls(building_id, floor_id, pose_dict)
