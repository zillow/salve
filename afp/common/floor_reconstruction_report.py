"""
Stores information about a floorplan reconstruction.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from argoverse.utils.sim2 import Sim2

from afp.common.posegraph2d import PoseGraph2d


@dataclass(frozen=True)
class FloorReconstructionReport:
    """Summary statistics about the reconstructed floorplan."""

    avg_abs_rot_err: float
    avg_abs_trans_err: float
    percent_panos_localized: float

    @classmethod
    def from_wSi_list(
        cls, wSi_list: List[Optional[Sim2]], gt_floor_pose_graph: PoseGraph2d, plot_save_dir: str
    ) -> "FloorReconstructionReport":
        """ """
        num_localized_panos = np.array([wSi is not None for wSi in wSi_list]).sum()
        num_floor_panos = len(gt_floor_pose_graph.nodes)
        percent_panos_localized = num_localized_panos / num_floor_panos * 100
        print(f"Localized {percent_panos_localized:.2f}% of panos: {num_localized_panos} / {num_floor_panos}")

        # TODO: try spanning tree version, vs. Shonan version
        wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
        wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

        est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pose_graph)

        mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_unaligned_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )
        print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")
        est_floor_pose_graph.render_estimated_layout(
            show_plot=False, save_plot=True, plot_save_dir=plot_save_dir, gt_floor_pg=gt_floor_pose_graph
        )

        print()
        print()

        return cls(
            avg_abs_rot_err=mean_abs_rot_err,
            avg_abs_trans_err=mean_abs_trans_err,
            percent_panos_localized=percent_panos_localized,
        )
