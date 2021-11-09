"""
Stores information about a floorplan reconstruction.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
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

        aligned_est_floor_pose_graph, _ = est_floor_pose_graph.align_by_Sim3_to_ref_pose_graph(ref_pose_graph=gt_floor_pose_graph)
        mean_abs_rot_err, mean_abs_trans_err = aligned_est_floor_pose_graph.measure_aligned_abs_pose_error(

        # mean_abs_rot_err, mean_abs_trans_err = est_floor_pose_graph.measure_unaligned_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )
        print(f"\tAvg translation error: {mean_abs_trans_err:.2f}")

        render_floorplans_side_by_side(
            est_floor_pose_graph=aligned_est_floor_pose_graph,
            show_plot=False,
            save_plot=True,
            plot_save_dir=plot_save_dir,
            gt_floor_pg=gt_floor_pose_graph
        )

        print()
        print()

        return cls(
            avg_abs_rot_err=mean_abs_rot_err,
            avg_abs_trans_err=mean_abs_trans_err,
            percent_panos_localized=percent_panos_localized,
        )


def render_floorplans_side_by_side(
    est_floor_pose_graph,
    show_plot: bool = True,
    save_plot: bool = False,
    plot_save_dir: str = "floorplan_renderings",
    gt_floor_pg: Optional[PoseGraph2d] = None,
    plot_save_fpath: Optional[str] = None,
) -> None:
    """
    Either render (show plot) or save plot to disk.

    Args:
        show_plot: boolean indicating whether to show via GUI a rendering of the plot.
        save_plot: boolean indicating whether to save a rendering of the plot.
        plot_save_dir: if only a general saving directory is provided for saving
        gt_floor_pg: ground truth pose graph
        plot_save_fpath: if a specifically desired file path is provided.
    """
    #import pdb; pdb.set_trace()
    building_id = est_floor_pose_graph.building_id
    floor_id = est_floor_pose_graph.floor_id

    if gt_floor_pg is not None:
        plt.suptitle("left: GT floorplan. Right: estimated floorplan.")
        ax1 = plt.subplot(1, 2, 1)
        render_floorplan(gt_floor_pg)
        #plt.axis("equal")
        ax1.set_aspect('equal')
    
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.set_aspect('equal')
    render_floorplan(est_floor_pose_graph)
    plt.title(f"Building {building_id}, {floor_id}")
    #plt.axis("equal")

    if save_plot:
        if plot_save_dir is not None and plot_save_fpath is None:
            os.makedirs(plot_save_dir, exist_ok=True)
            save_fpath = f"{plot_save_dir}/{building_id}_{floor_id}.jpg"
        elif plot_save_dir is None and plot_save_fpath is not None:
            save_fpath = plot_save_fpath
        plt.savefig(save_fpath, dpi=500)
        plt.close("all")

    if show_plot:
        #plt.axis("equal")
        plt.show()



def render_floorplan(pose_graph: PoseGraph2d):
    """Given global poses, render the floorplan by rendering each room layout in the global coordinate frame.

    Args:
        pose_graph: 2d pose graph, either estimated or ground truth.
    """
    for i, pano_obj in pose_graph.nodes.items():
        pano_obj.plot_room_layout(coord_frame="global", show_plot=False)


