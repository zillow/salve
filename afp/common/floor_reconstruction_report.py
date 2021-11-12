"""
Stores information about a floorplan reconstruction.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.sim2 import Sim2

import afp.utils.bev_rendering_utils as bev_rendering_utils
import afp.utils.iou_utils as iou_utils
from afp.common.bevparams import BEVParams
from afp.common.posegraph2d import PoseGraph2d


@dataclass(frozen=True)
class FloorReconstructionReport:
    """Summary statistics about the quality of the reconstructed floorplan."""

    avg_abs_rot_err: float
    avg_abs_trans_err: float
    percent_panos_localized: float
    floorplan_iou: Optional[float] = np.nan
    rotation_errors: Optional[np.ndarray] = None
    translation_errors: Optional[np.ndarray] = None


    def __repr__(self) -> str:
        """Concise summary of the class as a string."""
        summary_str = f"Abs. Rot err (deg) {avg_abs_rot_err:.1f}, "
        summary_str += f"Abs. trans err {avg_abs_trans_err:.2f}, "
        summary_str += f"%Localized {percent_panos_localized:.2f},"
        summary_str += f"Floorplan IoU {floorplan_iou:.2f}"
        return summary_str

    @classmethod
    def from_wSi_list(
        cls,
        wSi_list: List[Optional[Sim2]],
        gt_floor_pose_graph: PoseGraph2d,
        plot_save_dir: str,
        plot_save_fpath: str = None,
    ) -> "FloorReconstructionReport":
        """ """

        # TODO: try spanning tree version, vs. Shonan version
        wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
        wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

        est_floor_pose_graph = PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pose_graph)

        return FloorReconstructionReport.from_est_floor_pose_graph(
            est_floor_pose_graph=est_floor_pose_graph,
            gt_floor_pose_graph=gt_floor_pose_graph,
            plot_save_dir=plot_save_dir,
            plot_save_fpath=plot_save_fpath,
        )

    @classmethod
    def from_est_floor_pose_graph(
        cls,
        est_floor_pose_graph: PoseGraph2d,
        gt_floor_pose_graph: PoseGraph2d,
        plot_save_dir: str,
        plot_save_fpath: str,
    ) -> "FloorReconstructionReport":
        """Create a report from an estimated pose graph for a single floor."""
        num_localized_panos = len(est_floor_pose_graph.nodes)
        num_floor_panos = len(gt_floor_pose_graph.nodes)
        percent_panos_localized = num_localized_panos / num_floor_panos * 100
        print(f"Localized {percent_panos_localized:.2f}% of panos: {num_localized_panos} / {num_floor_panos}")

        #import pdb; pdb.set_trace()
        # aligned_est_floor_pose_graph = est_floor_pose_graph
        aligned_est_floor_pose_graph, _ = est_floor_pose_graph.align_by_Sim3_to_ref_pose_graph(
            ref_pose_graph=gt_floor_pose_graph
        )
        mean_abs_rot_err, mean_abs_trans_err, rot_errors, trans_errors = aligned_est_floor_pose_graph.measure_aligned_abs_pose_error(
            gt_floor_pg=gt_floor_pose_graph
        )

        # convert units to meters.
        worldmetric_s_worldnormalized = gt_floor_pose_graph.scale_meters_per_coordinate
        mean_abs_trans_err_m = worldmetric_s_worldnormalized * mean_abs_trans_err
        print(f"Mean rotation error: {mean_abs_rot_err:.2f}, Mean translation error: {mean_abs_trans_err_m:.2f}")

        render_floorplans_side_by_side(
            est_floor_pose_graph=aligned_est_floor_pose_graph,
            show_plot=False,
            save_plot=True,
            plot_save_dir=plot_save_dir,
            gt_floor_pg=gt_floor_pose_graph,
            plot_save_fpath=plot_save_fpath,
        )

        floorplan_iou = render_raster_occupancy(
            est_floor_pose_graph=aligned_est_floor_pose_graph,
            gt_floor_pg=gt_floor_pose_graph,
            plot_save_dir=plot_save_dir,
        )

        print()
        print()

        return cls(
            avg_abs_rot_err=mean_abs_rot_err,
            avg_abs_trans_err=mean_abs_trans_err_m,
            percent_panos_localized=percent_panos_localized,
            floorplan_iou=floorplan_iou,
            rotation_errors=rot_errors,
            translation_errors=trans_errors
        )


def render_floorplans_side_by_side(
    est_floor_pose_graph: PoseGraph2d,
    show_plot: bool = True,
    save_plot: bool = False,
    plot_save_dir: str = "floorplan_renderings",
    gt_floor_pg: Optional[PoseGraph2d] = None,
    plot_save_fpath: Optional[str] = None,
) -> None:
    """
    Either render (show plot) or save plot to disk.

    Args:
        est_floor_pose_graph: 
        show_plot: boolean indicating whether to show via GUI a rendering of the plot.
        save_plot: boolean indicating whether to save a rendering of the plot.
        plot_save_dir: if only a general saving directory is provided for saving
        gt_floor_pg: ground truth pose graph
        plot_save_fpath: if a specifically desired file path is provided.
    """
    # import pdb; pdb.set_trace()
    building_id = est_floor_pose_graph.building_id
    floor_id = est_floor_pose_graph.floor_id

    if gt_floor_pg is not None:
        plt.suptitle("left: GT floorplan. Right: estimated floorplan.")
        ax1 = plt.subplot(1, 2, 1)
        render_floorplan(gt_floor_pg, gt_floor_pg.scale_meters_per_coordinate)
        ax1.set_aspect("equal")

    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.set_aspect("equal")
    render_floorplan(est_floor_pose_graph, gt_floor_pg.scale_meters_per_coordinate)
    plt.title(f"Building {building_id}, {floor_id}")

    if save_plot:
        if plot_save_dir is not None and plot_save_fpath is None:
            os.makedirs(plot_save_dir, exist_ok=True)
            save_fpath = f"{plot_save_dir}/{building_id}_{floor_id}.jpg"
        elif plot_save_dir is None and plot_save_fpath is not None:
            save_fpath = plot_save_fpath
        plt.savefig(save_fpath, dpi=500)
        plt.close("all")

    if show_plot:
        # plt.axis("equal")
        plt.show()


def render_raster_occupancy(
    est_floor_pose_graph: PoseGraph2d, gt_floor_pg: PoseGraph2d, plot_save_dir: str, save_viz: bool = True
) -> None:
    """Compute raster IoU on occupancy."""
    # render side by side figures

    scale_meters_per_coordinate = gt_floor_pg.scale_meters_per_coordinate

    # not going to be larger than [-40,40] meters
    BUILDING_XLIMS_M = 25
    BUILDING_YLIMS_M = 25

    IOU_EVAL_METERS_PER_PX = 0.1 
    #IOU_EVAL_METERS_PER_PX = 0.01 #(no appreciable difference in IoU with this resolution)
    IOU_EVAL_PX_PER_METER = 1 / IOU_EVAL_METERS_PER_PX

    img_w = int(IOU_EVAL_PX_PER_METER * BUILDING_XLIMS_M * 2)
    img_h = int(IOU_EVAL_PX_PER_METER * BUILDING_YLIMS_M * 2)

    bev_params = BEVParams(img_h=img_h, img_w=img_w, meters_per_px=IOU_EVAL_METERS_PER_PX)

    est_mask = rasterize_room(bev_params, est_floor_pose_graph, scale_meters_per_coordinate)
    gt_mask = rasterize_room(bev_params, gt_floor_pg, scale_meters_per_coordinate)

    iou = iou_utils.binary_mask_iou(mask1=est_mask, mask2=gt_mask)
    print(f"IoU: {iou:.2f}")

    if save_viz:
        plt.subplot(1, 2, 1)
        plt.imshow(np.flipud(est_mask))
        plt.subplot(1, 2, 2)
        plt.imshow(np.flipud(gt_mask))
        plt.suptitle(f"{gt_floor_pg.building_id} {gt_floor_pg.floor_id} --> IoU {iou:.2f}")
        # plt.show()

        save_dir = f"{plot_save_dir}__floorplan_iou"
        os.makedirs(save_dir, exist_ok=True)
        save_fpath = f"{save_dir}/{gt_floor_pg.building_id}_{gt_floor_pg.floor_id}.jpg"
        plt.savefig(save_fpath, dpi=500)
        plt.close("all")

    return iou


def rasterize_room(
    bev_params: BEVParams, floor_pose_graph: PoseGraph2d, scale_meters_per_coordinate: float
) -> np.ndarray:
    """
    Args:
        bev_params
        floor_pose_graph
        scale_meters_per_coordinate

    Returns:
        occ_img: occupancy mask.
    """
    bev_img = np.zeros((bev_params.img_h + 1, bev_params.img_w + 1, 3))

    for i, pano_obj in floor_pose_graph.nodes.items():
        # convert to meters
        room_vertices_m = pano_obj.room_vertices_global_2d * scale_meters_per_coordinate

        bev_img = bev_rendering_utils.rasterize_polygon(
            polygon_xy=room_vertices_m, bev_img=bev_img, bevimg_Sim2_world=bev_params.bevimg_Sim2_world, color=[1, 1, 1]
        )

    occ_img = bev_img[:, :, 0]
    return occ_img


def render_floorplan(pose_graph: PoseGraph2d, scale_meters_per_coordinate: float) -> None:
    """Given global poses, render the floorplan by rendering each room layout in the global coordinate frame.

    Args:
        pose_graph: 2d pose graph, either estimated or ground truth.
    """
    for i, pano_obj in pose_graph.nodes.items():
        pano_obj.plot_room_layout(
            coord_frame="worldmetric", show_plot=False, scale_meters_per_coordinate=scale_meters_per_coordinate
        )


def summarize_reports(reconstruction_reports: List[FloorReconstructionReport]) -> None:
    """Given a report per floor, compute summary statistics for each error metric.

    Args:
        reconstruction_reports: report for every floor of each ZinD building in this split.
    """
    print()
    print()
    print(f"Test set contained {len(reconstruction_reports)} total floors.")
    if len(reconstruction_reports) == 0:
        print("Cannot compute error metrics, tested over zero homes.")
        return

    error_metrics = [
        "avg_abs_rot_err",
        "avg_abs_trans_err",
        "percent_panos_localized",
        "floorplan_iou"
    ]
    for error_metric in error_metrics:
        avg_val = np.nanmean([getattr(r, error_metric) for r in reconstruction_reports])
        print(f"Averaged over all tours, {error_metric} = {avg_val:.3f}")

        median_val = np.nanmedian([getattr(r, error_metric) for r in reconstruction_reports])
        print(f"Median over all tours, {error_metric} = {median_val:.3f}")

    # thresholded_trans_error_dict = {}
    # thresholded_trans_error_dict[0.2] = compute_translation_errors_against_threshold(reconstruction_reports, threshold=0.2)
    # thresholded_trans_error_dict[0.6] = compute_translation_errors_against_threshold(reconstruction_reports, threshold=0.6)
    # thresholded_trans_error_dict[1.0] = compute_translation_errors_against_threshold(reconstruction_reports, threshold=1.0)
    
    # print("Average position localization success rates: ", thresholded_trans_error_dict)
    print("======> Evaluation complete. ======>")


def compute_translation_errors_against_threshold(reconstruction_reports: List[FloorReconstructionReport], threshold: float) -> float:
    """Compute a success rate against a particular translation error threshold.

    See Shabani et al, ICCV 2021.

    Args:
        reconstruction_reports:
        threshold: maximum allowed translation error for each localized camera.

    Returns:
        avg_floor_success_rate: percent of cameras for which the translation error is below the specified threshold.
    """
    floor_success_rates = []
    for r in reconstruction_reports:
        if r.translation_errors is None:
            # no cameras were localized for this floor.
            print("No cameras for this FloorReconstructionReport.")
            continue
        floor_success_rate = (r.translation_errors < threshold).mean()
        floor_success_rates.append(floor_success_rate)

    avg_floor_success_rate = np.mean(floor_success_rates)
    print(f"Avg. Position Localization Floor Success Rate @ {threshold} = {avg_floor_success_rate:.3f}")
    return avg_floor_success_rate


def test_compute_translation_errors_against_threshold() -> None:
    """Ensure that translation localization success rate is computed correctly."""
    reconstruction_reports = [
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([0.0, 0.1, 0.19, 0.3, 0.4, 900]) # 3/6 are under threshold.
        ),
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([0.0, 0.1, 0.18, 0.19, 0.21]) # 4/5 are under threshold
        ),
        FloorReconstructionReport(
            avg_abs_rot_err=np.nan,
            avg_abs_trans_err=np.nan,
            percent_panos_localized=np.nan,
            floorplan_iou=np.nan,
            rotation_errors=None,
            translation_errors=np.array([800, 900, 1000]) # 0/3 are under threshold.
        )
    ]
    threshold = 0.2
    avg_success_rate = compute_translation_errors_against_threshold(reconstruction_reports, threshold)
    expected_avg_success_rate = np.mean([3/6, 4/5, 0/3])
    assert np.isclose(avg_success_rate, expected_avg_success_rate)
