"""
"""

import glob
from pathlib import Path

import argoverse.utils.json_utils as json_utils

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
from salve.common.floor_reconstruction_report import FloorReconstructionReport
from salve.dataset.zind_partition import DATASET_SPLITS
from read_prod_predictions import load_inferred_floor_pose_graphs


def eval_oraclepose_predictedlayout() -> None:
    """Evaluate floorplans.

    Measure accuracy of oracle pose graph w/ PREDICTED layout vs. GT
    """
    raw_dataset_dir = "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"

    # discover possible building ids and floors
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{raw_dataset_dir}/*") if Path(fpath).is_dir()]
    building_ids.sort()

    reconstruction_reports = []

    viz_save_dir = f"2021_11_11_oraclepose_predicted_layout"

    for building_id in building_ids:

        # for rendering test data only
        if building_id not in DATASET_SPLITS["test"]:
            continue

        json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
        if not Path(json_annot_fpath).exists():
            print(f"zind_data.json file missing for {building_id}")
            continue

        floor_map_json = json_utils.read_json_file(json_annot_fpath)

        if "merger" not in floor_map_json:
            print(f"No merger data in {building_id}: {json_annot_fpath}")
            continue

        # load up the inferred pose graph
        floor_pose_graphs = load_inferred_floor_pose_graphs(
            building_id=building_id, raw_dataset_dir=raw_dataset_dir
        )
        if floor_pose_graphs is None:
            continue

        merger_data = floor_map_json["merger"]
        for floor_id in merger_data.keys():

            gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

            est_floor_pose_graph = floor_pose_graphs[floor_id]
            report = FloorReconstructionReport.from_est_floor_pose_graph(
                est_floor_pose_graph=est_floor_pose_graph,
                gt_floor_pose_graph=gt_floor_pose_graph,
                # plot_save_dir=None,
                # plot_save_fpath=plot_save_fpath
                plot_save_dir=viz_save_dir,
                plot_save_fpath=None,
            )
            reconstruction_reports.append(report)

    floor_reconstruction_report.summarize_reports(reconstruction_reports)

    print("Evaluation complete (Oracle pose + predicted layout)")


if __name__ == "__main__":
    eval_oraclepose_predictedlayout()
