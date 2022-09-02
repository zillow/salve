"""Visualize model predictions on pano-pano edges, coloring by FP/FN/TN w.r.t. ground truth."""

import click

import salve.utils.pr_utils as pr_utils
from salve.common.edge_classification import edge_classification


def vis_edge_classifications(serialized_preds_json_dir: str, raw_dataset_dir: str) -> None:
    """Visualize model predictions on pano-pano edges, coloring by FP/FN/TN w.r.t. ground truth.

    Args:
        serialized_preds_json_dir:
        raw_dataset_dir: path to directory where the full ZinD dataset is stored (in raw form as downloaded from
            Bridge API).
    """
    floor_edgeclassifications_dict = edge_classification.get_edge_classifications_from_serialized_preds(
        serialized_preds_json_dir
    )

    color_dict = {"TP": "green", "FP": "red", "FN": "orange", "TN": "blue"}

    # loop over each building and floor
    for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

        # if building_id != '1490': # '1394':# '1635':
        # 	continue

        print(f"On building {building_id}, {floor_id}")
        gt_floor_pose_graph = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)

        # gather all of the edge classifications
        y_hat = np.array([m.y_hat for m in measurements])
        y_true = np.array([m.y_true for m in measurements])

        # classify into TPs, FPs, FNs, TNs
        is_TP, is_FP, is_FN, is_TN = pr_utils.assign_tp_fp_fn_tn(y_true, y_pred=y_hat)
        for m, is_tp, is_fp, is_fn, is_tn in zip(measurements, is_TP, is_FP, is_FN, is_TN):

            # then render the edges
            if is_tp:
                color = color_dict["TP"]
                # gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)
                print(f"\tFP: ({m.i1},{m.i2}) for pair {m.pair_idx}")

            elif is_fp:
                color = color_dict["FP"]

            elif is_fn:
                color = color_dict["FN"]
                # gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)

            elif is_tn:
                color = color_dict["TN"]
                gt_floor_pose_graph.draw_edge(m.i1, m.i2, color)

            # if m.i1 or m.i2 not in gt_floor_pose_graph.nodes:
            # 	import pdb; pdb.set_trace()

        # render the pose graph first
        gt_floor_pose_graph.render_estimated_layout(show_plot=True)
        # continue


@click.command(help="Script to render BEV texture maps for each feasible alignment hypothesis.")
@click.option(
    "--serialized_preds_json_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlam/Downloads/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20/2021_10_22_serialized_edge_classifications"  # noqa
    help="",
)
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
def run_vis_edge_classifications(serialized_preds_json_dir: str, raw_dataset_dir: str) -> None:
    """Click entry point for visualization of model predictions on pano-pano edges."""
    vis_edge_classifications(serialized_preds_json_dir=serialized_preds_json_dir, raw_dataset_dir=raw_dataset_dir)


if __name__ == "__main__":
    """ """
    run_vis_edge_classifications()
