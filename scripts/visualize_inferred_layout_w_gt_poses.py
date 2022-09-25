"""Script to visualize inferred layouts at oracle (ground-truth) camera poses."""

import click

import salve.common.posegraph2d as posegraph2d
import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader


def visualize_inferred_layouts_with_gt_poses(
    raw_dataset_dir: str,
    mhnet_predictions_data_root: str,
    rendering_save_dir: str,
) -> None:
    """
    For each of the 1575 ZinD homes, check to see if we have MHNet predictions. If we do, load
    the layout predictions as a PoseGraph2d object, and render them at oracle camera pose locations.

    Args:
        raw_dataset_dir: path to where ZInD dataset is stored on disk (after download from Bridge API).
        mhnet_predictions_data_root: path to directory containing ModfiedHorizonNet (MHNet) predictions.
        rendering_save_dir:
    """
    NUM_ZIND_BUILDINGS = 1575

    # Generate all possible building IDs for ZinD.
    building_ids = [str(v).zfill(4) for v in range(NUM_ZIND_BUILDINGS)]

    for building_id in building_ids:
        print(f"Loading {building_id}")
        floor_pose_graphs = hnet_prediction_loader.load_inferred_floor_pose_graphs(
            building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=mhnet_predictions_data_root
        )
        if floor_pose_graphs is None:
            # prediction files must have been missing, so we skip.
            raise ValueError
            # continue

        for floor_id, floor_pose_graph in floor_pose_graphs.items():

            # Load the GT pose graph to rip out the GT pose for each pano.
            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            floor_reconstruction_report.render_floorplans_side_by_side(
                est_floor_pose_graph=floor_pose_graph,
                show_plot=False,
                save_plot=True,
                plot_save_dir=rendering_save_dir,
                gt_floor_pg=gt_pose_graph,
            )


@click.command(help="Script to render MHNet inferred layouts at ground truth camera poses.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # default="/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"
    help="Path to where ZInD dataset is stored on disk (after download from Bridge API).",
)
@click.option(
    "--rendering_save_dir",
    type=str,
    required=True,
    # default="MHNet__oracle_pose_2022_09_25"
    help="Name of directory where renderings will be saved.",
)
@click.option(
    "--mhnet_predictions_data_root",
    type=click.Path(exists=True),
    # default="/Users/johnlambert/Downloads/zind_horizon_net_2022_09_05_unzipped_archive/ZInD_HorizonNet_predictions"
    required=True,
    help="Path to directory containing ModfiedHorizonNet (MHNet) predictions.",
)
def run_visualize_inferred_layouts_with_gt_poses(
    raw_dataset_dir: str, mhnet_predictions_data_root: str, rendering_save_dir: str
) -> None:
    """Click entry point for MHNet inferred layout rendering w/ oracle poses."""
    
    visualize_inferred_layouts_with_gt_poses(
        raw_dataset_dir=raw_dataset_dir,
        mhnet_predictions_data_root=mhnet_predictions_data_root,
        rendering_save_dir=rendering_save_dir,
    )


if __name__ == "__main__":
    run_visualize_inferred_layouts_with_gt_poses()
