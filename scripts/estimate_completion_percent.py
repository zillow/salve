"""Script to query completion progress of texture-map rendering during execution."""

import glob
from pathlib import Path

import click

EPS = 1e-10


def query_completion_progress(hypotheses_save_root: str, bev_save_root: str) -> None:
    """ """
    building_ids = [Path(d).name for d in glob.glob(f"{bev_save_root}/gt_alignment_approx/*")]
    building_ids.sort()
    for building_id in building_ids:

        # Count number of expected positives and number of currently rendered positives (match pairs).
        pos_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/gt_alignment_approx"
        positive_render_dirpath = f"{bev_save_root}/gt_alignment_approx/{building_id}"
        num_rendered_positives = len(glob.glob(f"{positive_render_dirpath}/*")) / 4
        expected_num_positives = len(glob.glob(f"{pos_hypotheses_dirpath}/*"))
        pos_rendering_percent = num_rendered_positives / (expected_num_positives + EPS) * 100

        # Count number of expected negatives and number of currently rendered negatives (mismatched pairs).
        neg_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/incorrect_alignment"
        negative_render_dirpath = f"{bev_save_root}/incorrect_alignment/{building_id}"
        num_rendered_negatives = len(glob.glob(f"{negative_render_dirpath}/*")) / 4
        expected_num_negatives = len(glob.glob(f"{neg_hypotheses_dirpath}/*"))
        neg_rendering_percent = num_rendered_negatives / (expected_num_negatives + EPS) * 100

        print(f"Building {building_id} Pos. {pos_rendering_percent:.2f}% Neg. {neg_rendering_percent:.2f}%")


@click.command(help="Script to query completion progress of texture-map rendering during execution.")
@click.option(
    "--hypotheses_save_root",
    type=click.Path(exists=True),
    required=True,
    # default = "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    help="Path to generated alignment hypotheses.",
)
@click.option(
    "--bev_save_root",
    type=click.Path(exists=True),
    required=True,
    # default = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"
    help="Path to BEV renderings.",
)
def run_query_completion_progress(hypotheses_save_root: str, bev_save_root: str) -> None:
    """Click entry point."""
    query_completion_progress(hypotheses_save_root=hypotheses_save_root, bev_save_root=bev_save_root)


if __name__ == "__main__":

    run_query_completion_progress()
    
