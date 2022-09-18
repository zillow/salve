"""Script to query completion progress of texture-map rendering during execution."""

import glob
from pathlib import Path


EPS = 1e-10


def main(hypotheses_save_root: str, bev_save_root: str) -> None:
    """ """
    building_ids = [Path(d).name for d in glob.glob(f"{bev_save_root}/gt_alignment_approx/*")]

    for building_id in building_ids:

        # find the floors
        pos_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/gt_alignment_approx"
        positive_render_dirpath = f"{bev_save_root}/gt_alignment_approx/{building_id}"
        num_rendered_positives = len(glob.glob(f"{positive_render_dirpath}/*")) / 4
        expected_num_positives = len(glob.glob(f"{pos_hypotheses_dirpath}/*"))
        pos_rendering_percent = num_rendered_positives / (expected_num_positives + EPS) * 100  # matches

        neg_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/incorrect_alignment"
        negative_render_dirpath = f"{bev_save_root}/incorrect_alignment/{building_id}"
        num_rendered_negatives = len(glob.glob(f"{negative_render_dirpath}/*")) / 4
        expected_num_negatives = len(glob.glob(f"{neg_hypotheses_dirpath}/*"))
        neg_rendering_percent = num_rendered_negatives / (expected_num_negatives + EPS) * 100  # mismatches

        print(f"Building {building_id} Pos. {pos_rendering_percent:.2f}% Neg. {neg_rendering_percent:.2f}%")


if __name__ == "__main__":

    hypotheses_save_root = (
        "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    )
    bev_save_root = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"
    main(hypotheses_save_root=hypotheses_save_root, bev_save_root=bev_save_root)
