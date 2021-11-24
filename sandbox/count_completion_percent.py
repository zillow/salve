"""Count what percent of data from a particular split has been rendered on a particular machine/node."""

import glob
from pathlib import Path

import numpy as np

from afp.dataset.zind_partition import DATASET_SPLITS


EPS = 1e-10


def check_completion() -> None:
    """ """
    desired_split = "test"

    # hypotheses_save_root = (
    #     "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    # )
    # bev_save_root = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"
    # bev_save_root = "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"

    hypotheses_save_root = "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"
    bev_save_root = "/home/johnlam/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8"

    building_ids = [Path(d).name for d in glob.glob(f"{bev_save_root}/gt_alignment_approx/*")]

    split_building_ids = DATASET_SPLITS[desired_split]
    for building_id in sorted(building_ids):

        if building_id not in split_building_ids:
            continue

        rendering_percent_dict = {}
        for label_type in ["gt_alignment_approx", "incorrect_alignment"]:

            # find the floors
            label_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/{label_type}"
            expected_num_label = len(glob.glob(f"{label_hypotheses_dirpath}/*"))

            label_render_dirpath = f"{bev_save_root}/{label_type}/{building_id}"
            num_rendered_label = len(glob.glob(f"{label_render_dirpath}/*")) / 4
            print(f"{label_type} {num_rendered_label}")
            
            label_rendering_percent = num_rendered_label / (expected_num_label + EPS) * 100  # matches

            rendering_percent_dict[label_type] = label_rendering_percent

        pos_rendering_percent = rendering_percent_dict["gt_alignment_approx"]
        neg_rendering_percent = rendering_percent_dict["incorrect_alignment"]
        print(f"Building {building_id} Pos. {pos_rendering_percent:.2f}% Neg. {neg_rendering_percent:.2f}%")


def check_negative_positive_ratio() -> None:
    """ """
    ratios = []

    desired_split = "train" #  "test"

    hypotheses_save_root = (
        "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    )
    bev_save_root = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"
    # bev_save_root = "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"

    # hypotheses_save_root = "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"
    # bev_save_root = "/home/johnlam/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8"

    building_ids = [Path(d).name for d in glob.glob(f"{bev_save_root}/gt_alignment_approx/*")]

    split_building_ids = DATASET_SPLITS[desired_split]
    for building_id in sorted(building_ids):

        if building_id not in split_building_ids:
            continue

        neg_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/incorrect_alignment"
        expected_num_neg = len(glob.glob(f"{neg_hypotheses_dirpath}/*"))

        pos_hypotheses_dirpath = f"{hypotheses_save_root}/{building_id}/*/gt_alignment_approx"
        expected_num_pos = len(glob.glob(f"{pos_hypotheses_dirpath}/*"))

        ratio = expected_num_neg / expected_num_pos
        print(f"Ratio on Building {building_id}: {ratio:.3f}")
        ratios.append(ratio)

    print()
    print(f"On split {desired_split}")
    print(f"Average ratio: {np.mean(ratios):.3f}")
    print(f"Min ratio: {np.amin(ratios):.3f}")
    print(f"Max ratio: {np.amax(ratios):.3f}")


if __name__ == "__main__":

    #check_completion()
    check_negative_positive_ratio()
