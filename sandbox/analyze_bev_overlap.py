"""
Analyze visibility between two panoramas, as captured by the BEV overlap of texture maps.
"""
import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np

import afp.dataset.zind_data as zind_data
import afp.utils.iou_utils as iou_utils


def main() -> None:
    """Analyze data generated with  alignments.
    
    Save 20-bin and 10-bin histograms showing distribution of visual overlap.
    """
    #data_root = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25" # Sim(3) alignments
    data_root = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres" # SE(2) alignments

    split_building_ids = zind_data.get_available_building_ids(dataset_root=f"{data_root}/gt_alignment_approx")

    args = SimpleNamespace(
        **{"data_root": "", "layout_data_root": "", "modalities": ["ceiling_rgb_texture", "floor_rgb_texture"]}
    )

    label_dict = {"gt_alignment_approx": 1, "incorrect_alignment": 0}  # is_match = True

    floor_ious = []

    # plot_range = [0.0, 0.1]
    # plot_range = [0.1, 0.2]
    # plot_range = [0.2, 0.3]
    # plot_range = [0.3, 0.4]
    # plot_range = [0.4, 0.5]
    # plot_range = [0.5, 0.6]
    # plot_range = [0.6, 0.7]
    # plot_range = [0.7, 0.8]
    # plot_range = [0.8, 0.9]
    # plot_range = [0.9, 1.0]

    for label_name, label_idx in label_dict.items():
        for b, building_id in enumerate(split_building_ids):

            if b > 100:
                continue

            print(f"Building {building_id}: {b}/{len(split_building_ids)}")
            for floor_id in ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04"]:
                fpaths = glob.glob(f"{data_root}/{label_name}/{building_id}/pair_*_rgb_{floor_id}_*.jpg")
                # here, pair_id will be unique
                if len(fpaths) == 0:
                    continue

                pair_tuples = zind_data.get_tuples_from_fpath_list(fpaths=fpaths, label_idx=label_idx, args=args)
                for pair_tuple in pair_tuples:
                    fp1c, fp2c, fp1f, fp2f, _ = pair_tuple

                    f1 = imageio.imread(fp1f)
                    f2 = imageio.imread(fp2f)

                    floor_iou = iou_utils.texture_map_iou(f1, f2)
                    floor_ious.append(floor_iou)

                    # print(fp1f)
                    # print(fp2f)

                    # if plot_range[0] < floor_iou and floor_iou < plot_range[1]:
                    #     c1 = imageio.imread(fp1c)
                    #     c2 = imageio.imread(fp2c)
                    #     show_quadruplet(c1, c2, f1, f2, title=f"IoU: {floor_iou:.2f}")

        classname = "Positive" if label_name == "gt_alignment_approx" else "Negative"

        plt.hist(floor_ious, bins=20)
        plt.xlabel("BEV Floor-Floor IoU")
        plt.ylabel("Image Pair Counts")
        plt.savefig(f"{Path(data_root).name}__{classname}_20bin_histogram.pdf", dpi=500)
        plt.close("all")

        bin_edges = np.linspace(0,1,11)
        bin_idxs = np.digitize(floor_ious, bins=bin_edges)
        iou_bins = np.bincount(bin_idxs - 1)

        # convert to percent
        normalized_iou_bins = iou_bins / len(floor_ious) * 100

        def format_bar_chart() -> None:
            """ """
            xtick_labels = []
            for i in range(len(bin_edges)-1):
                xtick_labels += [ f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})" ]
            plt.xticks(ticks=np.arange(10), labels=xtick_labels, rotation=20)
            plt.tight_layout()

        # bar chart.
        plt.bar(np.arange(10), normalized_iou_bins) # convert to percent
        plt.xlabel("Floor-Floor Texture Map IoU")
        plt.ylabel(f"Percent of {classname} Image Pairs")
        format_bar_chart()

        plt.savefig(f"{Path(data_root).name}__{classname}_10bin_histogram.pdf", dpi=500)
        plt.close("all")


def show_quadruplet(
    im1: np.ndarray, im2: np.ndarray, im3: np.ndarray, im4: np.ndarray, title: Optional[str] = None
) -> None:
    """ """
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(im1)

    plt.subplot(2, 2, 2)
    plt.imshow(im2)

    plt.subplot(2, 2, 3)
    plt.imshow(im3)

    plt.subplot(2, 2, 4)
    plt.imshow(im4)

    if title is not None:
        plt.suptitle(title)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
