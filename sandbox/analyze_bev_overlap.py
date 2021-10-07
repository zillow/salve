
"""
Analyze visibility between two panoramas, as captured by the BEV overlap of texture maps.
"""
import glob
from types import SimpleNamespace
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np

import afp.dataset.zind_data as zind_data


def main() -> None:
    """Analyze data generated with Sim(3) alignments."""
    data_root = "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25"

    split_building_ids = zind_data.get_available_building_ids(dataset_root=f"{data_root}/gt_alignment_approx")

    args = SimpleNamespace(
    **{
    "data_root": "",
    "layout_data_root":  "",
    "modalities": ["ceiling_rgb_texture", "floor_rgb_texture"]
    })

    label_dict = {"gt_alignment_approx": 1} #, "incorrect_alignment": 0}  # is_match = True

    floor_ious = []

    plot_range = [0.0, 0.1]
    #plot_range = [0.1, 0.2]
    #plot_range = [0.2, 0.3]
    #plot_range = [0.3, 0.4]
    #plot_range = [0.4, 0.5]
    #plot_range = [0.5, 0.6]
    #plot_range = [0.6, 0.7]
    #plot_range = [0.7, 0.8]
    #plot_range = [0.8, 0.9]
    #plot_range = [0.9, 1.0]

    for label_name, label_idx in label_dict.items():
        for b, building_id in enumerate(split_building_ids):
            print(f"Building {building_id}")
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

                    floor_iou = texture_map_iou(f1, f2)
                    floor_ious.append(floor_iou)

                    # print(fp1f)
                    # print(fp2f)

                    # if plot_range[0] < floor_iou and floor_iou < plot_range[1]:
                    #     c1 = imageio.imread(fp1c)
                    #     c2 = imageio.imread(fp2c)
                    #     show_quadruplet(c1, c2, f1, f2, title=f"IoU: {floor_iou:.2f}")


        plt.hist(floor_ious, bins=20)
        plt.xlabel("BEV Floor-Floor IoU")
        plt.ylabel("Counts")
        plt.show()
        plt.close("all")



def texture_map_iou(f1: np.ndarray, f2: np.ndarray) -> float:
    """floor texture maps"""

    f1_occ_mask = np.amax(f1, axis=2) > 0
    f2_occ_mask = np.amax(f2, axis=2) > 0

    iou = binary_mask_iou(f1_occ_mask, f2_occ_mask)
    return iou

def test_texture_map_iou() -> None:
    """
    f1 [1,0]
       [0,1]

    f2 [1,0]
       [1,0]
    """
    f1 = np.zeros((2,2,3)).astype(np.uint8)
    f1[0,0] = [0,100,0]
    f1[1,1] = [100,0,0]

    f2 = np.zeros((2,2,3)).astype(np.uint8)
    f2[0,0] = [0, 0, 100]
    f2[1,0] = [0, 100, 0]

    iou = texture_map_iou(f1, f2)
    assert np.isclose(iou, 1/3)

def binary_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """ """
    eps = 1e-12
    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return inter.sum() / (union.sum() + eps)

def test_binary_mask_iou() -> None:
    """ """
    # fmt: off
    mask1 = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])

    mask2 = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0]
        ])
    # fmt: on
    iou = binary_mask_iou(mask1, mask2)
    assert np.isclose(iou, 1/8)


def show_quadruplet(im1: np.ndarray, im2: np.ndarray, im3: np.ndarray, im4: np.ndarray, title: Optional[str] = None) -> None:
    """ """
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,1)
    plt.imshow(im1)

    plt.subplot(2,2,2)
    plt.imshow(im2)

    plt.subplot(2,2,3)
    plt.imshow(im3)

    plt.subplot(2,2,4)
    plt.imshow(im4)

    if title is not None:
        plt.suptitle(title)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()