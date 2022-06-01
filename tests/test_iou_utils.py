
"""Unit tests to ensure IoU is computed correctly between two masks/images."""

import numpy as np

import salve.utils.iou_utils as iou_utils


def test_texture_map_iou() -> None:
    """
    f1 [1,0]
       [0,1]

    f2 [1,0]
       [1,0]
    """
    f1 = np.zeros((2, 2, 3)).astype(np.uint8)
    f1[0, 0] = [0, 100, 0]
    f1[1, 1] = [100, 0, 0]

    f2 = np.zeros((2, 2, 3)).astype(np.uint8)
    f2[0, 0] = [0, 0, 100]
    f2[1, 0] = [0, 100, 0]

    iou = iou_utils.texture_map_iou(f1, f2)
    assert np.isclose(iou, 1 / 3)

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
    iou = iou_utils.binary_mask_iou(mask1, mask2)
    assert np.isclose(iou, 1 / 8)
