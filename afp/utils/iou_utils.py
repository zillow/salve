
"""
Measure IoU between two texture maps or two binary masks.
"""

import numpy as np

def texture_map_iou(f1: np.ndarray, f2: np.ndarray) -> float:
    """floor texture maps

    Args:
        f1: array of shape (H,W,C)
        f2: array of shape (H,W,C)

    Returns:
        iou: scalar in the range [0,1] representing intersection over union.
    """

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
    f1 = np.zeros((2, 2, 3)).astype(np.uint8)
    f1[0, 0] = [0, 100, 0]
    f1[1, 1] = [100, 0, 0]

    f2 = np.zeros((2, 2, 3)).astype(np.uint8)
    f2[0, 0] = [0, 0, 100]
    f2[1, 0] = [0, 100, 0]

    iou = texture_map_iou(f1, f2)
    assert np.isclose(iou, 1 / 3)


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
    assert np.isclose(iou, 1 / 8)
