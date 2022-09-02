"""Utilities to measure IoU between two texture maps or two binary masks."""

import numpy as np


def texture_map_iou(f1: np.ndarray, f2: np.ndarray) -> float:
    """Compute IoU between two floor texture maps.

    Args:
        f1: array of shape (H,W,C) representing floor RGB texture maps.
        f2: array of shape (H,W,C) representing floor RGB texture maps.

    Returns:
        iou: scalar in the range [0,1] representing intersection over union.
    """

    f1_occ_mask = np.amax(f1, axis=2) > 0
    f2_occ_mask = np.amax(f2, axis=2) > 0

    iou = binary_mask_iou(f1_occ_mask, f2_occ_mask)
    return iou


def binary_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    eps = 1e-12
    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return inter.sum() / (union.sum() + eps)
