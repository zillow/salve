"""Precision/recall computation utilities."""

import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from salve.common.edge_classification import EdgeClassification


_PathLike = Union[str, "os.PathLike[str]"]


EPS = 1e-7


def assign_tp_fp_fn_tn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    """Assign true positives, false positives, false negatives, true negatives.

    Args:
        y_true: integer array representing ground truth categories.
        y_pred: integer array representing predicted categories.

    Returns:
        is_TP: boolean mask for true positives.
        is_FP: boolean mask for false positives.
        is_FN: boolean mask for false negatives.
        is_TN: boolean mask for true negatives.
    """
    is_TP = np.logical_and(y_true == y_pred, y_pred == 1)
    is_FP = np.logical_and(y_true != y_pred, y_pred == 1)
    is_FN = np.logical_and(y_true != y_pred, y_pred == 0)
    is_TN = np.logical_and(y_true == y_pred, y_pred == 0)

    return is_TP, is_FP, is_FN, is_TN


def compute_tp_fp_fn_tn_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    """Compute counts of true positives, false positives, false negatives, true negatives.

    Args:
        y_true: integer array representing ground truth categories.
        y_pred: integer array representing predicted categories.

    Returns:
        is_TP: number of true positives.
        is_FP: number of false positives.
        is_FN: number of false negatives.
        is_TN: number of true negatives.
    """
    TP = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    FP = np.logical_and(y_true != y_pred, y_pred == 1).sum()

    FN = np.logical_and(y_true != y_pred, y_pred == 0).sum()
    TN = np.logical_and(y_true == y_pred, y_pred == 0).sum()

    return TP, FP, FN, TN


def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    """Compute precision and recall over a set of predictions, using ground-truth.

    Define 1 as the target class (positive).

    In confusion matrix, `actual` are along rows, `predicted` are columns:

              Predicted
          \\ P  N
    Actual P TP FN
           N FP TN

    Args:
        y_true: integer array representing ground truth categories.
        y_pred: integer array representing predicted categories (assumed to already be thresholded by confidence).

    Returns:
        prec: precision.
        rec: recall.
        mAcc: mean accuracy.
    """
    TP, FP, FN, TN = compute_tp_fp_fn_tn_counts(y_true, y_pred)

    # form a confusion matrix
    C = np.zeros((2,2))
    C[0,0] = TP
    C[0,1] = FN

    C[1,0] = FP
    C[1,1] = TN

    # Normalize the confusion matrix
    C[0] /= (C[0].sum() + EPS)
    C[1] /= (C[1].sum() + EPS)

    mAcc = np.mean(np.diag(C))

    prec = TP / (TP + FP + EPS)
    rec = TP / (TP + FN + EPS)

    if (TP + FN) == 0:
        # there were no positive GT elements
        print("Recall undefined...")
        #raise Warning("Recall undefined...")

    #import sklearn.metrics
    #prec, rec, _, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    return prec, rec, mAcc


def plot_precision_recall_curve_sklearn(measurements: List[EdgeClassification]) -> None:
    """Compute a P/R curve using the sklearn library.

    Args:
        measurements: list of length (K,) representing model predictions on pose graph edges.

    Returns:
        prec: array of shape (K,) representing monotonically increasing precision values such that element i is the
            precision of predictions with score >= thresholds[i] and the last element is 1.
            We do NOT force monotonicity.
        recall: array of shape (K,) representing decreasing recall values such that element i is the recall of predictions
            with score >= thresholds[i] and the last element is 0.
        thresholds: array of shape (K-1,) representing confidence thresholds for each precision and recall value.
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    """
    y_true_list = []
    probas_pred = []
    for m in measurements:

        y_true_list.append(m.y_true)

        if m.y_hat == 1:
            pos_prob = m.prob
        else:
            pos_prob = 1 - m.prob

        probas_pred.append(pos_prob)

    # from sklearn.metrics import PrecisionRecallDisplay
    prec, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=y_true_list, probas_pred=probas_pred, pos_label=1)
    # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    # plt.show()

    return prec, recall, thresholds
