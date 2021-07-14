
from typing import Tuple

import numpy as np


def assign_tp_fp_fn_tn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    """ """
    is_TP = np.logical_and(y_true == y_pred, y_pred == 1)
    is_FP = np.logical_and(y_true != y_pred, y_pred == 1)
    is_FN = np.logical_and(y_true != y_pred, y_pred == 0)
    is_TN = np.logical_and(y_true == y_pred, y_pred == 0)

    return is_TP, is_FP, is_FN, is_TN


def compute_tp_fp_fn_tn_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    """ """

    TP = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    FP = np.logical_and(y_true != y_pred, y_pred == 1).sum()

    FN = np.logical_and(y_true != y_pred, y_pred == 0).sum()
    TN = np.logical_and(y_true == y_pred, y_pred == 0).sum()

    return TP, FP, FN, TN


def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    """ Define 1 as the target class (positive)

    In confusion matrix, `actual` are along rows, `predicted` are columns:

              Predicted
          \\ P  N
    Actual P TP FN
           N FP TN

    Returns:
        prec:
        rec: 
        mAcc:
    """
    EPS = 1e-7

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


def test_compute_precision_recall_1() -> None:
    """All incorrect predictions."""
    y_pred = np.array([0,0,1])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    assert prec == 0.0
    assert rec == 0.0
    assert mAcc == 0.0
    

def test_compute_precision_recall_2() -> None:
    """All correct predictions."""
    y_pred = np.array([1,1,0])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    assert np.isclose(prec, 1.0)
    assert np.isclose(rec, 1.0)
    assert np.isclose(mAcc, 1.0)
    

# def test_compute_precision_recall_3() -> None:
#     """All correct predictions."""
#     y_pred = np.array([0,0,0])
#     y_true = np.array([1,1,0])

#     import pdb; pdb.set_trace()
#     prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    
#     # no false positives, but low recall
#     assert np.isclose(prec, 1.0)
#     assert np.isclose(rec, 1/3)


def test_compute_precision_recall_4() -> None:
    """All correct predictions."""
    y_pred = np.array([0,1,0,1])
    y_true = np.array([1,1,0,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    # no false positives, but low recall
    assert np.isclose(prec, 0.5)
    assert np.isclose(rec, 0.5)
    assert np.isclose(mAcc, 2.0)
