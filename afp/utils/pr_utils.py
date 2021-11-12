
"""Precision/recall computation utilities."""

import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from afp.common.edge_classification import EdgeClassification


_PathLike = Union[str, "os.PathLike[str]"]


class InterpType(Enum):
    ALL = auto()


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


def compute_prec_recall_curve(
    y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray, n_rec_samples: int = 101, figs_fpath: _PathLike = Path("pr_plots")
) -> None:
    """ """
    import pdb; pdb.set_trace()
    recalls_interp = np.linspace(0, 1, n_rec_samples)

    if not figs_fpath.is_dir():
        figs_fpath.mkdir(parents=True, exist_ok=True)

    # only 1 class of interest -- the match class
    is_TP, is_FP, is_FN, is_TN = assign_tp_fp_fn_tn(y_true, y_pred)
    ninst = is_TP.sum() + is_TN.sum()
    ranks = confidences.argsort()[::-1]  # sort by last column, i.e. confidences
    ninst = tp + fn 

    tp = cls_stats[:, i].astype(bool)
    ap_th, precisions_interp = calc_ap(tp, recalls_interp, ninst)

    # from sklearn.metrics import precision_recall_curve
    # y_true_array = np.array([m.y_true for m in measurements])
    # y_pred_probs_array = np.array([m.prob for m in measurements])
    # prec, rec, thresholds = precision_recall_curve(y_true_array, y_pred_probs_array)


def calc_ap(gt_ranked: np.ndarray, recalls_interp: np.ndarray, ninst: int) -> Tuple[float, np.ndarray]:
    """Compute precision and recall, interpolated over n fixed recall points.
    Args:
        gt_ranked: Ground truths, ranked by confidence.
        recalls_interp: Interpolated recall values.
        ninst: Number of instances of this class.
    Returns:
        avg_precision: Average precision.
        precisions_interp: Interpolated precision values.
    """
    tp = gt_ranked

    cumulative_tp = np.cumsum(tp, dtype=np.int)
    cumulative_fp = np.cumsum(~tp, dtype=np.int)
    cumulative_fn = ninst - cumulative_tp

    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
    recalls = cumulative_tp / (cumulative_tp + cumulative_fn)
    precisions = interp(precisions)
    precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)
    avg_precision = precisions_interp.mean()
    return avg_precision, precisions_interp


def interp(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels. See equation 2 in
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    for more information.
    Args:
        prec: Precision at all recall levels (N,).
        method: Accumulation method.
    Returns:
        prec_interp: Interpolated precision at all recall levels (N,).
    """
    if method == InterpType.ALL:
        prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return prec_interp



def rank(dts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank the detections in descending order according to score (detector confidence).
    Args:
        dts: Array of `ObjectLabelRecord` objects. (N,).
    Returns:
        ranked_dts: Array of `ObjectLabelRecord` objects ranked by score (N,) where N <= MAX_NUM_BOXES.
        ranked_scores: Array of floats sorted in descending order (N,) where N <= MAX_NUM_BOXES.
    """
    scores = np.array([dt.score for dt in dts.tolist()])
    ranks = scores.argsort()[::-1]

    ranked_dts = dts[ranks]
    ranked_scores = scores[ranks]

    # Ensure the number of boxes considered per class is at most `MAX_NUM_BOXES`.
    return ranked_dts[:MAX_NUM_BOXES], ranked_scores[:MAX_NUM_BOXES]


def plot(rec_interp: np.ndarray, prec_interp: np.ndarray, cls_name: str, figs_fpath: Path) -> Path:
    """Plot and save the precision recall curve.
    Args:
        rec_interp: Interpolated recall data of shape (N,).
        prec_interp: Interpolated precision data of shape (N,).
        cls_name: Class name.
        figs_fpath: Path to the folder which will contain the output figures.
    Returns:
        dst_fpath: Plot file path.
    """
    plt.plot(rec_interp, prec_interp)
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    dst_fpath = Path(f"{figs_fpath}/{cls_name}.png")
    plt.savefig(dst_fpath)
    plt.close()
    return dst_fpath



def test_compute_prec_recall_curve() -> None:
    """ """
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 1])
    confidences = np.array([1.0, 1.0, 0.5])

    import pdb; pdb.set_trace()
    ap = compute_prec_recall_curve(y_true, y_pred, confidences, n_rec_samples = 4, figs_fpath = Path("pr_plots"))
    assert ap == 0.67


def plot_precision_recall_curve_sklearn(measurements: List[EdgeClassification]) -> None:
    """
    Args:
        measurements: list of length (K,) representing model predictions.

    Returns:
        prec: array of shape (K,) represen
            Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1.
        recall: array of shape (K,) represen
            Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.
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


def make_dummy_edge_classification(prob: float, y_hat: int, y_true: int) -> EdgeClassification:
    """ """
    measurement = EdgeClassification(
        i1=0, # dummy val
        i2=1, # dummy val
        prob=prob,
        y_hat=y_hat,
        y_true=y_true,
        pair_idx=0, # dummy index
        wdo_pair_uuid="door_0_1", # dummy val
        configuration="identity" # dummy val
    )
    return measurement


def test_plot_precision_recall_curve_sklearn1() -> None:
    """For all correct predictions, with increasing amounts of confidence."""
    y_pred = np.array([1,1,1])
    y_true = np.array([1,1,1])
    probs =  np.array([0.1, 0.2, 0.3])

    measurements = []
    for i in range(3):
        measurements.append(make_dummy_edge_classification(prob=probs[i], y_hat=y_pred[i], y_true=y_true[i]))

    prec, rec, thresholds = plot_precision_recall_curve_sklearn(measurements)

    expected_prec = np.array([1., 1., 1., 1.])
    expected_rec = np.array([1., 0.667, 0.333, 0. ])
    expected_thresholds = np.array([0.1, 0.2, 0.3])

    assert np.allclose(prec, expected_prec)
    assert np.allclose(rec, expected_rec)
    assert np.allclose(thresholds, expected_thresholds)


def test_plot_precision_recall_curve_sklearn2() -> None:
    """For some incorrect and correct predictions.

    @ >= 0.3 thresh, 2 TPs, 1 FP. (prec 2/3). Recall 1.
    @ >= 0.8 thresh, 1 FP, 1 TP. (prec 1/2). Recall 1/2.
    @ >= 0.9 thresh, 0 FP, 1 TP (prec 1). Recall 1/2.
    """
    y_pred = np.array([1,1,1,1])
    y_true = np.array([1,0,1,0])
    probs =  np.array([0.9, 0.8, 0.3, 0.2])

    measurements = []
    for i in range(4):
        measurements.append(make_dummy_edge_classification(prob=probs[i], y_hat=y_pred[i], y_true=y_true[i]))

    prec, rec, thresholds = plot_precision_recall_curve_sklearn(measurements)

    expected_prec = np.array([0.66666667, 0.5 , 1. , 1. ])
    expected_rec = np.array([1. , 0.5, 0.5, 0. ])
    expected_thresholds = np.array([0.3, 0.8, 0.9])

    assert np.allclose(prec, expected_prec)
    assert np.allclose(rec, expected_rec)
    assert np.allclose(thresholds, expected_thresholds)


if __name__ == '__main__':
    #test_compute_prec_recall_curve()

    # test_compute_precision_recall_1()
    # test_compute_precision_recall_2()
    # #test_compute_precision_recall_3()
    # test_compute_precision_recall_4()

    #test_plot_precision_recall_curve_sklearn1()
    test_plot_precision_recall_curve_sklearn2()

