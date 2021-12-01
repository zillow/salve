

from pathlib import Path
from typing import List

import numpy as np

import afp.utils.pr_utils as pr_utils
from afp.common.edge_classification import EdgeClassification


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


def test_compute_precision_recall_1() -> None:
    """All incorrect predictions."""
    y_pred = np.array([0,0,1])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    assert prec == 0.0
    assert rec == 0.0
    assert mAcc == 0.0
    

def test_compute_precision_recall_2() -> None:
    """All correct predictions."""
    y_pred = np.array([1,1,0])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

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

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    # no false positives, but low recall
    assert np.isclose(prec, 0.5)
    assert np.isclose(rec, 0.5)
    assert np.isclose(mAcc, 2.0)


def test_compute_prec_recall_curve() -> None:
    """ """
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 1])
    confidences = np.array([1.0, 1.0, 0.5])

    import pdb; pdb.set_trace()
    ap = pr_utils.compute_prec_recall_curve(y_true, y_pred, confidences, n_rec_samples = 4, figs_fpath = Path("pr_plots"))
    assert ap == 0.67



def test_plot_precision_recall_curve_sklearn1() -> None:
    """For all correct predictions, with increasing amounts of confidence."""
    y_pred = np.array([1,1,1])
    y_true = np.array([1,1,1])
    probs =  np.array([0.1, 0.2, 0.3])

    measurements = []
    for i in range(3):
        measurements.append(make_dummy_edge_classification(prob=probs[i], y_hat=y_pred[i], y_true=y_true[i]))

    prec, rec, thresholds = pr_utils.plot_precision_recall_curve_sklearn(measurements)

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

    prec, rec, thresholds = pr_utils.plot_precision_recall_curve_sklearn(measurements)

    expected_prec = np.array([0.66666667, 0.5 , 1. , 1. ])
    expected_rec = np.array([1. , 0.5, 0.5, 0. ])
    expected_thresholds = np.array([0.3, 0.8, 0.9])

    assert np.allclose(prec, expected_prec)
    assert np.allclose(rec, expected_rec)
    assert np.allclose(thresholds, expected_thresholds)
