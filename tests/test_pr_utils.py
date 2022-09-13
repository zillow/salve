"""Unit tests on precision/recall computation utilities."""

from pathlib import Path
from typing import List

import numpy as np

import salve.utils.pr_utils as pr_utils
from salve.common.edge_classification import EdgeClassification


def _make_dummy_edge_classification(prob: float, y_hat: int, y_true: int) -> EdgeClassification:
    """Generates a dummy EdgeClassification object with the specified GT & predicted category, and confidence.

    Args:
        prob: confidence probability.
        y_hat: predicted category.
        y_true: ground truth category.
    """
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


def test_compute_precision_recall_all_incorrect_predictions() -> None:
    """Ensures that precision, recall, and accuracy are zero for all incorrect predictions."""
    y_pred = np.array([0,0,1])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    assert prec == 0.0
    assert rec == 0.0
    assert mAcc == 0.0
    

def test_compute_precision_recall_all_correct_predictions() -> None:
    """Ensures that precision, recall, and accuracy are one for all correct predictions."""
    y_pred = np.array([1,1,0])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    assert np.isclose(prec, 1.0)
    assert np.isclose(rec, 1.0)
    assert np.isclose(mAcc, 1.0)
    

def test_compute_precision_recall_no_positive_predictions() -> None:
    """Ensures that precision, recall, and mean accuracy are correct for mixed correct and incorrect predictions."""

    # Scenario data indicates no positive predictions. 2 incorrect and 1 correct prediction.
    y_pred = np.array([0,0,0])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    # No false positives (since no positive class 1 predictions), but no TPs, so low precision.
    assert np.isclose(prec, 0.0)

    # Two false negatives, but no TPs, so low recall.
    assert np.isclose(rec, 0.0)
    # Achieves 0% acc on class 1, but 100% acc on class 0.
    assert np.isclose(mAcc, 0.5)


def test_compute_precision_recall_mixed() -> None:
    """Ensures that precision, recall, and mean accuracy are correct for mixed correct and incorrect predictions."""
    y_pred = np.array([0,1,0,1])
    y_true = np.array([1,1,0,0])

    prec, rec, mAcc = pr_utils.compute_precision_recall(y_true, y_pred)

    # 1 FN, 1 TP, 1 FN, 1 TN. Precision is (1/2) and recall is (1/2).
    assert np.isclose(prec, 0.5)
    assert np.isclose(rec, 0.5)
    assert np.isclose(mAcc, 0.5)


def test_plot_precision_recall_curve_sklearn_all_correct() -> None:
    """Ensures that precision, recall, and thresholds from Sklearn (for P/R curve) are correct."""

    # We test a scenario with all correct predictions, with increasing amounts of confidence.
    y_pred = np.array([1,1,1])
    y_true = np.array([1,1,1])
    probs =  np.array([0.1, 0.2, 0.3])

    measurements = []
    for i in range(3):
        measurements.append(_make_dummy_edge_classification(prob=probs[i], y_hat=y_pred[i], y_true=y_true[i]))

    prec, rec, thresholds = pr_utils.plot_precision_recall_curve_sklearn(measurements)

    # There are no false positives, with 4 TPs, so precision is always perfect (equal to 1).
    expected_prec = np.array([1., 1., 1., 1.])

    # We can get false negatives if we require high-confidence; thus recall is not perfect
    # at all thresholds.
    expected_rec = np.array([1., 0.667, 0.333, 0. ])
    # At 0.1 threshold, get perfect recall, since no false positives.
    # At 0.2 threshold, 1 FP, so get 2/3 recall.
    # At 0.3 threshold, 2 FPs, so get 1/3 recall.
    expected_thresholds = np.array([0.1, 0.2, 0.3])

    assert np.allclose(prec, expected_prec)
    assert np.allclose(rec, expected_rec, 0.3)
    assert np.allclose(thresholds, expected_thresholds)


def test_plot_precision_recall_curve_sklearn_all_positive_predictions() -> None:
    """Ensures that precision, recall, and thresholds from Sklearn (for P/R curve) are correct.

    Represents all positive predictions (resulting in some incorrect and correct predictions).

    @ >= 0.2 thresh, 2 FP, 2 TP, 0 FN. (prec 1/2). Recall 1.
    @ >= 0.3 thresh, 1 FP, 2 TP, 0 FN. (prec 2/3). Recall 1.
    @ >= 0.8 thresh, 1 FP, 1 TP, 1 FN. (prec 1/2). Recall 1/2.
    @ >= 0.9 thresh, 0 FP, 1 TP, 1 FN. (prec 1). Recall 1/2.
    """
    # fmt: off
    y_pred = np.array([  1,   1,   1,   1])
    y_true = np.array([  1,   0,   1,   0])
    probs =  np.array([0.9, 0.8, 0.3, 0.2])
    # fmt: on

    measurements = []
    for i in range(4):
        measurements.append(_make_dummy_edge_classification(prob=probs[i], y_hat=y_pred[i], y_true=y_true[i]))

    prec, rec, thresholds = pr_utils.plot_precision_recall_curve_sklearn(measurements)

    # fmt: off
    expected_prec = np.array([      0.5, 0.667, 0.5,   1, 1 ])
    expected_rec = np.array([         1,     1, 0.5, 0.5, 0 ])
    expected_thresholds = np.array([0.2,   0.3, 0.8, 0.9])
    # fmt: on

    assert np.allclose(prec, expected_prec, atol=1e-3)
    assert np.allclose(rec, expected_rec)
    assert np.allclose(thresholds, expected_thresholds)
