"""Unit tests on alignment hypothesis comparisons / equality / de-duplication."""

import numpy as np

import salve.common.alignment_hypothesis as alignment_hypothesis_utils
from salve.common.alignment_hypothesis import AlignmentHypothesis
from salve.common.sim2 import Sim2


def test_prune_to_unique_sim2_objs() -> None:
    """Ensure that four alignment hypotheses are de-duplicated to just two unique instances."""
    wR1 = np.eye(2)
    wt1 = np.array([0, 1])
    ws1 = 1.5

    wR2 = np.array([[0, 1], [1, 0]])
    wt2 = np.array([1, 2])
    ws2 = 3.0

    # Instances 0, 1, and 3 are identical (duplicates).
    possible_alignment_info = [
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=1,
            i2_wdo_idx=5,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=2,
            i2_wdo_idx=6,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR2, wt2, ws2),
            wdo_alignment_object="window",
            i1_wdo_idx=3,
            i2_wdo_idx=7,
            configuration="identity",
        ),
        AlignmentHypothesis(
            i2Ti1=Sim2(wR1, wt1, ws1),
            wdo_alignment_object="window",
            i1_wdo_idx=4,
            i2_wdo_idx=8,
            configuration="identity",
        ),
    ]
    pruned_possible_alignment_info = alignment_hypothesis_utils.prune_to_unique_sim2_objs(possible_alignment_info)
    assert len(pruned_possible_alignment_info) == 2

    assert pruned_possible_alignment_info[0].i2Ti1.scale == 1.5
    assert pruned_possible_alignment_info[1].i2Ti1.scale == 3.0

