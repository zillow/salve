""" """

import numpy as np
from argoverse.utils.sim2 import Sim2

from salve.utils.wdo_alignment import AlignmentHypothesis

from scripts.export_alignment_hypotheses import obj_almost_equal  # TODO: fix
from scripts.export_alignment_hypotheses import prune_to_unique_sim2_objs # TODO: fix


def test_obj_almost_equal() -> None:
    """ """
    # fmt: off
    i2Ti1_pred = Sim2(
        R=np.array(
            [
                [-0.99928814, 0.03772511],
                [-0.03772511, -0.99928814]
            ], dtype=np.float32
        ),
        t=np.array([-3.0711207, -0.5683456], dtype=np.float32),
        s=1.0,
    )

    i2Ti1_gt = Sim2(
        R=np.array(
            [
                [-0.9999569, -0.00928213],
                [0.00928213, -0.9999569]
            ], dtype=np.float32
        ),
        t=np.array([-3.0890038, -0.5540818], dtype=np.float32),
        s=0.9999999999999999,
    )
    # fmt: on
    assert obj_almost_equal(i2Ti1_pred, i2Ti1_gt)
    assert obj_almost_equal(i2Ti1_gt, i2Ti1_pred)


def test_prune_to_unique_sim2_objs() -> None:
    """ """
    wR1 = np.eye(2)
    wt1 = np.array([0, 1])
    ws1 = 1.5

    wR2 = np.array([[0, 1], [1, 0]])
    wt2 = np.array([1, 2])
    ws2 = 3.0

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
    pruned_possible_alignment_info = prune_to_unique_sim2_objs(possible_alignment_info)
    assert len(pruned_possible_alignment_info) == 2

    assert pruned_possible_alignment_info[0].i2Ti1.scale == 1.5
    assert pruned_possible_alignment_info[1].i2Ti1.scale == 3.0



