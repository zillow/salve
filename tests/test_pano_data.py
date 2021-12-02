
""" """

import numpy as np

from afp.common.pano_data import WDO


def test_get_wd_normal_2d() -> None:
    """Verify that normal vector to a window/door object is computed correctly."""

    # flat horizontal line for window
    wd1 = WDO(global_Sim2_local=None, pt1=(-2, 0), pt2=(2, 0), bottom_z=-1, top_z=1, type="window")
    n1 = wd1.get_wd_normal_2d()
    gt_n1 = np.array([0, 1])
    assert np.allclose(n1, gt_n1)

    # upwards diagonal for window, y=x
    wd2 = WDO(global_Sim2_local=None, pt1=(0, 0), pt2=(3, 3), bottom_z=-1, top_z=1, type="window")
    n2 = wd2.get_wd_normal_2d()
    gt_n2 = np.array([-1, 1]) / np.sqrt(2)

    assert np.allclose(n2, gt_n2)
