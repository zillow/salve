"""Unit tests to ensure we can determine the zorder for rasterization of 3d points, from top to bottom"""

import numpy as np

import afp.utils.zorder_utils as zorder_utils


def test_choose_elevated_repeated_vals_1() -> None:
    """Single location is repeated"""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [0, 1, 5], [5, 6, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_2() -> None:
    """No location is repeated"""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [2, 3, 5], [3, 4, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([True, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_3() -> None:
    """Every location is repeated"""
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_4() -> None:
    """
    Some z-values are outside of the specified range
    """
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 10], [0, 1, 11]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, False, False])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_5() -> None:
    """
    Try for just 2 slices
    """
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=4, num_slices=2)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)


if __name__ == '__main__':
    # test_choose_elevated_repeated_vals_1()
    # test_choose_elevated_repeated_vals_2()
    # test_choose_elevated_repeated_vals_3()

    test_choose_elevated_repeated_vals_4()
    test_choose_elevated_repeated_vals_5()

