"""Unit tests to ensure we can determine the zorder for rasterization of 3d points, from bottom to top."""

import numpy as np

import salve.utils.zorder_utils as zorder_utils


def test_choose_elevated_repeated_vals_single_repeat() -> None:
    """Test when a single location is repeated, the highest value is assigned a true logical."""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [0, 1, 5], [5, 6, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_no_repeats() -> None:
    """Test that when no location is repeated, all logicals are true as most elevated value."""
    xyz = np.array([[0, 1, 0], [1, 2, 4], [2, 3, 5], [3, 4, 1]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([True, True, True, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_all_repeated() -> None:
    """Test that all when all points represent a single location, that the highest value is assigned a true logical."""
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_out_of_range_z_values() -> None:
    """Test that values outside of the valid z-value range are assigned `false` logicals."""
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 10], [0, 1, 11]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Point with z=10 excluded, since upper boundary is exclusive, not inclusive.
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=10, num_slices=5)
    gt_valid = np.array([False, True, False, False])
    assert np.allclose(valid, gt_valid)


def test_choose_elevated_repeated_vals_two_z_slices_only() -> None:
    """Test that zorder determination is correct when using just two z-slices."""
    xyz = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]])

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = zorder_utils.choose_elevated_repeated_vals(x, y, z, zmin=0, zmax=4, num_slices=2)
    gt_valid = np.array([False, False, False, True])
    assert np.allclose(valid, gt_valid)
