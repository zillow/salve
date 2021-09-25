"""
Unit tests to ensure ICP is working properly.
"""
import copy
import os
from pathlib import Path

import numpy as np
import open3d


import afp.baselines.open3d_icp as open3d_icp


TEST_DATA_ROOT = Path(__file__).resolve().parent / "test_data"


def test_register_point_cloud() -> None:
    """
    test data from https://github.com/isl-org/Open3D/tree/master/examples/test_data/ColoredICP
    """

    # Load two point clouds and show initial pose
    source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

    # draw initial alignment
    tTs = np.identity(4)
    # draw_registration_result_original_color(source, target, tTs)

    tTs = open3d_icp.register_point_clouds(source, target)
    # draw_registration_result_original_color(source, target, tTs)

    assert isinstance(tTs, np.ndarray)
    assert tTs.shape == (4, 4)


def test_register_colored_point_clouds() -> None:
    """ """
    source = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_115.ply"))
    target = open3d.io.read_point_cloud(os.path.join(TEST_DATA_ROOT, "ColoredICP", "frag_116.ply"))

    tTs = open3d_icp.register_colored_point_clouds(source, target)
    # draw_registration_result_original_color(source, target, tTs)
    assert isinstance(tTs, np.ndarray)
    assert tTs.shape == (4, 4)


def draw_registration_result_original_color(
    source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud, transformation: np.ndarray
) -> None:
    """ """
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries(
        [source_temp, target],
        zoom=0.5,
        front=[-0.2458, -0.8088, 0.5342],
        lookat=[1.7745, 2.2305, 0.9787],
        up=[0.3109, -0.5878, -0.7468],
    )
