"""Unit tests for checking utilities that computes layout overlap between two rooms."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

import salve.utils.overlap_utils as overlap_utils


def test_determine_invalid_wall_overlap1() -> None:
    """Ensure that wall overlap computation is correct for two rooms -- large and small horseshoe-shaped rooms.

    Pano 1's room indicated by .--. edges (large horse-shoe).
    Pano 2's room indicated by .xxx. edges (small horse-shoe).

    .---.---.
    |       |
    .   .xxx.
    |      x|
    .   .xxx.
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [2,4],
            [3,4],
            [3,3],
            [2,3]
        ])

    # fmt: on
    wall_buffer_m = 0.2  # 20 centimeter noise buffer

    is_valid = overlap_utils.determine_invalid_wall_overlap(
        pano1_room_vertices=pano1_room_vertices, pano2_room_vertices=pano2_room_vertices, shrink_factor=wall_buffer_m
    )
    # Wall pair is invalid, since the small horseshoe is located within the large horseshoe-shaped room.
    assert not is_valid


def test_determine_invalid_wall_overlap_identical_shape() -> None:
    """Ensure that wall overlap computation is correct for two identical shapes (a horseshoe).

    .---.---.
    |       |
    .       .
    |       |
    .       .
    |       |
    .       .

    """
    # fmt: off
    pano1_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])
    pano2_room_vertices = np.array(
        [
            [1,2],
            [1,5],
            [3,5],
            [3,2]
        ])

    # fmt: on

    # Noise butter.
    shrink_factor = 0.2

    is_valid = overlap_utils.determine_invalid_wall_overlap(
        pano1_room_vertices=pano1_room_vertices, pano2_room_vertices=pano2_room_vertices, shrink_factor=shrink_factor
    )
    # Identical shapes are OK. They would mean that we have two very accurate room layout predictions within
    # the same room. Original exterior vertices do not fall within the shrunken version.
    assert is_valid


def test_determine_invalid_wall_overlap3() -> None:
    """Ensures that two stacked shapes of obviously different size (square and then its half-rectangle) are invalid.
    
    panos (0,8) below. similar to

    TODO: determine why this was considered valid for Building "0003" and panos (0.7)
    "opening_0_7___step3_identity_2_0.jpg".
    """
    pano1_id = 0
    pano2_id = 8
    i = 2
    j = 0
    # 1 is in magenta.
    pano1_room_vertices = np.array(
        [[-1.20350544, 2.19687034], [-0.14832726, 3.12533515], [2.14896215, 0.51452036], [1.09378396, -0.41394445]]
    )
    # 2 is in green.
    pano2_room_vertices = np.array(
        [[-0.08913514, -1.02572344], [-2.17362494, 1.34324966], [-0.15560001, 3.11893567], [1.92888979, 0.74996256]]
    )
    shrink_factor = 0.1
    visualize = False

    freespace_is_valid = overlap_utils.determine_invalid_wall_overlap(
        pano1_room_vertices=pano1_room_vertices,
        pano2_room_vertices=pano2_room_vertices,
        shrink_factor=shrink_factor,
        pano1_id=pano1_id,
        pano2_id=pano2_id,
        i=i,
        j=j,
        visualize=visualize,
    )
    assert not freespace_is_valid


def test_shrink_polygon() -> None:
    """
    If not handled correctly, will see the following error:
        *** AttributeError: 'MultiPolygon' object has no attribute 'exterior'
    """

    def draw_polyline(polyline: np.ndarray, color: str) -> None:
        """ """
        plt.scatter(polyline[:, 0], polyline[:, 1], 10, color=color, marker=".")
        plt.plot(polyline[:, 0], polyline[:, 1], color=color)

    pano1_room_vertices = np.array(
        [
            [0.61807389, -1.0028074],
            [0.59331251, -0.48251453],
            [0.63846121, -0.38975602],
            [0.81566386, -0.02569123],
            [0.85433859, 0.05376642],
            [-1.9087475, 1.3986739],
            [-0.71553403, 3.85014409],
            [2.87482109, 2.10250285],
            [2.51753773, 1.36848825],
            [2.26585724, 1.49099615],
            [1.31355939, -0.46543567],
            [1.32937937, -1.00994635],
        ]
    )
    polygon1 = Polygon(pano1_room_vertices)
    shrunk_poly1 = overlap_utils.shrink_polygon(polygon1)

    assert np.isclose(shrunk_poly1.area, 6.275, atol=1e-3)

    # draw_polyline(pano1_room_vertices, "b")

    # colors = ["r", "g"]
    # for i in range(2):
    #     print(f"poly {i}: ", shrunk_poly1.geoms[i].area)
    #     poly = np.array(list(shrunk_poly1.geoms[i].exterior.coords))
    #     draw_polyline(poly, colors[i])
    # plt.show()

    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    # draw_polyline(shrunk_poly1_verts, "r")
    # plt.show()

    assert isinstance(shrunk_poly1_verts, np.ndarray)
