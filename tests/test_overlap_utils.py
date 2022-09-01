"""Unit tests for checking utilities that computes layout overlap between two rooms."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

import salve.utils.overlap_utils as overlap_utils
from salve.utils.interpolate import interp_arc


def test_determine_invalid_wall_overlap1() -> None:
    """Ensure that wall overlap computation is correct for two rooms -- large and small horseshoe-shaped rooms.

    .---.---.
    |       |
    .   .xxx.
    |       x|
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
    allowed_overlap_pct = 0.01  # TODO: allow 1% of violations ???

    is_valid = overlap_utils.determine_invalid_wall_overlap(
        pano1_room_vertices=pano1_room_vertices, pano2_room_vertices=pano2_room_vertices, shrink_factor=wall_buffer_m
    )
    assert not is_valid


def test_determine_invalid_wall_overlap_identical_shape() -> None:
    """Ensure that wall overlap computation is correct for two identical shapes.

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
    wall_buffer_m = 0.2  # 20 centimeter noise buffer
    allowed_overlap_pct = 0.01  # TODO: allow 1% of violations ???

    is_valid = overlap_utils.determine_invalid_wall_overlap(
        pano1_room_vertices=pano1_room_vertices, pano2_room_vertices=pano2_room_vertices, shrink_factor=wall_buffer_m
    )
    assert is_valid


def test_determine_invalid_wall_overlap3() -> None:
    """
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
    visualize = True

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

    assert False


def test_eliminate_duplicates_2d() -> None:
    """Ensure two duplicated waypoints are removed.

    Duplicates are at indices 4,5 and 6,7, so rows 5 and 7 should be removed.
    """
    polyline = np.array(
        [
            [3.41491678, 0.82735686],  # 0
            [2.5812492, -2.36060637],  # 1
            [0.2083626, -1.74008522],  # 2
            [0.53871724, -0.47680178],  # 3
            [0.40395381, -0.4415605],  # 4
            [0.40395381, -0.4415605],  # 5
            [-0.36244272, -0.24114416],  # 6
            [-0.36244272, -0.24114416],  # 7
            [-0.56108295, -0.18919879],  # 8
            [-0.14397634, 1.40582611],  # 9
            [0.06767395, 1.35047855],  # 10
            [0.15388028, 1.68013345],  # 11
        ]
    )
    px, py = overlap_utils.eliminate_duplicates_2d(px=polyline[:, 0], py=polyline[:, 1])
    polyline_no_dups = np.stack([px, py], axis=-1)

    expected_polyline_no_dups = np.array(
        [
            [3.41491678, 0.82735686],
            [2.5812492, -2.36060637],
            [0.2083626, -1.74008522],
            [0.53871724, -0.47680178],
            [0.40395381, -0.4415605],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634, 1.40582611],
            [0.06767395, 1.35047855],
            [0.15388028, 1.68013345],
        ]
    )
    assert np.allclose(polyline_no_dups, expected_polyline_no_dups)


# def test_interp_arc() -> None:
#     """ """
#     polyline = np.array(
#         [
#             [ 3.41491678,  0.82735686],
#             [ 2.5812492 , -2.36060637],
#             [ 0.2083626 , -1.74008522],
#             [ 0.53871724, -0.47680178],
#             [ 0.40395381, -0.4415605 ],
#             [ 0.40395381, -0.4415605 ],
#             [-0.36244272, -0.24114416],
#             [-0.36244272, -0.24114416],
#             [-0.56108295, -0.18919879],
#             [-0.14397634,  1.40582611],
#             [ 0.06767395,  1.35047855],
#             [ 0.15388028,  1.68013345]
#         ]
#     )
#     n_waypoints = 104
#     import pdb; pdb.set_trace()
#     interp_polyline = interp_arc(t=n_waypoints, px=polyline[:, 0], py=polyline[:, 1])

#     assert isinstance(interp_polyline, np.ndarray)


def test_interp_evenly_spaced_points() -> None:
    """ """
    pano2_room_vertices = np.array(
        [
            [3.41491678, 0.82735686],
            [2.5812492, -2.36060637],
            [0.2083626, -1.74008522],
            [0.53871724, -0.47680178],
            [0.40395381, -0.4415605],
            [0.40395381, -0.4415605],
            [-0.36244272, -0.24114416],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634, 1.40582611],
            [0.06767395, 1.35047855],
            [0.15388028, 1.68013345],
        ]
    )
    pano2_room_vertices_interp = overlap_utils.interp_evenly_spaced_points(
        pano2_room_vertices, interval_m=0.1
    )  # meters

    assert isinstance(pano2_room_vertices_interp, np.ndarray)
    assert pano2_room_vertices_interp.shape == (104, 2)


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


if __name__ == "__main__":
    # test_shrink_polygon()

    # test_interp_evenly_spaced_points()
    # test_interp_arc()
    # test_eliminate_duplicates_2d()

    # test_determine_invalid_wall_overlap1()
    # test_determine_invalid_wall_overlap2()
    test_determine_invalid_wall_overlap3()
