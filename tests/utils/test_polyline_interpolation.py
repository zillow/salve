"""Unit tests on polyline interpolation utilities."""

import numpy as np
import pytest

import salve.utils.polyline_interpolation as polyline_interp_utils


"""
def test_eliminate_duplicates_2d() -> None:
    #Ensure two duplicated polyline waypoints are removed.

    #Duplicates are located at indices 4,5 and 6,7, so rows 5 and 7 should be removed.
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

    px, py = polyline_interp_utils.eliminate_duplicates_2d(px=polyline[:, 0], py=polyline[:, 1])
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
    assert len(polyline_no_dups) == len(polyline) - 2
    assert np.allclose(polyline_no_dups, expected_polyline_no_dups)
"""


def test_interp_arc_with_consecutive_duplicates() -> None:
    """ """
    polyline = np.array(
        [
            [ 3.41491678,  0.82735686],
            [ 2.5812492 , -2.36060637],
            [ 0.2083626 , -1.74008522],
            [ 0.53871724, -0.47680178],
            [ 0.40395381, -0.4415605 ],
            [ 0.40395381, -0.4415605 ],
            [-0.36244272, -0.24114416],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634,  1.40582611],
            [ 0.06767395,  1.35047855],
            [ 0.15388028,  1.68013345]
        ]
    )
    n_waypoints = 104
    interp_polyline = polyline_interp_utils.interp_arc(t=n_waypoints, points=polyline)
    assert isinstance(interp_polyline, np.ndarray)


def test_interp_evenly_spaced_points_with_consecutive_duplicates() -> None:
    """Ensures that 2d polyline interpolation at a fixed-length waypoint discretization is successful."""

    # Assume vertices are given in units of meters.
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

    with pytest.raises(ValueError):
    	pano2_room_vertices_interp = polyline_interp_utils.interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)
    # assert isinstance(pano2_room_vertices_interp, np.ndarray)
    # assert pano2_room_vertices_interp.shape == (104, 2)

    # waypoint_separations = np.linalg.norm(np.diff(pano2_room_vertices_interp, axis=0), axis=1)
    # assert np.allclose(waypoint_separations, 0.1, atol=0.03)
    # assert np.isclose(np.mean(waypoint_separations), 0.1, atol=1e-3)


def test_interp_evenly_spaced_points_rectangle() -> None:
    """Should run successfully without issues."""

    # Layout boundary.
    layout_vertices = np.array([[1., 2.],
           [1., 5.],
           [3., 5.],
           [3., 2.],
           [1., 2.]])

    interp_layout_vertices = polyline_interp_utils.interp_evenly_spaced_points(layout_vertices, interval_m=0.1)



if __name__ == "__main__":

	pass
    #test_interp_evenly_spaced_points()
    #test_interp_arc()
    # test_eliminate_duplicates_2d()

