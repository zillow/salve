
"""
Utilities to verify if two set of walls intersect into another room's free-space, which is infeasible.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.interpolate import interp_arc, get_duplicate_indices_1d
from argoverse.utils.polyline_density import get_polyline_length
from shapely.geometry import MultiPolygon, Point, Polygon


def shrink_polygon(polygon: Polygon, shrink_factor: float = 0.10) -> Polygon:
    """
    Reference: https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates
    
    Args:
        shrink_factor: shrink by 10%
    """
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    # find the minimum volume enclosing bounding box, and treat its center as polygon centroid
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * shrink_factor

    polygon_shrunk = polygon.buffer(-shrink_distance)  # shrink

    # It's possible for a MultiPolygon to result as a result of the buffer operation, i.e. especially for unusual shapes
    # We choose the largest polygon inside the MultiPolygon, and return it
    if isinstance(polygon_shrunk, MultiPolygon):
        subpolygon_areas = [subpolygon.area for subpolygon in polygon_shrunk.geoms]
        largest_poly_idx = np.argmax(subpolygon_areas)
        polygon_shrunk = polygon_shrunk.geoms[largest_poly_idx]

    return polygon_shrunk


def test_shrink_polygon() -> None:
    """
    If not handled correctly, will see the following error:
        *** AttributeError: 'MultiPolygon' object has no attribute 'exterior'
    """
    def draw_polyline(polyline: np.ndarray, color: str) -> None:
        """ """
        plt.scatter(polyline[:,0], polyline[:,1], 10, color=color, marker='.')
        plt.plot(polyline[:,0], polyline[:,1], color=color)

    pano1_room_vertices = np.array(
        [
            [ 0.61807389, -1.0028074 ],
            [ 0.59331251, -0.48251453],
            [ 0.63846121, -0.38975602],
            [ 0.81566386, -0.02569123],
            [ 0.85433859,  0.05376642],
            [-1.9087475 ,  1.3986739 ],
            [-0.71553403,  3.85014409],
            [ 2.87482109,  2.10250285],
            [ 2.51753773,  1.36848825],
            [ 2.26585724,  1.49099615],
            [ 1.31355939, -0.46543567],
            [ 1.32937937, -1.00994635]
        ]
    )
    polygon1 = Polygon(pano1_room_vertices)
    shrunk_poly1 = shrink_polygon(polygon1)

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


def interp_evenly_spaced_points(polyline: np.ndarray, interval_m) -> np.ndarray:
    """Nx2 polyline to Mx2 polyline, for waypoint every `interval_m` meters"""

    length_m = get_polyline_length(polyline)
    n_waypoints = int(np.ceil(length_m / interval_m))
    px, py = eliminate_duplicates_2d(polyline[:, 0], py=polyline[:, 1])
    interp_polyline = interp_arc(t=n_waypoints, px=px, py=py)

    return interp_polyline


def test_interp_evenly_spaced_points() -> None:
    """ """
    pano2_room_vertices = np.array(
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
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)  # meters

    assert isinstance(pano2_room_vertices_interp, np.ndarray)
    assert pano2_room_vertices_interp.shape == (104,2)


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


def eliminate_duplicates_2d(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Note: Differs from the argoverse implementation.

    We compare indices to ensure that deleted values are exactly
    adjacent to each other in the polyline sequence.
    """
    num_pts = px.shape[0]
    assert px.shape[0] == py.shape[0]
    px_dup_inds = get_duplicate_indices_1d(px)
    py_dup_inds = get_duplicate_indices_1d(py)
    shared_dup_inds = np.intersect1d(px_dup_inds, py_dup_inds)

    px = np.delete(px, [shared_dup_inds])
    py = np.delete(py, [shared_dup_inds])

    return px, py



def test_eliminate_duplicates_2d() -> None:
    """Ensure two duplicated waypoints are removed.

    Duplicates are at indices 4,5 and 6,7, so rows 5 and 7 should be removed.
    """
    polyline = np.array(
        [
            [ 3.41491678,  0.82735686], # 0
            [ 2.5812492 , -2.36060637], # 1
            [ 0.2083626 , -1.74008522], # 2
            [ 0.53871724, -0.47680178], # 3
            [ 0.40395381, -0.4415605 ], # 4
            [ 0.40395381, -0.4415605 ], # 5
            [-0.36244272, -0.24114416], # 6
            [-0.36244272, -0.24114416], # 7
            [-0.56108295, -0.18919879], # 8
            [-0.14397634,  1.40582611], # 9
            [ 0.06767395,  1.35047855], # 10
            [ 0.15388028,  1.68013345] # 11
        ]
    )
    px, py = eliminate_duplicates_2d(px=polyline[:,0], py=polyline[:,1])
    polyline_no_dups = np.stack([px,py], axis=-1)

    expected_polyline_no_dups = np.array(
        [
            [ 3.41491678,  0.82735686],
            [ 2.5812492 , -2.36060637],
            [ 0.2083626 , -1.74008522],
            [ 0.53871724, -0.47680178],
            [ 0.40395381, -0.4415605 ],
            [-0.36244272, -0.24114416],
            [-0.56108295, -0.18919879],
            [-0.14397634,  1.40582611],
            [ 0.06767395,  1.35047855],
            [ 0.15388028,  1.68013345]
        ]
    )
    assert np.allclose(polyline_no_dups, expected_polyline_no_dups)


def count_verts_inside_poly(polygon: Polygon, query_verts: np.ndarray) -> int:
    """
    Args:
        polygon: reference shape.
        query_verts: query vertices that may or may not lie within the polygon's area

    Returns:
        num_violations: number of query vertices that lie within the polygon's area.
    """
    num_violations = 0
    for vert in query_verts:
        v_pt = Point(vert)
        if polygon.contains(v_pt) or polygon.contains(v_pt):
            num_violations += 1
    return num_violations


def determine_invalid_wall_overlap(
    pano1_id: int,
    pano2_id: int,
    i: int,
    j: int,
    pano1_room_vertices: np.ndarray,
    pano2_room_vertices: np.ndarray,
    shrink_factor: float = 0.1,
    visualize: bool = False,
) -> bool:
    """
    TODO: consider adding allowed_overlap_pct: float = 0.01
        Args:
            pano1_id: panorama id
            pano2_id: panorama id
            i: id of WDO to match from pano1
            j: id of WDO to match from pano2

        TODO: use `wall_buffer_m`

        Returns:
    """
    polygon1 = Polygon(pano1_room_vertices)
    polygon2 = Polygon(pano2_room_vertices)

    pano1_room_vertices_interp = interp_evenly_spaced_points(pano1_room_vertices, interval_m=0.1)  # meters
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)  # meters

    # # should be in the same coordinate frame, now
    # inter_poly = polygon1.intersection(polygon2)
    # inter_poly_area = inter_poly.area

    # inter_poly_verts = np.array(list(inter_poly.exterior.coords))

    shrunk_poly1 = shrink_polygon(polygon1, shrink_factor=shrink_factor)
    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    shrunk_poly2 = shrink_polygon(polygon2, shrink_factor=shrink_factor)
    shrunk_poly2_verts = np.array(list(shrunk_poly2.exterior.coords))

    # TODO: interpolate evenly spaced points along edges, if this gives any benefit?
    # Cannot be the case that poly1 vertices fall within a shrinken version of polygon2
    # also, cannot be the case that poly2 vertices fall within a shrunk version of polygon1
    pano1_violations = count_verts_inside_poly(shrunk_poly1, query_verts=pano2_room_vertices_interp)
    pano2_violations = count_verts_inside_poly(shrunk_poly2, query_verts=pano1_room_vertices_interp)
    num_violations = pano1_violations + pano2_violations

    is_valid = num_violations == 0

    if visualize:
        # plot the overlap region
        plt.close("all")

        # plot the interpolated points via scatter, but keep the original lines
        plt.scatter(pano1_room_vertices_interp[:, 0], pano1_room_vertices_interp[:, 1], 10, color="m")
        plt.plot(pano1_room_vertices[:, 0], pano1_room_vertices[:, 1], color="m", linewidth=20, alpha=0.1)

        plt.scatter(pano2_room_vertices_interp[:, 0], pano2_room_vertices_interp[:, 1], 10, color="g")
        plt.plot(pano2_room_vertices[:, 0], pano2_room_vertices[:, 1], color="g", linewidth=10, alpha=0.1)

        plt.scatter(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], 10, color="r")
        plt.plot(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], color="r", linewidth=1, alpha=0.1)

        plt.scatter(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], 10, color="b")
        plt.plot(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], color="b", linewidth=1, alpha=0.1)

        plt.title(f"Step 2: Number of violations: {num_violations}")
        plt.axis("equal")
        # plt.show()

        classification = "invalid" if num_violations > 0 else "valid"

        os.makedirs(f"debug_plots/{classification}", exist_ok=True)
        plt.savefig(f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step2_{i}_{j}.jpg")
        plt.close("all")

    return is_valid


def test_determine_invalid_wall_overlap1() -> None:
    """large horseshoe, and small horseshoe

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

    is_valid = determine_invalid_wall_overlap(pano1_room_vertices, pano2_room_vertices, wall_buffer_m)
    assert not is_valid


def test_determine_invalid_wall_overlap2() -> None:
    """identical shape

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

    is_valid = determine_invalid_wall_overlap(pano1_room_vertices, pano2_room_vertices, wall_buffer_m)
    assert is_valid


if __name__ == '__main__':
    pass
    # test_shrink_polygon()

    # test_interp_evenly_spaced_points()
    #test_interp_arc()
    # test_eliminate_duplicates_2d()

    # test_determine_invalid_wall_overlap1()
    # test_determine_invalid_wall_overlap2()
