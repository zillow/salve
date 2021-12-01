"""
Utilities to verify if two set of walls intersect into another room's free-space, which is infeasible.
"""

import os
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


def interp_evenly_spaced_points(polyline: np.ndarray, interval_m: float) -> np.ndarray:
    """Nx2 polyline to Mx2 polyline, for waypoint every `interval_m` meters"""

    length_m = get_polyline_length(polyline)
    n_waypoints = int(np.ceil(length_m / interval_m))
    px, py = eliminate_duplicates_2d(polyline[:, 0], py=polyline[:, 1])
    interp_polyline = interp_arc(t=n_waypoints, px=px, py=py)

    return interp_polyline


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

    Amount of intersection of the two polygons is not a useful signal.

    Args:
        pano1_id: ID for panorama 1
        pano2_id: ID for panorama 2
        i: ID of WDO to use for matching from pano1
        j: ID of WDO to use for matching from pano2
        pano1_room_vertices:
        pano2_room_vertices: in the same coordinate frame as `pano1_room_vertices'
        shrink_factor: related to amount of buffer away from boundaries (e.g. away from walls).

    Returns:
        is_valid: 
    """
    EPS = 1e-9 # must add eps to avoid being detected as a duplicate vertex
    # Note: polygon does not contain loop closure (no vertex is repeated twice). Thus, must add first
    # vertex to end of vertex list.
    pano1_room_vertices = np.vstack([pano1_room_vertices, pano1_room_vertices[0] + EPS])
    pano2_room_vertices = np.vstack([pano2_room_vertices, pano2_room_vertices[0] + EPS])

    polygon1 = Polygon(pano1_room_vertices)
    polygon2 = Polygon(pano2_room_vertices)

    pano1_room_vertices_interp = interp_evenly_spaced_points(pano1_room_vertices, interval_m=0.1)  # in normalized coords
    pano2_room_vertices_interp = interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)  # in normalized coords

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

        # render shrunk polygon 1 in red.
        plt.scatter(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], 10, color="r")
        plt.plot(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], color="r", linewidth=1, alpha=0.1)

        # render shrunk polygon 2 in blue.
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
