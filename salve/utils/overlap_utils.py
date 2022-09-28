"""Utilities to verify if two set of walls intersect into another room's free-space, which is infeasible."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

import salve.utils.polyline_interpolation as polyline_interpolation


EPS = 1e-9


def shrink_polygon(polygon: Polygon, shrink_factor: float = 0.10) -> Polygon:
    """Shrink a polygon's boundaries by a `shrink_factor`.

    Reference: https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates

    Args:
        polygon: polygon to shrink in size.
        shrink_factor: percentage by which to shrink polygon, e.g. 0.10 corresponds to shrinking by 10%

    Returns:
        polygon_shrunk:
    """
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    # Find the minimum volume enclosing bounding box, and treat its center as polygon centroid.
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * shrink_factor

    polygon_shrunk = polygon.buffer(-shrink_distance)  # shrink

    # It's possible for a MultiPolygon to result as a result of the buffer operation, i.e. especially for unusual
    # shapes. We choose the largest polygon inside the MultiPolygon, and return it.
    if isinstance(polygon_shrunk, MultiPolygon):
        subpolygon_areas = [subpolygon.area for subpolygon in polygon_shrunk.geoms]
        largest_poly_idx = np.argmax(subpolygon_areas)
        polygon_shrunk = polygon_shrunk.geoms[largest_poly_idx]

    return polygon_shrunk


def count_verts_inside_poly(polygon: Polygon, query_verts: np.ndarray) -> int:
    """Count the number of vertices located inside of a polygon.

    Args:
        polygon: reference shape.
        query_verts: query vertices that may or may not lie within the polygon's area

    Returns:
        num_violations: number of query vertices that lie within the polygon's area.
           These can be considered infeasible penetrations of freespace (violations).
    """
    num_violations = 0
    for vert in query_verts:
        v_pt = Point(vert)
        if polygon.contains(v_pt) or polygon.contains(v_pt):
            num_violations += 1
    return num_violations


def determine_invalid_wall_overlap(
    pano1_room_vertices: np.ndarray,
    pano2_room_vertices: np.ndarray,
    shrink_factor: float,
    pano1_id: Optional[int] = None,
    pano2_id: Optional[int] = None,
    i: Optional[int] = None,
    j: Optional[int] = None,
    visualize: bool = False,
) -> bool:
    """Determine whether two rooms have an invalid configuration form wall overlap/freespace penetration.

    First, we densely interpolate points (at 0.1 m discretization) for both room polygon boundaries. Second,
    we shrink both original polygons by `shrink_factor`, and then count the number of interpolated points
    along the original, outer boundar that fall into the other room's shrunk polygon. One room cannot be
    located within another room, as the layout prediction represents freespace up to the wall boundary.

    Note: the amount of intersection of the two polygons is not a useful signal because same-room (correct)
    alignments and incorrect (should be different rooms, but instead overlaid as one room) alignments
    both can yield high values.

    Args:
        pano1_room_vertices: room vertices of pano 1.
        pano2_room_vertices: room vertices of pano 2, in the same coordinate frame as `pano1_room_vertices'.
        shrink_factor: related to amount of buffer away from boundaries (e.g. away from walls). A good default
            is 0.1, i.e. 10%. This allows us to compensate for noisy room layouts.
        pano1_id: optional ID for panorama 1, used only for debug/visualization messages.
        pano2_id: optional ID for panorama 2, used only for debug/visualization messages.
        i: optional ID of WDO to use for matching from pano1, used only for debug/visualization messages.
        j: optional ID of WDO to use for matching from pano2, used only for debug/visualization messages.
        visualize: whether to save a visualization to disk.

    Returns:
        is_valid: boolean indicating whether room pair is a potentially valid configuration.
    """
    # Must add epsilon to avoid being detected as a duplicate vertex.
    # Note: polygon does not contain loop closure (no vertex is repeated twice). Thus, must add first
    # vertex to end of vertex list.
    pano1_room_vertices = np.vstack([pano1_room_vertices, pano1_room_vertices[0] + EPS])
    pano2_room_vertices = np.vstack([pano2_room_vertices, pano2_room_vertices[0] + EPS])

    polygon1 = Polygon(pano1_room_vertices)
    polygon2 = Polygon(pano2_room_vertices)

    # Vertices in two lines below are in normalized coords.
    # Iterpolate evenly spaced points along edges.
    pano1_room_vertices_interp = polyline_interpolation.interp_evenly_spaced_points(pano1_room_vertices, interval_m=0.1)
    pano2_room_vertices_interp = polyline_interpolation.interp_evenly_spaced_points(pano2_room_vertices, interval_m=0.1)

    shrunk_poly1 = shrink_polygon(polygon1, shrink_factor=shrink_factor)
    shrunk_poly1_verts = np.array(list(shrunk_poly1.exterior.coords))

    shrunk_poly2 = shrink_polygon(polygon2, shrink_factor=shrink_factor)
    shrunk_poly2_verts = np.array(list(shrunk_poly2.exterior.coords))

    # Cannot be the case that poly1 vertices fall within a shrinken version of polygon2
    # also, cannot be the case that poly2 vertices fall within a shrunk version of polygon1
    pano1_violations = count_verts_inside_poly(shrunk_poly1, query_verts=pano2_room_vertices_interp)
    pano2_violations = count_verts_inside_poly(shrunk_poly2, query_verts=pano1_room_vertices_interp)
    num_violations = pano1_violations + pano2_violations

    is_valid = num_violations == 0

    if visualize:
        _plot_overlap_visualization(
            pano1_room_vertices=pano1_room_vertices,
            pano2_room_vertices=pano2_room_vertices,
            pano1_room_vertices_interp=pano1_room_vertices_interp,
            pano2_room_vertices_interp=pano2_room_vertices_interp,
            shrunk_poly1_verts=shrunk_poly1_verts,
            shrunk_poly2_verts=shrunk_poly2_verts,
            num_violations=num_violations,
        )
    return is_valid


def _plot_overlap_visualization(
    pano1_room_vertices: np.ndarray,
    pano2_room_vertices: np.ndarray,
    pano1_room_vertices_interp: np.ndarray,
    pano2_room_vertices_interp: np.ndarray,
    shrunk_poly1_verts: np.ndarray,
    shrunk_poly2_verts: np.ndarray,
    num_violations: int
) -> None:
    """Visualize overlap regions by plotting original and shrunken layout polygons for two panos.

    Original room vertices are shown as thick lines.
    """
    plt.close("all")

    # Plot both the original lines (as thick lines) and also the interpolated points scattered on top.
    plt.scatter(pano1_room_vertices_interp[:, 0], pano1_room_vertices_interp[:, 1], 10, color="m")
    plt.plot(pano1_room_vertices[:, 0], pano1_room_vertices[:, 1], color="m", linewidth=20, alpha=0.1)

    plt.scatter(pano2_room_vertices_interp[:, 0], pano2_room_vertices_interp[:, 1], 10, color="g")
    plt.plot(pano2_room_vertices[:, 0], pano2_room_vertices[:, 1], color="g", linewidth=10, alpha=0.1)

    # Render shrunk polygon 1 in red (with thin lines).
    plt.scatter(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], 10, color="r")
    plt.plot(shrunk_poly1_verts[:, 0], shrunk_poly1_verts[:, 1], color="r", linewidth=1, alpha=0.1)

    # Render shrunk polygon 2 in blue (with thin lines).
    plt.scatter(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], 10, color="b")
    plt.plot(shrunk_poly2_verts[:, 0], shrunk_poly2_verts[:, 1], color="b", linewidth=1, alpha=0.1)

    plt.title(f"Step 2: Number of violations: {num_violations}")
    plt.axis("equal")
    plt.show()

    classification = "invalid" if num_violations > 0 else "valid"

    os.makedirs(f"debug_plots/{classification}", exist_ok=True)
    plt.savefig(f"debug_plots/{classification}/{pano1_id}_{pano2_id}___step2_{i}_{j}.jpg")
    plt.close("all")

