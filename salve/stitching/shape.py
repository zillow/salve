""" TODO: ADD DOCSTRING """

import json
import math
import os
from typing import Any, Dict, List, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from tqdm import tqdm

import salve.stitching.transform as transform_utils
from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT
from salve.stitching.draw import (
    draw_camera_in_top_down_canvas,
    draw_shape_in_top_down_canvas,
    draw_shape_in_top_down_canvas_fill,
    TANGO_COLOR_PALETTE,
)
from salve.stitching.models.feature2d import Feature2dU, Feature2dXy
from salve.stitching.models.locations import Point2d


def generate_shapely_polygon_from_room_shape_vertices(vertices: List[dict]) -> Polygon:
    """TODO

    Args:
        vertices: TODO

    Returns:
        Polygon representing ...
    """
    xys = []
    for vertex in vertices:
        xys.append([vertex["x"], vertex["y"]])
    return Polygon(xys)


def extract_coordinates_from_shapely_polygon(shape: Polygon) -> List[Point2d]:
    """TODO:

    Args:
        shape:

    Returns:
        coords:
    """
    coords = []
    xys = shape.boundary.xy
    for i in range(len(xys[0])):
        coords.append(Point2d(x=xys[0][i], y=xys[1][i]))
    return coords


def load_room_shape_polygon_from_predictions(
    room_shape_pred: Dict[str, Any], uncertainty=None, camera_height: float = DEFAULT_CAMERA_HEIGHT
) -> Polygon:
    """TODO

    Args:
        room_shape_pred
        uncertainty
        camera_height

    Returns:
        Polygon representing ...
    """
    flag = True
    xys = []
    xys_upper = []
    xys_lower = []

    uvs = []
    uvs_upper = []
    uvs_lower = []
    for i, corner in enumerate(room_shape_pred):
        if not flag:
            uvs.append([corner[0] + 0.5 / 1024, corner[1] + 0.5 / 512])
            if uncertainty:
                uvs_upper.append([corner[0] + 0.5 / 1024, corner[1] + 0.5 / 512 - uncertainty[i] / 512])
                # uvs_lower.append([corner[0]+0.5/1024, corner[1]+0.5/512+uncertainty[i]/512])
        flag = not flag
    xys = transform_utils.uv_to_xy_batch(uvs, camera_height)
    if uncertainty:
        xys_upper = transform_utils.uv_to_xy_batch(uvs_upper, camera_height)
        # xys_lower = transform_utils.uv_to_xy_batch(uvs_lower, camera_height)
        return Polygon(xys), Polygon(xys_upper)
    return Polygon(xys)


def generate_dense_shape(v_vals: List[Any], uncertainty: Any) -> Tuple[Any, Any]:
    """TODO

    Args:
        v_vals:
        uncertainty:

    Returns:
        polygon:
        distances:
    """
    vs = np.asarray(v_vals) / 512
    us = np.asarray(range(1024)) / 1024
    uvs = [[us[i], vs[i]] for i in range(1024)]
    polygon, poly_upper = load_room_shape_polygon_from_predictions(uvs, uncertainty)
    distances = []
    xys = polygon.boundary.xy
    xys_upper = poly_upper.boundary.xy
    for i in range(len(xys[0])):
        distances.append(math.sqrt((xys_upper[0][i] - xys[0][i]) ** 2 + (xys_upper[1][i] - xys[1][i]) ** 2))
    return polygon, distances


def group_panos_by_room(predictions: Any, location_panos: Any) -> List[Any]:
    """

    Args:
        predictions:
        location_panos:

    Returns:
        groups:
    """
    print("Running pano grouping by room ... ")
    shapes_global = {}
    graph = nx.Graph()
    for panoid in location_panos:
        pose = location_panos[panoid]
        shape = predictions[panoid]
        xys_transformed = []
        xys = extract_coordinates_from_shapely_polygon(shape)
        for xy in xys:
            xys_transformed.append(transform_utils.transform_xy_by_pose(xy, pose))
        shape_global = Polygon([[xy.x, xy.y] for xy in xys_transformed])
        shapes_global[panoid] = shape_global
        graph.add_node(panoid)

    panoids = [*location_panos.keys()]
    for i in range(len(panoids)):
        for j in range(i, len(panoids)):
            panoid1 = panoids[i]
            panoid2 = panoids[j]
            shape1 = shapes_global[panoid1]
            shape2 = shapes_global[panoid2]
            area_intersection = shape1.intersection(shape2).area
            area_union = shape1.union(shape2).area
            iou = area_intersection / area_union
            iou1 = area_intersection / shape1.area
            iou2 = area_intersection / shape2.area
            if iou > 0.1 or iou1 > 0.3 or iou2 > 0.3:
                graph.add_edge(panoid1, panoid2)
    groups = [[*c] for c in sorted(nx.connected_components(graph))]
    return groups


def refine_shape_group_start_with(
    group: Any, start_id: Any, predicted_shapes: Any, wall_confidences: Any, location_panos: Any
) -> Tuple[Any, Any]:
    """TODO

    Args:
        group: TODO
        start_id: TODO
        predicted_shapes: TODO
        wall_confidences: TODO
        location_panos: TODO

    Returns:
        xys1_final: TODO
        conf1_final: TODO
    """
    RES = 512
    original_us = np.arange(0.5 / RES, (RES + 0.5) / RES, 1.0 / RES)
    panoid = start_id
    # for panoid in group:
    current_shape = predicted_shapes[panoid]
    xys0 = extract_coordinates_from_shapely_polygon(current_shape)
    pose0 = location_panos[panoid]
    wall_conf0 = wall_confidences[panoid]
    uvs0 = []
    for xy0 in xys0:
        uvs0.append(transform_utils.xy_to_uv(xy0, DEFAULT_CAMERA_HEIGHT))

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)

    final_vs_all = {}
    final_cs_all = {}
    colors = {"9c5d07356f": "red", "84fe28b6c1": "blue", "404ed9fcfb": "yellow", "1a2445a9fe": "purple"}
    for panoid_1 in group:
        if panoid_1 == panoid:
            continue
        shape1 = predicted_shapes[panoid_1]
        pose1 = location_panos[panoid_1]
        wall_conf1 = wall_confidences[panoid_1]
        # print(group, panoid, panoid_1)

        xys1 = extract_coordinates_from_shapely_polygon(shape1)
        xys0 = extract_coordinates_from_shapely_polygon(current_shape)
        xys1_projected = []
        uvs1_projected = []
        for xy1 in xys1:
            xy1_transformed = transform_utils.transform_xy_by_pose(xy1, pose1)
            xy1_projected = transform_utils.project_xy_by_pose(xy1_transformed, pose0)
            xys1_projected.append(xy1_projected)
            uvs1_projected.append(transform_utils.xy_to_uv(xy1_projected, DEFAULT_CAMERA_HEIGHT))

        xys1_projected = [[xy.x, xy.y] for xy in xys1_projected]
        polygon_global = Polygon(xys1_projected)
        if not polygon_global.contains(Point(0, 0)):
            continue

        final_vs, final_cs = transform_utils.reproject_uvs_to(uvs1_projected, wall_conf1, panoid_1, start_id)

        final_vs_all[panoid_1] = final_vs
        final_cs_all[panoid_1] = final_cs

        xys_render = []
        for i, u in enumerate(original_us):
            xy_render = transform_utils.uv_to_xy(Point2d(x=u, y=final_vs[i]), DEFAULT_CAMERA_HEIGHT)
            xys_render.append(xy_render)

    #     color = colors[panoid_1] if panoid_1 in colors else 'red'
    #     draw_shape_in_top_down_canvas(
    #         axis, xys_render, color, pose=location_panos[start_id]
    #     )
    #     draw_camera_in_top_down_canvas(axis, location_panos[panoid_1], color, size=20)
    # color = colors[start_id] if start_id in colors else 'red'
    # draw_shape_in_top_down_canvas(axis, xys0, color, pose=location_panos[start_id])
    # draw_camera_in_top_down_canvas(axis, location_panos[start_id], color, size=20)
    # axis.set_aspect("equal")
    # path = f'./panoloc/scripts/outputs/2331de25-d580-7a65-f5cc-fc50f3f160e9/fused/cluster_0/group_0_c_{start_id}.png'
    # fig.savefig(path, dpi = 300)

    xys1_final = []
    conf1_final = []
    for i, u in enumerate(original_us):
        v = uvs0[i].y
        current_c = wall_conf0[i]
        for panoid_new in final_vs_all:
            if current_c > final_cs_all[panoid_new][i] and final_vs_all[panoid_new][i] != 0:
                v = final_vs_all[panoid_new][i]
                current_c = final_cs_all[panoid_new][i]
        xy1_final = transform_utils.uv_to_xy(Point2d(x=u, y=v), DEFAULT_CAMERA_HEIGHT)
        xys1_final.append(Point2d(x=xy1_final.x, y=xy1_final.y))
        if (
            i > 0
            and math.sqrt((xys1_final[i - 1].x - xy1_final.x) ** 2 + (xys1_final[i - 1].y - xy1_final.y) ** 2) > 0.03
        ):
            current_c = 0
        if (
            i < len(xys1_final) - 1
            and math.sqrt((xys1_final[i + 1].x - xy1_final.x) ** 2 + (xys1_final[i + 1].y - xy1_final.y) ** 2) > 0.03
        ):
            current_c = 0
        conf1_final.append(current_c)
    return xys1_final, conf1_final


def refine_predicted_shape(
    groups: Any,
    predicted_shapes: Any,
    wall_confidences: Any,
    location_panos: Any,
    cluster_dir: Any,
    tour_dir: Any = None,
) -> Tuple[Any, Any, Any]:
    """TODO

    Args:
        groups: TODO
        predicted_shapes: TODO
        wall_confidences: TODO
        location_panos: TODO
        cluster_dir: TODO
        tour_dir: TODO

    Returns:
        shape_fused_by_cluster: TODO
        fig2: TODO
        Cascaded union of ...
    """
    fig2 = Figure()
    axis2 = fig2.add_subplot(1, 2, 1)

    shape_fused_by_cluster = []
    shape_fused_by_cluster_poly = []
    pbar = tqdm(total=len(groups))

    # with open(os.path.join(tour_dir, 'colors.json')) as f:
    #     color_records = json.load(f)
    color_records = {}
    for i_group, group in enumerate(groups):
        shape_fused_by_group = []
        i_color = None
        for panoid in group:
            if panoid in color_records:
                i_color = color_records[panoid]
                break
        if i_color == None:
            i_color = ((8 - i_group) % 8) * 3 + int(i_group / 8)
            color = TANGO_COLOR_PALETTE[i_color % 24]
        else:
            color = TANGO_COLOR_PALETTE[(i_color) % 24]
        color = (color[0] / 255, color[1] / 255, color[2] / 255)
        # panoid = group[0]

        # fig1 = Figure()
        # axis1 = fig1.add_subplot(1, 1, 1)
        shapes = []
        for panoid in group:
            xys_fused, conf_fused = refine_shape_group_start_with(
                group, panoid, predicted_shapes, wall_confidences, location_panos
            )
            pose0 = location_panos[panoid]
            shape_fused_by_group.append([xys_fused, conf_fused, pose0])

            # path_output = os.path.join(cluster_dir, f'group_{i_group}_{panoid}.png')
            # fig = Figure()
            # axis = fig.add_subplot(1, 1, 1)

            xys_fused_transformed = []
            for xy in xys_fused:
                xys_fused_transformed.append(transform_utils.transform_xy_by_pose(xy, pose0))
            shapes.append(Polygon([[xy.x, xy.y] for xy in xys_fused_transformed]))

            # draw_shape_in_top_down_canvas(axis, xys_fused, 'black', pose=pose0)
            # draw_shape_in_top_down_canvas(axis1, xys_fused, 'black', pose=pose0)
            # if panoid == 'f6f605f86a':
            #     draw_camera_in_top_down_canvas(axis1, pose0, "blue", size=10)
            # else:
            #     draw_camera_in_top_down_canvas(axis1, pose0, "green", size=10)
            draw_shape_in_top_down_canvas_fill(axis2, xys_fused, color, pose=pose0)
            # axis.set_aspect("equal")
            # fig.savefig(path_output, dpi = 300)
        shape_fused_by_cluster.append(shape_fused_by_group)

        # axis1.set_aspect("equal")
        # path_output = os.path.join(cluster_dir, f'group_{i_group}.png')
        # fig1.savefig(path_output, dpi = 300)

        shape_fused_by_cluster_poly.append(cascaded_union(shapes))
        #
        # try:
        #     xys_final = extract_coordinates_from_shapely_polygon(shape_fused)
        #     draw_shape_in_top_down_canvas(axis2, xys_final, 'black')
        # except:
        #     print('union error')
        pbar.update(1)
    pbar.close()

    axis2.set_aspect("equal")
    path_output = os.path.join(cluster_dir, f"final.png")
    fig2.savefig(path_output, dpi=300)
    return shape_fused_by_cluster, fig2, cascaded_union(shape_fused_by_cluster_poly)
