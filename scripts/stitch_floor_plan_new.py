"""TODO new script written by John"""

import click
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import gtsfm.utils.io as io_utils
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.dataset.salve_sfm_result_loader as salve_sfm_result_loader
import salve.stitching.shape as shape_utils
import salve.stitching.transform as transform_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.salve_sfm_result_loader import EstimatedBoundaryType
from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT


# arbitrary image height, to match HNet model inference resolution.
IMAGE_WIDTH_PX = 1024
IMAGE_HEIGHT_PX = 512

MIN_LAYOUT_OVERLAP_RATIO = 0.3
MIN_LAYOUT_OVERLAP_IOU = 0.1



import json
from typing import Any, Dict, List, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.ops
from shapely.geometry import Point, Polygon
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
    """Convert a shapely Polygon object to a list of Point2d objects.

    Args:
        shape:

    Returns:
        coords: list of Point2d objects
    """
    coords = []
    xys = shape.boundary.xy
    for i in range(len(xys[0])):
        coords.append(Point2d(x=xys[0][i], y=xys[1][i]))
    return coords



def refine_shape_group_start_with(
    group: Any, start_id: Any, predicted_shapes: Any, wall_confidences: Any, location_panos: Any
) -> Tuple[Any, Any]:
    """Refine a room's shape using confidence.

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
    RES = IMAGE_HEIGHT_PX
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
        
        if (i > 0) and (xys1_final[i - 1].distance(xy1_final) > 0.03):
            current_c = 0
        if (i < len(xys1_final) - 1) and (xys1_final[i + 1].distance(xy1_final) > 0.03):
            current_c = 0
        conf1_final.append(current_c)
    return xys1_final, conf1_final


def refine_predicted_shape(
    groups: List[List[int]],
    predicted_shapes: Any,
    wall_confidences: Any,
    location_panos: Any,
    cluster_dir: Any,
    tour_dir: Any = None,
) -> Tuple[Any, Any, Any]:
    """Refine the predicted room shapes of each room (group) of a single building's floorplan.

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

        shape_fused_by_cluster_poly.append(shapely.ops.cascaded_union(shapes))
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
    return shape_fused_by_cluster, fig2, shapely.ops.cascaded_union(shape_fused_by_cluster_poly)


def load_room_shape_polygon_from_predictions(
    room_shape_pred: Iterable[Tuple[float, float]],
    uncertainty: Optional[Iterable[Tuple[float, float]]] = None,
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
) -> Union[Tuple[Polygon, Polygon], Polygon]:
    """TODO

    We subtract the uncertainty from v coordinates, moving the uv coordinates higher in each panorama,
    effectively extending the room at such coordinates.

    Args:
        room_shape_pred: corners or dense points along room boundary.
        uncertainty:
        camera_height:

    Returns:
        Polygon or tuple of two polygons, representing (N,2) coordinates in metric world coordinate system?
            polygon, and then potentially extended version of that polygon, according to uncertainty.
    """
    xys = []
    xys_upper = []

    uvs = []
    uvs_upper = []

    for i, (corner_u, corner_v) in enumerate(room_shape_pred[1::2]):
        uvs.append([corner_u + 0.5 / IMAGE_WIDTH_PX, corner_v + 0.5 / IMAGE_HEIGHT_PX])
        if uncertainty is not None:
            uvs_upper.append(
                [
                    corner_u + 0.5 / IMAGE_WIDTH_PX,
                    corner_v + 0.5 / IMAGE_HEIGHT_PX - uncertainty[i] / IMAGE_HEIGHT_PX,
                ]
            )

    xys = transform_utils.uv_to_xy_batch(uvs, camera_height)
    if uncertainty is not None:
        xys_upper = transform_utils.uv_to_xy_batch(uvs_upper, camera_height)
        return Polygon(xys), Polygon(xys_upper)
    return Polygon(xys)


def generate_dense_shape(v_vals: Iterable[float], uncertainty: Iterable[float]) -> Tuple[Polygon, np.ndarray]:
    """Use extended version of room (according to uncertainty) to determine distances by which each is extended.

    Args:
        v_vals: floor boundary "v" coordinates in pixels in the range [0, height], for each of 1024 image columns.
        uncertainty: floor boundary uncertainty, for each of 1024 image columns.

    Returns:
        polygon: coordinates subsampled in world metric coordinates (every 2nd coordinate), as shapely Polygon object.
        distances: distance from estimated boundary to extended boundary (according to uncertainty).
    """
    vs = np.asarray(v_vals) / IMAGE_HEIGHT_PX
    us = np.asarray(range(IMAGE_WIDTH_PX)) / IMAGE_WIDTH_PX
    uvs = [[us[i], vs[i]] for i in range(IMAGE_WIDTH_PX)]
    polygon, poly_upper = load_room_shape_polygon_from_predictions(uvs, uncertainty)

    x_room, y_room = polygon.boundary.xy
    x_extended, y_extended = poly_upper.boundary.xy
    distances = np.hypot(np.array(x_room) - np.array(x_extended), np.array(y_room) - np.array(y_extended))
    return polygon, distances


def group_panos_by_room(predictions: List[Polygon], est_pose_graph: PoseGraph2d) -> List[List[int]]:
    """Form per-room clusters of panoramas according to layout IoU and other overlap measures.
    Layouts that have high IoU, or have high intersection with either shape, are considered to belong to a single room.
    Args:
        predictions: room polygons in world metric system (using corner predictions, instead of dense boundary
            predictions).
        est_pose_graph: estimated 2d pose graph (as a result of executing SALVe's `run_sfm.py`).

    Returns:
        groups: list of connected components. Each connected component is represented
            by a list of panorama IDs.
    """
    import matplotlib.pyplot as plt

    print("Running pano grouping by room ... ")
    shapes_global = {}
    graph = nx.Graph()
    for pano_id in est_pose_graph.pano_ids():
        # pose = location_panos[panoid]
        # shape = predictions[panoid]
        # xys_transformed = []
        # xys = extract_coordinates_from_shapely_polygon(shape)
        # for xy in xys:
        #     xys_transformed.append(transform_utils.transform_xy_by_pose(xy, pose))
        # shape_global = Polygon([[xy.x, xy.y] for xy in xys_transformed])

        shapes_global[pano_id] = est_pose_graph.nodes[pano_id].room_vertices_global_2d
        graph.add_node(pano_id)

        plt.scatter(
            est_pose_graph.nodes[pano_id].room_vertices_global_2d[:,0],
            est_pose_graph.nodes[pano_id].room_vertices_global_2d[:,1],
            size=10,
            color=np.random.rand(3),
            marker="."
        )
    plt.axis("equal")
    plt.savefig("714.jpg", dpi=500)

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
            overlap_ratio1 = area_intersection / shape1.area
            overlap_ratio2 = area_intersection / shape2.area
            if (
                iou > MIN_LAYOUT_OVERLAP_IOU
                or overlap_ratio1 > MIN_LAYOUT_OVERLAP_RATIO
                or overlap_ratio2 > MIN_LAYOUT_OVERLAP_RATIO
            ):
                graph.add_edge(panoid1, panoid2)
    groups = [[*c] for c in sorted(nx.connected_components(graph))]
    return groups


def stitch_building_layouts(
    hnet_pred_dir: Path, raw_dataset_dir: str, est_localization_fpath: Path, output_dir: Path
) -> None:
    """ """
    building_id = "0715"

    output_dir.mkdir(exist_ok=True, parents=True)
    cluster_dir = os.path.join(output_dir, "fused")
    Path(cluster_dir).mkdir(exist_ok=True, parents=True)

    hnet_floor_predictions = hnet_prediction_loader.load_hnet_predictions(
        query_building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=hnet_pred_dir
    )
    est_pose_graph = salve_sfm_result_loader.load_estimated_pose_graph(
        json_fpath=Path(est_localization_fpath),
        boundary_type=EstimatedBoundaryType.HNET_CORNERS,
        raw_dataset_dir=raw_dataset_dir,
        predictions_data_root=hnet_pred_dir
    )

    # convert each pose in the JSON file to a Sim2 object.
    wall_confidences = {}
    predicted_shapes_raw = {}
    predicted_corner_shapes = {}

    for floor_id, floor_predictions in hnet_floor_predictions.items():

        if floor_id != "floor_01":
            continue

        for pano_id in floor_predictions.keys():

            # get the ceiling corners
            predicted_corner_shapes[pano_id] = load_room_shape_polygon_from_predictions(
                room_shape_pred=floor_predictions[pano_id].corners_in_uv
            )

            wall_confidences[pano_id] = floor_predictions[pano_id].floor_boundary_uncertainty
            # confidences are transformed from uv space to world metric space.
            predicted_shapes_raw[pano_id], wall_confidences[pano_id] = generate_dense_shape(
                v_vals=floor_predictions[pano_id].floor_boundary, uncertainty=wall_confidences[pano_id]
            )

        import pdb; pdb.set_trace()
        groups = group_panos_by_room(predicted_corner_shapes, est_pose_graph=est_pose_graph)

        print("Running shape refinement ... ")
        floor_shape_final, figure, floor_shape_fused_poly = refine_predicted_shape(
            groups=groups,
            predicted_shapes=predicted_shapes_raw,
            wall_confidences=wall_confidences,
            location_panos=location_panos,
            cluster_dir=cluster_dir,
            tour_dir=output_dir,
        )


@click.command(help="Script to run floorplan stitching algorithm, using previously localized poses.")
@click.option(
    "--raw_dataset_dir",
    type=click.Path(exists=True),
    required=True,
    # "/mnt/data/johnlam/zind_bridgeapi_2021_10_05"
    # "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05"
    # default="/home/johnlam/zind_bridgeapi_2021_10_05",
    help="where ZInD dataset is stored on disk (after download from Bridge API)",
)
@click.option(
    "--est-localization-fpath",
    required=True,
    help="Path to JSON file containing estimated panorama poses in a cluster, generated by SALVe + global optimization.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to directory where stitched outputs will be saved to.",
    type=str,  #
)
@click.option(
    "--hnet-pred-dir",
    required=True,
    help="Directory to where HorizonNet per-pano room shape and DWO predictions are stored.",
    type=click.Path(exists=True),
)
def run_stitch_building_layouts(
    raw_dataset_dir: str, est_localization_fpath: str, output_dir: str, hnet_pred_dir: str
) -> None:
    """Click entry point for ...


    Example usage:
    python scripts/stitch_floor_plan_new.py --output-dir 2022_07_01_stitching_output --est-localization-fpath 2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23___2022_02_01_pgo_floorplans_with_conf_0.93_door_window_opening_axisalignedTrue_serialized/0715__floor_01.json --hnet-pred-dir /srv/scratch/jlambert30/salve/zind2_john --raw_dataset_dir /srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05
    """
    stitch_building_layouts(
        hnet_pred_dir=Path(hnet_pred_dir),
        raw_dataset_dir=str(raw_dataset_dir),
        est_localization_fpath=Path(est_localization_fpath),
        output_dir=Path(output_dir),
    )


if __name__ == "__main__":
    run_stitch_building_layouts()
