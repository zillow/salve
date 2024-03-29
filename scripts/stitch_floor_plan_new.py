"""TODO new script written by John"""

import click
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import gtsfm.utils.io as io_utils
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
from matplotlib.figure import Figure
from shapely.geometry import Point, Polygon
from tqdm import tqdm

import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.dataset.salve_sfm_result_loader as salve_sfm_result_loader
import salve.stitching.shape as shape_utils
import salve.stitching.transform as transform_utils
import salve.utils.matplotlib_utils as matplotlib_utils
import salve.algorithms.room_merging as room_merging_algo
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.salve_sfm_result_loader import EstimatedBoundaryType
from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT
from salve.stitching.draw import (
    draw_camera_in_top_down_canvas,
    draw_shape_in_top_down_canvas,
    draw_shape_in_top_down_canvas_fill,
    TANGO_COLOR_PALETTE,
)
from salve.stitching.models.feature2d import Feature2dU, Feature2dXy
from salve.stitching.models.locations import Point2d


# arbitrary image height, to match HNet model inference resolution.
IMAGE_WIDTH_PX = 1024
IMAGE_HEIGHT_PX = 512

MIN_LAYOUT_OVERLAP_RATIO = 0.3
MIN_LAYOUT_OVERLAP_IOU = 0.1


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
    group: List[int],
    start_id: int,
    predicted_shapes: Dict[int, Polygon],
    wall_confidences: Dict[int, np.ndarray],
    est_pose_graph: PoseGraph2d,
) -> Tuple[Any, Any]:
    """Refine a room's shape using confidence.

    Args:
        group: panorama ID's that belong to one room.
        start_id: panorama ID to use as reference for this room, as i0
        predicted_shapes: Densely estimated layouts for all panos on this floor.
        wall_confidences: Confidences for all panos on this floor.
        est_pose_graph: Estimated poses for all panos on this floor.

    Returns:
        xys1_final: TODO
        conf1_final: TODO
    """
    RES = IMAGE_HEIGHT_PX
    # generate 512 image column indices from [start,stop).
    original_us = np.arange(start=0.5 / RES, stop=(RES + 0.5) / RES, step=1.0 / RES)
    i0 = start_id

    shape_i0 = predicted_shapes[i0]
    xys_0 = extract_coordinates_from_shapely_polygon(shape_i0)
    pose_i0 = est_pose_graph.nodes[i0].global_Sim2_local
    wall_conf0 = wall_confidences[i0]
    uvs0 = []
    for xy0 in xys0:
        uvs0.append(transform_utils.xy_to_uv(xy0, DEFAULT_CAMERA_HEIGHT))

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)

    final_vs_all = {}
    final_cs_all = {}
    colors = {"9c5d07356f": "red", "84fe28b6c1": "blue", "404ed9fcfb": "yellow", "1a2445a9fe": "purple"}
    for i1 in group:
        if i1 == i0:
            continue
        shape_i1 = predicted_shapes[i1]
        pose_i1 = est_pose_graph.nodes[i1].global_Sim2_local
        pose_i1 = location_panos[i1]
        wall_conf1 = wall_confidences[i1]

        xys1 = extract_coordinates_from_shapely_polygon(shape_i1)
        xys0 = extract_coordinates_from_shapely_polygon(shape_i0)
        xys1_projected = []
        uvs1_projected = []
        for xy1 in xys1:
            # Project contour i1 onto pano i0, to get (u,v) coordinates.
            xy1_transformed = transform_utils.transform_xy_by_pose(xy1, pose_i1)
            xy1_projected = transform_utils.project_xy_by_pose(xy1_transformed, pose_i0)
            xys1_projected.append(xy1_projected)
            uvs1_projected.append(transform_utils.xy_to_uv(xy1_projected, DEFAULT_CAMERA_HEIGHT))

        xys1_projected = [[xy.x, xy.y] for xy in xys1_projected]
        polygon_global = Polygon(xys1_projected)
        if not polygon_global.contains(Point(0, 0)):
            continue

        # In each image column of pano i, choose the most confident contour point from P???
        final_vs, final_cs = transform_utils.reproject_uvs_to(uvs1_projected, wall_conf1, i1, i0)

        final_vs_all[i1] = final_vs
        final_cs_all[i1] = final_cs

        xys_render = []
        for i, u in enumerate(original_us):
            xy_render = transform_utils.uv_to_xy(Point2d(x=u, y=final_vs[i]), DEFAULT_CAMERA_HEIGHT)
            xys_render.append(xy_render)

    #     color = colors[i1] if i1 in colors else 'red'
    #     draw_shape_in_top_down_canvas(
    #         axis, xys_render, color, pose=location_panos[start_id]
    #     )
    #     draw_camera_in_top_down_canvas(axis, location_panos[i1], color, size=20)
    # color = colors[start_id] if start_id in colors else 'red'
    # draw_shape_in_top_down_canvas(axis, xys0, color, pose=location_panos[start_id])
    # draw_camera_in_top_down_canvas(axis, location_panos[start_id], color, size=20)
    # axis.set_aspect("equal")
    # path = f'./panoloc/scripts/outputs/2331de25-d580-7a65-f5cc-fc50f3f160e9/fused/cluster_0/group_0_c_{start_id}.png'
    # fig.savefig(path, dpi = 300)

    # Convert most confident column to world metric space.
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
    predicted_shapes: List[Polygon],
    wall_confidences: np.ndarray,
    est_pose_graph: PoseGraph2d,
    cluster_dir: Path,
    tour_dir: Path = None,
) -> Tuple[Any, Any, Any]:
    """Refine the predicted room shapes of each room (group) of a single building's floorplan.

    Args:
        groups: list of connected components. Each connected components represents a single room,
           and consists of the corresponding panorama IDs.
        predicted_shapes: TODO
        wall_confidences: TODO
        est_pose_graph: estimated panorama poses with HorizonNet's densely estimated boundary predictions.
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
        # TODO (yuguangl): explain this logic
        for panoid in group:
            if panoid in color_records:
                i_color = color_records[panoid]
                break
        if i_color == None:
            i_color = ((8 - i_group) % 8) * 3 + int(i_group / 8)
            color = TANGO_COLOR_PALETTE[i_color % 24]
        else:
            color = TANGO_COLOR_PALETTE[i_color % 24]
        color = (color[0] / 255, color[1] / 255, color[2] / 255)

        shapes = []
        for panoid in group:
            xys_fused, conf_fused = refine_shape_group_start_with(
                group=group,
                start_id=panoid,
                predicted_shapes=predicted_shapes,
                wall_confidences=wall_confidences,
                est_pose_graph=est_pose_graph,
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


def stitch_building_layouts(
    building_id: str, hnet_pred_dir: Path, raw_dataset_dir: str, est_localization_fpath: Path, output_dir: Path
) -> None:
    """TODO

    Args:
        building_id:
        hnet_pred_dir:
        raw_dataset_dir:
        est_localization_fpath:
        output_dir:
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    cluster_dir = os.path.join(output_dir, "fused")
    Path(cluster_dir).mkdir(exist_ok=True, parents=True)

    hnet_floor_predictions = hnet_prediction_loader.load_hnet_predictions(
        query_building_id=building_id, raw_dataset_dir=raw_dataset_dir, predictions_data_root=hnet_pred_dir
    )
    # Room corner vertices will be used for room layout merging.
    est_pose_graph_corners = salve_sfm_result_loader.load_estimated_pose_graph(
        json_fpath=Path(est_localization_fpath),
        boundary_type=EstimatedBoundaryType.HNET_CORNERS,
        raw_dataset_dir=raw_dataset_dir,
        predictions_data_root=hnet_pred_dir,
    )

    est_pose_graph_dense_boundary = salve_sfm_result_loader.load_estimated_pose_graph(
        json_fpath=Path(est_localization_fpath),
        boundary_type=EstimatedBoundaryType.HNET_DENSE,
        raw_dataset_dir=raw_dataset_dir,
        predictions_data_root=hnet_pred_dir,
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

        groups = room_merging_algo.group_panos_by_room(est_pose_graph=est_pose_graph_corners)

        print("Running shape refinement ... ")
        import pdb

        pdb.set_trace()
        floor_shape_final, figure, floor_shape_fused_poly = refine_predicted_shape(
            groups=groups,
            predicted_shapes=predicted_shapes_raw,
            wall_confidences=wall_confidences,
            est_pose_graph=est_pose_graph_dense_boundary,
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
    """Click entry point for ...TODO

    Example usage:
    python scripts/stitch_floor_plan_new.py --output-dir 2022_07_01_stitching_output --est-localization-fpath 2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23___2022_02_01_pgo_floorplans_with_conf_0.93_door_window_opening_axisalignedTrue_serialized/0715__floor_01.json --hnet-pred-dir /srv/scratch/jlambert30/salve/zind2_john --raw_dataset_dir /srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05

    Improved localization:
    python scripts/stitch_floor_plan_new.py --output-dir 2022_07_05_stitching_output --est-localization-fpath 2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16___2022_07_05_pgo_floorplans_with_conf_0.93_door_window_opening_axisalignedTrue_serialized/0715__floor_01.json --hnet-pred-dir /srv/scratch/jlambert30/salve/zind2_john --raw_dataset_dir /srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05
    """
    building_ids = ["0715"]
    for building_id in building_ids:
        stitch_building_layouts(
            building_id=building_id,
            hnet_pred_dir=Path(hnet_pred_dir),
            raw_dataset_dir=str(raw_dataset_dir),
            est_localization_fpath=Path(est_localization_fpath),
            output_dir=Path(output_dir),
        )


if __name__ == "__main__":
    run_stitch_building_layouts()
