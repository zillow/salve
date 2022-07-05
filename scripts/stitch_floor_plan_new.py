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

import salve.stitching.shape as shape_utils
import salve.stitching.transform as transform_utils
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT
import salve.stitching.shape as shape_utils

# arbitrary image height?
IMAGE_WIDTH_PX = 1024
IMAGE_HEIGHT_PX = 512

MIN_LAYOUT_OVERLAP_RATIO = 0.3
MIN_LAYOUT_OVERLAP_IOU = 0.1


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
        polygon: coordinates subsampled (every 2nd), as shapely Polygon object.
        distances: distance from estimated boundary to extended boundary (according to uncertainty).
    """
    vs = np.asarray(v_vals) / IMAGE_HEIGHT_PX
    us = np.asarray(range(IMAGE_WIDTH_PX)) / IMAGE_WIDTH_PX
    uvs = [[us[i], vs[i]] for i in range(IMAGE_WIDTH_PX)]
    polygon, poly_upper = load_room_shape_polygon_from_predictions(uvs, uncertainty)
    distances = []
    x_room, y_room = polygon.boundary.xy
    x_extended, y_extended = poly_upper.boundary.xy
    distances = np.hypot(np.array(x_room) - np.array(x_extended), np.array(y_room) - np.array(y_extended))
    return polygon, distances


def group_panos_by_room(predictions: Any, location_panos: Any) -> List[List[int]]:
    """Form per-room clusters of panoramas according to layout IoU and other overlap measures.
    Layouts that have high IoU, or have high intersection with either shape, are considered to belong to a single room.
    Args:
        predictions:
        location_panos:

    Returns:
        groups: list of connected components. Each connected component is represented
            by a list of panorama IDs.
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

    localizations = io_utils.read_json_file(est_localization_fpath)

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
        groups = shape_utils.group_panos_by_room(predicted_corner_shapes, location_panos)

        print("Running shape refinement ... ")
        floor_shape_final, figure, floor_shape_fused_poly = shape_utils.refine_predicted_shape(
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
