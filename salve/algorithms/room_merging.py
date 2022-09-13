"""Merge rooms together based on layout overlap ratios."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib.figure import Figure

import salve.utils.matplotlib_utils as matplotlib_utils
from salve.common.posegraph2d import PoseGraph2d


# arbitrary image height, to match HNet model inference resolution.
IMAGE_WIDTH_PX = 1024
IMAGE_HEIGHT_PX = 512

MIN_LAYOUT_OVERLAP_RATIO = 0.3
MIN_LAYOUT_OVERLAP_IOU = 0.1


def group_panos_by_room(est_pose_graph: PoseGraph2d, visualize: bool = True) -> List[List[int]]:
    """Form per-room clusters of panoramas according to layout IoU and other overlap measures.

    Layouts that have high IoU, or have high intersection with either shape, are considered to belong to a single room.

    Args:
        est_pose_graph: estimated 2d pose graph (as a result of executing SALVe's `run_sfm.py`) containing
            estimated room polygons in world metric system (using corner predictions, instead of dense boundary
            predictions).
        visualize: whether to save visualizations to disk.

    Returns:
        groups: list of connected components. Each connected component is represented
            by a list of panorama IDs.
    """
    if visualize:
        plt.close("all")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

    pano_ids = est_pose_graph.pano_ids()

    print("Running pano grouping by room ... ")
    shapes_global = {}
    graph = nx.Graph()
    for pano_id in pano_ids:
        shapes_global[pano_id] = Polygon(est_pose_graph.nodes[pano_id].room_vertices_global_2d)
        graph.add_node(pano_id)

        if visualize:
            color = np.random.rand(3)
            x, y = np.mean(est_pose_graph.nodes[pano_id].room_vertices_global_2d, axis=0)
            ax.text(x, y, str(pano_id))
            matplotlib_utils.plot_polygon_patch_mpl(
                polygon_pts=est_pose_graph.nodes[pano_id].room_vertices_global_2d,
                ax=ax,
                color=color,
                alpha=0.2,
            )

    if visualize:
        plt.axis("equal")
        Path("cc_visualizations").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"cc_visualizations/{est_pose_graph.building_id}_{est_pose_graph.floor_id}.jpg", dpi=500)
        plt.close("all")

    for i in range(len(pano_ids)):
        for j in range(i, len(pano_ids)):
            panoid1 = pano_ids[i]
            panoid2 = pano_ids[j]
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
    print("Connected components: ", groups)
    return groups
