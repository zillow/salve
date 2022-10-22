"""Utilities for drawing graph topology, either for a multi-graph, or for a typical undirected graph."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.utils.colormap as colormap_utils
from salve.algorithms.cycle_consistency import TwoViewEstimationReport
from salve.common.edge_classification import EdgeClassification
from salve.common.posegraph2d import PoseGraph2d


# Colors that can be used for coloring nodes or edges.
PLUM1 = [173, 127, 168]
SKYBLUE = [135, 206, 250]
GREEN = [0, 140, 25]
CYAN = [0, 255, 255]


def generate_edge_colors_from_error_magnitudes(
    edges: List[Tuple[int, int]],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    max_saturated_error: float = 20.0,
    num_error_bins: int = 20,
) -> None:
    """Generate edge colors according to rotation error magnitude (quantize and clip, saturating at threshold).

    Args:
        edges
        two_view_reports_dict: dictionary containing R and t errors per pano-pano edge (i1,i2). E edges are given.

    Returns:
        edge_colors: array of shape (E,3) with values in [0,1]. Green is zero, and red means error saturated to
            threshold.
    """
    bin_edges = np.linspace(0, max_saturated_error, num_error_bins + 1)
    # Normalize RGB values to [0,1] range.
    colormap_by_magnitudes = colormap_utils.get_redgreen_colormap(N=num_error_bins) / 255
    # Reverse it so that green is lowest, and red is highest.
    colormap_by_magnitudes = colormap_by_magnitudes[::-1]
    R_errors = [two_view_reports_dict[(i1, i2)].R_error_deg for (i1, i2) in edges]
    # Quantize edge errors, and convert to zero-indexed.
    error_bins = np.digitize(R_errors, bins=bin_edges) - 1
    # Clamp values to [0,num_error_bins-1], because outliers appear in an extra bin outside interval.
    error_bins_clipped = np.clip(error_bins, a_min=0, a_max=num_error_bins - 1)
    edge_colors = colormap_by_magnitudes[error_bins_clipped]
    return edge_colors


def draw_graph_topology(
    edges: List[Tuple[int, int]],
    gt_floor_pose_graph: PoseGraph2d,
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    title: str,
    show_plot: bool = True,
    save_fpath: Optional[str] = None,
    color_scheme: str = "by_error_magnitude",
) -> None:
    """Draw the topology of an undirected graph, with vertices placed in their ground truth locations.

    False positive edges are colored red, and true positive edges are colored green.

    Args:
        edges: List of (i1,i2) pairs.
        gt_floor_pose_graph: ground truth 2d pose graph for this floor.
        two_view_reports_dict: dictionary containing R and t errors per pano-pano edge (i1,i2). E edges are given.
        title: desired title of figure.
        show_plot: whether to show the plot in the matplotlib GUI.
        save_fpath: file path where plot should be saved to disk (if None, will not be saved to disk).
        color_scheme: color scheme for edges "by_error_magnitude", "by_tp_fp"
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = list(G.nodes)
    GREEN = [0, 1, 0]
    RED = [1, 0, 0]

    if color_scheme == "by_tp_fp":
        edge_colors = [GREEN if two_view_reports_dict[edge].gt_class == 1 else RED for edge in edges]

    elif color_scheme == "by_error_magnitude":
        edge_colors = generate_edge_colors_from_error_magnitudes(edges=edges, two_view_reports_dict=two_view_reports_dict)

    plt.title(title)

    nx.drawing.nx_pylab.draw_networkx(
        G,
        edgelist=edges,
        edge_color=edge_colors,
        pos={v: gt_floor_pose_graph.nodes[v].global_Sim2_local.translation for v in nodes},
        arrows=True,
        with_labels=True,
    )
    # Show x and y axes for scale.
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal', adjustable='box')

    #plt.axis("equal")

    if save_fpath is not None:
        Path(save_fpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fpath, dpi=500)

    if show_plot:
        plt.show()
    plt.close("all")


def draw_multigraph(
    measurements: List[EdgeClassification],
    gt_floor_pose_graph: PoseGraph2d,
    inferred_floor_pose_graph: Optional[PoseGraph2d],
    use_gt_positions: bool,
    confidence_threshold: float = 0.5,
    save_dir: str = "./",
) -> None:
    """Draw the topology of a pose graph, with colored nodes and colored edges (allowed edges are conf.-thresholded).

    If the input pose graph is an estimated pose graph, some cameras may not be localized. In this case, we may
    want to render the remaining nodes at the GT locations.

    Args:
        measurements: possible relative pose hypotheses, before confidence thresholding is applied.
        gt_floor_pose_graph: poses where graph nodes will be plotted.
        inferred_floor_pose_graph
        use_gt_positions: whether to plot panos at GT poses.
        confidence_threshold: minimum required predicted confidence by model to plot an edge.
        save_dir: subdir to experiment directory, where visualization plots should be saved.
    """
    edges = []

    G = nx.MultiGraph()

    for m in measurements:
        # Find all of the predictions where pred class is 1.
        if m.y_hat != 1:
            continue

        if m.prob < confidence_threshold:
            continue

        # TODO: this should never happen bc sorted, figure out why it occurs
        if m.i1 >= m.i2:
            i2 = m.i1
            i1 = m.i2

            m.i1 = i1
            m.i2 = i2

        # EDGE_COLOR = SKYBLUE
        EDGE_COLOR = CYAN

        edge_color = [v / 255 for v in EDGE_COLOR]  # "g" if m.y_true == 1 else "r"
        weight = m.prob * 1.5
        # weight = 5 if m.y_true == 1 else 1
        G.add_edge(m.i1, m.i2, color=edge_color, weight=weight)

    NODE_COLOR = [v / 255 for v in GREEN]
    node_color_map = [NODE_COLOR for node in G]
    node_sizes = [15 for node in G]

    edges = G.edges()

    colors = []
    weight = []

    for (u, v, attrib_dict) in list(G.edges.data()):
        colors.append(attrib_dict["color"])
        weight.append(attrib_dict["weight"])

    nodes = list(G.nodes)
    node_positions = {}
    for v in nodes:
        if use_gt_positions and v in gt_floor_pose_graph.nodes:
            # Use ground truth pano pose.
            pos = gt_floor_pose_graph.nodes[v].global_Sim2_local.translation

        elif use_gt_positions and v not in gt_floor_pose_graph.nodes:
            raise ValueErorr(f"No ground truth pose exists for pano {v}.")

        elif not use_gt_positions and v in inferred_floor_pose_graph.nodes:
            # Use estimated pano pose.
            pos = inferred_floor_pose_graph[v].global_Sim2_local.translation

        elif not use_gt_positions and v not in inferred_floor_pose_graph.nodes:
            # Fall back to GT position if pano pose was not estimated by SfM.
            pos = gt_floor_pose_graph.nodes[v].global_Sim2_local.translation

        node_positions[v] = pos

    nx.drawing.nx_pylab.draw_networkx(
        G=G,
        pos=node_positions,
        edgelist=edges,
        edge_color=colors,
        width=weight,
        with_labels=False,
        node_color=node_color_map,
        node_size=node_sizes,
    )
    building_id = gt_floor_pose_graph.building_id
    floor_id = gt_floor_pose_graph.floor_id

    plt.axis("equal")
    # plt.show()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_fpath = os.path.join(save_dir, f"multigraph_{building_id}_{floor_id}.pdf")
    plt.savefig(save_fpath, dpi=500)
    plt.close("all")
